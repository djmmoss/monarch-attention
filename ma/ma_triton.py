import os
from math import sqrt

DEBUG = False

if DEBUG:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TRITON_INTERPRET"] = "1"

import torch
import triton
import triton.language as tl


# TMA allocator setup (required for TMA descriptor operations)
_tma_allocator_set = False


def _ensure_tma_allocator():
    """Set up TMA allocator if not already done."""
    global _tma_allocator_set
    if not _tma_allocator_set:
        def torch_allocator(size, align, stream):
            return torch.empty(size, dtype=torch.uint8, device='cuda').data_ptr()
        triton.set_allocator(torch_allocator)
        _tma_allocator_set = True


def check_inputs(q, k, v):
    pass


Tensor = torch.Tensor


# =============================================================================
# Autotuning configurations for tiled kernels (large-M / large-B hot path)
# Non-tiled kernels use fixed heuristics since they only run for small dims.
# =============================================================================

def _get_tiled_m_autotune_configs():
    """Configs for kernels with autotuned TILE_M (e.g., _z_kernel_tiled)."""
    configs = []
    for tile_m in [32, 64, 128]:
        for num_warps in [4, 8]:
            for num_stages in [2, 3, 4]:
                configs.append(triton.Config(
                    {'TILE_M': tile_m}, num_warps=num_warps, num_stages=num_stages,
                ))
    return configs


def _get_tiled_b_autotune_configs():
    """Configs for kernels with autotuned TILE_B (e.g., _al_cl_kernel_tiled)."""
    configs = []
    for tile_b in [32, 64, 128]:
        for num_warps in [4, 8]:
            for num_stages in [2, 3, 4]:
                configs.append(triton.Config(
                    {'TILE_B': tile_b}, num_warps=num_warps, num_stages=num_stages,
                ))
    return configs


def _get_warp_stage_autotune_configs():
    """Configs for kernels where only num_warps/num_stages are tuned."""
    configs = []
    for num_warps in [4, 8]:
        for num_stages in [2, 3, 4]:
            configs.append(triton.Config(
                {}, num_warps=num_warps, num_stages=num_stages,
            ))
    return configs


@triton.jit
def xlogx(x):
    return tl.where(x == 0, 0.0, x * tl.log(x))


@triton.jit
def _al_cl_kernel(
    ar_ptr,
    stride_ar_e,
    stride_ar_h,
    stride_ar_m,
    stride_ar_b,
    stride_ar_d,
    k_ptr,
    stride_k_e,
    stride_k_h,
    stride_k_m,
    stride_k_b,
    stride_k_d,
    cr_ptr,
    stride_cr_e,
    stride_cr_h,
    stride_cr_m,
    stride_cr_b,
    al_ptr,
    stride_al_e,
    stride_al_h,
    stride_al_m,
    stride_al_b,
    stride_al_d,
    cl_ptr,
    stride_cl_e,
    stride_cl_h,
    stride_cl_m,
    stride_cl_b,
    mask_ptr,
    stride_mask_e,
    stride_mask_m,
    stride_mask_b,
    sm_scale: float,
    HAS_ATTN_MASK: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PRE_PAD: tl.constexpr,
    EPS: tl.constexpr,
    IS_FIRST_CALL: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
):
    # 2D grid: (E*H, M) for better workload distribution
    idx_eh = tl.program_id(0)
    idx_m = tl.program_id(1)
    idx_e = idx_eh // H
    idx_h = idx_eh % H

    # Pre-compute constants (PRE_PAD is constexpr, so this is compile-time)
    pad_offset = M * B - N if PRE_PAD else 0
    block_start_n = B * idx_m  # Starting position in sequence

    range_b = tl.arange(0, BLOCK_B)
    range_d = tl.arange(0, BLOCK_D)

    # Simplified masks - avoid redundant comparisons when block matches dimension
    mask_b = range_b < B
    mask_d = range_d < D

    # k_mask_b: valid positions that are within the actual sequence
    # For PRE_PAD: positions where (block_start_n + range_b) >= pad_offset
    # For POST_PAD: positions where (block_start_n + range_b) < N
    if PRE_PAD:
        k_mask_b = mask_b & ((block_start_n + range_b) >= pad_offset)
    else:
        k_mask_b = mask_b & ((block_start_n + range_b) < N)

    # Pre-compute base pointers (reduces repeated arithmetic)
    # Common offset for tensors indexed by [e, h, m, ...]
    base_ehm = stride_ar_e * idx_e + stride_ar_h * idx_h + stride_ar_m * idx_m

    if HAS_ATTN_MASK:
        mask_base = mask_ptr + stride_mask_e * idx_e + stride_mask_m * idx_m
        mask_block_ptr = mask_base + stride_mask_b * (range_b - pad_offset)
        valid_token_mask = tl.load(mask_block_ptr, mask=k_mask_b, other=0)
        k_mask_b = k_mask_b & valid_token_mask

    # Pre-compute 2D offset pattern for [BLOCK_B, BLOCK_D] loads
    # Assumes stride_d = 1 (contiguous tensors) for coalesced access
    offset_2d = stride_ar_b * range_b[:, None] + range_d[None, :]

    # Load ar - use pre-computed base and offset
    ar_base = ar_ptr + base_ehm
    ar_offset = stride_ar_b * (range_b - (pad_offset if IS_FIRST_CALL else 0))[:, None] + range_d[None, :]
    ar = tl.load(
        ar_base + ar_offset,
        mask=(k_mask_b if IS_FIRST_CALL else mask_b)[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Load k - similar pattern with k's strides
    k_base = k_ptr + stride_k_e * idx_e + stride_k_h * idx_h + stride_k_m * idx_m
    k_offset = stride_k_b * (range_b - pad_offset)[:, None] + range_d[None, :]
    k = tl.load(
        k_base + k_offset,
        mask=k_mask_b[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Load cr - 1D load
    cr_base = cr_ptr + stride_cr_e * idx_e + stride_cr_h * idx_h + stride_cr_m * idx_m
    cr = tl.load(cr_base + stride_cr_b * range_b, mask=mask_b, other=1.0)

    # Attention matrix - use bf16 inputs for tensor cores, fp32 accumulator
    ar_bf16 = ar.to(tl.bfloat16)
    k_bf16 = k.to(tl.bfloat16)
    r = sm_scale * tl.dot(ar_bf16, tl.trans(k_bf16), out_dtype=tl.float32)
    r = r / (cr[:, None] + EPS)
    r = r + tl.where(k_mask_b[None, :], 0.0, float("-inf"))
    r = tl.exp(r - tl.clamp(tl.max(r, axis=1, keep_dims=True), EPS, float("inf")))
    r = r / (tl.sum(r, axis=1, keep_dims=True) + EPS)

    # Store cl - use pre-computed base (same pattern as cr)
    cl = tl.sum(xlogx(r), axis=1)
    cl_base = cl_ptr + stride_cl_e * idx_e + stride_cl_h * idx_h + stride_cl_m * idx_m
    tl.store(cl_base + stride_cl_b * range_b, cl, mask=mask_b)

    # Store al - use bf16 for tensor cores, cast back to input dtype
    al = (sm_scale * tl.dot(r.to(tl.bfloat16), k_bf16, out_dtype=tl.float32)).to(ar.dtype)
    al_base = al_ptr + stride_al_e * idx_e + stride_al_h * idx_h + stride_al_m * idx_m
    al_offset = stride_al_b * range_b[:, None] + range_d[None, :]  # Assumes stride_al_d = 1
    tl.store(al_base + al_offset, al, mask=mask_b[:, None] & mask_d[None, :])


@triton.autotune(configs=_get_tiled_b_autotune_configs(), key=['B', 'D'])
@triton.jit
def _al_cl_kernel_tiled(
    ar_ptr,
    stride_ar_e,
    stride_ar_h,
    stride_ar_m,
    stride_ar_b,
    stride_ar_d,
    k_ptr,
    stride_k_e,
    stride_k_h,
    stride_k_m,
    stride_k_b,
    stride_k_d,
    cr_ptr,
    stride_cr_e,
    stride_cr_h,
    stride_cr_m,
    stride_cr_b,
    al_ptr,
    stride_al_e,
    stride_al_h,
    stride_al_m,
    stride_al_b,
    stride_al_d,
    cl_ptr,
    stride_cl_e,
    stride_cl_h,
    stride_cl_m,
    stride_cl_b,
    mask_ptr,
    stride_mask_e,
    stride_mask_m,
    stride_mask_b,
    sm_scale: float,
    HAS_ATTN_MASK: tl.constexpr,
    TILE_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PRE_PAD: tl.constexpr,
    EPS: tl.constexpr,
    IS_FIRST_CALL: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
):
    """Tiled version of _al_cl_kernel for large B (>128).

    Uses online softmax to tile over key positions, avoiding B×B attention matrices.
    Grid: (E*H, M, cdiv(B, TILE_B))
    """
    idx_eh = tl.program_id(0)
    idx_m = tl.program_id(1)
    idx_q_tile = tl.program_id(2)
    idx_e = idx_eh // H
    idx_h = idx_eh % H

    pad_offset = M * B - N if PRE_PAD else 0
    block_start_n = B * idx_m

    # Query tile range
    q_start = idx_q_tile * TILE_B
    range_q = q_start + tl.arange(0, TILE_B)
    range_d = tl.arange(0, BLOCK_D)
    mask_q = range_q < B
    mask_d = range_d < D

    # Load ar (query rows for this tile)
    ar_base = ar_ptr + stride_ar_e * idx_e + stride_ar_h * idx_h + stride_ar_m * idx_m
    if IS_FIRST_CALL:
        ar_offset = stride_ar_b * (range_q - pad_offset)[:, None] + range_d[None, :]
        if PRE_PAD:
            q_load_mask = mask_q & ((block_start_n + range_q) >= pad_offset)
        else:
            q_load_mask = mask_q & ((block_start_n + range_q) < N)
        ar = tl.load(ar_base + ar_offset, mask=q_load_mask[:, None] & mask_d[None, :], other=0.0)
    else:
        ar_offset = stride_ar_b * range_q[:, None] + range_d[None, :]
        ar = tl.load(ar_base + ar_offset, mask=mask_q[:, None] & mask_d[None, :], other=0.0)
    ar_bf16 = ar.to(tl.bfloat16)

    # Load cr for query tile
    cr_base = cr_ptr + stride_cr_e * idx_e + stride_cr_h * idx_h + stride_cr_m * idx_m
    cr = tl.load(cr_base + stride_cr_b * range_q, mask=mask_q, other=1.0)

    # Online softmax accumulators
    max_rows = tl.full([TILE_B], float('-inf'), dtype=tl.float32)
    sum_rows = tl.zeros([TILE_B], dtype=tl.float32)
    al_acc = tl.zeros([TILE_B, BLOCK_D], dtype=tl.float32)
    score_acc = tl.zeros([TILE_B], dtype=tl.float32)

    # Loop over key tiles
    k_base = k_ptr + stride_k_e * idx_e + stride_k_h * idx_h + stride_k_m * idx_m
    range_k = tl.arange(0, TILE_B)

    for k_start in tl.range(0, B, TILE_B, num_stages=3):
        k_range = k_start + range_k
        k_mask = k_range < B
        if PRE_PAD:
            k_valid = k_mask & ((block_start_n + k_range) >= pad_offset)
        else:
            k_valid = k_mask & ((block_start_n + k_range) < N)

        if HAS_ATTN_MASK:
            mask_block_ptr = (
                mask_ptr + stride_mask_e * idx_e + stride_mask_m * idx_m
                + stride_mask_b * (k_range - pad_offset)
            )
            valid_token_mask = tl.load(mask_block_ptr, mask=k_valid, other=0)
            k_valid = k_valid & valid_token_mask

        # Load key tile [TILE_B, D]
        k_offset = stride_k_b * (k_range - pad_offset)[:, None] + range_d[None, :]
        k_tile = tl.load(k_base + k_offset, mask=k_valid[:, None] & mask_d[None, :], other=0.0)
        k_bf16 = k_tile.to(tl.bfloat16)

        # Scores [TILE_B_q, TILE_B_k]
        scores = sm_scale * tl.dot(ar_bf16, tl.trans(k_bf16), out_dtype=tl.float32)
        scores = scores / (cr[:, None] + EPS)
        scores = scores + tl.where(k_valid[None, :], 0.0, float("-inf"))

        # Online softmax update
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(max_rows, tile_max)
        scale = tl.exp(max_rows - new_max)

        al_acc = al_acc * scale[:, None]
        sum_rows = sum_rows * scale
        score_acc = score_acc * scale

        exp_scores = tl.exp(scores - new_max[:, None])
        al_acc = al_acc + tl.dot(exp_scores.to(tl.bfloat16), k_bf16, out_dtype=tl.float32)
        score_acc = score_acc + tl.sum(exp_scores * scores, axis=1)
        sum_rows = sum_rows + tl.sum(exp_scores, axis=1)
        max_rows = new_max

    # Finalize: al = sm_scale * (sum_j exp(s-max) * k_j) / sum = sm_scale * r @ k
    safe_sum = tl.where(sum_rows > 0, sum_rows, 1.0)
    al = (sm_scale * al_acc / safe_sum[:, None]).to(ar.dtype)
    # cl = sum_j(r_ij * log(r_ij)) = score_acc/sum - max - log(sum)
    cl = tl.where(sum_rows > 0, score_acc / safe_sum - max_rows - tl.log(safe_sum), 0.0)

    # Store al
    al_base = al_ptr + stride_al_e * idx_e + stride_al_h * idx_h + stride_al_m * idx_m
    al_offset = stride_al_b * range_q[:, None] + range_d[None, :]
    tl.store(al_base + al_offset, al, mask=mask_q[:, None] & mask_d[None, :])

    # Store cl
    cl_base = cl_ptr + stride_cl_e * idx_e + stride_cl_h * idx_h + stride_cl_m * idx_m
    tl.store(cl_base + stride_cl_b * range_q, cl, mask=mask_q)


@triton.jit
def _ar_cr_kernel(
    al_ptr,
    stride_al_e,
    stride_al_h,
    stride_al_m,
    stride_al_b,
    stride_al_d,
    q_ptr,
    stride_q_e,
    stride_q_h,
    stride_q_m,
    stride_q_b,
    stride_q_d,
    cl_ptr,
    stride_cl_e,
    stride_cl_h,
    stride_cl_m,
    stride_cl_b,
    ar_ptr,
    stride_ar_e,
    stride_ar_h,
    stride_ar_m,
    stride_ar_b,
    stride_ar_d,
    cr_ptr,
    stride_cr_e,
    stride_cr_h,
    stride_cr_m,
    stride_cr_b,
    mask_ptr,
    stride_mask_e,
    stride_mask_m,
    stride_mask_b,
    HAS_ATTN_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PRE_PAD: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
):
    # 2D grid: (E*H, B) for better workload distribution
    idx_eh = tl.program_id(0)
    idx_b = tl.program_id(1)
    idx_e = idx_eh // H
    idx_h = idx_eh % H

    # Pre-compute constants
    pad_offset = M * B - N if PRE_PAD else 0

    range_m = tl.arange(0, BLOCK_M)
    range_d = tl.arange(0, BLOCK_D)

    # Simplified masks
    mask_m = range_m < M
    mask_d = range_d < D

    # q_mask_m: valid query positions within actual sequence
    # range_n[i] = idx_b + B * i gives the sequence position for each m
    if PRE_PAD:
        q_mask_m = mask_m & ((idx_b + B * range_m) >= pad_offset)
    else:
        q_mask_m = mask_m & ((idx_b + B * range_m) < N)

    if HAS_ATTN_MASK:
        mask_block_ptr = (
            mask_ptr
            + stride_mask_e * idx_e
            + stride_mask_b * (idx_b - pad_offset)
            + stride_mask_m * range_m
        )
        valid_token_mask = tl.load(
            mask_block_ptr,
            mask=q_mask_m,
            other=0,
        )
        q_mask_m = q_mask_m & valid_token_mask

    # Pre-compute base pointers for coalesced memory access
    # 2D offset pattern assumes stride_d = 1 (contiguous tensors)
    offset_2d_m = stride_al_m * range_m[:, None] + range_d[None, :]

    # Load al - pre-computed base + offset
    al_base = al_ptr + stride_al_e * idx_e + stride_al_h * idx_h + stride_al_b * idx_b
    al = tl.load(al_base + offset_2d_m, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # Load q - different stride pattern for q
    q_base = q_ptr + stride_q_e * idx_e + stride_q_h * idx_h + stride_q_b * (idx_b - pad_offset)
    q_offset = stride_q_m * range_m[:, None] + range_d[None, :]
    q = tl.load(q_base + q_offset, mask=q_mask_m[:, None] & mask_d[None, :], other=0.0)

    # Load cl - 1D load
    cl_base = cl_ptr + stride_cl_e * idx_e + stride_cl_h * idx_h + stride_cl_b * idx_b
    cl = tl.load(cl_base + stride_cl_m * range_m, mask=mask_m, other=0.0)

    # Attention matrix - use bf16 inputs for tensor cores, fp32 accumulator
    al_bf16 = al.to(tl.bfloat16)
    q_bf16 = q.to(tl.bfloat16)
    l = tl.dot(al_bf16, tl.trans(q_bf16), out_dtype=tl.float32)
    l = l - cl[:, None]
    l = l + tl.where(mask_m[:, None], 0.0, float("-inf"))
    l = tl.exp(l - tl.max(l, axis=0, keep_dims=True))
    l = l / tl.sum(l, axis=0, keep_dims=True)
    l = q_mask_m[None, :] * l

    # Store cr - 1D store
    cr = tl.sum(l, axis=1)
    cr_base = cr_ptr + stride_cr_e * idx_e + stride_cr_h * idx_h + stride_cr_b * idx_b
    tl.store(cr_base + stride_cr_m * range_m, cr, mask=mask_m)

    # Store ar - use bf16 for tensor cores, cast back to input dtype
    ar = tl.dot(l.to(tl.bfloat16), q_bf16, out_dtype=tl.float32).to(al.dtype)
    ar_base = ar_ptr + stride_ar_e * idx_e + stride_ar_h * idx_h + stride_ar_b * idx_b
    ar_offset = stride_ar_m * range_m[:, None] + range_d[None, :]
    tl.store(ar_base + ar_offset, ar, mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def _al_y_cl_kernel(
    ar_ptr,
    stride_ar_e,
    stride_ar_h,
    stride_ar_m,
    stride_ar_b,
    stride_ar_d,
    k_ptr,
    stride_k_e,
    stride_k_h,
    stride_k_m,
    stride_k_b,
    stride_k_d,
    v_ptr,
    stride_v_e,
    stride_v_h,
    stride_v_m,
    stride_v_b,
    stride_v_d,
    cr_ptr,
    stride_cr_e,
    stride_cr_h,
    stride_cr_m,
    stride_cr_b,
    al_ptr,
    stride_al_e,
    stride_al_h,
    stride_al_m,
    stride_al_b,
    stride_al_d,
    y_ptr,
    stride_y_e,
    stride_y_h,
    stride_y_m,
    stride_y_b,
    stride_y_d,
    cl_ptr,
    stride_cl_e,
    stride_cl_h,
    stride_cl_m,
    stride_cl_b,
    mask_ptr,
    stride_mask_e,
    stride_mask_m,
    stride_mask_b,
    sm_scale: float,
    HAS_ATTN_MASK: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PRE_PAD: tl.constexpr,
    EPS: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
):
    # 2D grid: (E*H, M) for better workload distribution
    idx_eh = tl.program_id(0)
    idx_m = tl.program_id(1)
    idx_e = idx_eh // H
    idx_h = idx_eh % H

    # Pre-compute constants
    pad_offset = M * B - N if PRE_PAD else 0
    block_start_n = B * idx_m

    range_b = tl.arange(0, BLOCK_B)
    range_d = tl.arange(0, BLOCK_D)

    # Simplified masks
    mask_b = range_b < B
    mask_d = range_d < D

    # k_mask_b: valid key positions within actual sequence
    if PRE_PAD:
        k_mask_b = mask_b & ((block_start_n + range_b) >= pad_offset)
    else:
        k_mask_b = mask_b & ((block_start_n + range_b) < N)

    if HAS_ATTN_MASK:
        mask_block_ptr = (
            mask_ptr
            + stride_mask_e * idx_e
            + stride_mask_m * idx_m
            + stride_mask_b * (range_b - pad_offset)
        )
        valid_token_mask = tl.load(
            mask_block_ptr,
            mask=k_mask_b,
            other=0,
        )
        k_mask_b = k_mask_b & valid_token_mask

    # Load ar
    ar_block_ptr = (
        ar_ptr
        + stride_ar_e * idx_e
        + stride_ar_h * idx_h
        + stride_ar_m * idx_m
        + (stride_ar_b * range_b[:, None] + stride_ar_d * range_d[None, :])
    )
    ar = tl.load(
        ar_block_ptr,
        mask=mask_b[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Load k
    k_block_ptr = (
        k_ptr
        + stride_k_e * idx_e
        + stride_k_h * idx_h
        + stride_k_m * idx_m
        + (stride_k_b * (range_b - pad_offset)[:, None] + stride_k_d * range_d[None, :])
    )
    k = tl.load(
        k_block_ptr,
        mask=k_mask_b[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Load cr
    cr_block_ptr = (
        cr_ptr
        + stride_cr_e * idx_e
        + stride_cr_h * idx_h
        + stride_cr_m * idx_m
        + (stride_cr_b * range_b)
    )
    cr = tl.load(cr_block_ptr, mask=mask_b, other=1.0)

    # Attention matrix - use bf16 inputs for tensor cores, fp32 accumulator
    ar_bf16 = ar.to(tl.bfloat16)
    k_bf16 = k.to(tl.bfloat16)
    r = sm_scale * tl.dot(ar_bf16, tl.trans(k_bf16), out_dtype=tl.float32)
    r = r / (cr[:, None] + EPS)
    r = r + tl.where(k_mask_b[None, :], 0.0, float("-inf"))
    r = tl.exp(r - tl.clamp(tl.max(r, axis=1, keep_dims=True), EPS, float("inf")))
    r = r / (tl.sum(r, axis=1, keep_dims=True) + EPS)

    # Store cl
    cl = tl.sum(xlogx(r), axis=1)
    cl_block_ptr = (
        cl_ptr
        + stride_cl_e * idx_e
        + stride_cl_h * idx_h
        + stride_cl_m * idx_m
        + (stride_cl_b * range_b)
    )
    tl.store(cl_block_ptr, cl, mask=mask_b)

    # Store al - use bf16 for tensor cores, cast back to input dtype
    al = (sm_scale * tl.dot(r.to(tl.bfloat16), k_bf16, out_dtype=tl.float32)).to(ar.dtype)
    al_block_ptr = (
        al_ptr
        + stride_al_e * idx_e
        + stride_al_h * idx_h
        + stride_al_m * idx_m
        + (stride_al_b * range_b[:, None] + stride_al_d * range_d[None, :])
    )
    tl.store(
        al_block_ptr,
        al,
        mask=mask_b[:, None] & mask_d[None, :],
    )

    # Load v
    v_block_ptr = (
        v_ptr
        + stride_v_e * idx_e
        + stride_v_h * idx_h
        + stride_v_m * idx_m
        + (stride_v_b * (range_b - pad_offset)[:, None] + stride_v_d * range_d[None, :])
    )
    v = tl.load(
        v_block_ptr,
        mask=k_mask_b[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Store y - use bf16 for tensor cores, cast back to input dtype
    v_bf16 = v.to(tl.bfloat16)
    y = tl.dot(r.to(tl.bfloat16), v_bf16, out_dtype=tl.float32).to(ar.dtype)
    y_block_ptr = (
        y_ptr
        + stride_y_e * idx_e
        + stride_y_h * idx_h
        + stride_y_m * idx_m
        + (stride_y_b * range_b[:, None] + stride_y_d * range_d[None, :])
    )
    tl.store(
        y_block_ptr,
        y,
        mask=mask_b[:, None] & mask_d[None, :],
    )


@triton.autotune(configs=_get_tiled_b_autotune_configs(), key=['B', 'D'])
@triton.jit
def _al_y_cl_kernel_tiled(
    ar_ptr,
    stride_ar_e,
    stride_ar_h,
    stride_ar_m,
    stride_ar_b,
    stride_ar_d,
    k_ptr,
    stride_k_e,
    stride_k_h,
    stride_k_m,
    stride_k_b,
    stride_k_d,
    v_ptr,
    stride_v_e,
    stride_v_h,
    stride_v_m,
    stride_v_b,
    stride_v_d,
    cr_ptr,
    stride_cr_e,
    stride_cr_h,
    stride_cr_m,
    stride_cr_b,
    al_ptr,
    stride_al_e,
    stride_al_h,
    stride_al_m,
    stride_al_b,
    stride_al_d,
    y_ptr,
    stride_y_e,
    stride_y_h,
    stride_y_m,
    stride_y_b,
    stride_y_d,
    cl_ptr,
    stride_cl_e,
    stride_cl_h,
    stride_cl_m,
    stride_cl_b,
    mask_ptr,
    stride_mask_e,
    stride_mask_m,
    stride_mask_b,
    sm_scale: float,
    HAS_ATTN_MASK: tl.constexpr,
    TILE_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PRE_PAD: tl.constexpr,
    EPS: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
):
    """Tiled version of _al_y_cl_kernel for large B (>128).

    Uses online softmax to tile over key positions, also accumulates y = r @ v.
    Grid: (E*H, M, cdiv(B, TILE_B))
    """
    idx_eh = tl.program_id(0)
    idx_m = tl.program_id(1)
    idx_q_tile = tl.program_id(2)
    idx_e = idx_eh // H
    idx_h = idx_eh % H

    pad_offset = M * B - N if PRE_PAD else 0
    block_start_n = B * idx_m

    # Query tile range
    q_start = idx_q_tile * TILE_B
    range_q = q_start + tl.arange(0, TILE_B)
    range_d = tl.arange(0, BLOCK_D)
    mask_q = range_q < B
    mask_d = range_d < D

    # Load ar (query rows for this tile) - always from ar tensor (not q)
    ar_base = ar_ptr + stride_ar_e * idx_e + stride_ar_h * idx_h + stride_ar_m * idx_m
    ar_offset = stride_ar_b * range_q[:, None] + range_d[None, :]
    ar = tl.load(ar_base + ar_offset, mask=mask_q[:, None] & mask_d[None, :], other=0.0)
    ar_bf16 = ar.to(tl.bfloat16)

    # Load cr for query tile
    cr_base = cr_ptr + stride_cr_e * idx_e + stride_cr_h * idx_h + stride_cr_m * idx_m
    cr = tl.load(cr_base + stride_cr_b * range_q, mask=mask_q, other=1.0)

    # Online softmax accumulators
    max_rows = tl.full([TILE_B], float('-inf'), dtype=tl.float32)
    sum_rows = tl.zeros([TILE_B], dtype=tl.float32)
    al_acc = tl.zeros([TILE_B, BLOCK_D], dtype=tl.float32)
    y_acc = tl.zeros([TILE_B, BLOCK_D], dtype=tl.float32)
    score_acc = tl.zeros([TILE_B], dtype=tl.float32)

    # Loop over key/value tiles
    k_base = k_ptr + stride_k_e * idx_e + stride_k_h * idx_h + stride_k_m * idx_m
    v_base = v_ptr + stride_v_e * idx_e + stride_v_h * idx_h + stride_v_m * idx_m
    range_k = tl.arange(0, TILE_B)

    for k_start in tl.range(0, B, TILE_B, num_stages=3):
        k_range = k_start + range_k
        k_mask = k_range < B
        if PRE_PAD:
            k_valid = k_mask & ((block_start_n + k_range) >= pad_offset)
        else:
            k_valid = k_mask & ((block_start_n + k_range) < N)

        if HAS_ATTN_MASK:
            mask_block_ptr = (
                mask_ptr + stride_mask_e * idx_e + stride_mask_m * idx_m
                + stride_mask_b * (k_range - pad_offset)
            )
            valid_token_mask = tl.load(mask_block_ptr, mask=k_valid, other=0)
            k_valid = k_valid & valid_token_mask

        # Load key tile [TILE_B, D]
        k_offset = stride_k_b * (k_range - pad_offset)[:, None] + range_d[None, :]
        k_tile = tl.load(k_base + k_offset, mask=k_valid[:, None] & mask_d[None, :], other=0.0)
        k_bf16 = k_tile.to(tl.bfloat16)

        # Load value tile [TILE_B, D]
        v_offset = stride_v_b * (k_range - pad_offset)[:, None] + range_d[None, :]
        v_tile = tl.load(v_base + v_offset, mask=k_valid[:, None] & mask_d[None, :], other=0.0)
        v_bf16 = v_tile.to(tl.bfloat16)

        # Scores [TILE_B_q, TILE_B_k]
        scores = sm_scale * tl.dot(ar_bf16, tl.trans(k_bf16), out_dtype=tl.float32)
        scores = scores / (cr[:, None] + EPS)
        scores = scores + tl.where(k_valid[None, :], 0.0, float("-inf"))

        # Online softmax update
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(max_rows, tile_max)
        scale = tl.exp(max_rows - new_max)

        al_acc = al_acc * scale[:, None]
        y_acc = y_acc * scale[:, None]
        sum_rows = sum_rows * scale
        score_acc = score_acc * scale

        exp_scores = tl.exp(scores - new_max[:, None])
        al_acc = al_acc + tl.dot(exp_scores.to(tl.bfloat16), k_bf16, out_dtype=tl.float32)
        y_acc = y_acc + tl.dot(exp_scores.to(tl.bfloat16), v_bf16, out_dtype=tl.float32)
        score_acc = score_acc + tl.sum(exp_scores * scores, axis=1)
        sum_rows = sum_rows + tl.sum(exp_scores, axis=1)
        max_rows = new_max

    # Finalize outputs
    safe_sum = tl.where(sum_rows > 0, sum_rows, 1.0)
    al = (sm_scale * al_acc / safe_sum[:, None]).to(ar.dtype)
    y = (y_acc / safe_sum[:, None]).to(ar.dtype)
    cl = tl.where(sum_rows > 0, score_acc / safe_sum - max_rows - tl.log(safe_sum), 0.0)

    # Store al
    al_base = al_ptr + stride_al_e * idx_e + stride_al_h * idx_h + stride_al_m * idx_m
    al_offset = stride_al_b * range_q[:, None] + range_d[None, :]
    tl.store(al_base + al_offset, al, mask=mask_q[:, None] & mask_d[None, :])

    # Store y
    y_base = y_ptr + stride_y_e * idx_e + stride_y_h * idx_h + stride_y_m * idx_m
    y_offset = stride_y_b * range_q[:, None] + range_d[None, :]
    tl.store(y_base + y_offset, y, mask=mask_q[:, None] & mask_d[None, :])

    # Store cl
    cl_base = cl_ptr + stride_cl_e * idx_e + stride_cl_h * idx_h + stride_cl_m * idx_m
    tl.store(cl_base + stride_cl_b * range_q, cl, mask=mask_q)


@triton.jit
def _z_kernel(
    al_ptr,
    stride_al_e,
    stride_al_h,
    stride_al_m,
    stride_al_b,
    stride_al_d,
    q_ptr,
    stride_q_e,
    stride_q_h,
    stride_q_m,
    stride_q_b,
    stride_q_d,
    y_ptr,
    stride_y_e,
    stride_y_h,
    stride_y_m,
    stride_y_b,
    stride_y_d,
    cl_ptr,
    stride_cl_e,
    stride_cl_h,
    stride_cl_m,
    stride_cl_b,
    z_ptr,
    stride_z_e,
    stride_z_h,
    stride_z_m,
    stride_z_b,
    stride_z_d,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PRE_PAD: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
):
    # 2D grid: (E*H, B) for better workload distribution
    idx_eh = tl.program_id(0)
    idx_b = tl.program_id(1)
    idx_e = idx_eh // H
    idx_h = idx_eh % H

    # Pre-compute constants
    pad_offset = M * B - N if PRE_PAD else 0

    range_m = tl.arange(0, BLOCK_M)
    range_d = tl.arange(0, BLOCK_D)

    # Simplified masks
    mask_m = range_m < M
    mask_d = range_d < D

    # q_mask_m: valid query positions within actual sequence
    if PRE_PAD:
        q_mask_m = mask_m & ((idx_b + B * range_m) >= pad_offset)
    else:
        q_mask_m = mask_m & ((idx_b + B * range_m) < N)

    # Load al
    al_block_ptr = (
        al_ptr
        + stride_al_e * idx_e
        + stride_al_h * idx_h
        + stride_al_b * idx_b
        + (stride_al_m * range_m[:, None] + stride_al_d * range_d[None, :])
    )
    al = tl.load(
        al_block_ptr,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Load q
    q_block_ptr = (
        q_ptr
        + stride_q_e * idx_e
        + stride_q_h * idx_h
        + stride_q_b * (idx_b - pad_offset)
        + (stride_q_m * range_m[:, None] + stride_q_d * range_d[None, :])
    )
    q = tl.load(
        q_block_ptr,
        mask=q_mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Load cl
    cl_block_ptr = (
        cl_ptr
        + stride_cl_e * idx_e
        + stride_cl_h * idx_h
        + stride_cl_b * idx_b
        + (stride_cl_m * range_m)
    )
    cl = tl.load(cl_block_ptr, mask=mask_m, other=0.0)

    # Attention matrix - use bf16 inputs for tensor cores, fp32 accumulator
    q_bf16 = q.to(tl.bfloat16)
    al_bf16 = al.to(tl.bfloat16)
    l = tl.dot(q_bf16, tl.trans(al_bf16), out_dtype=tl.float32)
    l = l - cl[None, :]
    l = l + tl.where(mask_m[None, :], 0.0, float("-inf"))
    l = tl.exp(l - tl.max(l, axis=1, keep_dims=True))
    l = l / tl.sum(l, axis=1, keep_dims=True)

    # Load y
    y_block_ptr = (
        y_ptr
        + stride_y_e * idx_e
        + stride_y_h * idx_h
        + stride_y_b * idx_b
        + (stride_y_m * range_m[:, None] + stride_y_d * range_d[None, :])
    )
    y = tl.load(
        y_block_ptr,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Store z - use bf16 for tensor cores, cast back to input dtype
    y_bf16 = y.to(tl.bfloat16)
    z = tl.dot(l.to(tl.bfloat16), y_bf16, out_dtype=tl.float32).to(al.dtype)
    z_block_ptr = (
        z_ptr
        + stride_z_e * idx_e
        + stride_z_h * idx_h
        + stride_z_b * (idx_b - pad_offset)
        + (stride_z_m * range_m[:, None] + stride_z_d * range_d[None, :])
    )
    tl.store(
        z_block_ptr,
        z,
        mask=q_mask_m[:, None] & mask_d[None, :],
    )


# =============================================================================
# Tiled kernels for large M (sequence length > 4096 with block_size=32)
# These kernels avoid materializing the full MxM attention matrix by tiling
# =============================================================================


@triton.autotune(configs=_get_warp_stage_autotune_configs(), key=['M', 'D'])
@triton.jit
def _ar_cr_softmax_stats_kernel(
    al_ptr, stride_al_e, stride_al_h, stride_al_m, stride_al_b, stride_al_d,
    q_ptr, stride_q_e, stride_q_h, stride_q_m, stride_q_b, stride_q_d,
    cl_ptr, stride_cl_e, stride_cl_h, stride_cl_m, stride_cl_b,
    max_ptr, stride_max_e, stride_max_h, stride_max_m, stride_max_b,
    sum_ptr, stride_sum_e, stride_sum_h, stride_sum_m, stride_sum_b,
    TILE_M: tl.constexpr, BLOCK_D: tl.constexpr, PRE_PAD: tl.constexpr,
    H: tl.constexpr, M: tl.constexpr, B: tl.constexpr, D: tl.constexpr, N: tl.constexpr,
):
    """Phase 1: Compute softmax max and sum for each column j.

    For _ar_cr_kernel, softmax is over axis=0 (rows). Each column j has independent normalization.
    This kernel computes max_j and sum_j for all j in a tile, using online softmax over row tiles.
    """
    idx_eh = tl.program_id(0)
    idx_b = tl.program_id(1)
    idx_j_tile = tl.program_id(2)
    idx_e = idx_eh // H
    idx_h = idx_eh % H

    # Pre-compute constants
    pad_offset = M * B - N if PRE_PAD else 0

    # Column indices for this tile
    j_start = idx_j_tile * TILE_M
    j_range = j_start + tl.arange(0, TILE_M)
    j_mask = j_range < M

    range_d = tl.arange(0, BLOCK_D)
    mask_d = range_d < D

    # q_mask_j: valid query positions within actual sequence
    if PRE_PAD:
        q_mask_j = j_mask & ((idx_b + B * j_range) >= pad_offset)
    else:
        q_mask_j = j_mask & ((idx_b + B * j_range) < N)

    # Load q_tile [TILE_M, D] - the "key" positions for this column tile
    q_block_ptr = (
        q_ptr + stride_q_e * idx_e + stride_q_h * idx_h
        + stride_q_b * (idx_b - pad_offset)
        + (stride_q_m * j_range[:, None] + stride_q_d * range_d[None, :])
    )
    q_tile = tl.load(q_block_ptr, mask=q_mask_j[:, None] & mask_d[None, :], other=0.0)
    q_bf16 = q_tile.to(tl.bfloat16)

    # Initialize online softmax accumulators for each column
    max_cols = tl.full([TILE_M], float('-inf'), dtype=tl.float32)
    sum_cols = tl.zeros([TILE_M], dtype=tl.float32)

    # Iterate over row tiles (i dimension of al)
    range_i = tl.arange(0, TILE_M)
    for i_start in tl.range(0, M, TILE_M, num_stages=3):
        i_range = i_start + range_i
        i_mask = i_range < M

        # Load al_tile [TILE_M, D]
        al_block_ptr = (
            al_ptr + stride_al_e * idx_e + stride_al_h * idx_h
            + stride_al_b * idx_b
            + (stride_al_m * i_range[:, None] + stride_al_d * range_d[None, :])
        )
        al_tile = tl.load(al_block_ptr, mask=i_mask[:, None] & mask_d[None, :], other=0.0)

        # Load cl_tile [TILE_M]
        cl_block_ptr = (
            cl_ptr + stride_cl_e * idx_e + stride_cl_h * idx_h
            + stride_cl_b * idx_b + stride_cl_m * i_range
        )
        cl_tile = tl.load(cl_block_ptr, mask=i_mask, other=0.0)

        # Compute attention scores [TILE_M_i, TILE_M_j]
        al_bf16 = al_tile.to(tl.bfloat16)
        scores = tl.dot(al_bf16, tl.trans(q_bf16), out_dtype=tl.float32)  # [TILE_M, TILE_M]
        scores = scores - cl_tile[:, None]
        scores = tl.where(i_mask[:, None], scores, float('-inf'))

        # Online softmax update for each column
        tile_max = tl.max(scores, axis=0)  # [TILE_M]
        new_max = tl.maximum(max_cols, tile_max)

        # Rescale old sum
        scale = tl.exp(max_cols - new_max)
        sum_cols = sum_cols * scale

        # Add new contributions
        exp_scores = tl.exp(scores - new_max[None, :])
        sum_cols = sum_cols + tl.sum(exp_scores, axis=0)

        max_cols = new_max

    # Store max and sum for this column tile
    out_range = j_start + tl.arange(0, TILE_M)
    out_mask = out_range < M

    max_block_ptr = (
        max_ptr + stride_max_e * idx_e + stride_max_h * idx_h
        + stride_max_b * idx_b + stride_max_m * out_range
    )
    tl.store(max_block_ptr, max_cols, mask=out_mask)

    sum_block_ptr = (
        sum_ptr + stride_sum_e * idx_e + stride_sum_h * idx_h
        + stride_sum_b * idx_b + stride_sum_m * out_range
    )
    tl.store(sum_block_ptr, sum_cols, mask=out_mask)


@triton.autotune(configs=_get_warp_stage_autotune_configs(), key=['M', 'D'])
@triton.jit
def _ar_cr_accumulate_kernel(
    al_ptr, stride_al_e, stride_al_h, stride_al_m, stride_al_b, stride_al_d,
    q_ptr, stride_q_e, stride_q_h, stride_q_m, stride_q_b, stride_q_d,
    cl_ptr, stride_cl_e, stride_cl_h, stride_cl_m, stride_cl_b,
    max_ptr, stride_max_e, stride_max_h, stride_max_m, stride_max_b,
    sum_ptr, stride_sum_e, stride_sum_h, stride_sum_m, stride_sum_b,
    ar_ptr, stride_ar_e, stride_ar_h, stride_ar_m, stride_ar_b, stride_ar_d,
    cr_ptr, stride_cr_e, stride_cr_h, stride_cr_m, stride_cr_b,
    TILE_M: tl.constexpr, BLOCK_D: tl.constexpr, PRE_PAD: tl.constexpr,
    H: tl.constexpr, M: tl.constexpr, B: tl.constexpr, D: tl.constexpr, N: tl.constexpr,
):
    """Phase 2: Compute ar and cr using precomputed softmax stats.

    Parallelized over output row tiles. Each kernel computes ar[i_tile, :] and cr[i_tile].
    """
    idx_eh = tl.program_id(0)
    idx_b = tl.program_id(1)
    idx_i_tile = tl.program_id(2)
    idx_e = idx_eh // H
    idx_h = idx_eh % H

    pad_offset = M * B - N if PRE_PAD else 0

    # Row indices for this output tile
    i_start = idx_i_tile * TILE_M
    i_range = i_start + tl.arange(0, TILE_M)
    i_mask = i_range < M

    range_d = tl.arange(0, BLOCK_D)
    mask_d = range_d < D

    # Load al_tile [TILE_M, D] - this tile's rows
    al_block_ptr = (
        al_ptr + stride_al_e * idx_e + stride_al_h * idx_h
        + stride_al_b * idx_b
        + (stride_al_m * i_range[:, None] + stride_al_d * range_d[None, :])
    )
    al_tile = tl.load(al_block_ptr, mask=i_mask[:, None] & mask_d[None, :], other=0.0)
    al_bf16 = al_tile.to(tl.bfloat16)

    # Load cl_tile [TILE_M]
    cl_block_ptr = (
        cl_ptr + stride_cl_e * idx_e + stride_cl_h * idx_h
        + stride_cl_b * idx_b + stride_cl_m * i_range
    )
    cl_tile = tl.load(cl_block_ptr, mask=i_mask, other=0.0)

    # Initialize accumulators
    ar_acc = tl.zeros([TILE_M, BLOCK_D], dtype=tl.float32)
    cr_acc = tl.zeros([TILE_M], dtype=tl.float32)

    # Iterate over column tiles (j dimension)
    range_j = tl.arange(0, TILE_M)
    for j_start in tl.range(0, M, TILE_M, num_stages=3):
        j_range = j_start + range_j
        j_mask = j_range < M
        # Simplified mask computation
        if PRE_PAD:
            q_mask_j = j_mask & ((idx_b + B * j_range) >= pad_offset)
        else:
            q_mask_j = j_mask & ((idx_b + B * j_range) < N)

        # Load q_tile [TILE_M, D]
        q_block_ptr = (
            q_ptr + stride_q_e * idx_e + stride_q_h * idx_h
            + stride_q_b * (idx_b - pad_offset)
            + (stride_q_m * j_range[:, None] + stride_q_d * range_d[None, :])
        )
        q_tile = tl.load(q_block_ptr, mask=q_mask_j[:, None] & mask_d[None, :], other=0.0)
        q_bf16 = q_tile.to(tl.bfloat16)

        # Load precomputed max and sum for these columns
        max_block_ptr = (
            max_ptr + stride_max_e * idx_e + stride_max_h * idx_h
            + stride_max_b * idx_b + stride_max_m * j_range
        )
        max_cols = tl.load(max_block_ptr, mask=j_mask, other=0.0)

        sum_block_ptr = (
            sum_ptr + stride_sum_e * idx_e + stride_sum_h * idx_h
            + stride_sum_b * idx_b + stride_sum_m * j_range
        )
        sum_cols = tl.load(sum_block_ptr, mask=j_mask, other=1.0)  # Avoid div by zero

        # Compute attention scores [TILE_M, TILE_M]
        scores = tl.dot(al_bf16, tl.trans(q_bf16), out_dtype=tl.float32)
        scores = scores - cl_tile[:, None]

        # Compute normalized attention weights
        l_tile = tl.exp(scores - max_cols[None, :]) / sum_cols[None, :]
        l_tile = tl.where(j_mask[None, :], l_tile, 0.0)
        l_tile = tl.where(q_mask_j[None, :], l_tile, 0.0)

        # Accumulate ar and cr
        ar_acc = ar_acc + tl.dot(l_tile.to(tl.bfloat16), q_bf16, out_dtype=tl.float32)
        cr_acc = cr_acc + tl.sum(l_tile, axis=1)

    # Store results
    ar_block_ptr = (
        ar_ptr + stride_ar_e * idx_e + stride_ar_h * idx_h
        + stride_ar_b * idx_b
        + (stride_ar_m * i_range[:, None] + stride_ar_d * range_d[None, :])
    )
    tl.store(ar_block_ptr, ar_acc.to(al_tile.dtype), mask=i_mask[:, None] & mask_d[None, :])

    cr_block_ptr = (
        cr_ptr + stride_cr_e * idx_e + stride_cr_h * idx_h
        + stride_cr_b * idx_b + stride_cr_m * i_range
    )
    tl.store(cr_block_ptr, cr_acc, mask=i_mask)


@triton.jit
def _ar_cr_softmax_stats_kernel_tma(
    al_ptr, stride_al_e, stride_al_h, stride_al_m, stride_al_b, stride_al_d,
    q_ptr, stride_q_e, stride_q_h, stride_q_m, stride_q_b, stride_q_d,
    cl_ptr, stride_cl_e, stride_cl_h, stride_cl_m, stride_cl_b,
    max_ptr, stride_max_e, stride_max_h, stride_max_m, stride_max_b,
    sum_ptr, stride_sum_e, stride_sum_h, stride_sum_m, stride_sum_b,
    TILE_M: tl.constexpr, BLOCK_D: tl.constexpr, PRE_PAD: tl.constexpr,
    H: tl.constexpr, M: tl.constexpr, B: tl.constexpr, D: tl.constexpr, N: tl.constexpr,
):
    """TMA-optimized Phase 1: Compute softmax max and sum with pipelining."""
    idx_eh = tl.program_id(0)
    idx_b = tl.program_id(1)
    idx_j_tile = tl.program_id(2)
    idx_e = idx_eh // H
    idx_h = idx_eh % H

    pad_offset = M * B - N if PRE_PAD else 0

    # Column indices for this tile
    j_start = idx_j_tile * TILE_M
    j_range = j_start + tl.arange(0, TILE_M)
    j_mask = j_range < M

    range_d = tl.arange(0, BLOCK_D)
    mask_d = range_d < D

    if PRE_PAD:
        q_mask_j = j_mask & ((idx_b + B * j_range) >= pad_offset)
    else:
        q_mask_j = j_mask & ((idx_b + B * j_range) < N)

    # Load q_tile (only once)
    q_block_ptr = (
        q_ptr + stride_q_e * idx_e + stride_q_h * idx_h
        + stride_q_b * (idx_b - pad_offset)
        + (stride_q_m * j_range[:, None] + stride_q_d * range_d[None, :])
    )
    q_tile = tl.load(q_block_ptr, mask=q_mask_j[:, None] & mask_d[None, :], other=0.0)
    q_bf16 = q_tile.to(tl.bfloat16)

    # Base pointers for TMA
    al_base = al_ptr + stride_al_e * idx_e + stride_al_h * idx_h + stride_al_b * idx_b
    cl_base = cl_ptr + stride_cl_e * idx_e + stride_cl_h * idx_h + stride_cl_b * idx_b

    # Create TMA descriptor for al
    al_desc = tl.make_tensor_descriptor(
        al_base,
        shape=[M, D],
        strides=[stride_al_m, stride_al_d],
        block_shape=[TILE_M, BLOCK_D],
    )

    # Initialize accumulators
    max_cols = tl.full([TILE_M], float('-inf'), dtype=tl.float32)
    sum_cols = tl.zeros([TILE_M], dtype=tl.float32)

    num_tiles = tl.cdiv(M, TILE_M)
    range_i = tl.arange(0, TILE_M)

    # Prefetch first tile
    al_next = al_desc.load([0, 0])
    cl_next = tl.load(cl_base + stride_cl_m * range_i, mask=range_i < M, other=0.0)

    # Main loop with pipelining
    for tile_idx in range(num_tiles):
        i_start = tile_idx * TILE_M
        i_range = i_start + range_i
        i_mask = i_range < M

        al_tile = al_next
        cl_tile = cl_next

        # Prefetch next
        next_tile_idx = tile_idx + 1
        if next_tile_idx < num_tiles:
            next_i_start = next_tile_idx * TILE_M
            al_next = al_desc.load([next_i_start, 0])
            next_i_range = next_i_start + range_i
            cl_next = tl.load(cl_base + stride_cl_m * next_i_range, mask=next_i_range < M, other=0.0)

        # Compute scores
        al_bf16 = al_tile.to(tl.bfloat16)
        scores = tl.dot(al_bf16, tl.trans(q_bf16), out_dtype=tl.float32)
        scores = scores - cl_tile[:, None]
        scores = tl.where(i_mask[:, None], scores, float('-inf'))

        # Online softmax update
        tile_max = tl.max(scores, axis=0)
        new_max = tl.maximum(max_cols, tile_max)
        scale = tl.exp(max_cols - new_max)
        sum_cols = sum_cols * scale
        exp_scores = tl.exp(scores - new_max[None, :])
        sum_cols = sum_cols + tl.sum(exp_scores, axis=0)
        max_cols = new_max

    # Store results
    out_range = j_start + tl.arange(0, TILE_M)
    out_mask = out_range < M
    max_block_ptr = max_ptr + stride_max_e * idx_e + stride_max_h * idx_h + stride_max_b * idx_b + stride_max_m * out_range
    tl.store(max_block_ptr, max_cols, mask=out_mask)
    sum_block_ptr = sum_ptr + stride_sum_e * idx_e + stride_sum_h * idx_h + stride_sum_b * idx_b + stride_sum_m * out_range
    tl.store(sum_block_ptr, sum_cols, mask=out_mask)


@triton.jit
def _ar_cr_accumulate_kernel_tma(
    al_ptr, stride_al_e, stride_al_h, stride_al_m, stride_al_b, stride_al_d,
    q_ptr, stride_q_e, stride_q_h, stride_q_m, stride_q_b, stride_q_d,
    cl_ptr, stride_cl_e, stride_cl_h, stride_cl_m, stride_cl_b,
    max_ptr, stride_max_e, stride_max_h, stride_max_m, stride_max_b,
    sum_ptr, stride_sum_e, stride_sum_h, stride_sum_m, stride_sum_b,
    ar_ptr, stride_ar_e, stride_ar_h, stride_ar_m, stride_ar_b, stride_ar_d,
    cr_ptr, stride_cr_e, stride_cr_h, stride_cr_m, stride_cr_b,
    TILE_M: tl.constexpr, BLOCK_D: tl.constexpr, PRE_PAD: tl.constexpr,
    H: tl.constexpr, M: tl.constexpr, B: tl.constexpr, D: tl.constexpr, N: tl.constexpr,
):
    """TMA-optimized Phase 2: Compute ar and cr with pipelining."""
    idx_eh = tl.program_id(0)
    idx_b = tl.program_id(1)
    idx_i_tile = tl.program_id(2)
    idx_e = idx_eh // H
    idx_h = idx_eh % H

    pad_offset = M * B - N if PRE_PAD else 0

    # Row indices for this output tile
    i_start = idx_i_tile * TILE_M
    i_range = i_start + tl.arange(0, TILE_M)
    i_mask = i_range < M

    range_d = tl.arange(0, BLOCK_D)
    mask_d = range_d < D

    # Base pointers
    al_base = al_ptr + stride_al_e * idx_e + stride_al_h * idx_h + stride_al_b * idx_b
    cl_base = cl_ptr + stride_cl_e * idx_e + stride_cl_h * idx_h + stride_cl_b * idx_b
    max_base = max_ptr + stride_max_e * idx_e + stride_max_h * idx_h + stride_max_b * idx_b
    sum_base = sum_ptr + stride_sum_e * idx_e + stride_sum_h * idx_h + stride_sum_b * idx_b

    # Load this tile's al and cl (constant across j iterations)
    al_block_ptr = al_base + (stride_al_m * i_range[:, None] + stride_al_d * range_d[None, :])
    al_tile = tl.load(al_block_ptr, mask=i_mask[:, None] & mask_d[None, :], other=0.0)
    al_bf16 = al_tile.to(tl.bfloat16)
    cl_tile = tl.load(cl_base + stride_cl_m * i_range, mask=i_mask, other=0.0)

    # TMA descriptor for q
    q_base = q_ptr + stride_q_e * idx_e + stride_q_h * idx_h + stride_q_b * (idx_b - pad_offset)
    q_desc = tl.make_tensor_descriptor(
        q_base,
        shape=[M, D],
        strides=[stride_q_m, stride_q_d],
        block_shape=[TILE_M, BLOCK_D],
    )

    # Initialize accumulators
    ar_acc = tl.zeros([TILE_M, BLOCK_D], dtype=tl.float32)
    cr_acc = tl.zeros([TILE_M], dtype=tl.float32)

    num_tiles = tl.cdiv(M, TILE_M)
    range_j = tl.arange(0, TILE_M)

    # Prefetch first tile
    q_next = q_desc.load([0, 0])
    max_next = tl.load(max_base + stride_max_m * range_j, mask=range_j < M, other=0.0)
    sum_next = tl.load(sum_base + stride_sum_m * range_j, mask=range_j < M, other=1.0)

    # Main loop with pipelining
    for tile_idx in range(num_tiles):
        j_start = tile_idx * TILE_M
        j_range = j_start + range_j
        j_mask = j_range < M

        if PRE_PAD:
            q_mask_j = j_mask & ((idx_b + B * j_range) >= pad_offset)
        else:
            q_mask_j = j_mask & ((idx_b + B * j_range) < N)

        q_tile = q_next
        max_cols = max_next
        sum_cols = sum_next

        # Prefetch next
        next_tile_idx = tile_idx + 1
        if next_tile_idx < num_tiles:
            next_j_start = next_tile_idx * TILE_M
            q_next = q_desc.load([next_j_start, 0])
            next_j_range = next_j_start + range_j
            max_next = tl.load(max_base + stride_max_m * next_j_range, mask=next_j_range < M, other=0.0)
            sum_next = tl.load(sum_base + stride_sum_m * next_j_range, mask=next_j_range < M, other=1.0)

        # Compute scores
        q_bf16 = q_tile.to(tl.bfloat16)
        scores = tl.dot(al_bf16, tl.trans(q_bf16), out_dtype=tl.float32)
        scores = scores - cl_tile[:, None]

        # Normalize
        l_tile = tl.exp(scores - max_cols[None, :]) / sum_cols[None, :]
        l_tile = tl.where(j_mask[None, :], l_tile, 0.0)
        l_tile = tl.where(q_mask_j[None, :], l_tile, 0.0)

        # Accumulate
        ar_acc = ar_acc + tl.dot(l_tile.to(tl.bfloat16), q_bf16, out_dtype=tl.float32)
        cr_acc = cr_acc + tl.sum(l_tile, axis=1)

    # Store results
    ar_block_ptr = (ar_ptr + stride_ar_e * idx_e + stride_ar_h * idx_h + stride_ar_b * idx_b
                    + (stride_ar_m * i_range[:, None] + stride_ar_d * range_d[None, :]))
    tl.store(ar_block_ptr, ar_acc.to(al_tile.dtype), mask=i_mask[:, None] & mask_d[None, :])
    cr_block_ptr = cr_ptr + stride_cr_e * idx_e + stride_cr_h * idx_h + stride_cr_b * idx_b + stride_cr_m * i_range
    tl.store(cr_block_ptr, cr_acc, mask=i_mask)


@triton.autotune(configs=_get_tiled_m_autotune_configs(), key=['M', 'D'])
@triton.jit
def _z_kernel_tiled(
    al_ptr, stride_al_e, stride_al_h, stride_al_m, stride_al_b, stride_al_d,
    q_ptr, stride_q_e, stride_q_h, stride_q_m, stride_q_b, stride_q_d,
    y_ptr, stride_y_e, stride_y_h, stride_y_m, stride_y_b, stride_y_d,
    cl_ptr, stride_cl_e, stride_cl_h, stride_cl_m, stride_cl_b,
    z_ptr, stride_z_e, stride_z_h, stride_z_m, stride_z_b, stride_z_d,
    TILE_M: tl.constexpr, BLOCK_D: tl.constexpr, PRE_PAD: tl.constexpr,
    H: tl.constexpr, M: tl.constexpr, B: tl.constexpr, D: tl.constexpr, N: tl.constexpr,
):
    """Tiled _z_kernel using Flash Attention style online softmax.

    For _z_kernel, softmax is over axis=1 (columns). Each row i has independent normalization.
    This is like standard attention, so we can use online softmax while iterating over j tiles.
    """
    idx_eh = tl.program_id(0)
    idx_b = tl.program_id(1)
    idx_i_tile = tl.program_id(2)
    idx_e = idx_eh // H
    idx_h = idx_eh % H

    # Pre-compute constants
    pad_offset = M * B - N if PRE_PAD else 0

    # Row indices for this tile (output rows)
    i_start = idx_i_tile * TILE_M
    i_range = i_start + tl.arange(0, TILE_M)
    i_mask = i_range < M

    range_d = tl.arange(0, BLOCK_D)
    mask_d = range_d < D

    # q_mask_i: valid query positions within actual sequence
    if PRE_PAD:
        q_mask_i = i_mask & ((idx_b + B * i_range) >= pad_offset)
    else:
        q_mask_i = i_mask & ((idx_b + B * i_range) < N)

    # Load q_tile [TILE_M, D] - query positions for this output tile
    q_block_ptr = (
        q_ptr + stride_q_e * idx_e + stride_q_h * idx_h
        + stride_q_b * (idx_b - pad_offset)
        + (stride_q_m * i_range[:, None] + stride_q_d * range_d[None, :])
    )
    q_tile = tl.load(q_block_ptr, mask=q_mask_i[:, None] & mask_d[None, :], other=0.0)

    # Initialize online softmax accumulators (per row)
    max_rows = tl.full([TILE_M], float('-inf'), dtype=tl.float32)
    sum_rows = tl.zeros([TILE_M], dtype=tl.float32)
    z_acc = tl.zeros([TILE_M, BLOCK_D], dtype=tl.float32)

    # Iterate over column tiles (j dimension - al/y positions)
    range_j = tl.arange(0, TILE_M)
    for j_start in tl.range(0, M, TILE_M, num_stages=3):
        j_range = j_start + range_j
        j_mask = j_range < M

        # Load al_tile [TILE_M, D]
        al_block_ptr = (
            al_ptr + stride_al_e * idx_e + stride_al_h * idx_h
            + stride_al_b * idx_b
            + (stride_al_m * j_range[:, None] + stride_al_d * range_d[None, :])
        )
        al_tile = tl.load(al_block_ptr, mask=j_mask[:, None] & mask_d[None, :], other=0.0)

        # Load cl_tile [TILE_M]
        cl_block_ptr = (
            cl_ptr + stride_cl_e * idx_e + stride_cl_h * idx_h
            + stride_cl_b * idx_b + stride_cl_m * j_range
        )
        cl_tile = tl.load(cl_block_ptr, mask=j_mask, other=0.0)

        # Load y_tile [TILE_M, D]
        y_block_ptr = (
            y_ptr + stride_y_e * idx_e + stride_y_h * idx_h
            + stride_y_b * idx_b
            + (stride_y_m * j_range[:, None] + stride_y_d * range_d[None, :])
        )
        y_tile = tl.load(y_block_ptr, mask=j_mask[:, None] & mask_d[None, :], other=0.0)

        # Compute attention scores [TILE_M_i, TILE_M_j]
        q_bf16 = q_tile.to(tl.bfloat16)
        al_bf16 = al_tile.to(tl.bfloat16)
        scores = tl.dot(q_bf16, tl.trans(al_bf16), out_dtype=tl.float32)  # [TILE_M, TILE_M]
        scores = scores - cl_tile[None, :]
        scores = tl.where(j_mask[None, :], scores, float('-inf'))

        # Online softmax update (Flash Attention style)
        tile_max = tl.max(scores, axis=1)  # [TILE_M] - max per row
        new_max = tl.maximum(max_rows, tile_max)

        # Rescale old accumulator and sum
        scale = tl.exp(max_rows - new_max)
        z_acc = z_acc * scale[:, None]
        sum_rows = sum_rows * scale

        # Compute exp scores with new max
        exp_scores = tl.exp(scores - new_max[:, None])

        # Accumulate weighted values
        y_bf16 = y_tile.to(tl.bfloat16)
        z_acc = z_acc + tl.dot(exp_scores.to(tl.bfloat16), y_bf16, out_dtype=tl.float32)
        sum_rows = sum_rows + tl.sum(exp_scores, axis=1)

        max_rows = new_max

    # Final normalization
    z = z_acc / sum_rows[:, None]

    # Store z
    z_block_ptr = (
        z_ptr + stride_z_e * idx_e + stride_z_h * idx_h
        + stride_z_b * (idx_b - pad_offset)
        + (stride_z_m * i_range[:, None] + stride_z_d * range_d[None, :])
    )
    tl.store(z_block_ptr, z.to(q_tile.dtype), mask=q_mask_i[:, None] & mask_d[None, :])


@triton.jit
def _z_kernel_tiled_tma(
    al_ptr, stride_al_e, stride_al_h, stride_al_m, stride_al_b, stride_al_d,
    q_ptr, stride_q_e, stride_q_h, stride_q_m, stride_q_b, stride_q_d,
    y_ptr, stride_y_e, stride_y_h, stride_y_m, stride_y_b, stride_y_d,
    cl_ptr, stride_cl_e, stride_cl_h, stride_cl_m, stride_cl_b,
    z_ptr, stride_z_e, stride_z_h, stride_z_m, stride_z_b, stride_z_d,
    TILE_M: tl.constexpr, BLOCK_D: tl.constexpr, PRE_PAD: tl.constexpr,
    H: tl.constexpr, M: tl.constexpr, B: tl.constexpr, D: tl.constexpr, N: tl.constexpr,
):
    """TMA-optimized tiled _z_kernel with software pipelining.

    Uses TMA descriptors for bulk memory transfers and prefetches next tiles
    while computing current tiles.
    """
    idx_eh = tl.program_id(0)
    idx_b = tl.program_id(1)
    idx_i_tile = tl.program_id(2)
    idx_e = idx_eh // H
    idx_h = idx_eh % H

    pad_offset = M * B - N if PRE_PAD else 0

    # Row indices for this tile (output rows)
    i_start = idx_i_tile * TILE_M
    i_range = i_start + tl.arange(0, TILE_M)
    i_mask = i_range < M

    range_d = tl.arange(0, BLOCK_D)
    mask_d = range_d < D

    if PRE_PAD:
        q_mask_i = i_mask & ((idx_b + B * i_range) >= pad_offset)
    else:
        q_mask_i = i_mask & ((idx_b + B * i_range) < N)

    # Base pointers for this (e, h, b) slice
    al_base = al_ptr + stride_al_e * idx_e + stride_al_h * idx_h + stride_al_b * idx_b
    y_base = y_ptr + stride_y_e * idx_e + stride_y_h * idx_h + stride_y_b * idx_b
    cl_base = cl_ptr + stride_cl_e * idx_e + stride_cl_h * idx_h + stride_cl_b * idx_b

    # Create TMA descriptors for al and y (2D tensors: [M, D])
    # Note: TMA requires contiguous memory, so we work with [M, D] slices
    al_desc = tl.make_tensor_descriptor(
        al_base,
        shape=[M, D],
        strides=[stride_al_m, stride_al_d],
        block_shape=[TILE_M, BLOCK_D],
    )
    y_desc = tl.make_tensor_descriptor(
        y_base,
        shape=[M, D],
        strides=[stride_y_m, stride_y_d],
        block_shape=[TILE_M, BLOCK_D],
    )

    # Load q_tile (only once, stays in registers)
    q_block_ptr = (
        q_ptr + stride_q_e * idx_e + stride_q_h * idx_h
        + stride_q_b * (idx_b - pad_offset)
        + (stride_q_m * i_range[:, None] + stride_q_d * range_d[None, :])
    )
    q_tile = tl.load(q_block_ptr, mask=q_mask_i[:, None] & mask_d[None, :], other=0.0)
    q_bf16 = q_tile.to(tl.bfloat16)

    # Initialize online softmax accumulators
    max_rows = tl.full([TILE_M], float('-inf'), dtype=tl.float32)
    sum_rows = tl.zeros([TILE_M], dtype=tl.float32)
    z_acc = tl.zeros([TILE_M, BLOCK_D], dtype=tl.float32)

    num_tiles = tl.cdiv(M, TILE_M)
    range_j = tl.arange(0, TILE_M)

    # Software pipelining: prefetch first tiles
    al_next = al_desc.load([0, 0])
    y_next = y_desc.load([0, 0])
    cl_next = tl.load(cl_base + stride_cl_m * range_j, mask=range_j < M, other=0.0)

    # Main loop with pipelining
    for tile_idx in range(num_tiles):
        j_start = tile_idx * TILE_M
        j_range = j_start + range_j
        j_mask = j_range < M

        # Use prefetched data
        al_tile = al_next
        y_tile = y_next
        cl_tile = cl_next

        # Prefetch next tiles (if not last iteration)
        next_tile_idx = tile_idx + 1
        if next_tile_idx < num_tiles:
            next_j_start = next_tile_idx * TILE_M
            al_next = al_desc.load([next_j_start, 0])
            y_next = y_desc.load([next_j_start, 0])
            next_j_range = next_j_start + range_j
            cl_next = tl.load(cl_base + stride_cl_m * next_j_range, mask=next_j_range < M, other=0.0)

        # Compute attention scores
        al_bf16 = al_tile.to(tl.bfloat16)
        scores = tl.dot(q_bf16, tl.trans(al_bf16), out_dtype=tl.float32)
        scores = scores - cl_tile[None, :]
        scores = tl.where(j_mask[None, :], scores, float('-inf'))

        # Online softmax update
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(max_rows, tile_max)

        scale = tl.exp(max_rows - new_max)
        z_acc = z_acc * scale[:, None]
        sum_rows = sum_rows * scale

        exp_scores = tl.exp(scores - new_max[:, None])

        y_bf16 = y_tile.to(tl.bfloat16)
        z_acc = z_acc + tl.dot(exp_scores.to(tl.bfloat16), y_bf16, out_dtype=tl.float32)
        sum_rows = sum_rows + tl.sum(exp_scores, axis=1)

        max_rows = new_max

    # Final normalization
    z = z_acc / sum_rows[:, None]

    # Store z
    z_block_ptr = (
        z_ptr + stride_z_e * idx_e + stride_z_h * idx_h
        + stride_z_b * (idx_b - pad_offset)
        + (stride_z_m * i_range[:, None] + stride_z_d * range_d[None, :])
    )
    tl.store(z_block_ptr, z.to(q_tile.dtype), mask=q_mask_i[:, None] & mask_d[None, :])


def get_optimal_num_warps(block_b: int, block_d: int) -> int:
    """Determine optimal num_warps based on block sizes for B200."""
    # B200 has 4 warp schedulers per SM and larger register file
    # Larger blocks benefit from more warps for better occupancy
    if block_b >= 128 and block_d >= 128:
        return 16
    elif block_b >= 128 and block_d >= 64:
        return 8
    elif block_b >= 64 and block_d >= 64:
        return 8
    elif block_b >= 32 and block_d >= 64:
        return 4
    else:
        return 4


def get_optimal_num_stages(block_b: int, block_d: int) -> int:
    """Determine optimal num_stages for B200 (larger shared memory)."""
    # B200 has 228KB shared memory per SM - can use more stages for pipelining
    # Higher stages hide memory latency better
    if block_b >= 128 and block_d >= 128:
        return 5
    elif block_b >= 64 and block_d >= 64:
        return 4
    elif block_b >= 32 and block_d >= 64:
        return 3
    else:
        return 2


def monarch_attention_triton(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None,
    T: int,
    B: int,
    pre_pad: bool,
    eps: float = 0.0,
) -> Tensor:
    assert T > 1
    check_inputs(q, k, v)

    # Ensure inputs are contiguous for coalesced memory access
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    E, H, N, D = q.shape
    M = triton.cdiv(N, B)

    # 2D grids for better workload distribution on B200's 148 SMs
    grid_ehm = (E * H, M)
    grid_ehb = (E * H, B)

    # Block sizes must be powers of 2 and >= actual dimensions for Triton
    BLOCK_D = max(triton.next_power_of_2(D), 16)

    # For large B, use tiled within-block kernels with online softmax
    # TILE_B for tiled kernels is autotuned by @triton.autotune
    MAX_BLOCK_B = 128
    use_tiled_b_kernels = B > MAX_BLOCK_B
    BLOCK_B = max(triton.next_power_of_2(B), 16) if not use_tiled_b_kernels else 64

    # For large M, cap BLOCK_M and use tiled kernels
    # The non-tiled kernels compute MxM attention matrices which exceed memory for large M
    raw_block_m = max(triton.next_power_of_2(M), 16)
    MAX_BLOCK_M = 128  # Maximum single-block M size for non-tiled kernels
    # For tiled kernels, TILE_M is autotuned by @triton.autotune for _z_kernel_tiled.
    # For the ar_cr pair, TILE_M must match between stats and accumulate kernels.
    TILED_BLOCK_M = 64
    use_tiled_kernels = raw_block_m > MAX_BLOCK_M
    BLOCK_M = min(raw_block_m, MAX_BLOCK_M)

    # TMA with software pipelining benefits from larger tile sizes (128 vs 64)
    # At TILE_M=128, TMA shows 2.64x speedup for M>=4096
    # But TILE_M=128 causes register pressure issues in _ar_cr kernels
    # TMA also requires contiguous [M, D] memory which our [E,H,M,B,D] layout doesn't provide
    # Future work: transpose to [E,H,B,M,D] layout for TMA benefits
    use_tma = False

    # Warp and stage counts for non-autotuned kernels (small M/B path)
    num_warps_b = get_optimal_num_warps(BLOCK_B, BLOCK_D)
    num_warps_m = get_optimal_num_warps(BLOCK_M, BLOCK_D)
    num_stages = get_optimal_num_stages(BLOCK_B, BLOCK_D)

    sm_scale = 1 / sqrt(D)

    q_strides = (q.stride(0), q.stride(1), B * q.stride(2), q.stride(2), q.stride(3))
    k_strides = (k.stride(0), k.stride(1), B * k.stride(2), k.stride(2), k.stride(3))
    v_strides = (v.stride(0), v.stride(1), B * v.stride(2), v.stride(2), v.stride(3))

    ar = torch.empty(E, H, M, B, D, device=q.device, dtype=q.dtype)
    al = torch.empty_like(ar)

    ar_strides = (ar.stride(0), ar.stride(1), ar.stride(2), ar.stride(3), ar.stride(4))
    al_strides = (al.stride(0), al.stride(1), al.stride(2), al.stride(3), al.stride(4))

    cr = torch.ones(E, H, M, B, device=q.device, dtype=torch.float)
    cl = torch.empty_like(cr)

    cr_strides = (cr.stride(0), cr.stride(1), cr.stride(2), cr.stride(3))
    cl_strides = (cl.stride(0), cl.stride(1), cl.stride(2), cl.stride(3))

    attn_mask_strides = (
        (attn_mask.stride(0), B * attn_mask.stride(1), attn_mask.stride(1))
        if attn_mask is not None
        else (0, 0, 0)
    )

    for t in range(T - 1):
        is_first_call = t == 0
        _ar = q if is_first_call else ar
        if use_tiled_b_kernels:
            # Grid uses lambda — TILE_B comes from autotuning
            grid_al_cl_tiled = lambda META: (E * H, M, triton.cdiv(B, META['TILE_B']))
            _al_cl_kernel_tiled[grid_al_cl_tiled](
                _ar,
                *ar_strides,
                k,
                *k_strides,
                cr,
                *cr_strides,
                al,
                *al_strides,
                cl,
                *cl_strides,
                attn_mask,
                *attn_mask_strides,
                sm_scale,
                HAS_ATTN_MASK=attn_mask is not None,  # type: ignore
                BLOCK_D=BLOCK_D,  # type: ignore
                PRE_PAD=pre_pad,  # type: ignore
                EPS=eps,  # type: ignore
                IS_FIRST_CALL=is_first_call,  # type: ignore
                H=H,  # type: ignore
                M=M,  # type: ignore
                B=B,  # type: ignore
                D=D,  # type: ignore
                N=N,  # type: ignore
            )
        else:
            _al_cl_kernel[grid_ehm](
                _ar,
                *ar_strides,
                k,
                *k_strides,
                cr,
                *cr_strides,
                al,
                *al_strides,
                cl,
                *cl_strides,
                attn_mask,
                *attn_mask_strides,
                sm_scale,
                HAS_ATTN_MASK=attn_mask is not None,  # type: ignore
                BLOCK_B=BLOCK_B,  # type: ignore
                BLOCK_D=BLOCK_D,  # type: ignore
                PRE_PAD=pre_pad,  # type: ignore
                EPS=eps,  # type: ignore
                IS_FIRST_CALL=is_first_call,  # type: ignore
                H=H,  # type: ignore
                M=M,  # type: ignore
                B=B,  # type: ignore
                D=D,  # type: ignore
                N=N,  # type: ignore
                num_warps=num_warps_b,
                num_stages=num_stages,
            )

        if use_tiled_kernels:
            # Use tiled kernels for large M
            # TILE_M must match between stats and accumulate kernels
            num_m_tiles = triton.cdiv(M, TILED_BLOCK_M)
            grid_tiled = (E * H, B, num_m_tiles)

            # Allocate intermediate buffers for softmax stats
            softmax_max = torch.empty(E, H, M, B, device=q.device, dtype=torch.float32)
            softmax_sum = torch.empty_like(softmax_max)
            max_strides = (softmax_max.stride(0), softmax_max.stride(1), softmax_max.stride(2), softmax_max.stride(3))
            sum_strides = (softmax_sum.stride(0), softmax_sum.stride(1), softmax_sum.stride(2), softmax_sum.stride(3))

            if use_tma:
                num_warps_tiled = get_optimal_num_warps(TILED_BLOCK_M, BLOCK_D)
                # Use TMA-optimized kernels with software pipelining for M > 4096
                # Phase 1: Compute softmax stats with TMA
                _ar_cr_softmax_stats_kernel_tma[grid_tiled](
                    al, *al_strides,
                    q, *q_strides,
                    cl, *cl_strides,
                    softmax_max, *max_strides,
                    softmax_sum, *sum_strides,
                    TILE_M=TILED_BLOCK_M,  # type: ignore
                    BLOCK_D=BLOCK_D,  # type: ignore
                    PRE_PAD=pre_pad,  # type: ignore
                    H=H,  # type: ignore
                    M=M,  # type: ignore
                    B=B,  # type: ignore
                    D=D,  # type: ignore
                    N=N,  # type: ignore
                    num_warps=num_warps_tiled,
                    num_stages=num_stages,
                )

                # Phase 2: Accumulate ar and cr with TMA
                _ar_cr_accumulate_kernel_tma[grid_tiled](
                    al, *al_strides,
                    q, *q_strides,
                    cl, *cl_strides,
                    softmax_max, *max_strides,
                    softmax_sum, *sum_strides,
                    ar, *ar_strides,
                    cr, *cr_strides,
                    TILE_M=TILED_BLOCK_M,  # type: ignore
                    BLOCK_D=BLOCK_D,  # type: ignore
                    PRE_PAD=pre_pad,  # type: ignore
                    H=H,  # type: ignore
                    M=M,  # type: ignore
                    B=B,  # type: ignore
                    D=D,  # type: ignore
                    N=N,  # type: ignore
                    num_warps=num_warps_tiled,
                    num_stages=num_stages,
                )
            else:
                # Autotuned tiled kernels (num_warps/num_stages chosen by @triton.autotune)
                # Phase 1: Compute softmax stats
                _ar_cr_softmax_stats_kernel[grid_tiled](
                    al, *al_strides,
                    q, *q_strides,
                    cl, *cl_strides,
                    softmax_max, *max_strides,
                    softmax_sum, *sum_strides,
                    TILE_M=TILED_BLOCK_M,  # type: ignore
                    BLOCK_D=BLOCK_D,  # type: ignore
                    PRE_PAD=pre_pad,  # type: ignore
                    H=H,  # type: ignore
                    M=M,  # type: ignore
                    B=B,  # type: ignore
                    D=D,  # type: ignore
                    N=N,  # type: ignore
                )

                # Phase 2: Accumulate ar and cr
                _ar_cr_accumulate_kernel[grid_tiled](
                    al, *al_strides,
                    q, *q_strides,
                    cl, *cl_strides,
                    softmax_max, *max_strides,
                    softmax_sum, *sum_strides,
                    ar, *ar_strides,
                    cr, *cr_strides,
                    TILE_M=TILED_BLOCK_M,  # type: ignore
                    BLOCK_D=BLOCK_D,  # type: ignore
                    PRE_PAD=pre_pad,  # type: ignore
                    H=H,  # type: ignore
                    M=M,  # type: ignore
                    B=B,  # type: ignore
                    D=D,  # type: ignore
                    N=N,  # type: ignore
                )
        else:
            _ar_cr_kernel[grid_ehb](
                al,
                *al_strides,
                q,
                *q_strides,
                cl,
                *cl_strides,
                ar,
                *ar_strides,
                cr,
                *cr_strides,
                attn_mask,
                *attn_mask_strides,
                HAS_ATTN_MASK=attn_mask is not None,  # type: ignore
                BLOCK_M=BLOCK_M,  # type: ignore
                BLOCK_D=BLOCK_D,  # type: ignore
                PRE_PAD=pre_pad,  # type: ignore
                H=H,  # type: ignore
                M=M,  # type: ignore
                B=B,  # type: ignore
                D=D,  # type: ignore
                N=N,  # type: ignore
                num_warps=num_warps_m,
                num_stages=num_stages,
            )

    y = torch.empty_like(al)
    y_strides = (y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4))

    if use_tiled_b_kernels:
        # Grid uses lambda — TILE_B comes from autotuning
        grid_al_y_cl_tiled = lambda META: (E * H, M, triton.cdiv(B, META['TILE_B']))
        _al_y_cl_kernel_tiled[grid_al_y_cl_tiled](
            ar,
            *ar_strides,
            k,
            *k_strides,
            v,
            *v_strides,
            cr,
            *cr_strides,
            al,
            *al_strides,
            y,
            *y_strides,
            cl,
            *cl_strides,
            attn_mask,
            *attn_mask_strides,
            sm_scale,
            HAS_ATTN_MASK=attn_mask is not None,  # type: ignore
            BLOCK_D=BLOCK_D,  # type: ignore
            PRE_PAD=pre_pad,  # type: ignore
            EPS=eps,  # type: ignore
            H=H,  # type: ignore
            M=M,  # type: ignore
            B=B,  # type: ignore
            D=D,  # type: ignore
            N=N,  # type: ignore
        )
    else:
        _al_y_cl_kernel[grid_ehm](
            ar,
            *ar_strides,
            k,
            *k_strides,
            v,
            *v_strides,
            cr,
            *cr_strides,
            al,
            *al_strides,
            y,
            *y_strides,
            cl,
            *cl_strides,
            attn_mask,
            *attn_mask_strides,
            sm_scale,
            HAS_ATTN_MASK=attn_mask is not None,  # type: ignore
            BLOCK_B=BLOCK_B,  # type: ignore
            BLOCK_D=BLOCK_D,  # type: ignore
            PRE_PAD=pre_pad,  # type: ignore
            EPS=eps,  # type: ignore
            H=H,  # type: ignore
            M=M,  # type: ignore
            B=B,  # type: ignore
            D=D,  # type: ignore
            N=N,  # type: ignore
            num_warps=num_warps_b,
            num_stages=num_stages,
        )

    z = torch.empty_like(v)
    z_strides = (z.stride(0), z.stride(1), B * z.stride(2), z.stride(2), z.stride(3))

    if use_tiled_kernels:
        if use_tma:
            # Use TMA-optimized kernel with software pipelining for M > 4096
            num_m_tiles = triton.cdiv(M, TILED_BLOCK_M)
            grid_tiled_z = (E * H, B, num_m_tiles)
            num_warps_tiled = get_optimal_num_warps(TILED_BLOCK_M, BLOCK_D)
            _z_kernel_tiled_tma[grid_tiled_z](
                al, *al_strides,
                q, *q_strides,
                y, *y_strides,
                cl, *cl_strides,
                z, *z_strides,
                TILE_M=TILED_BLOCK_M,  # type: ignore
                BLOCK_D=BLOCK_D,  # type: ignore
                PRE_PAD=pre_pad,  # type: ignore
                H=H,  # type: ignore
                M=M,  # type: ignore
                B=B,  # type: ignore
                D=D,  # type: ignore
                N=N,  # type: ignore
                num_warps=num_warps_tiled,
                num_stages=num_stages,
            )
        else:
            # Autotuned tiled z kernel — TILE_M, num_warps, num_stages from @triton.autotune
            grid_z_tiled = lambda META: (E * H, B, triton.cdiv(M, META['TILE_M']))
            _z_kernel_tiled[grid_z_tiled](
                al, *al_strides,
                q, *q_strides,
                y, *y_strides,
                cl, *cl_strides,
                z, *z_strides,
                BLOCK_D=BLOCK_D,  # type: ignore
                PRE_PAD=pre_pad,  # type: ignore
                H=H,  # type: ignore
                M=M,  # type: ignore
                B=B,  # type: ignore
                D=D,  # type: ignore
                N=N,  # type: ignore
            )
    else:
        _z_kernel[grid_ehb](
            al,
            *al_strides,
            q,
            *q_strides,
            y,
            *y_strides,
            cl,
            *cl_strides,
            z,
            *z_strides,
            BLOCK_M=BLOCK_M,  # type: ignore
            BLOCK_D=BLOCK_D,  # type: ignore
            PRE_PAD=pre_pad,  # type: ignore
            H=H,  # type: ignore
            M=M,  # type: ignore
            B=B,  # type: ignore
            D=D,  # type: ignore
            N=N,  # type: ignore
            num_warps=num_warps_m,
            num_stages=num_stages,
        )

    return z
