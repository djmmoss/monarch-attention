"""Benchmark individual kernels to identify optimization targets."""

import torch
import triton
from math import sqrt


def benchmark_monarch_kernels():
    """Benchmark each kernel phase to identify bottlenecks."""
    from ma.ma_triton import (
        _al_cl_kernel, _ar_cr_kernel, _al_y_cl_kernel, _z_kernel,
        _ar_cr_softmax_stats_kernel, _ar_cr_accumulate_kernel, _z_kernel_tiled,
        get_optimal_num_warps, get_optimal_num_stages,
    )

    print("=" * 70)
    print("Monarch Attention Kernel Benchmarks")
    print("=" * 70)

    # Test configurations
    configs = [
        # (E, H, N, D, B, T) - typical configs
        (2, 8, 512, 64, 32, 2),      # Small
        (2, 8, 4096, 64, 32, 2),     # Medium
        (2, 8, 16384, 64, 32, 2),    # Large (uses tiled kernels)
        (2, 8, 65536, 64, 32, 2),    # Very large
    ]

    for E, H, N, D, B, T in configs:
        M = triton.cdiv(N, B)
        print(f"\nConfig: E={E}, H={H}, N={N}, D={D}, B={B}, T={T}, M={M}")
        print("-" * 60)

        # Block sizes
        BLOCK_B = max(triton.next_power_of_2(B), 16)
        BLOCK_D = max(triton.next_power_of_2(D), 16)
        raw_block_m = max(triton.next_power_of_2(M), 16)
        MAX_BLOCK_M = 128
        use_tiled = raw_block_m > MAX_BLOCK_M
        BLOCK_M = min(raw_block_m, MAX_BLOCK_M)

        num_warps_b = get_optimal_num_warps(BLOCK_B, BLOCK_D)
        num_warps_m = get_optimal_num_warps(BLOCK_M, BLOCK_D)
        num_stages = get_optimal_num_stages(BLOCK_B, BLOCK_D)

        sm_scale = 1 / sqrt(D)

        # Allocate tensors
        q = torch.randn(E, H, N, D, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(E, H, N, D, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(E, H, N, D, device='cuda', dtype=torch.bfloat16)

        ar = torch.empty(E, H, M, B, D, device='cuda', dtype=torch.bfloat16)
        al = torch.empty_like(ar)
        y = torch.empty_like(ar)
        z = torch.empty(E, H, N, D, device='cuda', dtype=torch.bfloat16)

        cr = torch.ones(E, H, M, B, device='cuda', dtype=torch.float32)
        cl = torch.empty_like(cr)

        # Strides
        q_strides = (q.stride(0), q.stride(1), B * q.stride(2), q.stride(2), q.stride(3))
        k_strides = (k.stride(0), k.stride(1), B * k.stride(2), k.stride(2), k.stride(3))
        v_strides = (v.stride(0), v.stride(1), B * v.stride(2), v.stride(2), v.stride(3))
        ar_strides = (ar.stride(0), ar.stride(1), ar.stride(2), ar.stride(3), ar.stride(4))
        al_strides = (al.stride(0), al.stride(1), al.stride(2), al.stride(3), al.stride(4))
        y_strides = (y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4))
        z_strides = (z.stride(0), z.stride(1), B * z.stride(2), z.stride(2), z.stride(3))
        cr_strides = (cr.stride(0), cr.stride(1), cr.stride(2), cr.stride(3))
        cl_strides = (cl.stride(0), cl.stride(1), cl.stride(2), cl.stride(3))

        grid_ehm = (E * H, M)
        grid_ehb = (E * H, B)

        # Benchmark _al_cl_kernel
        def run_al_cl():
            _al_cl_kernel[grid_ehm](
                q, *ar_strides,  # ar input (using q as stand-in for first call)
                k, *k_strides,
                cr, *cr_strides,
                al, *al_strides,
                cl, *cl_strides,
                None, 0, 0, 0,  # mask
                sm_scale,
                HAS_ATTN_MASK=False, BLOCK_B=BLOCK_B, BLOCK_D=BLOCK_D,
                PRE_PAD=True, EPS=0.0, IS_FIRST_CALL=True,
                H=H, M=M, B=B, D=D, N=N,
                num_warps=num_warps_b, num_stages=num_stages,
            )

        ms_al_cl = triton.testing.do_bench(run_al_cl, warmup=10, rep=50)
        print(f"  _al_cl_kernel:     {ms_al_cl:.4f} ms")

        # Benchmark _ar_cr_kernel (or tiled)
        if use_tiled:
            num_m_tiles = triton.cdiv(M, BLOCK_M)
            grid_tiled = (E * H, B, num_m_tiles)

            softmax_max = torch.empty(E, H, M, B, device='cuda', dtype=torch.float32)
            softmax_sum = torch.empty_like(softmax_max)
            max_strides = (softmax_max.stride(0), softmax_max.stride(1), softmax_max.stride(2), softmax_max.stride(3))
            sum_strides = (softmax_sum.stride(0), softmax_sum.stride(1), softmax_sum.stride(2), softmax_sum.stride(3))

            def run_ar_cr_stats():
                _ar_cr_softmax_stats_kernel[grid_tiled](
                    al, *al_strides,
                    q, *q_strides,
                    cl, *cl_strides,
                    softmax_max, *max_strides,
                    softmax_sum, *sum_strides,
                    TILE_M=BLOCK_M, BLOCK_D=BLOCK_D, PRE_PAD=True,
                    H=H, M=M, B=B, D=D, N=N,
                    num_warps=num_warps_m, num_stages=num_stages,
                )

            def run_ar_cr_accum():
                _ar_cr_accumulate_kernel[grid_tiled](
                    al, *al_strides,
                    q, *q_strides,
                    cl, *cl_strides,
                    softmax_max, *max_strides,
                    softmax_sum, *sum_strides,
                    ar, *ar_strides,
                    cr, *cr_strides,
                    TILE_M=BLOCK_M, BLOCK_D=BLOCK_D, PRE_PAD=True,
                    H=H, M=M, B=B, D=D, N=N,
                    num_warps=num_warps_m, num_stages=num_stages,
                )

            ms_ar_cr_stats = triton.testing.do_bench(run_ar_cr_stats, warmup=10, rep=50)
            ms_ar_cr_accum = triton.testing.do_bench(run_ar_cr_accum, warmup=10, rep=50)
            print(f"  _ar_cr_stats:      {ms_ar_cr_stats:.4f} ms (tiled)")
            print(f"  _ar_cr_accum:      {ms_ar_cr_accum:.4f} ms (tiled)")
            ms_ar_cr_total = ms_ar_cr_stats + ms_ar_cr_accum
        else:
            def run_ar_cr():
                _ar_cr_kernel[grid_ehb](
                    al, *al_strides,
                    q, *q_strides,
                    cl, *cl_strides,
                    ar, *ar_strides,
                    cr, *cr_strides,
                    None, 0, 0, 0,  # mask
                    HAS_ATTN_MASK=False, BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
                    PRE_PAD=True, H=H, M=M, B=B, D=D, N=N,
                    num_warps=num_warps_m, num_stages=num_stages,
                )

            ms_ar_cr = triton.testing.do_bench(run_ar_cr, warmup=10, rep=50)
            print(f"  _ar_cr_kernel:     {ms_ar_cr:.4f} ms")
            ms_ar_cr_total = ms_ar_cr

        # Benchmark _al_y_cl_kernel
        def run_al_y_cl():
            _al_y_cl_kernel[grid_ehm](
                ar, *ar_strides,
                k, *k_strides,
                v, *v_strides,
                cr, *cr_strides,
                al, *al_strides,
                y, *y_strides,
                cl, *cl_strides,
                None, 0, 0, 0,  # mask
                sm_scale,
                HAS_ATTN_MASK=False, BLOCK_B=BLOCK_B, BLOCK_D=BLOCK_D,
                PRE_PAD=True, EPS=0.0, H=H, M=M, B=B, D=D, N=N,
                num_warps=num_warps_b, num_stages=num_stages,
            )

        ms_al_y_cl = triton.testing.do_bench(run_al_y_cl, warmup=10, rep=50)
        print(f"  _al_y_cl_kernel:   {ms_al_y_cl:.4f} ms")

        # Benchmark _z_kernel (or tiled)
        if use_tiled:
            num_m_tiles = triton.cdiv(M, BLOCK_M)
            grid_tiled_z = (E * H, B, num_m_tiles)

            def run_z_tiled():
                _z_kernel_tiled[grid_tiled_z](
                    al, *al_strides,
                    q, *q_strides,
                    y, *y_strides,
                    cl, *cl_strides,
                    z, *z_strides,
                    TILE_M=BLOCK_M, BLOCK_D=BLOCK_D, PRE_PAD=True,
                    H=H, M=M, B=B, D=D, N=N,
                    num_warps=num_warps_m, num_stages=num_stages,
                )

            ms_z = triton.testing.do_bench(run_z_tiled, warmup=10, rep=50)
            print(f"  _z_kernel_tiled:   {ms_z:.4f} ms")
        else:
            def run_z():
                _z_kernel[grid_ehb](
                    al, *al_strides,
                    q, *q_strides,
                    y, *y_strides,
                    cl, *cl_strides,
                    z, *z_strides,
                    BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D, PRE_PAD=True,
                    H=H, M=M, B=B, D=D, N=N,
                    num_warps=num_warps_m, num_stages=num_stages,
                )

            ms_z = triton.testing.do_bench(run_z, warmup=10, rep=50)
            print(f"  _z_kernel:         {ms_z:.4f} ms")

        # Calculate totals
        # T-1 iterations of (al_cl + ar_cr), then 1 (al_y_cl + z)
        iter_time = ms_al_cl + ms_ar_cr_total
        final_time = ms_al_y_cl + ms_z
        total_estimated = (T - 1) * iter_time + final_time

        print(f"\n  Estimated total (T={T}): {total_estimated:.4f} ms")
        print(f"    - {T-1}x iteration: {(T-1) * iter_time:.4f} ms")
        print(f"    - final step:      {final_time:.4f} ms")

        # Identify bottleneck
        times = {
            '_al_cl_kernel': ms_al_cl * (T - 1),
            '_ar_cr (total)': ms_ar_cr_total * (T - 1),
            '_al_y_cl_kernel': ms_al_y_cl,
            '_z_kernel': ms_z,
        }
        bottleneck = max(times, key=times.get)
        print(f"  Bottleneck: {bottleneck} ({times[bottleneck]:.4f} ms)")


if __name__ == "__main__":
    benchmark_monarch_kernels()
