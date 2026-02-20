#!/usr/bin/env python3
"""
FP8 correctness tests for Monarch Attention Triton kernels.

Tests fp8 (float8_e4m3fn) tensor core support against the bf16 torch reference.
These tests are expected to FAIL until fp8 support is added to the kernels (TDD).

Usage:
    python -m ma.test_fp8_correctness
"""

import torch

from ma.ma_torch import monarch_attention_torch
from ma.ma_triton import monarch_attention_triton

# Standard test configuration
E = 2   # batch size
H = 8   # number of heads
D = 64  # head dimension
T = 2   # decomposition steps


def _make_inputs(N, B, q_dtype, kv_dtype):
    """Create q, k, v tensors with specified dtypes.

    Generates bf16 random tensors first, then casts to the target dtypes.
    This ensures fp8 values are in a reasonable range (bf16 -> fp8 clamps).
    """
    torch.manual_seed(42)
    q_bf16 = torch.randn(E, H, N, D, device="cuda", dtype=torch.bfloat16)
    k_bf16 = torch.randn(E, H, N, D, device="cuda", dtype=torch.bfloat16)
    v_bf16 = torch.randn(E, H, N, D, device="cuda", dtype=torch.bfloat16)

    q = q_bf16.to(q_dtype)
    k = k_bf16.to(kv_dtype)
    v = v_bf16.to(kv_dtype)

    return q, k, v, q_bf16, k_bf16, v_bf16


def test_fp8_correctness(N: int = 512, B: int = 32, verbose: bool = True) -> dict:
    """Compare fp8 triton output against bf16 torch reference.

    Tests three configurations:
      1. bf16 baseline: triton bf16 vs torch bf16 (max_diff < 0.1)
      2. fp8 KV with bf16 Q: triton fp8-kv vs torch bf16 (max_diff < 0.5)
      3. All fp8: triton all-fp8 vs torch bf16 (max_diff < 0.5)

    Args:
        N: sequence length
        B: block size
        verbose: print results
    Returns:
        dict with test results
    """
    results = {}
    pre_pad = True

    if verbose:
        print(f"\n{'='*60}")
        print(f"FP8 Correctness Test: N={N}, B={B}, E={E}, H={H}, D={D}, T={T}")
        print(f"{'='*60}")

    # --- bf16 baseline ---
    torch.manual_seed(42)
    q_bf16 = torch.randn(E, H, N, D, device="cuda", dtype=torch.bfloat16)
    k_bf16 = torch.randn(E, H, N, D, device="cuda", dtype=torch.bfloat16)
    v_bf16 = torch.randn(E, H, N, D, device="cuda", dtype=torch.bfloat16)

    ref = monarch_attention_torch(q_bf16, k_bf16, v_bf16, None, T, B, pre_pad)
    out_bf16 = monarch_attention_triton(q_bf16, k_bf16, v_bf16, None, T, B, pre_pad)

    bf16_diff = (ref - out_bf16).abs().max().item()
    bf16_pass = bf16_diff < 0.1
    results["bf16_max_diff"] = bf16_diff
    results["bf16_passed"] = bf16_pass

    if verbose:
        print(f"  bf16 baseline:     max_diff={bf16_diff:.6f}  {'PASS' if bf16_pass else 'FAIL'}")

    # --- fp8 KV with bf16 Q ---
    k_fp8 = k_bf16.to(torch.float8_e4m3fn)
    v_fp8 = v_bf16.to(torch.float8_e4m3fn)

    out_fp8kv = monarch_attention_triton(q_bf16, k_fp8, v_fp8, None, T, B, pre_pad)

    fp8kv_diff = (ref - out_fp8kv.to(ref.dtype)).abs().max().item()
    fp8kv_pass = fp8kv_diff < 0.5
    results["fp8kv_max_diff"] = fp8kv_diff
    results["fp8kv_passed"] = fp8kv_pass

    if verbose:
        print(f"  fp8 KV + bf16 Q:   max_diff={fp8kv_diff:.6f}  {'PASS' if fp8kv_pass else 'FAIL'}")

    # --- all fp8 ---
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)

    out_fp8all = monarch_attention_triton(q_fp8, k_fp8, v_fp8, None, T, B, pre_pad)

    fp8all_diff = (ref - out_fp8all.to(ref.dtype)).abs().max().item()
    fp8all_pass = fp8all_diff < 0.5
    results["fp8all_max_diff"] = fp8all_diff
    results["fp8all_passed"] = fp8all_pass

    if verbose:
        print(f"  all fp8:           max_diff={fp8all_diff:.6f}  {'PASS' if fp8all_pass else 'FAIL'}")

    results["all_passed"] = bf16_pass and fp8kv_pass and fp8all_pass
    return results


def test_fp8_large_n(verbose: bool = True) -> dict:
    """Test fp8 at large N where tiled kernels are exercised.

    Configurations:
      - N=32768,  B=128
      - N=65536,  B=256
      - N=131072, B=256
    """
    configs = [
        (32768,  128),
        (65536,  256),
        (131072, 256),
    ]

    if verbose:
        print(f"\n{'='*60}")
        print("FP8 Large-N Tests (tiled kernels)")
        print(f"{'='*60}")

    results = {}
    all_passed = True

    for N, B in configs:
        pre_pad = True

        torch.manual_seed(42)
        q_bf16 = torch.randn(E, H, N, D, device="cuda", dtype=torch.bfloat16)
        k_bf16 = torch.randn(E, H, N, D, device="cuda", dtype=torch.bfloat16)
        v_bf16 = torch.randn(E, H, N, D, device="cuda", dtype=torch.bfloat16)

        ref = monarch_attention_torch(q_bf16, k_bf16, v_bf16, None, T, B, pre_pad)

        # fp8 KV with bf16 Q
        k_fp8 = k_bf16.to(torch.float8_e4m3fn)
        v_fp8 = v_bf16.to(torch.float8_e4m3fn)

        out = monarch_attention_triton(q_bf16, k_fp8, v_fp8, None, T, B, pre_pad)

        diff = (ref - out.to(ref.dtype)).abs().max().item()
        passed = diff < 0.5

        results[f"N{N}_B{B}_max_diff"] = diff
        results[f"N{N}_B{B}_passed"] = passed
        all_passed = all_passed and passed

        if verbose:
            print(f"  N={N:>6}, B={B:>3}: max_diff={diff:.6f}  {'PASS' if passed else 'FAIL'}")

    results["all_passed"] = all_passed
    return results


def run_all():
    """Run all fp8 correctness tests."""
    print("=" * 60)
    print("FP8 CORRECTNESS TESTS FOR MONARCH ATTENTION")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    # Standard sizes
    small_configs = [
        (256,  32),
        (512,  32),
        (1024, 32),
        (2048, 64),
        (4096, 64),
    ]

    all_passed = True

    for N, B in small_configs:
        try:
            r = test_fp8_correctness(N=N, B=B)
            all_passed = all_passed and r["all_passed"]
        except Exception as e:
            print(f"  ERROR at N={N}, B={B}: {e}")
            all_passed = False

    # Large-N tiled kernel tests
    try:
        r = test_fp8_large_n()
        all_passed = all_passed and r["all_passed"]
    except Exception as e:
        print(f"  ERROR in large-N tests: {e}")
        all_passed = False

    print(f"\n{'='*60}")
    print(f"OVERALL: {'ALL PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print(f"{'='*60}")

    return all_passed


if __name__ == "__main__":
    run_all()
