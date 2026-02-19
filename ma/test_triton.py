#!/usr/bin/env python3
"""
Verification script for Triton implementation with fp16/bf16 support.

Tests:
1. Correctness across all dtypes (fp32, fp16, bf16) against torch reference
2. Performance benchmarks comparing torch vs triton
3. Various sequence lengths and configurations
"""

import torch
import triton

from ma.ma_torch import monarch_attention_torch
from ma.ma_triton import monarch_attention_triton
from ma.monarch_attention import MonarchAttention, PadType


def test_correctness(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 512,
    head_dim: int = 64,
    block_size: int = 32,
    num_steps: int = 2,
    dtype: torch.dtype = torch.float32,
    pre_pad: bool = True,
    verbose: bool = True,
) -> dict:
    """Test correctness of Triton implementation against torch reference."""
    torch.manual_seed(42)

    # Create inputs
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)

    # Torch reference (always in same dtype)
    out_torch = monarch_attention_torch(q, k, v, None, num_steps, block_size, pre_pad)

    # Triton implementation
    out_triton = monarch_attention_triton(q, k, v, None, num_steps, block_size, pre_pad)

    # Compute differences
    abs_diff = (out_torch - out_triton).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    # Relative error (avoid division by zero)
    rel_diff = abs_diff / (out_torch.abs() + 1e-8)
    max_rel_diff = rel_diff.max().item()

    results = {
        "dtype": str(dtype),
        "seq_len": seq_len,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "max_rel_diff": max_rel_diff,
        "passed": max_diff < 0.01 if dtype == torch.float32 else max_diff < 0.1,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Correctness Test: dtype={dtype}, seq_len={seq_len}")
        print(f"{'='*60}")
        print(f"Max absolute diff: {max_diff:.6f}")
        print(f"Mean absolute diff: {mean_diff:.6f}")
        print(f"Max relative diff: {max_rel_diff:.6f}")
        print(f"Status: {'PASSED' if results['passed'] else 'FAILED'}")

    return results


def test_module_dtype(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 512,
    head_dim: int = 64,
    block_size: int = 32,
    num_steps: int = 2,
    input_dtype: torch.dtype = torch.float32,
    compute_dtype: torch.dtype = torch.float16,
    verbose: bool = True,
) -> dict:
    """Test MonarchAttention module with dtype conversion."""
    torch.manual_seed(42)

    # Create inputs in input_dtype
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=input_dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=input_dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=input_dtype)

    # Create module with dtype conversion
    ma = MonarchAttention(
        block_size=block_size,
        num_steps=num_steps,
        pad_type=PadType.pre,
        impl="triton",
        dtype=compute_dtype,
    )

    # Run forward pass
    out = ma(q, k, v)

    # Check output dtype matches input dtype
    dtype_matches = out.dtype == input_dtype

    results = {
        "input_dtype": str(input_dtype),
        "compute_dtype": str(compute_dtype),
        "output_dtype": str(out.dtype),
        "dtype_matches": dtype_matches,
        "passed": dtype_matches,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Module dtype test: input={input_dtype}, compute={compute_dtype}")
        print(f"{'='*60}")
        print(f"Output dtype: {out.dtype}")
        print(f"Dtype matches input: {dtype_matches}")
        print(f"Status: {'PASSED' if results['passed'] else 'FAILED'}")

    return results


def benchmark_performance(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 512,
    head_dim: int = 64,
    block_size: int = 32,
    num_steps: int = 2,
    dtype: torch.dtype = torch.float32,
    warmup: int = 10,
    rep: int = 100,
    verbose: bool = True,
) -> dict:
    """Benchmark torch vs triton performance."""
    torch.manual_seed(42)

    # Create inputs
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)

    # Benchmark torch
    def torch_fn():
        return monarch_attention_torch(q, k, v, None, num_steps, block_size, True)

    # Benchmark triton
    def triton_fn():
        return monarch_attention_triton(q, k, v, None, num_steps, block_size, True)

    # Warmup and measure
    ms_torch = triton.testing.do_bench(torch_fn, warmup=warmup, rep=rep)
    ms_triton = triton.testing.do_bench(triton_fn, warmup=warmup, rep=rep)

    speedup = ms_torch / ms_triton

    results = {
        "dtype": str(dtype),
        "seq_len": seq_len,
        "torch_ms": ms_torch,
        "triton_ms": ms_triton,
        "speedup": speedup,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Performance Benchmark: dtype={dtype}, seq_len={seq_len}")
        print(f"{'='*60}")
        print(f"Torch:  {ms_torch:.3f} ms")
        print(f"Triton: {ms_triton:.3f} ms")
        print(f"Speedup: {speedup:.2f}x")

    return results


def benchmark_cuda_graph(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 512,
    head_dim: int = 64,
    block_size: int = 32,
    num_steps: int = 2,
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 10,
    rep: int = 100,
    verbose: bool = True,
) -> dict:
    """Benchmark CUDA graph vs standard Triton."""
    torch.manual_seed(42)

    # Create inputs
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)

    # Create modules
    ma_standard = MonarchAttention(block_size, num_steps, PadType.pre, impl="triton", use_cuda_graph=False)
    ma_graph = MonarchAttention(block_size, num_steps, PadType.pre, impl="triton", use_cuda_graph=True)

    # Warmup
    _ = ma_standard(q, k, v)
    _ = ma_graph(q, k, v)

    # Benchmark
    ms_standard = triton.testing.do_bench(lambda: ma_standard(q, k, v), warmup=warmup, rep=rep)
    ms_graph = triton.testing.do_bench(lambda: ma_graph(q, k, v), warmup=warmup, rep=rep)

    results = {
        "dtype": str(dtype),
        "seq_len": seq_len,
        "standard_ms": ms_standard,
        "graph_ms": ms_graph,
        "graph_speedup": ms_standard / ms_graph,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"CUDA Graph Benchmark: dtype={dtype}, seq_len={seq_len}")
        print(f"{'='*60}")
        print(f"Standard Triton: {ms_standard:.3f} ms")
        print(f"With CUDA Graph: {ms_graph:.3f} ms")
        print(f"Graph speedup:   {results['graph_speedup']:.2f}x")

    return results


def run_all_tests():
    """Run all verification tests."""
    print("=" * 70)
    print("MONARCH ATTENTION TRITON VERIFICATION")
    print("=" * 70)

    # Check GPU info
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    all_results = {"correctness": [], "module_dtype": [], "performance": []}

    # Test 1: Correctness across dtypes
    print("\n" + "=" * 70)
    print("CORRECTNESS TESTS")
    print("=" * 70)

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        for seq_len in [256, 512, 1024]:
            try:
                result = test_correctness(dtype=dtype, seq_len=seq_len)
                all_results["correctness"].append(result)
            except Exception as e:
                print(f"ERROR: dtype={dtype}, seq_len={seq_len}: {e}")

    # Test 2: Module dtype conversion
    print("\n" + "=" * 70)
    print("MODULE DTYPE CONVERSION TESTS")
    print("=" * 70)

    dtype_pairs = [
        (torch.float32, torch.float16),
        (torch.float32, torch.bfloat16),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
    ]

    for input_dtype, compute_dtype in dtype_pairs:
        try:
            result = test_module_dtype(input_dtype=input_dtype, compute_dtype=compute_dtype)
            all_results["module_dtype"].append(result)
        except Exception as e:
            print(f"ERROR: input={input_dtype}, compute={compute_dtype}: {e}")

    # Test 3: Performance benchmarks
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 70)

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        for seq_len in [256, 512, 1024, 2048, 4096]:
            try:
                result = benchmark_performance(dtype=dtype, seq_len=seq_len)
                all_results["performance"].append(result)
            except Exception as e:
                print(f"ERROR: dtype={dtype}, seq_len={seq_len}: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    correctness_passed = sum(1 for r in all_results["correctness"] if r["passed"])
    correctness_total = len(all_results["correctness"])
    print(f"Correctness: {correctness_passed}/{correctness_total} passed")

    module_passed = sum(1 for r in all_results["module_dtype"] if r["passed"])
    module_total = len(all_results["module_dtype"])
    print(f"Module dtype: {module_passed}/{module_total} passed")

    if all_results["performance"]:
        avg_speedup = sum(r["speedup"] for r in all_results["performance"]) / len(
            all_results["performance"]
        )
        print(f"Average speedup: {avg_speedup:.2f}x")

    return all_results


if __name__ == "__main__":
    run_all_tests()
