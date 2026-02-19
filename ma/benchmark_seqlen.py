"""Benchmark Monarch Attention: Torch vs Triton across sequence lengths."""

import torch
import triton
import time


def benchmark_monarch_attention():
    """Compare torch vs triton monarch attention across sequence lengths."""
    from ma.ma_torch import monarch_attention_torch as ma_torch
    from ma.ma_triton import monarch_attention_triton as ma_triton
    from ma.monarch_attention import optimal_block_size

    print("=" * 90)
    print("Monarch Attention: Torch vs Triton Benchmark")
    print("=" * 90)

    # Fixed parameters
    E, H, D, T = 2, 8, 64, 2

    # Sequence lengths to test (up to 1M)
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

    print(f"\nConfig: E={E}, H={H}, D={D}, T={T}, block_size=optimal")
    print(f"{'N':>10} {'B':>5} {'M':>6} {'Torch (ms)':>12} {'Triton (ms)':>12} {'Speedup':>10} {'Max Diff':>10}")
    print("-" * 90)

    for N in seq_lengths:
        B = optimal_block_size(N)
        M = triton.cdiv(N, B)

        # Create input tensors
        Q = torch.randn(E, H, N, D, device='cuda', dtype=torch.bfloat16)
        K = torch.randn(E, H, N, D, device='cuda', dtype=torch.bfloat16)
        V = torch.randn(E, H, N, D, device='cuda', dtype=torch.bfloat16)

        # Warmup and correctness check
        try:
            # Try torch first (may OOM at large N)
            try:
                out_torch = ma_torch(Q, K, V, None, T, B, pre_pad=True)
                torch.cuda.synchronize()
                torch_ok = True
            except torch.cuda.OutOfMemoryError:
                torch_ok = False
                torch.cuda.empty_cache()

            out_triton = ma_triton(Q, K, V, None, T, B, pre_pad=True)
            torch.cuda.synchronize()

            def run_triton():
                return ma_triton(Q, K, V, None, T, B, pre_pad=True)

            ms_triton = triton.testing.do_bench(run_triton, warmup=5, rep=20)

            if torch_ok:
                max_diff = (out_torch - out_triton).abs().max().item()

                def run_torch():
                    return ma_torch(Q, K, V, None, T, B, pre_pad=True)

                ms_torch = triton.testing.do_bench(run_torch, warmup=5, rep=20)
                speedup = ms_torch / ms_triton
                print(f"{N:>10} {B:>5} {M:>6} {ms_torch:>12.3f} {ms_triton:>12.3f} {speedup:>10.2f}x {max_diff:>10.6f}")
            else:
                print(f"{N:>10} {B:>5} {M:>6} {'OOM':>12} {ms_triton:>12.3f} {'--':>10} {'--':>10}")

        except torch.cuda.OutOfMemoryError:
            print(f"{N:>10} {B:>5} {M:>6} {'OOM':>12} {'OOM':>12}")
        except Exception as e:
            print(f"{N:>10} {B:>5} {M:>6} ERROR: {str(e)[:40]}")

        # Clear cache
        torch.cuda.empty_cache()


def benchmark_vs_softmax():
    """Compare monarch attention vs standard softmax attention."""
    print("\n" + "=" * 90)
    print("Monarch Attention vs Softmax Attention")
    print("=" * 90)

    from ma.ma_triton import monarch_attention_triton as ma_triton
    from ma.monarch_attention import optimal_block_size

    E, H, D, T = 2, 8, 64, 2
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

    print(f"\nConfig: E={E}, H={H}, D={D}, block_size=optimal")
    print(f"{'N':>10} {'Softmax (ms)':>14} {'Monarch (ms)':>14} {'Ratio':>10}")
    print("-" * 60)

    for N in seq_lengths:
        B = optimal_block_size(N)
        Q = torch.randn(E, H, N, D, device='cuda', dtype=torch.bfloat16)
        K = torch.randn(E, H, N, D, device='cuda', dtype=torch.bfloat16)
        V = torch.randn(E, H, N, D, device='cuda', dtype=torch.bfloat16)

        try:
            # Standard softmax attention (scaled dot product)
            def run_softmax():
                scale = 1.0 / (D ** 0.5)
                scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
                attn = torch.softmax(scores, dim=-1)
                return torch.matmul(attn, V)

            def run_monarch():
                return ma_triton(Q, K, V, None, T, B, pre_pad=True)

            # Warmup
            run_softmax()
            run_monarch()
            torch.cuda.synchronize()

            ms_softmax = triton.testing.do_bench(run_softmax, warmup=5, rep=20)
            ms_monarch = triton.testing.do_bench(run_monarch, warmup=5, rep=20)

            ratio = ms_monarch / ms_softmax
            print(f"{N:>10} {ms_softmax:>14.3f} {ms_monarch:>14.3f} {ratio:>10.2f}x")

        except torch.cuda.OutOfMemoryError:
            # Softmax OOMs at large N, try monarch alone
            try:
                ms_monarch = triton.testing.do_bench(run_monarch, warmup=5, rep=20)
                print(f"{N:>10} {'OOM':>14} {ms_monarch:>14.3f} {'--':>10}")
            except Exception:
                print(f"{N:>10} {'OOM':>14} {'OOM':>14}")
        except Exception as e:
            print(f"{N:>10} ERROR: {str(e)[:40]}")

        torch.cuda.empty_cache()


if __name__ == "__main__":
    benchmark_monarch_attention()
    benchmark_vs_softmax()
