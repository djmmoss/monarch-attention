# Triton Kernel Optimization History

Benchmark results across optimization stages on the `triton-optimization` branch.

**Config:** E=2, H=8, D=64, T=2, dtype=bfloat16, GPU=NVIDIA B200

## Results (ms)

| N | 1: B=32 fixed | 2: opt B (cap 128) | | 3: tiled B (no cap) | | 4: autotune + pipeline | |
|---:|---:|---:|:---:|---:|:---:|---:|:---:|
| 512 | 0.110 | 0.112 | 0.98x | 0.110 | 1.00x | 0.113 | 0.97x |
| 1024 | 0.109 | 0.111 | 0.98x | 0.109 | 1.00x | 0.111 | 0.98x |
| 2048 | 0.109 | 0.110 | 0.99x | 0.109 | 1.00x | 0.112 | 0.97x |
| 4096 | 0.108 | 0.109 | 0.99x | 0.109 | 0.99x | 0.112 | 0.96x |
| 8192 | 0.277 | 0.138 | **2.01x** | 0.138 | 2.01x | 0.138 | 2.01x |
| 16384 | 0.850 | 0.326 | **2.61x** | 0.326 | 2.61x | 0.326 | 2.61x |
| 32768 | 2.975 | 1.251 | 2.38x | 0.771 | **3.86x** | 0.677 | **4.39x** |
| 65536 | 11.238 | 3.733 | 3.01x | 2.780 | **4.04x** | 2.225 | **5.05x** |
| 131072 | 43.196 | 12.668 | 3.41x | 8.041 | **5.37x** | 6.607 | **6.54x** |
| 262144 | 177.886 | 46.184 | 3.85x | 18.526 | **9.60x** | 15.471 | **11.50x** |
| 524288 | 717.383 | 181.234 | 3.96x | 45.816 | **15.66x** | 37.274 | **19.24x** |
| 1048576 | OOM | 722.851 | — | 204.756 | — | 179.188 | — |

Speedups are relative to column 1 (B=32 fixed baseline).

## Commits

| # | Commit | Description |
|---|--------|-------------|
| 1 | `795d699` | Initial optimized Triton kernels with tiled online softmax for B200 |
| 2 | `3c04342` | Auto-select optimal block_size (B ≈ √N), capped at B=128 |
| 3 | `27e9da2` | Tiled within-block (B×B) kernels to remove B=128 cap |
| 4 | `6bc4cd2` | @triton.autotune and software pipelining (num_stages=3) on tiled kernels |

## Key Observations

- **N ≤ 4096**: All versions ~0.11ms — kernel launch overhead dominates
- **N = 8192–16384**: Optimal B selection (step 2) gave 2–2.6x; no further gains from tiling/autotuning since M ≤ 128 stays on non-tiled path
- **N ≥ 32768**: Each optimization compounds — total speedup from step 1→4 ranges from 4.4x (32k) to 19.2x (524k)
- **N = 1M**: Steps 2→4 achieved 4.0x speedup (722.9ms → 179.2ms); step 1 OOMs with B=32 (M=32768)
