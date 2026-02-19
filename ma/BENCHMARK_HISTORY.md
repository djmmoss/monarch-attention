# Triton Kernel Optimization History

Benchmark results across optimization stages on the `triton-optimization` branch.

**Config:** E=2, H=8, D=64, T=2, dtype=bfloat16, GPU=NVIDIA B200

## Results (ms)

| N | 0: main (B=32) | 1: B=32 optimized | | 2: opt B (cap 128) | | 3: tiled B (no cap) | | 4: autotune + pipeline | |
|---:|---:|---:|:---:|---:|:---:|---:|:---:|---:|:---:|
| 512 | 0.105 | 0.110 | 0.95x | 0.112 | 0.94x | 0.110 | 0.95x | 0.113 | 0.93x |
| 1024 | 0.103 | 0.109 | 0.94x | 0.111 | 0.93x | 0.109 | 0.94x | 0.111 | 0.93x |
| 2048 | 0.103 | 0.109 | 0.94x | 0.110 | 0.94x | 0.109 | 0.94x | 0.112 | 0.92x |
| 4096 | 0.102 | 0.108 | 0.94x | 0.109 | 0.94x | 0.109 | 0.94x | 0.112 | 0.91x |
| 8192 | 1.506 | 0.277 | **5.44x** | 0.138 | **10.91x** | 0.138 | **10.91x** | 0.138 | **10.91x** |
| 16384 | FAIL | 0.850 | — | 0.326 | — | 0.326 | — | 0.326 | — |
| 32768 | FAIL | 2.975 | — | 1.251 | — | 0.771 | — | 0.677 | — |
| 65536 | FAIL | 11.238 | — | 3.733 | — | 2.780 | — | 2.225 | — |
| 131072 | FAIL | 43.196 | — | 12.668 | — | 8.041 | — | 6.607 | — |
| 262144 | FAIL | 177.886 | — | 46.184 | — | 18.526 | — | 15.471 | — |
| 524288 | FAIL | 717.383 | — | 181.234 | — | 45.816 | — | 37.274 | — |
| 1048576 | FAIL | OOM | — | 722.851 | — | 204.756 | — | 179.188 | — |

Speedups are relative to column 0 (main branch baseline). FAIL = Triton compiler crash (no tiling, M exceeds max tensor numel).

## Commits

| # | Commit | Description |
|---|--------|-------------|
| 0 | `cfe80d1` (main) | Original Triton kernels — no tiling, no block size optimization, B=32 fixed |
| 1 | `795d699` | Optimized Triton kernels with tiled online softmax (M-tiling) for B200 |
| 2 | `3c04342` | Auto-select optimal block_size (B ≈ √N), capped at B=128 |
| 3 | `27e9da2` | Tiled within-block (B×B) kernels to remove B=128 cap |
| 4 | `6bc4cd2` | @triton.autotune and software pipelining (num_stages=3) on tiled kernels |

## Key Observations

- **Main branch (col 0)**: Only works up to N=8192 with B=32; larger M values crash the Triton compiler (no tiling support)
- **N ≤ 4096**: All optimized versions ~0.11ms — slightly slower than main (0.10ms) due to additional kernel features; kernel launch overhead dominates
- **N = 8192**: Main branch takes 1.5ms; optimized versions achieve 10.9x speedup (0.138ms) via optimal B + M-tiling
- **N ≥ 16384**: Main branch fails entirely; step 1 enables these sizes, steps 2–4 progressively improve performance
- **N = 524k**: Steps 1→4 achieved 19.2x cumulative speedup (717ms → 37ms)
- **N = 1M**: Steps 2→4 achieved 4.0x speedup (723ms → 179ms); step 1 OOMs with B=32 (M=32768)
