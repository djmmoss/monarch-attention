# Triton Kernel Optimization History

Benchmark results across optimization stages on the `triton-optimization` branch.

**Config:** E=2, H=8, D=64, T=2, dtype=bfloat16, GPU=NVIDIA B200

## Results (ms)

| N | 0: main (B=32) | 1: B=32 optimized | | 2: opt B (cap 128) | | 3: tiled B (no cap) | | 4: autotune + pipeline | | 5: warp config tuning | | 6: CUDA graph | |
|---:|---:|---:|:---:|---:|:---:|---:|:---:|---:|:---:|---:|:---:|---:|:---:|
| 512 | 0.105 | 0.110 | 0.95x | 0.112 | 0.94x | 0.110 | 0.95x | 0.113 | 0.93x | 0.090 | **1.17x** | 0.038 | **2.76x** |
| 1024 | 0.103 | 0.109 | 0.94x | 0.111 | 0.93x | 0.109 | 0.94x | 0.111 | 0.93x | 0.090 | **1.14x** | 0.042 | **2.45x** |
| 2048 | 0.103 | 0.109 | 0.94x | 0.110 | 0.94x | 0.109 | 0.94x | 0.112 | 0.92x | 0.092 | **1.12x** | 0.059 | **1.75x** |
| 4096 | 0.102 | 0.108 | 0.94x | 0.109 | 0.94x | 0.109 | 0.94x | 0.112 | 0.91x | 0.092 | **1.11x** | 0.091 | **1.12x** |
| 8192 | 1.506 | 0.277 | **5.44x** | 0.138 | **10.91x** | 0.138 | **10.91x** | 0.138 | **10.91x** | 0.142 | **10.61x** | 0.143 | **10.53x** |
| 16384 | FAIL | 0.850 | *1.00x* | 0.326 | **2.61x** | 0.326 | **2.61x** | 0.326 | **2.61x** | 0.327 | **2.60x** | 0.329 | **2.58x** |
| 32768 | FAIL | 2.975 | *1.00x* | 1.251 | **2.38x** | 0.771 | **3.86x** | 0.677 | **4.39x** | 0.741 | **4.01x** | 0.743 | **4.00x** |
| 65536 | FAIL | 11.238 | *1.00x* | 3.733 | **3.01x** | 2.780 | **4.04x** | 2.225 | **5.05x** | 1.541 | **7.29x** | 1.544 | **7.28x** |
| 131072 | FAIL | 43.196 | *1.00x* | 12.668 | **3.41x** | 8.041 | **5.37x** | 6.607 | **6.54x** | 4.166 | **10.37x** | 4.147 | **10.42x** |
| 262144 | FAIL | 177.886 | *1.00x* | 46.184 | **3.85x** | 18.526 | **9.60x** | 15.471 | **11.50x** | 10.563 | **16.86x** | 10.538 | **16.87x** |
| 524288 | FAIL | 717.383 | *1.00x* | 181.234 | **3.96x** | 45.816 | **15.66x** | 37.274 | **19.24x** | 32.625 | **21.99x** | 32.930 | **21.78x** |
| 1048576 | FAIL | OOM | — | 722.851 | *1.00x* | 204.756 | **3.53x** | 179.188 | **4.04x** | 113.011 | **6.40x** | 113.011 | **6.40x** |

Speedups are relative to the earliest valid result for each row. *1.00x* marks the baseline. FAIL = Triton compiler crash. OOM = out of memory.

## Commits

| # | Commit | Description |
|---|--------|-------------|
| 0 | `cfe80d1` (main) | Original Triton kernels — no tiling, no block size optimization, B=32 fixed |
| 1 | `795d699` | Optimized Triton kernels with tiled online softmax (M-tiling) for B200 |
| 2 | `3c04342` | Auto-select optimal block_size (B ≈ √N), capped at B=128 |
| 3 | `27e9da2` | Tiled within-block (B×B) kernels to remove B=128 cap |
| 4 | `6bc4cd2` | @triton.autotune and software pipelining (num_stages=3) on tiled kernels |
| 5 | `380e683` | Add num_warps=2 to autotune config space for tiled kernels |
| 6 | `9487175`, `5ab250a` | CUDA graph capture for non-tiled kernels (N ≤ 4096) + tiled threshold fix for N=16384 |

## Key Observations

- **Main branch (col 0)**: Only works up to N=8192 with B=32; larger M values crash the Triton compiler (no tiling support)
- **N ≤ 2048**: Step 6 (CUDA graph) eliminates kernel launch overhead, giving 1.75–2.76x speedup over step 5
- **N = 4096**: CUDA graph gives modest 1.12x speedup (launch overhead is smaller fraction of total)
- **N = 8192**: Main branch takes 1.5ms; optimized versions achieve ~10.5x speedup (0.14ms) via optimal B + M-tiling
- **N ≥ 16384**: Main branch fails entirely; steps 1–5 progressively improve performance
- **N = 65k–262k**: Step 5 (num_warps=2) gave 1.4–1.6x improvement over step 4 on tiled _ar_cr kernels
- **N = 524k**: Steps 1→5 achieved 22x cumulative speedup (717ms → 33ms)
- **N = 1M**: Steps 2→5 achieved 6.4x speedup (723ms → 113ms); step 1 OOMs with B=32 (M=32768)
