# Triton Kernel Optimization History

Benchmark results across optimization stages on the `triton-optimization` branch.

**Config:** E=2, H=8, D=64, T=2, dtype=bfloat16, GPU=NVIDIA B200
**Triton:** Triton 3.6.0

Values are bf16 ms with speedup vs baseline. `[fp8 ms, speedup vs baseline]` = fp8 k,v inputs with bf16 q. **Bold** speedup = best for that row. FP8 bolded only if it beats the best bf16.

## Results (ms)

| N | 0: main (B=32) | 1: tiled softmax (B=32) | 2: opt B (cap 128) | 3: tiled B (no cap) | 4: autotune | 5: warp tuning | 6: CUDA graph | 7: fp8 | 8: block ptrs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.130 (*1.00x*) | 0.091 (1.43x) | 0.092 (1.41x) | 0.091 (1.43x) | 0.092 (1.41x) | 0.095 (1.37x) | 0.046 (**2.83x**) | 0.046 (**2.83x**) [0.061, 2.13x] | 0.046 (**2.83x**) [0.060, 2.17x] |
| 1024 | 0.085 (*1.00x*) | 0.091 (0.93x) | 0.090 (0.94x) | 0.091 (0.93x) | 0.092 (0.92x) | 0.094 (0.90x) | 0.053 (**1.60x**) | 0.054 (1.57x) [0.067, 1.27x] | 0.053 (**1.60x**) [0.067, 1.27x] |
| 2048 | 0.086 (*1.00x*) | 0.091 (0.95x) | 0.090 (0.96x) | 0.091 (0.95x) | 0.091 (0.95x) | 0.095 (0.91x) | 0.069 (**1.25x**) | 0.069 (**1.25x**) [0.085, 1.01x] | 0.070 (1.23x) [0.085, 1.01x] |
| 4096 | 0.110 (*1.00x*) | 0.094 (1.17x) | 0.091 (**1.21x**) | 0.091 (**1.21x**) | 0.092 (1.20x) | 0.096 (1.15x) | 0.106 (1.04x) | 0.105 (1.05x) [0.123, 0.89x] | 0.106 (1.04x) [0.123, 0.89x] |
| 8192 | 1.415 (*1.00x*) | 0.292 (4.85x) | 0.146 (**9.69x**) | 0.146 (**9.69x**) | 0.146 (**9.69x**) | 0.146 (**9.69x**) | 0.166 (8.52x) | 0.164 (8.63x) [0.168, 8.42x] | 0.164 (8.63x) [0.168, 8.42x] |
| 16384 | FAIL | 0.895 (*1.00x*) | 0.341 (**2.63x**) | 0.341 (**2.63x**) | 0.342 (2.62x) | 0.342 (2.62x) | 0.348 (2.57x) | 0.348 (2.57x) [0.338, **2.65x**] | 0.351 (2.55x) [0.342, 2.62x] |
| 32768 | FAIL | 3.131 (*1.00x*) | 1.314 (2.38x) | 1.314 (2.38x) | 1.136 (2.76x) | 0.773 (**4.05x**) | 0.819 (3.82x) | 0.818 (3.83x) [0.889, 3.52x] | 0.841 (3.72x) [0.846, 3.70x] |
| 65536 | FAIL | 11.770 (*1.00x*) | 8.926 (1.32x) | 2.916 (4.04x) | 2.334 (5.04x) | 1.611 (**7.31x**) | 1.697 (6.93x) | 1.697 (6.93x) [1.691, 6.96x] | 1.771 (6.65x) [1.652, **7.12x**] |
| 131072 | FAIL | 45.492 (*1.00x*) | 19.828 (2.29x) | 8.441 (5.39x) | 6.952 (6.54x) | 4.354 (**10.45x**) | 4.514 (10.08x) | 4.543 (10.01x) [4.632, 9.82x] | 4.662 (9.76x) [4.461, **10.20x**] |
| 262144 | FAIL | FAIL | FAIL | 19.453 (*1.00x*) | 16.298 (1.19x) | 11.091 (1.75x) | 11.507 (1.69x) | 11.507 (1.69x) [10.638, 1.83x] | 10.966 (1.77x) [10.347, **1.88x**] |
| 524288 | FAIL | FAIL | FAIL | 61.909 (*1.00x*) | 53.171 (1.16x) | 33.260 (**1.86x**) | 34.157 (1.81x) | 34.887 (1.77x) [32.386, 1.91x] | 35.214 (1.76x) [30.754, **2.01x**] |
| 1048576 | FAIL | FAIL | FAIL | 215.574 (*1.00x*) | 188.359 (1.14x) | 117.119 (1.84x) | 118.374 (1.82x) | 119.866 (1.80x) [108.599, 1.99x] | 108.785 (1.98x) [101.714, **2.12x**] |

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
| 6 | `7b63aa4` | CUDA graph capture + refactor (includes 9487175, 5ab250a) |
| 7 | `9738489` | FP8 tensor cores + CUDA graph for fp8 + expanded autotune configs |
| 8 | `a727008` | Block pointer loads (tl.make_block_ptr) for TMA on Blackwell |
