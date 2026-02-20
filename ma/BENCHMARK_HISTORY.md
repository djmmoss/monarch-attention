# Triton Kernel Optimization History

Benchmark results across optimization stages on the `triton-optimization` branch.

**Config:** E=2, H=8, D=64, T=2, dtype=bfloat16, GPU=NVIDIA B200
**Triton:** Triton 3.6.0

Values are bf16 ms. [brackets] = fp8 ms (fp8 k,v inputs, bf16 q).

## Results (ms)

| N | 0: main | 1: tiled softmax | 2: opt B (cap 128) | 3: tiled B (no cap) | 4: autotune | 5: warp tuning | 6: CUDA graph | 7: fp8 | 8: block ptrs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.086 (*1.00x*) | 0.091 (0.95x) | 0.092 (0.93x) | 0.091 (0.95x) | 0.092 (0.93x) | 0.095 (0.91x) | 0.046 (**1.87x**) | 0.046 (**1.87x**) [0.061] | 0.046 (**1.87x**) [0.060] |
| 1024 | 0.085 (*1.00x*) | 0.090 (0.94x) | 0.090 (0.94x) | 0.091 (0.93x) | 0.092 (0.92x) | 0.094 (0.90x) | 0.053 (**1.60x**) | 0.054 (**1.57x**) [0.067] | 0.053 (**1.60x**) [0.067] |
| 2048 | 0.086 (*1.00x*) | 0.090 (0.96x) | 0.090 (0.96x) | 0.091 (0.95x) | 0.091 (0.95x) | 0.095 (0.91x) | 0.069 (**1.25x**) | 0.069 (**1.25x**) [0.085] | 0.070 (**1.23x**) [0.085] |
| 4096 | 0.087 (*1.00x*) | 0.092 (0.95x) | 0.091 (0.96x) | 0.091 (0.96x) | 0.092 (0.95x) | 0.096 (0.91x) | 0.106 (0.82x) | 0.105 (0.83x) [0.123] | 0.106 (0.82x) [0.123] |
| 8192 | 0.190 (*1.00x*) | 0.147 (**1.29x**) | 0.146 (**1.30x**) | 0.146 (**1.30x**) | 0.146 (**1.30x**) | 0.146 (**1.30x**) | 0.166 (**1.14x**) | 0.164 (**1.16x**) [0.168] | 0.164 (**1.16x**) [0.168] |
| 16384 | 0.492 (*1.00x*) | 0.341 (**1.44x**) | 0.341 (**1.44x**) | 0.341 (**1.44x**) | 0.342 (**1.44x**) | 0.342 (**1.44x**) | 0.348 (**1.41x**) | 0.348 (**1.41x**) [0.338] | 0.351 (**1.40x**) [0.342] |
| 32768 | 5.668 (*1.00x*) | 1.314 (**4.31x**) | 1.314 (**4.31x**) | 1.314 (**4.31x**) | 1.136 (**4.99x**) | 0.773 (**7.33x**) | 0.819 (**6.92x**) | 0.818 (**6.93x**) [0.889] | 0.841 (**6.74x**) [0.846] |
| 65536 | 26.396 (*1.00x*) | 8.918 (**2.96x**) | 8.926 (**2.96x**) | 2.916 (**9.05x**) | 2.334 (**11.31x**) | 1.611 (**16.38x**) | 1.697 (**15.55x**) | 1.697 (**15.55x**) [1.691] | 1.771 (**14.90x**) [1.652] |
| 131072 | FAIL | 19.811 (*1.00x*) | 19.828 (1.00x) | 8.441 (**2.35x**) | 6.952 (**2.85x**) | 4.354 (**4.55x**) | 4.514 (**4.39x**) | 4.543 (**4.36x**) [4.632] | 4.662 (**4.25x**) [4.461] |
| 262144 | FAIL | FAIL | FAIL | 19.453 (*1.00x*) | 16.298 (**1.19x**) | 11.091 (**1.75x**) | 11.507 (**1.69x**) | 11.507 (**1.69x**) [10.638] | 10.966 (**1.77x**) [10.347] |
| 524288 | FAIL | FAIL | FAIL | 61.909 (*1.00x*) | 53.171 (**1.16x**) | 33.260 (**1.86x**) | 34.157 (**1.81x**) | 34.887 (**1.77x**) [32.386] | 35.214 (**1.76x**) [30.754] |
| 1048576 | FAIL | FAIL | FAIL | 215.574 (*1.00x*) | 188.359 (**1.14x**) | 117.119 (**1.84x**) | 118.374 (**1.82x**) | 119.866 (**1.80x**) [108.599] | 108.785 (**1.98x**) [101.714] |

Speedups are relative to the earliest valid result for each row. *1.00x* marks the baseline. FAIL = Triton compiler crash. OOM = out of memory.

## Commits

| # | Commit | Description |
|---|--------|-------------|
| 0 | `cfe80d1` | Original Triton kernels — no tiling, no block size optimization, B=32 fixed |
| 1 | `795d699` | Optimized Triton kernels with tiled online softmax (M-tiling) for B200 |
| 2 | `3c04342` | Auto-select optimal block_size (B ≈ √N), capped at B=128 |
| 3 | `27e9da2` | Tiled within-block (B×B) kernels to remove B=128 cap |
| 4 | `6bc4cd2` | @triton.autotune and software pipelining (num_stages=3) on tiled kernels |
| 5 | `380e683` | Add num_warps=2 to autotune config space for tiled kernels |
| 6 | `7b63aa4` | CUDA graph capture + refactor (includes 9487175, 5ab250a) |
| 7 | `9738489` | FP8 tensor cores + CUDA graph for fp8 + expanded autotune configs |
| 8 | `a727008` | Block pointer loads (tl.make_block_ptr) for TMA on Blackwell |

