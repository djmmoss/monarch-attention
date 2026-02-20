"""Rebuild BENCHMARK_HISTORY.md by benchmarking each optimization commit with current Triton."""

import subprocess
import sys
import os
import shutil
import tempfile

# Commits to benchmark (oldest to newest)
# Each entry: (label, commit_hash, description, has_fp8, fixed_B)
# fixed_B: if set, use this block size instead of optimal_block_size (for pre-optimization commits)
COMMITS = [
    ("0: main", "cfe80d1", "Original Triton kernels — no tiling, no block size optimization, B=32 fixed", False, 32),
    ("1: tiled softmax", "795d699", "Optimized Triton kernels with tiled online softmax (M-tiling) for B200", False, 32),
    ("2: opt B (cap 128)", "3c04342", "Auto-select optimal block_size (B ≈ √N), capped at B=128", False, None),
    ("3: tiled B (no cap)", "27e9da2", "Tiled within-block (B×B) kernels to remove B=128 cap", False, None),
    ("4: autotune", "6bc4cd2", "@triton.autotune and software pipelining (num_stages=3) on tiled kernels", False, None),
    ("5: warp tuning", "380e683", "Add num_warps=2 to autotune config space for tiled kernels", False, None),
    ("6: CUDA graph", "7b63aa4", "CUDA graph capture + refactor (includes 9487175, 5ab250a)", False, None),
    ("7: fp8", "9738489", "FP8 tensor cores + CUDA graph for fp8 + expanded autotune configs", True, None),
    ("8: block ptrs", "a727008", "Block pointer loads (tl.make_block_ptr) for TMA on Blackwell", True, None),
]

SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

BENCH_KERNEL = '''
import torch
import triton
import sys

E, H, D, T = 2, 8, 64, 2

sys.path.insert(0, "{repo_root}")
# Force reimport
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("ma."):
        del sys.modules[mod_name]

from ma.ma_triton import monarch_attention_triton as ma_triton
from ma.monarch_attention import optimal_block_size

use_fp8 = {use_fp8}
fixed_B = {fixed_B}
seq_lengths = {seq_lengths}

for N in seq_lengths:
    B = fixed_B if fixed_B else optimal_block_size(N)
    M = triton.cdiv(N, B)
    Q = torch.randn(E, H, N, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(E, H, N, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(E, H, N, D, device="cuda", dtype=torch.bfloat16)
    if use_fp8:
        K = K.to(torch.float8_e4m3fn)
        V = V.to(torch.float8_e4m3fn)
    try:
        # Warmup
        out = ma_triton(Q, K, V, None, T, B, pre_pad=True)
        torch.cuda.synchronize()
        def run():
            return ma_triton(Q, K, V, None, T, B, pre_pad=True)
        ms = triton.testing.do_bench(run, warmup=5, rep=20)
        print(f"{{N}},{{B}},{{M}},{{ms:.3f}}")
    except torch.cuda.OutOfMemoryError:
        print(f"{{N}},{{B}},{{M}},OOM")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"{{N}},{{B}},{{M}},FAIL:{{str(e)[:60]}}")
        torch.cuda.empty_cache()
'''


def parse_bench_output(stdout):
    """Parse benchmark subprocess output into {N: value} dict."""
    results = {}
    for line in stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) == 4:
            N = int(parts[0])
            val = parts[3]
            if val.startswith("FAIL") or val == "OOM":
                results[N] = val.split(":")[0]
            else:
                results[N] = float(val)
    return results


def run_bench_subprocess(repo_root, use_fp8=False, fixed_B=None):
    """Run benchmark in a subprocess, return parsed results."""
    script = BENCH_KERNEL.format(
        repo_root=repo_root, seq_lengths=SEQ_LENGTHS, use_fp8=use_fp8,
        fixed_B=fixed_B or "None"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=1200,
        env={**os.environ, "PYTHONPATH": repo_root}
    )
    if result.stderr.strip():
        for line in result.stderr.strip().split("\n"):
            if "autotuning" in line.lower() or "error" in line.lower():
                print(f"  {line.strip()}", file=sys.stderr)
    return parse_bench_output(result.stdout)


def run_benchmark_for_commit(label, commit, has_fp8, fixed_B, repo_root, ma_triton_backup):
    """Checkout ma_triton.py from commit, run benchmark(s), restore."""
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Benchmarking: {label} ({commit})", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    ma_triton_path = os.path.join(repo_root, "ma", "ma_triton.py")

    # Checkout ma_triton.py from this commit
    result = subprocess.run(
        ["git", "checkout", commit, "--", "ma/ma_triton.py"],
        cwd=repo_root, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  git checkout failed: {result.stderr.strip()}", file=sys.stderr)
        shutil.copy2(ma_triton_backup, ma_triton_path)
        subprocess.run(["git", "checkout", "HEAD", "--", "ma/ma_triton.py"], cwd=repo_root)
        empty = {N: "FAIL" for N in SEQ_LENGTHS}
        return (empty, None)

    # Clear triton cache to force recompile
    triton_cache = os.path.expanduser("~/.triton/cache")
    if os.path.exists(triton_cache):
        shutil.rmtree(triton_cache)

    # Run bf16 benchmark
    print("  Running bf16...", file=sys.stderr)
    bf16_results = run_bench_subprocess(repo_root, use_fp8=False, fixed_B=fixed_B)
    for N in SEQ_LENGTHS:
        val = bf16_results.get(N, "MISSING")
        suffix = f" {val:.3f} ms" if isinstance(val, float) else f" {val}"
        print(f"  N={N:>8}:{suffix}", file=sys.stderr)

    # Run fp8 benchmark if supported
    fp8_results = None
    if has_fp8:
        # Clear cache again for fp8 autotuning
        if os.path.exists(triton_cache):
            shutil.rmtree(triton_cache)
        print("  Running fp8...", file=sys.stderr)
        fp8_results = run_bench_subprocess(repo_root, use_fp8=True, fixed_B=fixed_B)
        for N in SEQ_LENGTHS:
            val = fp8_results.get(N, "MISSING")
            suffix = f" {val:.3f} ms" if isinstance(val, float) else f" {val}"
            print(f"  N={N:>8}:{suffix} (fp8)", file=sys.stderr)

    # Restore ma_triton.py
    shutil.copy2(ma_triton_backup, ma_triton_path)
    subprocess.run(["git", "checkout", "HEAD", "--", "ma/ma_triton.py"], cwd=repo_root)

    return (bf16_results, fp8_results)


def format_cell(val, fp8_val, first_valid):
    """Format a table cell with optional fp8 value in brackets."""
    if isinstance(val, str):
        # FAIL or OOM
        return val
    if val is None:
        return "—"

    # Format bf16 value with speedup
    if first_valid and isinstance(first_valid, float):
        speedup = first_valid / val
        if speedup >= 1.05:
            bf16_str = f"{val:.3f} (**{speedup:.2f}x**)"
        elif first_valid == val:
            bf16_str = f"{val:.3f} (*1.00x*)"
        else:
            bf16_str = f"{val:.3f} ({speedup:.2f}x)"
    else:
        bf16_str = f"{val:.3f}"

    # Add fp8 in brackets if available
    if fp8_val is not None and isinstance(fp8_val, float):
        bf16_str += f" [{fp8_val:.3f}]"

    return bf16_str


def build_markdown_table(all_results):
    """Build the markdown table for BENCHMARK_HISTORY.md."""
    lines = []
    lines.append("# Triton Kernel Optimization History")
    lines.append("")
    lines.append("Benchmark results across optimization stages on the `triton-optimization` branch.")
    lines.append("")
    lines.append("**Config:** E=2, H=8, D=64, T=2, dtype=bfloat16, GPU=NVIDIA B200")
    lines.append(f"**Triton:** {get_triton_version()}")
    lines.append("")
    lines.append("Values are bf16 ms. [brackets] = fp8 ms (fp8 k,v inputs, bf16 q).")
    lines.append("")
    lines.append("## Results (ms)")
    lines.append("")

    # Header row
    header = "| N |"
    sep = "|---:|"
    for label, *_ in COMMITS:
        header += f" {label} |"
        sep += "---:|"
    lines.append(header)
    lines.append(sep)

    # Data rows
    for N in SEQ_LENGTHS:
        # Find first valid bf16 result for speedup calculation
        first_valid = None
        for label, *_ in COMMITS:
            bf16, _ = all_results.get(label, ({}, None))
            val = bf16.get(N)
            if isinstance(val, float):
                first_valid = val
                break

        row = f"| {N} |"
        for label, *_ in COMMITS:
            bf16, fp8 = all_results.get(label, ({}, None))
            val = bf16.get(N)
            fp8_val = fp8.get(N) if fp8 else None
            cell = format_cell(val, fp8_val, first_valid)
            row += f" {cell} |"
        lines.append(row)

    lines.append("")
    lines.append("Speedups are relative to the earliest valid result for each row. *1.00x* marks the baseline. FAIL = Triton compiler crash. OOM = out of memory.")
    lines.append("")

    # Commits table
    lines.append("## Commits")
    lines.append("")
    lines.append("| # | Commit | Description |")
    lines.append("|---|--------|-------------|")
    for label, commit, desc, *_ in COMMITS:
        num = label.split(":")[0]
        lines.append(f"| {num} | `{commit}` | {desc} |")
    lines.append("")

    return "\n".join(lines)


def get_triton_version():
    try:
        import triton
        return f"Triton {triton.__version__}"
    except Exception:
        return "unknown"


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ma_triton_path = os.path.join(repo_root, "ma", "ma_triton.py")

    # Backup current ma_triton.py
    backup = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
    shutil.copy2(ma_triton_path, backup.name)
    backup.close()

    all_results = {}
    try:
        for label, commit, desc, has_fp8, fixed_B in COMMITS:
            bf16, fp8 = run_benchmark_for_commit(
                label, commit, has_fp8, fixed_B, repo_root, backup.name
            )
            all_results[label] = (bf16, fp8)
    finally:
        # Always restore
        shutil.copy2(backup.name, ma_triton_path)
        subprocess.run(["git", "checkout", "HEAD", "--", "ma/ma_triton.py"], cwd=repo_root)
        os.unlink(backup.name)

    # Build and write markdown
    md = build_markdown_table(all_results)
    history_path = os.path.join(repo_root, "ma", "BENCHMARK_HISTORY.md")
    with open(history_path, "w") as f:
        f.write(md + "\n")

    print(f"\n\nResults written to {history_path}", file=sys.stderr)
    print("\n" + md)


if __name__ == "__main__":
    main()
