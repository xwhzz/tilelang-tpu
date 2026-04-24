"""
Batch benchmark runner for tilelang vs PPL on TPU.

Each test case runs in a subprocess for isolation (avoids TPU runtime
memory corruption across multiple kernel launches in the same process).

Usage:
    cd /mnt2/users/tilelanguser-xxw/tilelang-tpu
    python tpu_benchmark/bench_all.py
"""

import os
import sys
import re
import json
import subprocess

BENCHMARK_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BENCHMARK_ROOT)

SIZES = [64, 128]
DTYPES = ["float32", "bfloat16", "float16"]

# ── build config ──────────────────────────────────────────────────────────────

BENCHMARK_CONFIG = []
for S in SIZES:
    for dt in DTYPES:
        atol = 1e-2 if dt == "float32" else 1e-1
        BENCHMARK_CONFIG.append(("swiglu", dt, {"C": S, "W": S}, {"block_C": 32, "block_W": 32}, atol, atol))
        BENCHMARK_CONFIG.append(("rope",   dt, {"C": S, "W": S}, {"block_C": 64, "block_W": 16}, atol, atol))
        # atol_add = 1e-5 if dt == "float32" else 1e-3 if dt == "float16" else 1e-2
        # BENCHMARK_CONFIG.append(("add",    dt, {"M": S, "N": S}, {"block_M": 32, "block_N": 32}, atol_add, atol_add))


# ── subprocess worker script ──────────────────────────────────────────────────

WORKER_SCRIPT = '''\
import os, sys, json, re, torch, torch.nn.functional as F

BENCHMARK_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCHMARK_ROOT)

import tilelang
from swiglu.kernels import build_tl_swiglu
from rope.kernels import build_tl_rope
from elementwise_add.kernels import build_tl_add
from ppl_utils import compile_ppl_kernel, generate_pl

TORCH_DTYPE_MAP = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

def make_rope_cos_sin(C, W, dtype):
    half = W // 2
    k = torch.arange(half, dtype=dtype)
    inv_freq = (10000.0 ** (-2 * k / W)).to(dtype)
    pos = torch.arange(C, dtype=dtype).unsqueeze(1)
    theta = pos * inv_freq
    c, s = torch.cos(theta), torch.sin(theta)
    cos = torch.empty(C, W, dtype=dtype); sin = torch.empty(C, W, dtype=dtype)
    cos[:, 0::2] = c; cos[:, 1::2] = c
    sin[:, 0::2] = s; sin[:, 1::2] = s
    return cos.contiguous(), sin.contiguous()

def make_test_data(op, dtype, shape):
    dt = TORCH_DTYPE_MAP[dtype]
    if op == "swiglu":
        C, W = shape["C"], shape["W"]
        a, b = torch.randn(C, W, dtype=dt), torch.randn(C, W, dtype=dt)
        ref = (b.float() * F.silu(a.float())).to(dt)
        return [a, b], ref, [2], (C, W)
    elif op == "rope":
        C, W = shape["C"], shape["W"]
        x = torch.randn(C, W, dtype=dt)
        cos, sin = make_rope_cos_sin(C, W, dt)
        out_ref = torch.empty_like(x)
        out_ref[:, 0::2] = x[:, 0::2]*cos[:, 0::2] - x[:, 1::2]*sin[:, 0::2]
        out_ref[:, 1::2] = x[:, 1::2]*cos[:, 1::2] + x[:, 0::2]*sin[:, 1::2]
        return [x, cos, sin], out_ref, [3], (C, W)
    elif op == "add":
        M, N = shape["M"], shape["N"]
        a, b = torch.randn(M, N, dtype=dt), torch.randn(M, N, dtype=dt)
        return [a, b], a + b, [2], (M, N)

def build_tl_func(op, dtype, shape, block):
    if op == "swiglu": return build_tl_swiglu(dtype, shape["C"], shape["W"], block["block_C"], block["block_W"])
    elif op == "rope": return build_tl_rope(dtype, shape["C"], shape["W"], block["block_C"], block["block_W"])
    elif op == "add":  return build_tl_add(dtype, shape["M"], shape["N"], block["block_M"], block["block_N"])

def build_arg_specs(op, dtype, shape):
    dt = TORCH_DTYPE_MAP[dtype]
    if op == "swiglu": return [((shape["C"], shape["W"]), dt)] * 3
    elif op == "rope": return [((shape["C"], shape["W"]), dt)] * 4
    elif op == "add":  return [((shape["M"], shape["N"]), dt)] * 3

def parse_avg_time(text):
    m = re.search(r"Average execution time:\\s+([\\d.]+)\\s+ms", text)
    return float(m.group(1)) if m else None

cfg = json.loads(sys.argv[1])
op, dtype, shape, block, atol, rtol = cfg["op"], cfg["dtype"], cfg["shape"], cfg["block"], cfg["atol"], cfg["rtol"]
backend = cfg["backend"]  # "tilelang" or "ppl"

dt = TORCH_DTYPE_MAP[dtype]
inputs, ref, result_idx, out_shape = make_test_data(op, dtype, shape)
out = torch.zeros(out_shape, dtype=dt)

if backend == "tilelang":
    tl_func = build_tl_func(op, dtype, shape, block)
    kernel = tilelang.compile(tl_func, out_idx=-1, target="tpu")
    kernel(*inputs, out)
else:
    pl_path = generate_pl(op, dtype, shape)
    arg_specs = build_arg_specs(op, dtype, shape)
    forward = compile_ppl_kernel(pl_path, arg_specs, result_idx=result_idx)
    forward(*inputs, out)

correct = torch.allclose(out, ref, atol=atol, rtol=rtol)
max_diff = (out.float() - ref.float()).abs().max().item()

result = {"correct": correct, "max_diff": max_diff}
sys.stderr.write("RESULT:" + json.dumps(result) + "\\n")
sys.stderr.flush()
'''


def run_subprocess(op, dtype, shape, block, atol, rtol, backend):
    """Run a single benchmark in a subprocess, return (correct, avg_ms)."""
    cfg = json.dumps({
        "op": op, "dtype": dtype, "shape": shape, "block": block,
        "atol": atol, "rtol": rtol, "backend": backend,
    })

    env = os.environ.copy()
    env.setdefault("PPL_PROJECT_ROOT", "")

    worker_path = os.path.join(BENCHMARK_ROOT, "_bench_worker.py")
    with open(worker_path, "w") as f:
        f.write(WORKER_SCRIPT)

    try:
        proc = subprocess.run(
            [sys.executable, worker_path, cfg],
            capture_output=True, timeout=300,
            cwd=BENCHMARK_ROOT, env=env,
        )
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", None

    # Parse timing from stdout (C printf goes to stdout)
    avg_ms = None
    m = re.search(r"Average execution time:\s+([\d.]+)\s+ms", stdout)
    if m:
        avg_ms = float(m.group(1))

    # Parse correctness from stderr
    correct = None
    for line in stderr.splitlines():
        if line.startswith("RESULT:"):
            try:
                r = json.loads(line[7:])
                correct = r.get("correct")
            except json.JSONDecodeError:
                pass

    if correct is None:
        if returncode != 0:
            correct = "CRASH"
            import sys as _sys
            _sys.stderr.write(f"\n--- CRASH for {op} {dtype} {shape} {backend} (rc={returncode}) ---\n")
            _sys.stderr.write(f"STDOUT:\n{stdout[-500:]}\n")
            _sys.stderr.write(f"STDERR:\n{stderr[-500:]}\n")
            _sys.stderr.write("--- END CRASH ---\n")
        else:
            correct = "?"

    return correct, avg_ms


def shape_str(op, shape):
    if op in ("swiglu", "rope"):
        return f"{shape['C']}x{shape['W']}"
    return f"{shape['M']}x{shape['N']}"


def main():
    results = []
    total = len(BENCHMARK_CONFIG)

    print("=" * 100)
    print("  TPU Benchmark: tilelang vs PPL")
    print(f"  Sizes: {SIZES}   Dtypes: {DTYPES}   Total: {total * 2} runs ({total} configs x 2 backends)")
    print("=" * 100)

    for i, (op, dtype, shape, block, atol, rtol) in enumerate(BENCHMARK_CONFIG):
        tag = f"[{i+1}/{total}] {op:<8} {dtype:<10} {shape_str(op, shape)}"
        print(f"\n  {tag}", end="", flush=True)

        tl_correct, tl_ms = run_subprocess(op, dtype, shape, block, atol, rtol, "tilelang")
        print(f"  TL={tl_ms:.3f}ms" if tl_ms else "  TL=?", end="", flush=True)

        ppl_correct, ppl_ms = run_subprocess(op, dtype, shape, block, atol, rtol, "ppl")
        print(f"  PPL={ppl_ms:.3f}ms" if ppl_ms else "  PPL=?", end="", flush=True)

        results.append((op, dtype, shape, tl_correct, tl_ms, ppl_correct, ppl_ms))

    # ── Summary ──
    print("\n\n")
    print("=" * 100)
    print(f"{'Op':<10} {'Dtype':<10} {'Shape':<12} "
          f"{'TL ok':<8} {'TL ms':<10} "
          f"{'PPL ok':<8} {'PPL ms':<10} "
          f"{'Speedup':<10}")
    print("-" * 100)
    for op, dtype, shape, tl_c, tl_ms, ppl_c, ppl_ms in results:
        tl_c_s = "Y" if tl_c is True else "N" if tl_c is False else str(tl_c or "-")
        ppl_c_s = "Y" if ppl_c is True else "N" if ppl_c is False else str(ppl_c or "-")
        tl_t = f"{tl_ms:.3f}" if tl_ms else "-"
        ppl_t = f"{ppl_ms:.3f}" if ppl_ms else "-"
        speedup = f"{ppl_ms / tl_ms:.2f}x" if tl_ms and ppl_ms and tl_ms > 0 else "-"
        print(f"{op:<10} {dtype:<10} {shape_str(op, shape):<12} "
              f"{tl_c_s:<8} {tl_t:<10} "
              f"{ppl_c_s:<8} {ppl_t:<10} "
              f"{speedup:<10}")
    print("=" * 100)

    # cleanup
    worker_path = os.path.join(BENCHMARK_ROOT, "_bench_worker.py")
    if os.path.exists(worker_path):
        os.remove(worker_path)


if __name__ == "__main__":
    main()
