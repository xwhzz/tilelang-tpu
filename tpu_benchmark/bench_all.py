"""
Batch benchmark runner for tilelang vs PPL on TPU.

Configure test cases in BENCHMARK_CONFIG below, then run:
    cd /mnt2/users/tilelanguser-xxw/tilelang-tpu
    python tpu_benchmark/bench_all.py

The .pl files for PPL are auto-generated from templates for any shape/dtype.
"""

import os
import sys
import re
import io
import contextlib
import torch
import torch.nn.functional as F

BENCHMARK_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCHMARK_ROOT)

import tilelang

from swiglu.kernels import build_tl_swiglu
from rope.kernels import build_tl_rope
from elementwise_add.kernels import build_tl_add
from ppl_utils import compile_ppl_kernel, generate_pl

TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

SIZES = [64, 128, 256, 512, 1024]
DTYPES = ["float32", "bfloat16", "float16"]

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit this to add/remove tests
# ═══════════════════════════════════════════════════════════════════════════════

BENCHMARK_CONFIG = []

for S in SIZES:
    for dt in DTYPES:
        atol = 1e-2 if dt == "float32" else 1e-1
        rtol = atol
        BENCHMARK_CONFIG.append(("swiglu", dt, {"C": S, "W": S}, {"block_C": 32, "block_W": 32}, atol, rtol))
        BENCHMARK_CONFIG.append(("rope",   dt, {"C": S, "W": S}, {"block_C": 64, "block_W": 16}, atol, rtol))
        atol_add = 1e-5 if dt == "float32" else 1e-3 if dt == "float16" else 1e-2
        BENCHMARK_CONFIG.append(("add",    dt, {"M": S, "N": S}, {"block_M": 32, "block_N": 32}, atol_add, atol_add))


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST DATA + REFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def make_rope_cos_sin(C, W, dtype):
    half = W // 2
    k = torch.arange(half, dtype=dtype)
    inv_freq = (10000.0 ** (-2 * k / W)).to(dtype)
    pos = torch.arange(C, dtype=dtype).unsqueeze(1)
    theta = pos * inv_freq
    c, s = torch.cos(theta), torch.sin(theta)
    cos = torch.empty(C, W, dtype=dtype)
    sin = torch.empty(C, W, dtype=dtype)
    cos[:, 0::2] = c; cos[:, 1::2] = c
    sin[:, 0::2] = s; sin[:, 1::2] = s
    return cos.contiguous(), sin.contiguous()


def make_test_data(op, dtype, shape):
    dt = TORCH_DTYPE_MAP[dtype]
    if op == "swiglu":
        C, W = shape["C"], shape["W"]
        a = torch.randn(C, W, dtype=dt)
        b = torch.randn(C, W, dtype=dt)
        ref = (b.float() * F.silu(a.float())).to(dt)
        return [a, b], ref, [2], (C, W)
    elif op == "rope":
        C, W = shape["C"], shape["W"]
        x = torch.randn(C, W, dtype=dt)
        cos, sin = make_rope_cos_sin(C, W, dt)
        out_ref = torch.empty_like(x)
        out_ref[:, 0::2] = x[:, 0::2] * cos[:, 0::2] - x[:, 1::2] * sin[:, 0::2]
        out_ref[:, 1::2] = x[:, 1::2] * cos[:, 1::2] + x[:, 0::2] * sin[:, 1::2]
        return [x, cos, sin], out_ref, [3], (C, W)
    elif op == "add":
        M, N = shape["M"], shape["N"]
        a = torch.randn(M, N, dtype=dt)
        b = torch.randn(M, N, dtype=dt)
        return [a, b], a + b, [2], (M, N)
    raise ValueError(f"Unknown op: {op}")


# ═══════════════════════════════════════════════════════════════════════════════
#  RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_avg_time(captured: str):
    m = re.search(r"Average execution time:\s+([\d.]+)\s+ms", captured)
    return float(m.group(1)) if m else None


def build_tl_func(op, dtype, shape, block):
    if op == "swiglu":
        return build_tl_swiglu(dtype, shape["C"], shape["W"], block["block_C"], block["block_W"])
    elif op == "rope":
        return build_tl_rope(dtype, shape["C"], shape["W"], block["block_C"], block["block_W"])
    elif op == "add":
        return build_tl_add(dtype, shape["M"], shape["N"], block["block_M"], block["block_N"])
    raise ValueError(f"Unknown op: {op}")


def build_ppl_arg_specs(op, dtype, shape):
    dt = TORCH_DTYPE_MAP[dtype]
    if op == "swiglu":
        return [((shape["C"], shape["W"]), dt)] * 3
    elif op == "rope":
        return [((shape["C"], shape["W"]), dt)] * 4
    elif op == "add":
        return [((shape["M"], shape["N"]), dt)] * 3
    raise ValueError(f"Unknown op: {op}")


def run_one(op, dtype, shape, block, atol, rtol):
    dt = TORCH_DTYPE_MAP[dtype]
    inputs, ref, result_idx, out_shape = make_test_data(op, dtype, shape)

    # ── tilelang ──
    tl_func = build_tl_func(op, dtype, shape, block)
    tl_kernel = tilelang.compile(tl_func, out_idx=-1, target="tpu")
    out_tl = torch.zeros(out_shape, dtype=dt)
    captured_tl = io.StringIO()
    with contextlib.redirect_stdout(captured_tl):
        tl_kernel(*inputs, out_tl)
    tl_correct = torch.allclose(out_tl, ref, atol=atol, rtol=rtol)
    tl_max_diff = (out_tl.float() - ref.float()).abs().max().item()
    tl_avg_ms = parse_avg_time(captured_tl.getvalue())

    # ── PPL ──
    ppl_correct, ppl_max_diff, ppl_avg_ms = None, None, None
    try:
        pl_path = generate_pl(op, dtype, shape)
        arg_specs = build_ppl_arg_specs(op, dtype, shape)
        ppl_forward = compile_ppl_kernel(pl_path, arg_specs, result_idx=result_idx)
        out_ppl = torch.zeros(out_shape, dtype=dt)
        captured_ppl = io.StringIO()
        with contextlib.redirect_stdout(captured_ppl):
            ppl_forward(*inputs, out_ppl)
        ppl_correct = torch.allclose(out_ppl, ref, atol=atol, rtol=rtol)
        ppl_max_diff = (out_ppl.float() - ref.float()).abs().max().item()
        ppl_avg_ms = parse_avg_time(captured_ppl.getvalue())
    except Exception as e:
        ppl_correct = f"ERR"
        ppl_max_diff = None
        ppl_avg_ms = None
        print(f"    PPL error: {e}")

    return {
        "tl_correct": tl_correct, "tl_max_diff": tl_max_diff, "tl_avg_ms": tl_avg_ms,
        "ppl_correct": ppl_correct, "ppl_max_diff": ppl_max_diff, "ppl_avg_ms": ppl_avg_ms,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def shape_str(op, shape):
    if op in ("swiglu", "rope"):
        return f"{shape['C']}x{shape['W']}"
    return f"{shape['M']}x{shape['N']}"


def main():
    results = []

    print("=" * 95)
    print("  TPU Benchmark: tilelang vs PPL")
    print(f"  Sizes: {SIZES}   Dtypes: {DTYPES}")
    print("=" * 95)

    for i, (op, dtype, shape, block, atol, rtol) in enumerate(BENCHMARK_CONFIG):
        tag = f"[{i+1}/{len(BENCHMARK_CONFIG)}] {op:<8} {dtype:<10} {shape_str(op, shape)}"
        print(f"\n  {tag} ...", end="", flush=True)
        try:
            r = run_one(op, dtype, shape, block, atol, rtol)
        except Exception as e:
            r = {"tl_correct": f"ERR", "tl_max_diff": None, "tl_avg_ms": None,
                 "ppl_correct": None, "ppl_max_diff": None, "ppl_avg_ms": None}
            print(f"\n    Error: {e}")
        tl_t = f"{r['tl_avg_ms']:.3f}ms" if r['tl_avg_ms'] else "?"
        ppl_t = f"{r['ppl_avg_ms']:.3f}ms" if r.get('ppl_avg_ms') else "?"
        print(f"  TL={tl_t}  PPL={ppl_t}")
        results.append((op, dtype, shape, r))

    # ── Summary Table ──
    print("\n\n" + "=" * 95)
    print(f"{'Op':<10} {'Dtype':<10} {'Shape':<12} "
          f"{'TL ok':<8} {'TL ms':<10} "
          f"{'PPL ok':<8} {'PPL ms':<10} "
          f"{'Speedup':<10}")
    print("-" * 95)
    for op, dtype, shape, r in results:
        tl_c = "Y" if r["tl_correct"] is True else "N" if r["tl_correct"] is False else str(r["tl_correct"])
        tl_t = f"{r['tl_avg_ms']:.3f}" if r['tl_avg_ms'] else "-"
        ppl_c = "Y" if r["ppl_correct"] is True else "N" if r["ppl_correct"] is False else str(r["ppl_correct"]) if r["ppl_correct"] is not None else "-"
        ppl_t = f"{r['ppl_avg_ms']:.3f}" if r.get("ppl_avg_ms") else "-"
        if r.get("tl_avg_ms") and r.get("ppl_avg_ms") and r["ppl_avg_ms"] > 0:
            speedup = f"{r['ppl_avg_ms'] / r['tl_avg_ms']:.2f}x"
        else:
            speedup = "-"
        print(f"{op:<10} {dtype:<10} {shape_str(op, shape):<12} "
              f"{tl_c:<8} {tl_t:<10} "
              f"{ppl_c:<8} {ppl_t:<10} "
              f"{speedup:<10}")
    print("=" * 95)


if __name__ == "__main__":
    main()
