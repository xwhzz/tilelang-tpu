"""
Batch benchmark runner for tilelang vs PPL on TPU.

Define test cases in BENCHMARK_CONFIG, then run:
    cd /mnt2/users/tilelanguser-xxw/tilelang-tpu
    python tpu_benchmark/bench_all.py

Each entry specifies: operator, dtype, shape, block size, and tolerances.
Results are collected into a summary table at the end.
"""

import os
import sys
import time
import re
import io
import contextlib
import torch
import torch.nn.functional as F

BENCHMARK_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCHMARK_ROOT)

import tilelang
import tilelang.language as T

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
# Each entry: (op, dtype, shape_dict, block_dict, atol, rtol)
#   op:          "swiglu" | "rope" | "add"
#   dtype:       "float32" | "bfloat16" | "float16"
#   shape_dict:  operator-specific shape params
#   block_dict:  operator-specific block params

BENCHMARK_CONFIG = [
    # ── SwiGLU ────────────────────────────────────────────────
    ("swiglu", "float32",  {"C": 64, "W": 64},   {"block_C": 32, "block_W": 32}, 1e-2, 1e-2),
    ("swiglu", "bfloat16", {"C": 64, "W": 64},   {"block_C": 32, "block_W": 32}, 1e-1, 1e-1),
    ("swiglu", "float16",  {"C": 64, "W": 64},   {"block_C": 32, "block_W": 32}, 1e-1, 1e-1),

    # ── RoPE ──────────────────────────────────────────────────
    ("rope", "float32",  {"C": 128, "W": 64},  {"block_C": 64, "block_W": 16}, 1e-2, 1e-2),
    ("rope", "bfloat16", {"C": 128, "W": 64},  {"block_C": 64, "block_W": 16}, 1e-1, 1e-1),
    ("rope", "float16",  {"C": 128, "W": 64},  {"block_C": 64, "block_W": 16}, 1e-1, 1e-1),

    # ── Elementwise Add ──────────────────────────────────────
    ("add", "float32",  {"M": 64, "N": 64},   {"block_M": 32, "block_N": 32}, 1e-5, 1e-5),
    ("add", "bfloat16", {"M": 64, "N": 64},   {"block_M": 32, "block_N": 32}, 1e-2, 1e-2),
    ("add", "float16",  {"M": 64, "N": 64},   {"block_M": 32, "block_N": 32}, 1e-3, 1e-3),
]

TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

PL_DIR = os.path.join(BENCHMARK_ROOT)

# ═══════════════════════════════════════════════════════════════════════════════
#  TILELANG KERNEL BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_tl_swiglu(dtype, C, W, block_C, block_W):
    is_lowp = dtype in ("bfloat16", "float16")
    accum_dtype = "float32"
    global_shape = (C, W)

    if not is_lowp:
        @T.prim_func
        def main_kernel_inner(
            G_in: T.Tensor(global_shape, dtype),
            G_right: T.Tensor(global_shape, dtype),
            G_out: T.Tensor(global_shape, dtype),
        ):
            with T.Kernel(T.ceildiv(C, block_C), T.ceildiv(W, block_W), is_cpu=True) as (bx, by):
                bs = (block_C, block_W)
                x = T.alloc_shared(bs, accum_dtype)
                right = T.alloc_shared(bs, accum_dtype)
                x_neg_exp = T.alloc_shared(bs, accum_dtype)
                ones = T.alloc_shared(bs, accum_dtype)
                T.ppl_fill(ones, T.float32(1.0))
                x_neg_exp_1 = T.alloc_shared(bs, accum_dtype)
                x_neg_exp_1_div = T.alloc_shared(bs, accum_dtype)
                out = T.alloc_shared(bs, accum_dtype)
                T.ppl_copy(G_in[bx * block_C, by * block_W], x)
                T.ppl_copy(G_right[bx * block_C, by * block_W], right)
                T.ppl_mul_C(x_neg_exp, x, T.float32(-1.0))
                w0 = T.alloc_shared([block_C, block_W], accum_dtype)
                w1 = T.alloc_shared([block_C, block_W], accum_dtype)
                coeff = T.alloc_shared([64, 32], accum_dtype)
                table = T.alloc_shared([64, 192], accum_dtype)
                T.ppl_exp2(x_neg_exp, w0, w1, coeff, table)
                T.ppl_add(x_neg_exp_1, x_neg_exp, ones)
                T.ppl_div(x_neg_exp_1_div, x, x_neg_exp_1)
                T.ppl_mul(out, right, x_neg_exp_1_div)
                T.ppl_copy(out, G_out[bx * block_C, by * block_W])
    else:
        @T.prim_func
        def main_kernel_inner(
            G_in: T.Tensor(global_shape, dtype),
            G_right: T.Tensor(global_shape, dtype),
            G_out: T.Tensor(global_shape, dtype),
        ):
            with T.Kernel(T.ceildiv(C, block_C), T.ceildiv(W, block_W), is_cpu=True) as (bx, by):
                bs = (block_C, block_W)
                x_ori = T.alloc_shared(bs, dtype)
                right_ori = T.alloc_shared(bs, dtype)
                x = T.alloc_shared(bs, accum_dtype)
                right = T.alloc_shared(bs, accum_dtype)
                x_neg_exp = T.alloc_shared(bs, accum_dtype)
                ones = T.alloc_shared(bs, accum_dtype)
                T.ppl_fill(ones, T.float32(1.0))
                x_neg_exp_1 = T.alloc_shared(bs, accum_dtype)
                x_neg_exp_1_div = T.alloc_shared(bs, accum_dtype)
                out = T.alloc_shared(bs, accum_dtype)
                out_ori = T.alloc_shared(bs, dtype)
                T.ppl_copy(G_in[bx * block_C, by * block_W], x_ori)
                T.ppl_copy(G_right[bx * block_C, by * block_W], right_ori)
                T.ppl_copy(x_ori, x)
                T.ppl_copy(right_ori, right)
                T.ppl_mul_C(x_neg_exp, x, T.float32(-1.0))
                w0 = T.alloc_shared([block_C, block_W], accum_dtype)
                w1 = T.alloc_shared([block_C, block_W], accum_dtype)
                coeff = T.alloc_shared([64, 32], accum_dtype)
                table = T.alloc_shared([64, 192], accum_dtype)
                T.ppl_exp2(x_neg_exp, w0, w1, coeff, table)
                T.ppl_add(x_neg_exp_1, x_neg_exp, ones)
                T.ppl_div(x_neg_exp_1_div, x, x_neg_exp_1)
                T.ppl_mul(out, right, x_neg_exp_1_div)
                T.ppl_copy(out, out_ori)
                T.ppl_copy(out_ori, G_out[bx * block_C, by * block_W])

    return main_kernel_inner


def build_tl_rope(dtype, C, W, block_C, block_W):
    accum_dtype = dtype
    global_shape = (C, W)

    @T.prim_func
    def main_kernel_inner(
        G_in: T.Tensor(global_shape, dtype),
        G_cos: T.Tensor(global_shape, dtype),
        G_sin: T.Tensor(global_shape, dtype),
        G_out: T.Tensor(global_shape, accum_dtype),
    ):
        with T.Kernel(T.ceildiv(C, block_C), T.ceildiv(W, block_W), is_cpu=True) as (bx, by):
            bs = (block_C, block_W)
            in_x = T.alloc_shared(bs, dtype)
            in_cos = T.alloc_shared(bs, dtype)
            in_sin = T.alloc_shared(bs, dtype)
            T.ppl_copy(G_in[bx * block_C, by * block_W], in_x)
            T.ppl_copy(G_cos[bx * block_C, by * block_W], in_cos)
            T.ppl_copy(G_sin[bx * block_C, by * block_W], in_sin)
            x_cos = T.alloc_shared(bs, dtype)
            T.ppl_mul(x_cos, in_x, in_cos)
            x_sin = T.alloc_shared(bs, dtype)
            T.ppl_mul(x_sin, in_x, in_sin)
            x_neg = T.alloc_shared(bs, dtype)
            T.ppl_mul_C(x_neg, in_x, T.float16(-1.0))
            x_neg_sin = T.alloc_shared(bs, dtype)
            T.ppl_mul(x_neg_sin, x_neg, in_sin)
            out = T.alloc_shared(bs, accum_dtype)
            T.ppl_rope_add(out, x_cos, x_neg_sin, x_cos, x_sin)
            T.ppl_copy(out, G_out[bx * block_C, by * block_W])

    return main_kernel_inner


def build_tl_add(dtype, M, N, block_M, block_N):
    @T.prim_func
    def main_kernel_inner(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
            A_s = T.alloc_shared((block_M, block_N), dtype)
            B_s = T.alloc_shared((block_M, block_N), dtype)
            C_s = T.alloc_shared((block_M, block_N), dtype)
            T.ppl_copy(A[by * block_M, bx * block_N], A_s)
            T.ppl_copy(B[by * block_M, bx * block_N], B_s)
            T.ppl_add(C_s, A_s, B_s)
            T.ppl_copy(C_s, C[by * block_M, bx * block_N])

    return main_kernel_inner


# ═══════════════════════════════════════════════════════════════════════════════
#  PPL .pl PATH RESOLVER
# ═══════════════════════════════════════════════════════════════════════════════

DTYPE_SHORT = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}

def get_pl_path(op, dtype, shape):
    ds = DTYPE_SHORT[dtype]
    if op == "swiglu":
        return os.path.join(PL_DIR, "swiglu", "pl", f"swiglu_{ds}_{shape['C']}x{shape['W']}.pl")
    elif op == "rope":
        return os.path.join(PL_DIR, "rope", "pl", f"rope_{ds}_{shape['C']}x{shape['W']}.pl")
    elif op == "add":
        return os.path.join(PL_DIR, "elementwise_add", "pl", f"add_{ds}_{shape['M']}x{shape['N']}.pl")
    raise ValueError(f"Unknown op: {op}")


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
        return [a, b], ref, [2]
    elif op == "rope":
        C, W = shape["C"], shape["W"]
        x = torch.randn(C, W, dtype=dt)
        cos, sin = make_rope_cos_sin(C, W, dt)
        out_ref = torch.empty_like(x)
        out_ref[:, 0::2] = x[:, 0::2] * cos[:, 0::2] - x[:, 1::2] * sin[:, 0::2]
        out_ref[:, 1::2] = x[:, 1::2] * cos[:, 1::2] + x[:, 0::2] * sin[:, 1::2]
        return [x, cos, sin], out_ref, [3]
    elif op == "add":
        M, N = shape["M"], shape["N"]
        a = torch.randn(M, N, dtype=dt)
        b = torch.randn(M, N, dtype=dt)
        return [a, b], a + b, [2]
    raise ValueError(f"Unknown op: {op}")


# ═══════════════════════════════════════════════════════════════════════════════
#  RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_avg_time(captured: str):
    """Extract 'Average execution time: X.XXX ms' from captured stdout."""
    m = re.search(r"Average execution time:\s+([\d.]+)\s+ms", captured)
    return float(m.group(1)) if m else None


def run_one(op, dtype, shape, block, atol, rtol):
    dt = TORCH_DTYPE_MAP[dtype]
    inputs, ref, result_idx = make_test_data(op, dtype, shape)

    # ── build tilelang kernel ──
    if op == "swiglu":
        tl_func = build_tl_swiglu(dtype, shape["C"], shape["W"], block["block_C"], block["block_W"])
        out_shape = (shape["C"], shape["W"])
        arg_specs = [((shape["C"], shape["W"]), dt)] * 3
    elif op == "rope":
        tl_func = build_tl_rope(dtype, shape["C"], shape["W"], block["block_C"], block["block_W"])
        out_shape = (shape["C"], shape["W"])
        arg_specs = [((shape["C"], shape["W"]), dt)] * 4
    elif op == "add":
        tl_func = build_tl_add(dtype, shape["M"], shape["N"], block["block_M"], block["block_N"])
        out_shape = (shape["M"], shape["N"])
        arg_specs = [((shape["M"], shape["N"]), dt)] * 3

    # ── tilelang ──
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
        from ppl_utils import compile_ppl_kernel
        pl_path = get_pl_path(op, dtype, shape)
        if os.path.isfile(pl_path):
            ppl_forward = compile_ppl_kernel(pl_path, arg_specs, result_idx=result_idx)
            out_ppl = torch.zeros(out_shape, dtype=dt)
            captured_ppl = io.StringIO()
            with contextlib.redirect_stdout(captured_ppl):
                ppl_forward(*inputs, out_ppl)
            ppl_correct = torch.allclose(out_ppl, ref, atol=atol, rtol=rtol)
            ppl_max_diff = (out_ppl.float() - ref.float()).abs().max().item()
            ppl_avg_ms = parse_avg_time(captured_ppl.getvalue())
        else:
            ppl_correct = "N/A (no .pl)"
    except Exception as e:
        ppl_correct = f"ERR: {e}"

    return {
        "tl_correct": tl_correct,
        "tl_max_diff": tl_max_diff,
        "tl_avg_ms": tl_avg_ms,
        "ppl_correct": ppl_correct,
        "ppl_max_diff": ppl_max_diff,
        "ppl_avg_ms": ppl_avg_ms,
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

    print("=" * 90)
    print("  TPU Benchmark: tilelang vs PPL")
    print("=" * 90)

    for i, (op, dtype, shape, block, atol, rtol) in enumerate(BENCHMARK_CONFIG):
        tag = f"[{i+1}/{len(BENCHMARK_CONFIG)}] {op} {dtype} {shape_str(op, shape)}"
        print(f"\n  Running {tag} ...", flush=True)
        try:
            r = run_one(op, dtype, shape, block, atol, rtol)
        except Exception as e:
            r = {"tl_correct": f"ERR: {e}", "tl_max_diff": None, "tl_avg_ms": None,
                 "ppl_correct": None, "ppl_max_diff": None, "ppl_avg_ms": None}
        results.append((op, dtype, shape, r))
        # progress
        tl_t = f"{r['tl_avg_ms']:.3f}ms" if r['tl_avg_ms'] else "?"
        ppl_t = f"{r['ppl_avg_ms']:.3f}ms" if r.get('ppl_avg_ms') else "?"
        print(f"    tilelang={tl_t}  PPL={ppl_t}")

    # ── Summary Table ──
    print("\n")
    print("=" * 90)
    print(f"{'Op':<12} {'Dtype':<10} {'Shape':<10} "
          f"{'TL correct':<12} {'TL avg(ms)':<12} "
          f"{'PPL correct':<12} {'PPL avg(ms)':<12}")
    print("-" * 90)
    for op, dtype, shape, r in results:
        tl_c = str(r["tl_correct"])
        tl_t = f"{r['tl_avg_ms']:.3f}" if r['tl_avg_ms'] else "-"
        ppl_c = str(r["ppl_correct"]) if r["ppl_correct"] is not None else "-"
        ppl_t = f"{r['ppl_avg_ms']:.3f}" if r.get("ppl_avg_ms") else "-"
        print(f"{op:<12} {dtype:<10} {shape_str(op, shape):<10} "
              f"{tl_c:<12} {tl_t:<12} "
              f"{ppl_c:<12} {ppl_t:<12}")
    print("=" * 90)


if __name__ == "__main__":
    main()
