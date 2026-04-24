"""
Generate standalone benchmark scripts for every (op, dtype, shape) combination.

Each generated script is self-contained (kernel inline, no kernels.py import)
and can be run directly:  python tpu_benchmark/swiglu/bench_swiglu_bf16_128x128.py

Usage:
    cd /mnt2/users/tilelanguser-xxw/tilelang-tpu
    python tpu_benchmark/bench_gen_all.py                     # generate all
    python tpu_benchmark/bench_gen_all.py --sizes 64 128      # subset of sizes
    python tpu_benchmark/bench_gen_all.py --ops swiglu rope    # subset of ops
"""

import argparse
import os

BENCHMARK_ROOT = os.path.dirname(os.path.abspath(__file__))

DTYPES = ["float32", "bfloat16", "float16"]

DTYPE_SHORT = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}
TORCH_DTYPE = {"float32": "torch.float32", "bfloat16": "torch.bfloat16", "float16": "torch.float16"}

OP_CONFIG = {
    "swiglu": {"dir": "swiglu", "block": (32, 32), "dims": ("C", "W")},
    "rope":   {"dir": "rope",   "block": (64, 16), "dims": ("C", "W")},
    "add":    {"dir": "elementwise_add", "block": (32, 32), "dims": ("M", "N")},
}

ATOL_MAP = {
    "swiglu":  {"float32": "1e-2",  "bfloat16": "1e-1",  "float16": "1e-1"},
    "rope":    {"float32": "1e-2",  "bfloat16": "1e-1",  "float16": "1e-1"},
    "add":     {"float32": "1e-5",  "bfloat16": "1e-2",  "float16": "1e-3"},
}

# ── Templates ────────────────────────────────────────────────────────────────

SWIGLU_FP32_TEMPLATE = '''\
"""
Benchmark: SwiGLU FP32 {C}x{W} — tilelang vs PPL
"""

import os, sys, torch, torch.nn.functional as F

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_ROOT = os.path.dirname(BENCH_DIR)
sys.path.insert(0, BENCHMARK_ROOT)

import tilelang
import tilelang.language as T

C, W = {C}, {W}
BLOCK_C, BLOCK_W = {block_C}, {block_W}
ATOL, RTOL = {atol}, {atol}


def tl_swiglu(Block_w, Block_c, C, W, dtype="float32", accum_dtype="float32"):
    global_shape = (C, W)

    @T.prim_func
    def main_kernel_inner(
        G_in: T.Tensor(global_shape, dtype),
        G_right: T.Tensor(global_shape, dtype),
        G_out: T.Tensor(global_shape, accum_dtype),
    ):
        with T.Kernel(T.ceildiv(C, Block_c), T.ceildiv(W, Block_w), is_cpu=True) as (bx, by):
            block_shape = (Block_c, Block_w)
            x = T.alloc_shared(block_shape, accum_dtype)
            right = T.alloc_shared(block_shape, accum_dtype)
            x_neg_exp = T.alloc_shared(block_shape, accum_dtype)
            ones = T.alloc_shared(block_shape, accum_dtype)
            T.ppl_fill(ones, T.float32(1.0))
            x_neg_exp_1 = T.alloc_shared(block_shape, accum_dtype)
            x_neg_exp_1_div = T.alloc_shared(block_shape, accum_dtype)
            out = T.alloc_shared(block_shape, accum_dtype)

            T.ppl_copy(G_in[bx * Block_c, by * Block_w], x)
            T.ppl_copy(G_right[bx * Block_c, by * Block_w], right)
            T.ppl_mul_C(x_neg_exp, x, T.float32(-1.0))

            work0 = T.alloc_shared([Block_c, Block_w], accum_dtype)
            work1 = T.alloc_shared([Block_c, Block_w], accum_dtype)
            coeff = T.alloc_shared([64, 32], accum_dtype)
            table = T.alloc_shared([64, 192], accum_dtype)
            T.ppl_exp2(x_neg_exp, work0, work1, coeff, table)
            T.ppl_add(x_neg_exp_1, x_neg_exp, ones)
            T.ppl_div(x_neg_exp_1_div, x, x_neg_exp_1)
            T.ppl_mul(out, right, x_neg_exp_1_div)

            T.ppl_copy(out, G_out[bx * Block_c, by * Block_w])

    return main_kernel_inner


def torch_ref(a, b):
    return b * F.silu(a)


def run_and_check(name, kernel_func, a, b, ref):
    c = torch.zeros(C, W, dtype=torch.float32)
    kernel_func(a, b, c)
    correct = torch.allclose(c, ref, atol=ATOL, rtol=RTOL)
    max_diff = (c - ref).abs().max().item()
    avg_diff = (c - ref).abs().mean().item()
    print(f"  [{{name}}] correct={{correct}}  max_diff={{max_diff:.6f}}  avg_diff={{avg_diff:.8f}}")
    return correct


def main():
    a = torch.randn(C, W, dtype=torch.float32)
    b = torch.randn(C, W, dtype=torch.float32)
    ref = torch_ref(a, b)

    print("=" * 60)
    print(f"SwiGLU FP32  C={{C}} W={{W}}  block={{BLOCK_C}}x{{BLOCK_W}}")
    print("=" * 60)

    print("\\n--- tilelang ---")
    tl_kernel = tilelang.compile(
        tl_swiglu(BLOCK_W, BLOCK_C, C, W), out_idx=-1, target="tpu")
    run_and_check("tilelang", tl_kernel, a, b, ref)

    print("\\n--- PPL ---")
    try:
        from ppl_utils import compile_ppl_kernel, generate_pl
        pl_path = generate_pl("swiglu", "float32", {{"C": C, "W": W}})
        arg_specs = [((C, W), torch.float32)] * 3
        ppl_forward = compile_ppl_kernel(pl_path, arg_specs, result_idx=[2])
        run_and_check("PPL", ppl_forward, a, b, ref)
    except Exception as e:
        print(f"  PPL failed: {{e}}")
        import traceback; traceback.print_exc()

    print("\\n" + "=" * 60)


if __name__ == "__main__":
    main()
'''

SWIGLU_LOWP_TEMPLATE = '''\
"""
Benchmark: SwiGLU {dtype_upper} {C}x{W} — tilelang vs PPL
"""

import os, sys, torch, torch.nn.functional as F

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_ROOT = os.path.dirname(BENCH_DIR)
sys.path.insert(0, BENCHMARK_ROOT)

import tilelang
import tilelang.language as T

C, W = {C}, {W}
BLOCK_C, BLOCK_W = {block_C}, {block_W}
ATOL, RTOL = {atol}, {atol}


def tl_swiglu_lowp(Block_w, Block_c, C, W):
    dtype = "{dtype}"
    accum_dtype = "float32"
    global_shape = (C, W)

    @T.prim_func
    def main_kernel_inner(
        G_in: T.Tensor(global_shape, dtype),
        G_right: T.Tensor(global_shape, dtype),
        G_out: T.Tensor(global_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(C, Block_c), T.ceildiv(W, Block_w), is_cpu=True) as (bx, by):
            block_shape = (Block_c, Block_w)
            x_ori = T.alloc_shared(block_shape, dtype)
            right_ori = T.alloc_shared(block_shape, dtype)
            x = T.alloc_shared(block_shape, accum_dtype)
            right = T.alloc_shared(block_shape, accum_dtype)
            x_neg_exp = T.alloc_shared(block_shape, accum_dtype)
            ones = T.alloc_shared(block_shape, accum_dtype)
            T.ppl_fill(ones, T.float32(1.0))
            x_neg_exp_1 = T.alloc_shared(block_shape, accum_dtype)
            x_neg_exp_1_div = T.alloc_shared(block_shape, accum_dtype)
            out = T.alloc_shared(block_shape, accum_dtype)
            out_ori = T.alloc_shared(block_shape, dtype)

            T.ppl_copy(G_in[bx * Block_c, by * Block_w], x_ori)
            T.ppl_copy(G_right[bx * Block_c, by * Block_w], right_ori)
            T.ppl_copy(x_ori, x)
            T.ppl_copy(right_ori, right)
            T.ppl_mul_C(x_neg_exp, x, T.float32(-1.0))
            work0 = T.alloc_shared([Block_c, Block_w], accum_dtype)
            work1 = T.alloc_shared([Block_c, Block_w], accum_dtype)
            coeff = T.alloc_shared([64, 32], accum_dtype)
            table = T.alloc_shared([64, 192], accum_dtype)
            T.ppl_exp2(x_neg_exp, work0, work1, coeff, table)
            T.ppl_add(x_neg_exp_1, x_neg_exp, ones)
            T.ppl_div(x_neg_exp_1_div, x, x_neg_exp_1)
            T.ppl_mul(out, right, x_neg_exp_1_div)
            T.ppl_copy(out, out_ori)
            T.ppl_copy(out_ori, G_out[bx * Block_c, by * Block_w])

    return main_kernel_inner


def torch_ref(a, b):
    return (b.float() * F.silu(a.float())).to({torch_dtype})


def run_and_check(name, kernel_func, a, b, ref):
    c = torch.zeros(C, W, dtype={torch_dtype})
    kernel_func(a, b, c)
    correct = torch.allclose(c, ref, atol=ATOL, rtol=RTOL)
    max_diff = (c.float() - ref.float()).abs().max().item()
    avg_diff = (c.float() - ref.float()).abs().mean().item()
    print(f"  [{{name}}] correct={{correct}}  max_diff={{max_diff:.6f}}  avg_diff={{avg_diff:.8f}}")


def main():
    a = torch.randn(C, W, dtype={torch_dtype})
    b = torch.randn(C, W, dtype={torch_dtype})
    ref = torch_ref(a, b)

    print("=" * 60)
    print(f"SwiGLU {dtype_upper}  C={{C}} W={{W}}  block={{BLOCK_C}}x{{BLOCK_W}}")
    print("=" * 60)

    print("\\n--- tilelang ---")
    tl_kernel = tilelang.compile(
        tl_swiglu_lowp(BLOCK_W, BLOCK_C, C, W), out_idx=-1, target="tpu")
    run_and_check("tilelang", tl_kernel, a, b, ref)

    print("\\n--- PPL ---")
    try:
        from ppl_utils import compile_ppl_kernel, generate_pl
        pl_path = generate_pl("swiglu", "{dtype}", {{"C": C, "W": W}})
        arg_specs = [((C, W), {torch_dtype})] * 3
        ppl_forward = compile_ppl_kernel(pl_path, arg_specs, result_idx=[2])
        run_and_check("PPL", ppl_forward, a, b, ref)
    except Exception as e:
        print(f"  PPL failed: {{e}}")
        import traceback; traceback.print_exc()

    print("\\n" + "=" * 60)


if __name__ == "__main__":
    main()
'''

ROPE_FP32_TEMPLATE = '''\
"""
Benchmark: RoPE FP32 {C}x{W} — tilelang vs PPL
"""

import os, sys, torch

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_ROOT = os.path.dirname(BENCH_DIR)
sys.path.insert(0, BENCHMARK_ROOT)

import tilelang
import tilelang.language as T

C, W = {C}, {W}
BLOCK_C, BLOCK_W = {block_C}, {block_W}
ATOL, RTOL = {atol}, {atol}


def tl_rope(Block_c, Block_w, C, W, dtype="float32", accum_dtype="float32"):
    global_shape = (C, W)

    @T.prim_func
    def main_kernel_inner(
        G_in: T.Tensor(global_shape, dtype),
        G_cos: T.Tensor(global_shape, dtype),
        G_sin: T.Tensor(global_shape, dtype),
        G_out: T.Tensor(global_shape, accum_dtype),
    ):
        with T.Kernel(T.ceildiv(C, Block_c), T.ceildiv(W, Block_w), is_cpu=True) as (bx, by):
            block_shape = (Block_c, Block_w)
            in_x = T.alloc_shared(block_shape, dtype)
            in_cos = T.alloc_shared(block_shape, dtype)
            in_sin = T.alloc_shared(block_shape, dtype)
            T.ppl_copy(G_in[bx * Block_c, by * Block_w], in_x)
            T.ppl_copy(G_cos[bx * Block_c, by * Block_w], in_cos)
            T.ppl_copy(G_sin[bx * Block_c, by * Block_w], in_sin)
            x_cos = T.alloc_shared(block_shape, dtype)
            T.ppl_mul(x_cos, in_x, in_cos)
            x_sin = T.alloc_shared(block_shape, dtype)
            T.ppl_mul(x_sin, in_x, in_sin)
            x_neg = T.alloc_shared(block_shape, dtype)
            T.ppl_mul_C(x_neg, in_x, T.float32(-1.0))
            x_neg_sin = T.alloc_shared(block_shape, dtype)
            T.ppl_mul(x_neg_sin, x_neg, in_sin)
            out = T.alloc_shared(block_shape, accum_dtype)
            T.ppl_rope_add(out, x_cos, x_neg_sin, x_cos, x_sin)
            T.ppl_copy(out, G_out[bx * Block_c, by * Block_w])

    return main_kernel_inner


def torch_rope_ref(x, cos, sin):
    out = torch.zeros_like(x)
    out[:, 0::2] = x[:, 0::2] * cos[:, 0::2] - x[:, 1::2] * sin[:, 0::2]
    out[:, 1::2] = x[:, 1::2] * cos[:, 1::2] + x[:, 0::2] * sin[:, 1::2]
    return out


def run_and_check(name, kernel_func, x, cos_val, sin_val, ref):
    out = torch.zeros(C, W, dtype=torch.float32)
    kernel_func(x, cos_val, sin_val, out)
    correct = torch.allclose(out, ref, atol=ATOL, rtol=RTOL)
    max_diff = (out - ref).abs().max().item()
    avg_diff = (out - ref).abs().mean().item()
    print(f"  [{{name}}] correct={{correct}}  max_diff={{max_diff:.6f}}  avg_diff={{avg_diff:.8f}}")
    return correct


def make_rope_cos_sin(C, W, dtype=torch.float32):
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


def main():
    x = torch.randn(C, W, dtype=torch.float32)
    cos_val, sin_val = make_rope_cos_sin(C, W)
    ref = torch_rope_ref(x, cos_val, sin_val)

    print("=" * 60)
    print(f"RoPE FP32  C={{C}} W={{W}}  block={{BLOCK_C}}x{{BLOCK_W}}")
    print("=" * 60)

    print("\\n--- tilelang ---")
    tl_kernel = tilelang.compile(
        tl_rope(BLOCK_C, BLOCK_W, C, W), out_idx=-1, target="tpu")
    run_and_check("tilelang", tl_kernel, x, cos_val, sin_val, ref)

    print("\\n--- PPL ---")
    try:
        from ppl_utils import compile_ppl_kernel, generate_pl
        pl_path = generate_pl("rope", "float32", {{"C": C, "W": W}})
        arg_specs = [((C, W), torch.float32)] * 4
        ppl_forward = compile_ppl_kernel(pl_path, arg_specs, result_idx=[3])
        run_and_check("PPL", ppl_forward, x, cos_val, sin_val, ref)
    except Exception as e:
        print(f"  PPL failed: {{e}}")
        import traceback; traceback.print_exc()

    print("\\n" + "=" * 60)


if __name__ == "__main__":
    main()
'''

ROPE_LOWP_TEMPLATE = '''\
"""
Benchmark: RoPE {dtype_upper} {C}x{W} — tilelang vs PPL
"""

import os, sys, torch

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_ROOT = os.path.dirname(BENCH_DIR)
sys.path.insert(0, BENCHMARK_ROOT)

import tilelang
import tilelang.language as T

C, W = {C}, {W}
BLOCK_C, BLOCK_W = {block_C}, {block_W}
ATOL, RTOL = {atol}, {atol}


def tl_rope_lowp(Block_c, Block_w, C, W):
    dtype = "{dtype}"
    accum_dtype = "{dtype}"
    global_shape = (C, W)

    @T.prim_func
    def main_kernel_inner(
        G_in: T.Tensor(global_shape, dtype),
        G_cos: T.Tensor(global_shape, dtype),
        G_sin: T.Tensor(global_shape, dtype),
        G_out: T.Tensor(global_shape, accum_dtype),
    ):
        with T.Kernel(T.ceildiv(C, Block_c), T.ceildiv(W, Block_w), is_cpu=True) as (bx, by):
            block_shape = (Block_c, Block_w)
            in_x = T.alloc_shared(block_shape, dtype)
            in_cos = T.alloc_shared(block_shape, dtype)
            in_sin = T.alloc_shared(block_shape, dtype)
            T.ppl_copy(G_in[bx * Block_c, by * Block_w], in_x)
            T.ppl_copy(G_cos[bx * Block_c, by * Block_w], in_cos)
            T.ppl_copy(G_sin[bx * Block_c, by * Block_w], in_sin)
            x_cos = T.alloc_shared(block_shape, dtype)
            T.ppl_mul(x_cos, in_x, in_cos)
            x_sin = T.alloc_shared(block_shape, dtype)
            T.ppl_mul(x_sin, in_x, in_sin)
            x_neg = T.alloc_shared(block_shape, dtype)
            T.ppl_mul_C(x_neg, in_x, T.float16(-1.0))
            x_neg_sin = T.alloc_shared(block_shape, dtype)
            T.ppl_mul(x_neg_sin, x_neg, in_sin)
            out = T.alloc_shared(block_shape, accum_dtype)
            T.ppl_rope_add(out, x_cos, x_neg_sin, x_cos, x_sin)
            T.ppl_copy(out, G_out[bx * Block_c, by * Block_w])

    return main_kernel_inner


def rope_ref(x, cos, sin):
    out = torch.empty_like(x)
    out[:, 0::2] = x[:, 0::2] * cos[:, 0::2] - x[:, 1::2] * sin[:, 0::2]
    out[:, 1::2] = x[:, 1::2] * cos[:, 1::2] + x[:, 0::2] * sin[:, 1::2]
    return out


def run_and_check(name, kernel_func, x, cos_val, sin_val, ref):
    out = torch.zeros(C, W, dtype={torch_dtype})
    kernel_func(x, cos_val, sin_val, out)
    correct = torch.allclose(out, ref, atol=ATOL, rtol=RTOL)
    max_diff = (out.float() - ref.float()).abs().max().item()
    avg_diff = (out.float() - ref.float()).abs().mean().item()
    print(f"  [{{name}}] correct={{correct}}  max_diff={{max_diff:.6f}}  avg_diff={{avg_diff:.8f}}")


def make_rope_cos_sin(C, W, dtype={torch_dtype}):
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


def main():
    x = torch.randn(C, W, dtype={torch_dtype})
    cos_val, sin_val = make_rope_cos_sin(C, W)
    ref = rope_ref(x, cos_val, sin_val)

    print("=" * 60)
    print(f"RoPE {dtype_upper}  C={{C}} W={{W}}  block={{BLOCK_C}}x{{BLOCK_W}}")
    print("=" * 60)

    print("\\n--- tilelang ---")
    tl_kernel = tilelang.compile(
        tl_rope_lowp(BLOCK_C, BLOCK_W, C, W), out_idx=-1, target="tpu")
    run_and_check("tilelang", tl_kernel, x, cos_val, sin_val, ref)

    print("\\n--- PPL ---")
    try:
        from ppl_utils import compile_ppl_kernel, generate_pl
        pl_path = generate_pl("rope", "{dtype}", {{"C": C, "W": W}})
        arg_specs = [((C, W), {torch_dtype})] * 4
        ppl_forward = compile_ppl_kernel(pl_path, arg_specs, result_idx=[3])
        run_and_check("PPL", ppl_forward, x, cos_val, sin_val, ref)
    except Exception as e:
        print(f"  PPL failed: {{e}}")
        import traceback; traceback.print_exc()

    print("\\n" + "=" * 60)


if __name__ == "__main__":
    main()
'''

ADD_FP32_TEMPLATE = '''\
"""
Benchmark: Elementwise Add FP32 {M}x{N} — tilelang vs PPL
"""

import os, sys, torch

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_ROOT = os.path.dirname(BENCH_DIR)
sys.path.insert(0, BENCHMARK_ROOT)

import tilelang
import tilelang.language as T

M, N = {M}, {N}
BLOCK_M, BLOCK_N = {block_M}, {block_N}
ATOL, RTOL = {atol}, {atol}


def tl_add(M, N, block_M, block_N, dtype="float32"):
    @T.prim_func
    def main_kernel_inner(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)
            T.ppl_copy(A[by * block_M, bx * block_N], A_shared)
            T.ppl_copy(B[by * block_M, bx * block_N], B_shared)
            T.ppl_add(C_shared, A_shared, B_shared)
            T.ppl_copy(C_shared, C[by * block_M, bx * block_N])

    return main_kernel_inner


def run_and_check(name, kernel_func, a, b, ref):
    c = torch.zeros(M, N, dtype=torch.float32)
    kernel_func(a, b, c)
    correct = torch.allclose(c, ref, atol=ATOL, rtol=RTOL)
    max_diff = (c - ref).abs().max().item()
    avg_diff = (c - ref).abs().mean().item()
    print(f"  [{{name}}] correct={{correct}}  max_diff={{max_diff:.6f}}  avg_diff={{avg_diff:.8f}}")
    return correct


def main():
    a = torch.randn(M, N, dtype=torch.float32)
    b = torch.randn(M, N, dtype=torch.float32)
    ref = a + b

    print("=" * 60)
    print(f"Elementwise Add FP32  M={{M}} N={{N}}  block={{BLOCK_M}}x{{BLOCK_N}}")
    print("=" * 60)

    print("\\n--- tilelang ---")
    tl_kernel = tilelang.compile(
        tl_add(M, N, BLOCK_M, BLOCK_N), out_idx=-1, target="tpu")
    run_and_check("tilelang", tl_kernel, a, b, ref)

    print("\\n--- PPL ---")
    try:
        from ppl_utils import compile_ppl_kernel, generate_pl
        pl_path = generate_pl("add", "float32", {{"M": M, "N": N}})
        arg_specs = [((M, N), torch.float32)] * 3
        ppl_forward = compile_ppl_kernel(pl_path, arg_specs, result_idx=[2])
        run_and_check("PPL", ppl_forward, a, b, ref)
    except Exception as e:
        print(f"  PPL failed: {{e}}")
        import traceback; traceback.print_exc()

    print("\\n" + "=" * 60)


if __name__ == "__main__":
    main()
'''

ADD_LOWP_TEMPLATE = '''\
"""
Benchmark: Elementwise Add {dtype_upper} {M}x{N} — tilelang vs PPL
"""

import os, sys, torch

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_ROOT = os.path.dirname(BENCH_DIR)
sys.path.insert(0, BENCHMARK_ROOT)

import tilelang
import tilelang.language as T

M, N = {M}, {N}
BLOCK_M, BLOCK_N = {block_M}, {block_N}
ATOL, RTOL = {atol}, {atol}


def tl_add_lowp(M, N, block_M, block_N):
    dtype = "{dtype}"

    @T.prim_func
    def main_kernel_inner(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)
            T.ppl_copy(A[by * block_M, bx * block_N], A_shared)
            T.ppl_copy(B[by * block_M, bx * block_N], B_shared)
            T.ppl_add(C_shared, A_shared, B_shared)
            T.ppl_copy(C_shared, C[by * block_M, bx * block_N])

    return main_kernel_inner


def run_and_check(name, kernel_func, a, b, ref):
    c = torch.zeros(M, N, dtype={torch_dtype})
    kernel_func(a, b, c)
    correct = torch.allclose(c, ref, atol=ATOL, rtol=RTOL)
    max_diff = (c.float() - ref.float()).abs().max().item()
    avg_diff = (c.float() - ref.float()).abs().mean().item()
    print(f"  [{{name}}] correct={{correct}}  max_diff={{max_diff:.6f}}  avg_diff={{avg_diff:.8f}}")


def main():
    a = torch.randn(M, N, dtype={torch_dtype})
    b = torch.randn(M, N, dtype={torch_dtype})
    ref = a + b

    print("=" * 60)
    print(f"Elementwise Add {dtype_upper}  M={{M}} N={{N}}  block={{BLOCK_M}}x{{BLOCK_N}}")
    print("=" * 60)

    print("\\n--- tilelang ---")
    tl_kernel = tilelang.compile(
        tl_add_lowp(M, N, BLOCK_M, BLOCK_N), out_idx=-1, target="tpu")
    run_and_check("tilelang", tl_kernel, a, b, ref)

    print("\\n--- PPL ---")
    try:
        from ppl_utils import compile_ppl_kernel, generate_pl
        pl_path = generate_pl("add", "{dtype}", {{"M": M, "N": N}})
        arg_specs = [((M, N), {torch_dtype})] * 3
        ppl_forward = compile_ppl_kernel(pl_path, arg_specs, result_idx=[2])
        run_and_check("PPL", ppl_forward, a, b, ref)
    except Exception as e:
        print(f"  PPL failed: {{e}}")
        import traceback; traceback.print_exc()

    print("\\n" + "=" * 60)


if __name__ == "__main__":
    main()
'''


def get_template(op, dtype):
    if op == "swiglu":
        return SWIGLU_FP32_TEMPLATE if dtype == "float32" else SWIGLU_LOWP_TEMPLATE
    elif op == "rope":
        return ROPE_FP32_TEMPLATE if dtype == "float32" else ROPE_LOWP_TEMPLATE
    elif op == "add":
        return ADD_FP32_TEMPLATE if dtype == "float32" else ADD_LOWP_TEMPLATE


def generate_one(op, dtype, size):
    cfg = OP_CONFIG[op]
    d0, d1 = cfg["dims"]
    block_0, block_1 = cfg["block"]
    ds = DTYPE_SHORT[dtype]
    atol = ATOL_MAP[op][dtype]

    tpl = get_template(op, dtype)

    params = {
        d0: size, d1: size,
        f"block_{d0}": block_0, f"block_{d1}": block_1,
        "dtype": dtype,
        "dtype_upper": ds.upper(),
        "torch_dtype": TORCH_DTYPE[dtype],
        "atol": atol,
    }

    content = tpl.format(**params)

    op_dir = os.path.join(BENCHMARK_ROOT, cfg["dir"])
    if op == "add":
        fname = f"bench_add_{ds}_{size}x{size}.py"
    else:
        fname = f"bench_{op}_{ds}_{size}x{size}.py"
    out_path = os.path.join(op_dir, fname)
    with open(out_path, "w") as f:
        f.write(content)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark scripts")
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256, 512, 1024])
    parser.add_argument("--dtypes", nargs="+", default=DTYPES)
    parser.add_argument("--ops", nargs="+", default=list(OP_CONFIG.keys()))
    args = parser.parse_args()

    generated = []
    for op in args.ops:
        if op not in OP_CONFIG:
            print(f"Unknown op: {op}, skipping")
            continue
        for size in args.sizes:
            for dtype in args.dtypes:
                path = generate_one(op, dtype, size)
                generated.append(path)

    print(f"Generated {len(generated)} benchmark scripts:")
    for p in generated:
        rel = os.path.relpath(p, os.path.dirname(BENCHMARK_ROOT))
        print(f"  {rel}")


if __name__ == "__main__":
    main()
