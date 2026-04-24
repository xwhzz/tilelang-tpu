"""
Benchmark: SwiGLU BF16 512x512 — tilelang vs PPL
"""

import os, sys, torch, torch.nn.functional as F

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_ROOT = os.path.dirname(BENCH_DIR)
sys.path.insert(0, BENCHMARK_ROOT)

import tilelang
import tilelang.language as T

C, W = 512, 512
BLOCK_C, BLOCK_W = 32, 32
ATOL, RTOL = 1e-1, 1e-1


def tl_swiglu_lowp(Block_w, Block_c, C, W):
    dtype = "bfloat16"
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
    return (b.float() * F.silu(a.float())).to(torch.bfloat16)


def run_and_check(name, kernel_func, a, b, ref):
    c = torch.zeros(C, W, dtype=torch.bfloat16)
    kernel_func(a, b, c)
    correct = torch.allclose(c, ref, atol=ATOL, rtol=RTOL)
    max_diff = (c.float() - ref.float()).abs().max().item()
    avg_diff = (c.float() - ref.float()).abs().mean().item()
    print(f"  [{name}] correct={correct}  max_diff={max_diff:.6f}  avg_diff={avg_diff:.8f}")


def main():
    a = torch.randn(C, W, dtype=torch.bfloat16)
    b = torch.randn(C, W, dtype=torch.bfloat16)
    ref = torch_ref(a, b)

    print("=" * 60)
    print(f"SwiGLU BF16  C={C} W={W}  block={BLOCK_C}x{BLOCK_W}")
    print("=" * 60)

    print("\n--- tilelang ---")
    tl_kernel = tilelang.compile(
        tl_swiglu_lowp(BLOCK_W, BLOCK_C, C, W), out_idx=-1, target="tpu")
    run_and_check("tilelang", tl_kernel, a, b, ref)

    print("\n--- PPL ---")
    try:
        from ppl_utils import compile_ppl_kernel, generate_pl
        pl_path = generate_pl("swiglu", "bfloat16", {"C": C, "W": W})
        arg_specs = [((C, W), torch.bfloat16)] * 3
        ppl_forward = compile_ppl_kernel(pl_path, arg_specs, result_idx=[2])
        run_and_check("PPL", ppl_forward, a, b, ref)
    except Exception as e:
        print(f"  PPL failed: {e}")
        import traceback; traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
