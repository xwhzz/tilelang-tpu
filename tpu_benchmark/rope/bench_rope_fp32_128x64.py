"""
Benchmark: RoPE FP32 128x64 — tilelang vs PPL

Usage:
    cd /mnt2/users/tilelanguser-xxw/tilelang-tpu
    python tpu_benchmark/rope/bench_rope_fp32_128x64.py
"""

import os
import sys
import torch

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_ROOT = os.path.dirname(BENCH_DIR)
sys.path.insert(0, BENCHMARK_ROOT)

import tilelang
import tilelang.language as T

C, W = 128, 64
BLOCK_C, BLOCK_W = 64, 16
ATOL, RTOL = 1e-2, 1e-2


# ── tilelang kernel ──────────────────────────────────────────────────────────
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


# ── reference ────────────────────────────────────────────────────────────────
def torch_rope_ref(x, cos, sin):
    """Standard RoPE: rotate pairs of elements."""
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
    print(f"  [{name}] correct={correct}  max_diff={max_diff:.6f}  avg_diff={avg_diff:.8f}")
    return correct


def main():
    x = torch.randn(C, W, dtype=torch.float32)
    cos_val = torch.randn(C, W, dtype=torch.float32)
    sin_val = torch.randn(C, W, dtype=torch.float32)
    ref = torch_rope_ref(x, cos_val, sin_val)

    print("=" * 60)
    print(f"RoPE FP32  C={C} W={W}  block={BLOCK_C}x{BLOCK_W}")
    print("=" * 60)

    # ── tilelang ──
    print("\n--- tilelang ---")
    tl_kernel = tilelang.compile(
        tl_rope(BLOCK_C, BLOCK_W, C, W), out_idx=-1, target="tpu")
    run_and_check("tilelang", tl_kernel, x, cos_val, sin_val, ref)

    # ── PPL ──
    print("\n--- PPL ---")
    try:
        from ppl_utils import compile_ppl_kernel
        pl_path = os.path.join(BENCH_DIR, "pl", "rope_fp32_128x64.pl")
        arg_specs = [
            ((C, W), torch.float32),
            ((C, W), torch.float32),
            ((C, W), torch.float32),
            ((C, W), torch.float32),
        ]
        ppl_forward = compile_ppl_kernel(pl_path, arg_specs, result_idx=[3])
        run_and_check("PPL", ppl_forward, x, cos_val, sin_val, ref)
    except Exception as e:
        print(f"  PPL failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
