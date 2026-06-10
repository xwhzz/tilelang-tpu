"""
Benchmark: RoPE BF16 128x64 — tilelang vs PPL

Usage:
    cd /mnt2/users/tilelanguser-xxw/tilelang-tpu
    python tpu_benchmark/rope/bench_rope_bf16_128x64.py
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
ATOL, RTOL = 1e-1, 1e-1


def tl_rope_bf16(Block_c, Block_w, C, W):
    dtype = "bfloat16"
    accum_dtype = "bfloat16"
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
            T.copy(G_in[bx * Block_c, by * Block_w], in_x)
            T.copy(G_cos[bx * Block_c, by * Block_w], in_cos)
            T.copy(G_sin[bx * Block_c, by * Block_w], in_sin)

            x_cos = T.alloc_shared(block_shape, dtype)
            T.mul(x_cos, in_x, in_cos)
            x_sin = T.alloc_shared(block_shape, dtype)
            T.mul(x_sin, in_x, in_sin)
            x_neg = T.alloc_shared(block_shape, dtype)
            T.mul_C(x_neg, in_x, T.float16(-1.0))
            x_neg_sin = T.alloc_shared(block_shape, dtype)
            T.mul(x_neg_sin, x_neg, in_sin)

            out = T.alloc_shared(block_shape, accum_dtype)
            T.rope_add(out, x_cos, x_neg_sin, x_cos, x_sin)
            T.copy(out, G_out[bx * Block_c, by * Block_w])

    return main_kernel_inner


def rope_ref(x, cos, sin):
    out = torch.empty_like(x)
    out[:, 0::2] = x[:, 0::2] * cos[:, 0::2] - x[:, 1::2] * sin[:, 0::2]
    out[:, 1::2] = x[:, 1::2] * cos[:, 1::2] + x[:, 0::2] * sin[:, 1::2]
    return out


def run_and_check(name, kernel_func, x, cos_val, sin_val, ref):
    out = torch.zeros(C, W, dtype=torch.bfloat16)
    kernel_func(x, cos_val, sin_val, out)
    correct = torch.allclose(out, ref, atol=ATOL, rtol=RTOL)
    max_diff = (out.float() - ref.float()).abs().max().item()
    avg_diff = (out.float() - ref.float()).abs().mean().item()
    print(f"  [{name}] correct={correct}  max_diff={max_diff:.6f}  avg_diff={avg_diff:.8f}")


def make_rope_cos_sin(C, W, dtype=torch.bfloat16):
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
    x = torch.randn(C, W, dtype=torch.bfloat16)
    cos_val, sin_val = make_rope_cos_sin(C, W)
    ref = rope_ref(x, cos_val, sin_val)

    print("=" * 60)
    print(f"RoPE BF16  C={C} W={W}  block={BLOCK_C}x{BLOCK_W}")
    print("=" * 60)

    print("\n--- tilelang ---")
    tl_kernel = tilelang.compile(
        tl_rope_bf16(BLOCK_C, BLOCK_W, C, W), out_idx=-1, target="tpu")
    run_and_check("tilelang", tl_kernel, x, cos_val, sin_val, ref)

    print("\n--- PPL ---")
    try:
        from ppl_utils import compile_ppl_kernel
        pl_path = os.path.join(BENCH_DIR, "pl", "rope_bf16_128x64.pl")
        arg_specs = [((C, W), torch.bfloat16)] * 4
        ppl_forward = compile_ppl_kernel(pl_path, arg_specs, result_idx=[3])
        run_and_check("PPL", ppl_forward, x, cos_val, sin_val, ref)
    except Exception as e:
        print(f"  PPL failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
