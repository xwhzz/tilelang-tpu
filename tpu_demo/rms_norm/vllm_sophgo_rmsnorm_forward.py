# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T
import torch


def _choose_blk_m(M, max_blk_m=128):
    blk_m = min(M, max_blk_m)
    while M % blk_m != 0:
        blk_m -= 1
    return blk_m

def vllm_sophgo_rmsnorm_forward(M, N, blk_m=None, eps=1e-6, dtype="float16"):
    if blk_m is None:
        blk_m = _choose_blk_m(M)

    @T.prim_func
    def main_kernel_inner(
            X: T.Tensor((M, N), dtype),
            Weight: T.Tensor((1, N), dtype),
            Output: T.Tensor((M, N), dtype),
    ):
        reciprocal_N = T.float32(1.0 / N)

        with T.Kernel(T.ceildiv(M, blk_m), is_cpu=True) as (bx,):
            X_shared = T.alloc_shared((blk_m, N), dtype)
            X_fp32 = T.alloc_shared((blk_m, N), "float32")
            X_pow2 = T.alloc_shared((blk_m, N), "float32")
            X_powsum = T.alloc_shared((blk_m, 1), "float32")
            Weight_shared = T.alloc_shared((1, N), dtype)

            T.ppl_copy(X[bx * blk_m:(bx + 1) * blk_m, :], X_shared)
            T.ppl_copy(X_shared, X_fp32)
            T.ppl_mul(X_pow2, X_fp32, X_fp32)
            T.ppl_reduce_sum(X_pow2, X_powsum, dim=1)
            T.ppl_mul_C(X_powsum, X_powsum, reciprocal_N)
            T.ppl_add_C(X_powsum, X_powsum, T.float32(eps))
            T.ppl_rsqrt(X_powsum, X_powsum)
            T.ppl_mul(X_fp32, X_fp32, X_powsum)
            T.ppl_copy(X_fp32, X_shared)
            T.ppl_copy(Weight[0:1, :], Weight_shared)
            T.ppl_mul(X_shared, X_shared, Weight_shared)
            T.ppl_copy(X_shared, Output[bx * blk_m:(bx + 1) * blk_m, :])

    return main_kernel_inner


if __name__ == "__main__":
    M = 128
    N = 128
    eps = 1e-6
    blk_m = _choose_blk_m(M)
    dtype = "float16"
    kernel = tilelang.compile(
        vllm_sophgo_rmsnorm_forward(M, N, blk_m, eps, dtype),
        out_idx=2,
        target="tpu",
        mode="cmodel",
    )

    x = torch.randn(M, N).half()
    weight = torch.randn(1, N).half()
    output = torch.zeros(M, N).half()

    kernel(x, weight, output)

    ref = x.float() * torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps)
    ref = (ref * weight.float()).to(x.dtype)

    diff = ref.float() - output.float()
    max_diff = torch.max(torch.abs(diff))
    avg_diff = torch.mean(torch.abs(diff))

    print("output:")
    print(output)
    print("ref:")
    print(ref)
    print(f"max diff: {max_diff}")
    print(f"avg diff: {avg_diff}")
    print("check close:")
    print(torch.allclose(output.float(), ref.float(), atol=5e-2, rtol=5e-2))
