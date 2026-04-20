# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T
import torch


def rms_norm(M, N, blk_m, dtype="float32"):

    @T.prim_func
    def main_kernel_inner(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        reciprocal_N = T.float32(1.0 / N)

        with T.Kernel(T.ceildiv(M, blk_m), is_cpu=True) as (bx,):
            A_shared = T.alloc_shared((blk_m, N), dtype)
            A_pow2 = T.alloc_shared((blk_m, N), dtype)
            A_powsum = T.alloc_shared((blk_m, 1), dtype)
            T.ppl_copy(A[bx * blk_m, 0], A_shared)
            T.ppl_mul(A_pow2, A_shared, A_shared)
            T.ppl_reduce_sum(A_pow2, A_powsum, dim=1)
            T.ppl_mul_C(A_powsum, A_powsum, reciprocal_N)
            T.ppl_add_C(A_powsum, A_powsum, T.float32(1e-12))
            T.ppl_rsqrt(A_powsum, A_powsum)
            T.ppl_mul(A_shared, A_shared, A_powsum)
            T.ppl_copy(A_shared, B[bx * blk_m, 0])

    return main_kernel_inner


M = 128
N = 128
blk_m = 32
kernel = tilelang.compile(rms_norm(M, N, blk_m), out_idx=-1, target="tpu")

a = torch.randn(M, N).float()
b = torch.zeros(M, N).float()

res = kernel(a, b)
print(res)
print("output:")
print(b)

eps = 1e-12
ref = a / torch.sqrt(torch.mean(a * a, dim=1, keepdim=True) + eps)
print("ref:")
print(ref)

diff = ref - b
max_diff = torch.max(torch.abs(diff))
avg_diff = torch.mean(torch.abs(diff))

print(f"\n=== 差异分析 ===")
print(f"最大差异: {max_diff}")
print(f"平均差异: {avg_diff}")
print("check close:")
print(torch.allclose(b, ref, atol=1e-2, rtol=1e-2))
