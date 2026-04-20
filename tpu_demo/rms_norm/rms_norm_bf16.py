# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T
import torch


def rms_norm(M, N, blk_m, dtype="bfloat16"):

    @T.prim_func
    def main_kernel_inner(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        reciprocal_N = T.float32(1.0 / N)

        with T.Kernel(T.ceildiv(M, blk_m), is_cpu=True) as (bx,):
            A_shared = T.alloc_shared((blk_m, N), dtype)
            A_shared_fp32 = T.alloc_shared((blk_m, N), "float32")
            A_pow2 = T.alloc_shared((blk_m, N), "float32")
            A_powsum = T.alloc_shared((blk_m, 1), "float32")

            T.ppl_copy(A[bx * blk_m, 0], A_shared)
            T.ppl_copy(A_shared, A_shared_fp32)
            T.ppl_mul(A_pow2, A_shared_fp32, A_shared_fp32)
            T.ppl_reduce_sum(A_pow2, A_powsum, dim=1)
            T.ppl_mul_C(A_powsum, A_powsum, reciprocal_N)
            T.ppl_add_C(A_powsum, A_powsum, T.float32(1e-12))
            T.ppl_rsqrt(A_powsum, A_powsum)
            T.ppl_mul(A_shared_fp32, A_shared_fp32, A_powsum)
            T.ppl_copy(A_shared_fp32, A_shared)
            T.ppl_copy(A_shared, B[bx * blk_m, 0])

    return main_kernel_inner


M = 128
N = 128
blk_m = 32
kernel = tilelang.compile(rms_norm(M, N, blk_m), out_idx=-1, target="tpu")

a = torch.randn(M, N).bfloat16()
b = torch.zeros(M, N).bfloat16()

res = kernel(a, b)
print(res)
print("output:")
print(b)

eps = 1e-12
a_fp32 = a.float()
ref_fp32 = a_fp32 / torch.sqrt(torch.mean(a_fp32 * a_fp32, dim=1, keepdim=True) + eps)
ref = ref_fp32.to(a.dtype)
print("ref:")
print(ref)

diff = ref.float() - b.float()
max_diff = torch.max(torch.abs(diff))
avg_diff = torch.mean(torch.abs(diff))

print(f"\n=== 差异分析 ===")
print(f"最大差异: {max_diff}")
print(f"平均差异: {avg_diff}")
print("check close:")
print(torch.allclose(b.float(), ref.float(), atol=5e-2, rtol=5e-2))
