# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T


def rms_norm(M, N, blk_m, dtype):

    @T.prim_func
    def main_kernel_inner(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        reciprocal_N = T.float32(1.0 / N)

        with T.Kernel(T.ceildiv(M, blk_m), is_cpu=True) as (bx,):
            A_shared = T.alloc_shared((blk_m, N), dtype)
            A_pow2 = T.alloc_shared((blk_m, N), "float32")
            A_powsum = T.alloc_shared((blk_m, 1), "float32")
            T.copy(A[bx * blk_m:(bx + 1) * blk_m, :], A_shared)
            A_shared_fp32 = T.alloc_shared((blk_m, N), "float32")
            T.copy(A_shared, A_shared_fp32)

            T.mul(A_pow2, A_shared_fp32, A_shared_fp32)
            T.reduce_sum(A_pow2, A_powsum, dim=1)
            T.mul_C(A_powsum, A_powsum, reciprocal_N)
            T.add_C(A_powsum, A_powsum, T.float32(1e-12))  # 给A_powsum加1e-12，避免除0
            T.rsqrt(A_powsum, A_powsum)
            T.mul(A_shared_fp32, A_shared_fp32, A_powsum)
            T.copy(A_shared_fp32, A_shared)
            T.copy(A_shared, B[bx * blk_m:(bx + 1) * blk_m, :])

    return main_kernel_inner


dtype = "bfloat16"
func = rms_norm(128, 128, 32, dtype)
mod = tilelang.lower(func)
