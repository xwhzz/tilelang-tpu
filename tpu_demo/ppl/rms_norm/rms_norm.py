# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T


def rms_norm(M, N, blk_m, dtype):

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        reciprocal_N = T.float32(1.0 / N)

        with T.Kernel(T.ceildiv(M, blk_m), is_cpu=True) as (bx,):
            A_shared = T.alloc_shared((blk_m, N), dtype)
            A_pow2 = T.alloc_shared((blk_m, N), dtype)
            A_powsum = T.alloc_shared((blk_m, 1), dtype)
            T.ppl_copy(A[bx * blk_m:(bx + 1) * blk_m, :], A_shared)
            T.ppl_mul(A_pow2, A_shared, A_shared)
            T.ppl_reduce_sum(A_pow2, A_powsum, dim=1)
            T.ppl_mul_C(A_powsum, A_powsum, reciprocal_N)
            T.ppl_add_C(A_powsum, A_powsum, T.float32(1e-12))  # 给A_powsum加1e-12，避免除0
            T.ppl_rsqrt(A_powsum, A_powsum)
            T.ppl_mul(A_shared, A_shared, A_powsum)
            T.ppl_copy(A_shared, B[bx * blk_m:(bx + 1) * blk_m, :])

    return main


dtype = "float"
func = rms_norm(2048, 2048, 64, dtype)
mod = tilelang.lower(func)
