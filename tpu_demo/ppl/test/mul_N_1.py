# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T


def rms_norm(M, N, blk_m, dtype):

    @T.prim_func
    def main(C: T.Tensor((M, N), dtype), A: T.Tensor((M, N), dtype), B: T.Tensor((M, 1), dtype)):

        with T.Kernel(T.ceildiv(M, blk_m), is_cpu=True) as (bx,):
            A_shared = T.alloc_shared((blk_m, N), dtype)
            B_shared = T.alloc_shared((blk_m, 1), dtype)
            T.ppl_copy(A[bx * blk_m:(bx + 1) * blk_m, :], A_shared)
            T.ppl_copy(B[bx * blk_m:(bx + 1) * blk_m, 0], B_shared)
            T.ppl_mul(A_shared, A_shared, B_shared)
            T.ppl_copy(A_shared, C[bx * blk_m:(bx + 1) * blk_m, :])

    return main


dtype = "float"
func = rms_norm(8192, 8192, 64, dtype)
mod = tilelang.lower(func)
