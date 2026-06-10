# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T


def elementwise_c(M, N, blk_m, blk_n, dtype):

    @T.prim_func
    def main_kernel_inner(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):

        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, blk_n), is_cpu=True) as (bx, by):
            A_shared = T.alloc_shared((blk_m, blk_n), dtype)
            T.copy(A[bx * blk_m, by * blk_n], A_shared)
            T.mul_C(A_shared, A_shared, T.float32(1 / 8192))
            T.add_C(A_shared, A_shared, T.float32(1e-12))
            T.copy(A_shared, B[bx * blk_m, by * blk_n])

    return main_kernel_inner


dtype = "float"
func = elementwise_c(16, 4, 4, 2, dtype)
mod = tilelang.lower(func)
