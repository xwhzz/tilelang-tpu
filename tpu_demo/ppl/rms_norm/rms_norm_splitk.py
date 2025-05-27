# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T


def rms_norm_splitk(M, N, blk_m, blk_k, dtype):

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        reciprocal_N = T.float32(1.0 / N)

        with T.Kernel(T.ceildiv(M, blk_m), is_cpu=True) as (bx,):
            A_shared = T.alloc_shared((blk_m, blk_k), dtype)
            A_pow2 = T.alloc_shared((blk_m, blk_k), dtype)
            A_powsum = T.alloc_shared((blk_m, 1), dtype)
            A_temp = T.alloc_shared((blk_m, 1), dtype)  # 临时累加

            T.ppl_fill(A_powsum, T.float32(0.0))  # 初始化为0
            num_k_step = T.ceildiv(N, blk_k)
            for k in T.Pipelined(num_k_step, num_stages=0):
                T.ppl_copy(A[bx * blk_m, k * blk_k], A_shared)
                T.ppl_mul(A_pow2, A_shared, A_shared)
                T.ppl_reduce_sum(A_pow2, A_temp, dim=1)
                T.ppl_add(A_powsum, A_powsum, A_temp)  # 累加

            T.ppl_mul_C(A_powsum, A_powsum, reciprocal_N)
            T.ppl_add_C(A_powsum, A_powsum, T.float32(1e-12))  # 避免除0
            T.ppl_rsqrt(A_powsum, A_powsum)

            for k in T.Pipelined(num_k_step, num_stages=0):  # 倒序遍历提高cache命中率
                T.ppl_copy(A[bx * blk_m, (num_k_step - 1 - k) * blk_k], A_shared)
                T.ppl_mul(A_shared, A_shared, A_powsum)
                T.ppl_copy(A_shared, B[bx * blk_m, (num_k_step - 1 - k) * blk_k])

    return main


dtype = "float"
func = rms_norm_splitk(512, 512, 16, 16, dtype)
mod = tilelang.lower(func)
