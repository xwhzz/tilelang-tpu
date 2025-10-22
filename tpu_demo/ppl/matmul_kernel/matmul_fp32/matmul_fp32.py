# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T


def matmul(M, N, K, block_M, block_N, block_K, stage, dtype="float32", accum_dtype="float32"):

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            T.ppl_fill(C_shared, T.float32(0))
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=stage):
                T.ppl_copy(A[by * block_M, k * block_K], A_shared)
                T.ppl_copy(B[k * block_K, bx * block_N], B_shared)
                T.ppl_gemm(A_shared, B_shared, C_shared)
            
            T.ppl_copy(C_shared, C[by * block_M, bx * block_N])

    return main


func =  matmul(384, 768, 768, 128, 128, 64, 2)
mod = tilelang.lower(func)
