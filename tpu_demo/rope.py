# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import tilelang
import tilelang.language as T


def matmul(M, N, block_M, block_N, stage, dtype="float16", accum_dtype="float"):
    assert N / 2 % block_N == 0

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            cos: T.Tensor((M, N), dtype),
            sin: T.Tensor((M, N), dtype)
    ):
      with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_N), accum_dtype)
        T.copy(A_shared, A[by * block_M, bx * block_N])
        # 
        x_cos = T.ppl_mul()




    return main


# func =  matmul(4096, 8192, 1024, 1024, 512, 128, 2)
func =  matmul(T.symbolic("m"), T.symbolic("n"), 128, 128, 2)
mod = tilelang.lower(func)
