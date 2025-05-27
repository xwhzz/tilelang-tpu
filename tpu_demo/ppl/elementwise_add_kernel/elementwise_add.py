# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T
import torch


def elementwise_add(
    M,
    N,
    block_M,
    block_N,
    in_dtype,
    out_dtype,
    threads,
):

    @T.prim_func
    def main(
            A: T.Tensor((M, N), in_dtype),
            B: T.Tensor((M, N), in_dtype),
            C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), in_dtype)
            B_shared = T.alloc_shared((block_M, block_N), in_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            T.ppl_copy(A[by * block_M, bx * block_N], A_shared)
            T.ppl_copy(B[by * block_M, bx * block_N], B_shared)
            T.ppl_add(C_shared, A_shared, B_shared)

            T.ppl_copy(C_shared, C[by * block_M, bx * block_N])

    return main


def ref_add(a, b):
    return torch.add(a, b)


a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
print("Ref:", ref_add(a, b))
func = elementwise_add(512, 1024, 128, 256, "float32", "float32", 128)
mod = tilelang.lower(func)
