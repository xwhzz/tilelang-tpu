"""Elementwise Add tilelang kernel builders, parameterized by dtype/shape."""

import tilelang.language as T


def build_tl_add(dtype, M, N, block_M, block_N):

    @T.prim_func
    def main_kernel_inner(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
            A_s = T.alloc_shared((block_M, block_N), dtype)
            B_s = T.alloc_shared((block_M, block_N), dtype)
            C_s = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A[by * block_M, bx * block_N], A_s)
            T.copy(B[by * block_M, bx * block_N], B_s)
            T.add(C_s, A_s, B_s)
            T.copy(C_s, C[by * block_M, bx * block_N])

    return main_kernel_inner
