"""Matmul tilelang kernel builders, parameterized by dtype/shape."""

import tilelang.language as T


def build_tl_matmul(dtype, M, N, K, block_M, block_N, block_K):
    accum_dtype = "float32"

    if dtype == "float32":
        @T.prim_func
        def main_kernel_inner(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                A_tmp = T.alloc_shared((block_M, block_K), "float16")
                B_tmp = T.alloc_shared((block_K, block_N), "float16")
                C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

                T.fill(C_shared, T.float32(0))
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.copy(A_shared, A_tmp)
                    T.copy(B_shared, B_tmp)
                    T.gemm(A_tmp, B_tmp, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])
    else:
        @T.prim_func
        def main_kernel_inner(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
                C_shared_ori = T.alloc_shared((block_M, block_N), dtype)

                T.fill(C_shared, T.float32(0))
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_shared)
                T.copy(C_shared, C_shared_ori)
                T.copy(C_shared_ori, C[by * block_M, bx * block_N])

    return main_kernel_inner
