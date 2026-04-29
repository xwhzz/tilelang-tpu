"""RMSNorm tilelang kernel builders, parameterized by dtype/shape."""

import tilelang.language as T


def build_tl_rmsnorm(dtype, M, N, blk_m):

    if dtype == "float32":
        @T.prim_func
        def main_kernel_inner(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
        ):
            reciprocal_N = T.float32(1.0 / N)
            with T.Kernel(T.ceildiv(M, blk_m), is_cpu=True) as (bx,):
                A_shared = T.alloc_shared((blk_m, N), dtype)
                A_pow2 = T.alloc_shared((blk_m, N), dtype)
                A_powsum = T.alloc_shared((blk_m, 1), dtype)
                T.ppl_copy(A[bx * blk_m, 0], A_shared)
                T.ppl_mul(A_pow2, A_shared, A_shared)
                T.ppl_reduce_sum(A_pow2, A_powsum, dim=1)
                T.ppl_mul_C(A_powsum, A_powsum, reciprocal_N)
                T.ppl_add_C(A_powsum, A_powsum, T.float32(1e-12))
                T.ppl_rsqrt(A_powsum, A_powsum)
                T.ppl_mul(A_shared, A_shared, A_powsum)
                T.ppl_copy(A_shared, B[bx * blk_m, 0])
    else:
        @T.prim_func
        def main_kernel_inner(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
        ):
            reciprocal_N = T.float32(1.0 / N)
            with T.Kernel(T.ceildiv(M, blk_m), is_cpu=True) as (bx,):
                A_shared = T.alloc_shared((blk_m, N), dtype)
                A_shared_fp32 = T.alloc_shared((blk_m, N), "float32")
                A_pow2 = T.alloc_shared((blk_m, N), "float32")
                A_powsum = T.alloc_shared((blk_m, 1), "float32")

                T.ppl_copy(A[bx * blk_m, 0], A_shared)
                T.ppl_copy(A_shared, A_shared_fp32)
                T.ppl_mul(A_pow2, A_shared_fp32, A_shared_fp32)
                T.ppl_reduce_sum(A_pow2, A_powsum, dim=1)
                T.ppl_mul_C(A_powsum, A_powsum, reciprocal_N)
                T.ppl_add_C(A_powsum, A_powsum, T.float32(1e-12))
                T.ppl_rsqrt(A_powsum, A_powsum)
                T.ppl_mul(A_shared_fp32, A_shared_fp32, A_powsum)
                T.ppl_copy(A_shared_fp32, A_shared)
                T.ppl_copy(A_shared, B[bx * blk_m, 0])

    return main_kernel_inner
