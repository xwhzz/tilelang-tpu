# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""TileLang kernel factory for W8A16 dequantized MLP.

Strategy: Dequantize int8 weights to fp16 on CPU, then run standard
fp16 matmul kernels on TPU. This avoids per-tile runtime scale handling
in the kernel while still benefiting from TPU matmul acceleration.

The Python wrapper handles:
  1. CPU: dequantize int8 weight -> fp16 using per-block scales
  2. TPU kernel 1: gate = x @ gate_w_fp16^T
  3. TPU kernel 2: up   = x @ up_w_fp16^T
  4. CPU: act = silu(gate) * up
  5. TPU kernel 3: out  = act @ down_w_fp16^T
"""

import tilelang
import tilelang.language as T


def fp16_matmul_kernel(M, N, K, block_M, block_N, block_K, dtype="float16"):
    """Generate a standard fp16 matmul kernel: C[M,N] = A[M,K] @ B[N,K]^T

    Parameters
    ----------
    M, N, K : int
        Matrix dimensions. C[M,N] = A[M,K] @ B[N,K]^T
    block_M, block_N, block_K : int
        Tile sizes.
    dtype : str
        Data type for activations and weights.

    Returns
    -------
    T.prim_func
    """
    accum_dtype = "float32"

    @T.prim_func
    def main_kernel_inner(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_accum = T.alloc_shared((block_M, block_N), accum_dtype)
            C_temp = T.alloc_shared((block_M, block_N), accum_dtype)
            C_out = T.alloc_shared((block_M, block_N), dtype)

            T.ppl_fill(C_accum, T.float32(0))

            for k in T.serial(T.ceildiv(K, block_K)):
                T.ppl_copy(A[by * block_M, k * block_K], A_shared)
                T.ppl_copy(B[bx * block_N, k * block_K], B_shared)
                # transpose_B=True: C_temp = A_shared @ B_shared^T
                T.ppl_gemm(A_shared, B_shared, C_temp, transpose_B=True)
                T.ppl_add(C_accum, C_accum, C_temp)

            T.ppl_copy(C_accum, C_out)
            T.ppl_copy(C_out, C[by * block_M, bx * block_N])

    return main_kernel_inner


def make_matmul_kernel(M, N, K, block_M=32, block_N=128, block_K=128,
                        target="tpu", mode="cmodel"):
    """Compile a fp16 matmul kernel with the given dimensions."""
    return tilelang.compile(
        fp16_matmul_kernel(M, N, K, block_M, block_N, block_K),
        out_idx=-1,
        target=target,
        mode=mode,
    )
