# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Single W8A16 dequantized matrix multiplication — cmodel verification test.

Computes:  C[M, N] = A[M, K] @ (W_int8[N, K] * scale)^T

This is the foundational building block for the full MLP operator.

Phase 2a: Test int8→fp16 cast + matmul with UNIFORM scale (compile-time constant).
Phase 2b: Add per-tile runtime scale support.
"""

import tilelang
import tilelang.language as T
import torch

from tpu_demo.mlp_w8a16.torch_ref import dequant_matmul_ref


def dequant_matmul_kernel_uniform(M, N, K, block_M, block_N, block_K,
                                   dtype="float16", uniform_scale=1.0):
    """Phase 2a: Dequant matmul with uniform compile-time scale.

    Verifies:
      1. int8 buffer allocation works
      2. tpu_bdc_cast(DT_INT8 → DT_FP16) works
      3. fp16 matmul with fp32 accumulation works

    Parameters
    ----------
    uniform_scale : float
        Single scale applied to all weight elements (compile-time constant).
    """
    accum_dtype = "float32"
    scale_val = T.float32(uniform_scale)

    @T.prim_func
    def main_kernel_inner(
        A: T.Tensor((M, K), dtype),
        W_int8: T.Tensor((N, K), "int8"),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
            # Local tile buffers
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            W_int8_buf = T.alloc_shared((block_N, block_K), "int8")
            W_fp16_buf = T.alloc_shared((block_N, block_K), dtype)
            C_accum = T.alloc_shared((block_M, block_N), accum_dtype)
            C_temp = T.alloc_shared((block_M, block_N), accum_dtype)
            C_out = T.alloc_shared((block_M, block_N), dtype)

            T.ppl_fill(C_accum, T.float32(0))

            for k in T.serial(T.ceildiv(K, block_K)):
                # Step 1: DMA load fp16 activation tile
                T.ppl_copy(A[by * block_M, k * block_K], A_shared)

                # Step 2: DMA load int8 weight tile (same dtype, pure DMA)
                T.ppl_copy(W_int8[bx * block_N, k * block_K], W_int8_buf)

                # Step 3: Cast int8 → fp16 (triggers tpu_bdc_cast)
                T.ppl_copy(W_int8_buf, W_fp16_buf)

                # Step 4: Apply uniform scale (compile-time constant)
                T.ppl_mul_C(W_fp16_buf, W_fp16_buf, scale_val)

                # Step 5: Fp16 matmul with transpose_B=True
                #   R_trans computes C[M,N] = A[M,K] @ B[N,K]^T (no result_add)
                #   We accumulate manually via C_accum += C_temp
                T.ppl_gemm(A_shared, W_fp16_buf, C_temp, transpose_B=True)
                T.ppl_add(C_accum, C_accum, C_temp)

            # Write accumulated fp32 result → fp16 → global memory
            T.ppl_copy(C_accum, C_out)
            T.ppl_copy(C_out, C[by * block_M, bx * block_N])

    return main_kernel_inner


if __name__ == "__main__":
    # Small test dimensions for cmodel verification
    M, N, K = 32, 128, 256
    block_M, block_N, block_K = 32, 128, 128
    blocksize = 128
    uniform_scale = 1.5  # non-trivial uniform scale for verification

    print(f"=== Phase 2a: Uniform Scale Dequant Matmul ===")
    print(f"M={M}, N={N}, K={K}, block_M={block_M}, block_N={block_N}, block_K={block_K}")
    print(f"uniform_scale={uniform_scale}")

    # Compile kernel in cmodel mode
    kernel = tilelang.compile(
        dequant_matmul_kernel_uniform(M, N, K, block_M, block_N, block_K,
                                       uniform_scale=uniform_scale),
        out_idx=-1,
        target="tpu",
        mode="cmodel",
    )

    # Prepare test data
    torch.manual_seed(42)
    a = torch.randn(M, K).half()
    w_int8 = torch.randint(-128, 127, (N, K), dtype=torch.int8)

    # Reference: uniform scale means W_scale is all uniform_scale
    num_N_blocks = (N + blocksize - 1) // blocksize
    num_K_blocks = (K + blocksize - 1) // blocksize
    w_scale = torch.full((num_N_blocks, num_K_blocks), uniform_scale).half()
    c = torch.zeros(M, N).half()

    # Run kernel
    print("Running kernel...")
    kernel(a, w_int8, c)

    # Compare with PyTorch reference
    ref = dequant_matmul_ref(a, w_int8, w_scale, blocksize)

    max_diff = (ref - c).abs().max().item()
    avg_diff = (ref - c).abs().mean().item()

    print(f"\n=== Results ===")
    print(f"TPU output sample: {c[0, :5]}")
    print(f"Ref  output sample: {ref[0, :5]}")
    print(f"最大差异 (max diff): {max_diff}")
    print(f"平均差异 (avg diff): {avg_diff}")
    passed = torch.allclose(c, ref, atol=1e-2, rtol=1e-2)
    print(f"check close (atol=1e-2, rtol=1e-2): {passed}")

    if not passed:
        print("\n!!! TEST FAILED !!!")
        exit(1)
    else:
        print("\n=== Phase 2a PASSED ===")
