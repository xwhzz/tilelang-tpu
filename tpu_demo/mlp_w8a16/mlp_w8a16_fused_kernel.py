# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Fused W8A16 dequantized MLP kernel for BM1690 TPU.

Computes in a SINGLE kernel launch:
    output = down_proj(silu(gate_proj(x)) * up_proj(x))

All three matmuls + SiLU + element-wise multiply happen entirely in
local (NPU) memory with zero CPU round-trips. The caller handles
CPU-side int8→fp16 dequantization before invoking this kernel.

Reference: PPL tgi/mlp.py fused MLP pattern.
"""

import tilelang.language as T


def mlp_w8a16_fused_kernel(
    M, hidden, intermediate,
    block_M=32, block_N=128, block_K=128, block_D=128,
    dtype="float16",
):
    """Generate a fused MLP kernel.

    Computes: Output[M, hidden] = down(silu(gate(x)) * up(x))

    Parameters
    ----------
    M : int
        Batch/sequence dimension.
    hidden : int
        Input and output feature dimension (e.g. 4096).
    intermediate : int
        Intermediate MLP dimension (e.g. 11008).
    block_M : int
        Tile size for M dimension. Default 32.
    block_N : int
        Tile size for intermediate (N) dimension. Default 128.
    block_K : int
        Tile size for K accumulation dimension. Default 128.
    block_D : int
        Tile size for output hidden (D) dimension. Must equal block_N
        to reuse W_shared buffer. Default 128.
    dtype : str
        Data type for activations and dequantized weights. Default "float16".
    """
    assert block_D == block_N, (
        "block_D must equal block_N for W_shared buffer reuse. "
        f"Got block_D={block_D}, block_N={block_N}"
    )

    accum_dtype = "float32"

    @T.prim_func
    def main_kernel_inner(
        X: T.Tensor((M, hidden), dtype),
        W_gate: T.Tensor((intermediate, hidden), dtype),
        W_up: T.Tensor((intermediate, hidden), dtype),
        W_down: T.Tensor((hidden, intermediate), dtype),
        Output: T.Tensor((M, hidden), dtype),
    ):
        # Grid: by iterates M tiles. D-tile loop is INSIDE (serial) to avoid
        # code explosion from replicating the full loop nest per output tile.
        with T.Kernel(1, T.ceildiv(M, block_M), is_cpu=True) as (_, by):

            # --- Activation tile (reused across N and K loops) ---
            A_shared = T.alloc_shared((block_M, block_K), dtype)

            # --- Weight tile (reused for gate, up, down) ---
            W_shared = T.alloc_shared((block_N, block_K), dtype)

            # --- Gate accumulation (per N-tile) ---
            gate_accum = T.alloc_shared((block_M, block_N), accum_dtype)

            # --- Up accumulation (per N-tile) ---
            up_accum = T.alloc_shared((block_M, block_N), accum_dtype)

            # --- GEMM temporary for R_trans without result_add ---
            C_temp = T.alloc_shared((block_M, block_N), accum_dtype)

            # --- Down accumulation (across all N-tiles, per D-tile) ---
            down_accum = T.alloc_shared((block_M, block_D), accum_dtype)

            # --- Down GEMM temporary ---
            down_temp = T.alloc_shared((block_M, block_D), accum_dtype)

            # --- SiLU: ones constant ---
            ones = T.alloc_shared((block_M, block_N), accum_dtype)
            T.ppl_fill(ones, T.float32(1.0))

            # --- SiLU: exp2 work buffers (pattern from tpu_test_swiglu_fp32.py) ---
            # exp_work0 reused as negate buffer before exp2 call
            exp_work0 = T.alloc_shared((block_M, block_N), accum_dtype)
            exp_work1 = T.alloc_shared((block_M, block_N), accum_dtype)
            # coeff and table shapes from BM1690 hardware spec (npu=64)
            exp_coeff = T.alloc_shared((64, 32), accum_dtype)
            exp_table = T.alloc_shared((64, 192), accum_dtype)

            # --- Activation fp16 (for down GEMM input) ---
            act_fp16 = T.alloc_shared((block_M, block_N), dtype)

            # --- Down output temporary (fp32→fp16 before global store) ---
            down_out = T.alloc_shared((block_M, block_D), dtype)

            # ============================================================
            # Main computation: iterate over output hidden (D) tiles,
            # then intermediate (N) tiles, then K tiles for accumulation.
            # ============================================================
            for d_tile in T.serial(T.ceildiv(hidden, block_D)):
                # Initialize down accumulator for this D-tile
                T.ppl_fill(down_accum, T.float32(0))

                for n_tile in T.serial(T.ceildiv(intermediate, block_N)):
                    # Initialize gate and up accumulators for this N-tile
                    T.ppl_fill(gate_accum, T.float32(0))
                    T.ppl_fill(up_accum, T.float32(0))

                    # K-dimension accumulation for gate and up projections
                    for k_tile in T.serial(T.ceildiv(hidden, block_K)):
                        # Load activation tile (reused for gate and up)
                        T.ppl_copy(X[by * block_M, k_tile * block_K], A_shared)

                        # --- Gate projection: gate_accum += A @ W_gate^T ---
                        T.ppl_copy(W_gate[n_tile * block_N, k_tile * block_K], W_shared)
                        T.ppl_gemm(A_shared, W_shared, C_temp, transpose_B=True)
                        T.ppl_add(gate_accum, gate_accum, C_temp)

                        # --- Up projection: up_accum += A @ W_up^T ---
                        T.ppl_copy(W_up[n_tile * block_N, k_tile * block_K], W_shared)
                        T.ppl_gemm(A_shared, W_shared, C_temp, transpose_B=True)
                        T.ppl_add(up_accum, up_accum, C_temp)

                    # =============================================
                    # SiLU(gate_accum) * up_accum (on-NPU, local)
                    # Pattern verified in tpu_test_swiglu_fp32.py
                    # =============================================
                    # Step 1: Backup gate_accum → C_temp (for later div)
                    T.ppl_copy(gate_accum, C_temp)
                    # Step 2: exp_work0 = -gate_accum
                    T.ppl_mul_C(exp_work0, gate_accum, T.float32(-1.0))
                    # Step 3: exp_work0 = exp2(-gate), using exp_work1+gate_accum as scratch
                    T.ppl_exp2(exp_work0, exp_work1, gate_accum, exp_coeff, exp_table)
                    # Step 4: exp_work1 = exp(-gate) + 1
                    T.ppl_add(exp_work1, exp_work0, ones)
                    # Step 5: gate_accum = C_temp / exp_work1 = SiLU(gate)
                    T.ppl_div(gate_accum, C_temp, exp_work1)
                    # Step 6: C_temp = SiLU(gate) * up
                    T.ppl_mul(C_temp, gate_accum, up_accum)
                    # Cast to fp16 for down GEMM
                    T.ppl_copy(C_temp, act_fp16)

                    # --- Down projection (accumulate across N-tiles) ---
                    T.ppl_copy(W_down[d_tile * block_D, n_tile * block_N], W_shared)
                    T.ppl_gemm(act_fp16, W_shared, down_temp, transpose_B=True)
                    T.ppl_add(down_accum, down_accum, down_temp)

                # Store accumulated down result for this D-tile
                T.ppl_copy(down_accum, down_out)
                T.ppl_copy(down_out, Output[by * block_M, d_tile * block_D])

    return main_kernel_inner
