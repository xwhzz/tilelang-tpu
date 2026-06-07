# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Fused W8A16 dequantized MLP kernel for BM1690 TPU.

Computes in a SINGLE kernel launch:
    output = down_proj(silu(gate_proj(x)) * up_proj(x))

All three matmuls + SiLU + element-wise multiply happen entirely in
local (NPU) memory with zero CPU round-trips. The caller handles
CPU-side int8→fp16 dequantization before invoking this kernel.

The D-tile (output hidden) dimension is mapped to the TPU grid so that
each grid block computes one (D-tile × M-tile) output slice. This avoids
TIR code explosion from serial D-tile loops and keeps compilation time
practical even for large hidden sizes (e.g. DeepSeek-V3 h=4096).

Buffer layout (8×fp32 + 3×fp16 + 2 fixed = 13 buffers, ~189KB at block_M=32):
  fp32: gate_accum, up_accum, C_temp, down_accum,
        exp_work0, exp_work1, exp_coeff, exp_table
  fp16: A_shared, W_shared, down_out
  (act_fp16 reused from A_shared, down_temp reused from C_temp, ones elided
   via ppl_add_C)
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
        for W_shared buffer reuse. Default 128.
    dtype : str
        Data type for activations and dequantized weights. Default "float16".
    """
    assert block_D == block_N, (
        "block_D must equal block_N for W_shared buffer reuse. "
        f"Got block_D={block_D}, block_N={block_N}"
    )

    accum_dtype = "float32"
    D_tiles = T.ceildiv(hidden, block_D)
    M_tiles = T.ceildiv(M, block_M)

    @T.prim_func
    def main_kernel_inner(
        X: T.Tensor((M, hidden), dtype),
        W_gate: T.Tensor((intermediate, hidden), dtype),
        W_up: T.Tensor((intermediate, hidden), dtype),
        W_down: T.Tensor((hidden, intermediate), dtype),
        Output: T.Tensor((M, hidden), dtype),
    ):
        # Grid: D-tile × M-tile.  Each grid block computes one output slice.
        # D-tile is at grid level (not an inner serial loop) to avoid code
        # explosion for large hidden sizes (e.g. DSv3 32 D-tiles).
        with T.Kernel(D_tiles, M_tiles, is_cpu=True) as (d_tile, by):

            # --- Activation tile (block_M × block_K, fp16) ---
            A_shared = T.alloc_shared((block_M, block_K), dtype)

            # --- Weight tile (block_N × block_K, fp16) — shared by gate/up/down ---
            W_shared = T.alloc_shared((block_N, block_K), dtype)

            # --- Gate / Up accumulators (per N-tile) ---
            gate_accum = T.alloc_shared((block_M, block_N), accum_dtype)
            up_accum = T.alloc_shared((block_M, block_N), accum_dtype)

            # --- GEMM / SiLU / down scratch (multi-use) ---
            # C_temp is reused across phases:
            #   K-loop:  GEMM result accumulation temp (gate + up)
            #   SiLU:    gate backup → silu(gate)*up result
            #   N-loop:  down GEMM result temp (replaces down_temp)
            C_temp = T.alloc_shared((block_M, block_N), accum_dtype)

            # --- Down accumulation (across N-tiles for this D-tile) ---
            down_accum = T.alloc_shared((block_M, block_D), accum_dtype)

            # --- SiLU: exp2 work buffers ---
            exp_work0 = T.alloc_shared((block_M, block_N), accum_dtype)
            exp_work1 = T.alloc_shared((block_M, block_N), accum_dtype)
            # coeff and table shapes from BM1690 hardware spec (npu=64)
            exp_coeff = T.alloc_shared((64, 32), accum_dtype)
            exp_table = T.alloc_shared((64, 192), accum_dtype)

            # --- Down output (fp32→fp16 before global store) ---
            down_out = T.alloc_shared((block_M, block_D), dtype)

            # ============================================================
            # Main computation for this (D-tile, M-tile).
            # ============================================================
            T.ppl_fill(down_accum, T.float32(0))

            for n_tile in T.serial(T.ceildiv(intermediate, block_N)):
                T.ppl_fill(gate_accum, T.float32(0))
                T.ppl_fill(up_accum, T.float32(0))

                # K-dimension accumulation for gate and up
                for k_tile in T.serial(T.ceildiv(hidden, block_K)):
                    T.ppl_copy(X[by * block_M, k_tile * block_K], A_shared)

                    # Gate: gate_accum += A @ W_gate^T
                    T.ppl_copy(W_gate[n_tile * block_N, k_tile * block_K], W_shared)
                    T.ppl_gemm(A_shared, W_shared, C_temp, transpose_B=True)
                    T.ppl_add(gate_accum, gate_accum, C_temp)

                    # Up: up_accum += A @ W_up^T
                    T.ppl_copy(W_up[n_tile * block_N, k_tile * block_K], W_shared)
                    T.ppl_gemm(A_shared, W_shared, C_temp, transpose_B=True)
                    T.ppl_add(up_accum, up_accum, C_temp)

                # =============================================
                # SiLU(gate_accum) * up_accum (on-NPU, local)
                # =============================================
                # Step 1: Backup gate → C_temp
                T.ppl_copy(gate_accum, C_temp)
                # Step 2: exp_work0 = -gate
                T.ppl_mul_C(exp_work0, gate_accum, T.float32(-1.0))
                # Step 3: exp_work0 = exp2(-gate), scratch: exp_work1+gate_accum
                T.ppl_exp2(exp_work0, exp_work1, gate_accum, exp_coeff, exp_table)
                # Step 4: exp_work1 = exp(-gate) + 1  (ppl_add_C replaces ones buffer)
                T.ppl_add_C(exp_work1, exp_work0, T.float32(1.0))
                # Step 5: gate_accum = C_temp / exp_work1 = silu(gate)
                T.ppl_div(gate_accum, C_temp, exp_work1)
                # Step 6: C_temp = silu(gate) * up  (C_temp reused as result)
                T.ppl_mul(C_temp, gate_accum, up_accum)

                # Step 7: Clamp to fp16 representable range [-65500, 65500].
                # This prevents INF in the down GEMM when intermediate values
                # overflow fp16.  Uses exp_work0 as fill scratch and exp_work1
                # as clamp temp (both are free after SiLU steps 1-6).
                fp16_max = T.float32(65500.0)
                T.ppl_fill(exp_work0, fp16_max)
                T.ppl_min(exp_work1, C_temp, exp_work0)
                T.ppl_fill(exp_work0, T.float32(-65500.0))
                T.ppl_max(C_temp, exp_work1, exp_work0)

                # --- Down projection (accumulate across N-tiles) ---
                # Cast silu*up to fp16 via A_shared (free after K-loop).
                # A_shared and the N-tile have compatible shapes when
                # block_K >= block_N, which holds for all real workloads
                # (both saturate at 128).
                T.ppl_copy(C_temp, A_shared)
                T.ppl_copy(
                    W_down[d_tile * block_D, n_tile * block_N], W_shared)
                # C_temp reused as down GEMM output (replaces down_temp)
                T.ppl_gemm(A_shared, W_shared, C_temp, transpose_B=True)
                T.ppl_add(down_accum, down_accum, C_temp)

            T.ppl_copy(down_accum, down_out)
            T.ppl_copy(down_out, Output[by * block_M, d_tile * block_D])

    return main_kernel_inner
