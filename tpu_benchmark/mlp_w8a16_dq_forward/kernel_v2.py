# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Fused W8A16 MLP kernel v2 — D-tile redundancy eliminated.

Key difference from v1:
  v1: Grid = (D_tiles, M_tiles). Each grid block redundantly computes full
      gate+up+SiLU, causing D_tiles× redundant GEMMs. Fast for small models
      but 6.6× slower than PPL for DSv3 (8 D-tiles).

  v2: Grid = (M_tiles,). Gate+up+SiLU computed ONCE for all N-tiles and
      stored to an intermediate Act buffer (global memory). Down projection
      then loops over D-tiles serially, reading from the Act buffer.
      Eliminates redundant gate+up computation entirely.

Tradeoff:
  - Extra global memory traffic for Act buffer (M × intermediate × 2 bytes).
    For DSv3 (M=1,I=2752): ~5.5KB — negligible.
  - D-tiles serial (inner loop) instead of in grid, but gate+up runs once
    instead of D_tiles times — net win for D_tiles ≥ 2.

SiLU steps (12): backup → negate → upper-bound clamp → exp → add → div → mul → min/max clamp.
"""

import tilelang.language as T


def _largest_supported_divisor(values, preferred):
    for candidate in (preferred, 128, 64, 32, 16, 8, 4, 2, 1):
        if candidate > 0 and all(value % candidate == 0 for value in values):
            return candidate
    raise ValueError(f"Cannot find a common tile divisor for {values}")


def mlp_w8a16_fused_kernel_v2(
    M, hidden, intermediate,
    block_M=32, block_N=128, block_K=128, block_D=128,
    dtype="float16",
):
    """Generate a fused MLP kernel v2 (no D-tile redundancy).

    Computes: Output[M, hidden] = down(silu(gate(x)) * up(x))

    The caller must provide an Act buffer of shape (M, intermediate) in fp16
    for intermediate storage between the gate/up/SiLU and down phases.

    Parameters
    ----------
    M : int
        Batch/sequence dimension.
    hidden : int
        Input and output feature dimension.
    intermediate : int
        Intermediate MLP dimension.
    block_M : int
        Tile size for M dimension. Default 32.
    block_N : int
        Tile size for intermediate (N) dimension. Default 128.
    block_K : int
        Tile size for K accumulation dimension. Default 128.
    block_D : int
        Tile size for output hidden (D) dimension. Must equal block_N.
    dtype : str
        Data type for activations and weights. Default "float16".
    """
    if M % block_M != 0:
        block_M = _largest_supported_divisor((M,), block_M)

    # PPL copy currently has no implicit tail masking.  Hidden tiles must divide
    # the static shape.  Intermediate tails are handled explicitly below.
    if (
        block_N != block_K
        or block_D != block_N
        or hidden % block_K != 0
        or hidden % block_D != 0
    ):
        common_block = _largest_supported_divisor((hidden, intermediate), block_N)
        block_N = common_block
        block_K = common_block
        block_D = common_block

    accum_dtype = "float32"
    M_tiles = T.ceildiv(M, block_M)
    N_tiles_full = intermediate // block_N
    N_tail = intermediate - N_tiles_full * block_N
    N_tail_alloc = N_tail if N_tail != 0 else 1
    K_tiles = T.ceildiv(hidden, block_K)
    D_tiles = T.ceildiv(hidden, block_D)

    @T.prim_func
    def main_kernel_inner(
        X: T.Tensor((M, hidden), dtype),
        W_gate: T.Tensor((intermediate, hidden), dtype),
        W_up: T.Tensor((intermediate, hidden), dtype),
        W_down: T.Tensor((hidden, intermediate), dtype),
        Act: T.Tensor((M, intermediate), dtype),
        Output: T.Tensor((M, hidden), dtype),
    ):
        # Grid: M-tile only (no D-tile at grid level → no redundancy).
        with T.Kernel(M_tiles, is_cpu=True) as (by,):

            # --- Activation tile (block_M × block_K, fp16) ---
            A_shared = T.alloc_shared((block_M, block_K), dtype)

            # --- Weight tile (block_N × block_K, fp16) — shared by gate/up/down ---
            W_shared = T.alloc_shared((block_N, block_K), dtype)

            # --- Gate / Up accumulators (per N-tile) ---
            gate_accum = T.alloc_shared((block_M, block_N), accum_dtype)
            up_accum = T.alloc_shared((block_M, block_N), accum_dtype)

            # --- GEMM / SiLU / down scratch ---
            C_temp = T.alloc_shared((block_M, block_N), accum_dtype)

            # --- SiLU: exp work buffers ---
            exp_work0 = T.alloc_shared((block_M, block_N), accum_dtype)
            exp_work1 = T.alloc_shared((block_M, block_N), accum_dtype)

            # --- Down accumulation buffer (reuses gate_accum after Phase 1) ---
            down_accum = T.alloc_shared((block_M, block_D), accum_dtype)

            A_tail = T.alloc_shared((block_M, N_tail_alloc), dtype)
            W_tail = T.alloc_shared((N_tail_alloc, block_K), dtype)
            W_down_tail = T.alloc_shared((block_D, N_tail_alloc), dtype)
            gate_tail = T.alloc_shared((block_M, N_tail_alloc), accum_dtype)
            up_tail = T.alloc_shared((block_M, N_tail_alloc), accum_dtype)
            C_tail = T.alloc_shared((block_M, N_tail_alloc), accum_dtype)
            tail_work0 = T.alloc_shared((block_M, N_tail_alloc), accum_dtype)
            tail_work1 = T.alloc_shared((block_M, N_tail_alloc), accum_dtype)

            # ============================================================
            # Phase 1: gate+up+SiLU → Act buffer (computed ONCE).
            # ============================================================
            for n_tile in T.serial(N_tiles_full):
                T.fill(gate_accum, T.float32(0))
                T.fill(up_accum, T.float32(0))

                # K-dimension accumulation for gate and up
                for k_tile in T.serial(K_tiles):
                    T.copy(X[by * block_M, k_tile * block_K], A_shared)

                    # Gate: gate_accum += A @ W_gate^T
                    T.copy(W_gate[n_tile * block_N, k_tile * block_K], W_shared)
                    T.gemm(A_shared, W_shared, C_temp, transpose_B=True)
                    T.add(gate_accum, gate_accum, C_temp)

                    # Up: up_accum += A @ W_up^T
                    T.copy(W_up[n_tile * block_N, k_tile * block_K], W_shared)
                    T.gemm(A_shared, W_shared, C_temp, transpose_B=True)
                    T.add(up_accum, up_accum, C_temp)

                # =============================================
                # SiLU(gate_accum) * up_accum (on-NPU, local)
                # =============================================
                # Step 1: Backup gate → C_temp
                T.copy(gate_accum, C_temp)
                # Step 2: exp_work0 = -gate
                T.mul_C(exp_work0, gate_accum, T.float32(-1.0))
                # Step 2b: Clamp -gate to 88.72 = ln(FLT_MAX) to prevent fp32 overflow
                T.fill(exp_work1, T.float32(88.72))
                T.mul_C(exp_work0, exp_work0, T.float32(-1.0))
                T.mul_C(exp_work1, exp_work1, T.float32(-1.0))
                T.max(exp_work0, exp_work0, exp_work1)
                T.mul_C(exp_work0, exp_work0, T.float32(-1.0))
                # Step 3: exp_work0 = exp(-gate)
                T.exp(exp_work0)
                # Step 4: exp_work1 = exp(-gate) + 1
                T.add_C(exp_work1, exp_work0, T.float32(1.0))
                # Step 5: gate_accum = gate / (exp(-gate) + 1) = silu(gate)
                T.div(gate_accum, C_temp, exp_work1)
                # Step 6: C_temp = silu(gate) * up
                T.mul(C_temp, gate_accum, up_accum)

                # Step 7: Clamp to fp16 representable range [-65500, 65500]
                fp16_max = T.float32(65500.0)
                T.mul_C(exp_work1, C_temp, T.float32(-1.0))
                T.fill(exp_work0, -fp16_max)
                T.max(exp_work1, exp_work1, exp_work0)
                T.mul_C(exp_work1, exp_work1, T.float32(-1.0))
                T.fill(exp_work0, T.float32(-65500.0))
                T.max(C_temp, exp_work1, exp_work0)

                # Store to Act buffer (fp32 → fp16 via A_shared)
                T.copy(C_temp, A_shared)
                T.copy(A_shared, Act[by * block_M, n_tile * block_N])

            if N_tail != 0:
                T.fill(gate_tail, T.float32(0))
                T.fill(up_tail, T.float32(0))
                for k_tile in T.serial(K_tiles):
                    T.copy(X[by * block_M, k_tile * block_K], A_shared)
                    T.copy(W_gate[N_tiles_full * block_N, k_tile * block_K], W_tail)
                    T.gemm(A_shared, W_tail, C_tail, transpose_B=True)
                    T.add(gate_tail, gate_tail, C_tail)
                    T.copy(W_up[N_tiles_full * block_N, k_tile * block_K], W_tail)
                    T.gemm(A_shared, W_tail, C_tail, transpose_B=True)
                    T.add(up_tail, up_tail, C_tail)

                T.copy(gate_tail, C_tail)
                T.mul_C(tail_work0, gate_tail, T.float32(-1.0))
                T.fill(tail_work1, T.float32(88.72))
                T.mul_C(tail_work0, tail_work0, T.float32(-1.0))
                T.mul_C(tail_work1, tail_work1, T.float32(-1.0))
                T.max(tail_work0, tail_work0, tail_work1)
                T.mul_C(tail_work0, tail_work0, T.float32(-1.0))
                T.exp(tail_work0)
                T.add_C(tail_work1, tail_work0, T.float32(1.0))
                T.div(gate_tail, C_tail, tail_work1)
                T.mul(C_tail, gate_tail, up_tail)

                fp16_max = T.float32(65500.0)
                T.mul_C(tail_work1, C_tail, T.float32(-1.0))
                T.fill(tail_work0, -fp16_max)
                T.max(tail_work1, tail_work1, tail_work0)
                T.mul_C(tail_work1, tail_work1, T.float32(-1.0))
                T.fill(tail_work0, T.float32(-65500.0))
                T.max(C_tail, tail_work1, tail_work0)

                T.copy(C_tail, A_tail)
                T.copy(A_tail, Act[by * block_M, N_tiles_full * block_N])

            # ============================================================
            # Phase 2: Down projection (D-tile serial loop).
            # Reads pre-computed Act from global memory.
            # ============================================================
            for d_tile in T.serial(D_tiles):
                T.fill(down_accum, T.float32(0))

                for n_tile in T.serial(N_tiles_full):
                    # Load pre-computed SiLU(gate)*up for this N-tile
                    T.copy(Act[by * block_M, n_tile * block_N], A_shared)
                    # Load W_down slice for this D-tile, N-tile
                    T.copy(
                        W_down[d_tile * block_D, n_tile * block_N], W_shared)
                    T.gemm(A_shared, W_shared, C_temp, transpose_B=True)
                    T.add(down_accum, down_accum, C_temp)

                if N_tail != 0:
                    T.copy(Act[by * block_M, N_tiles_full * block_N], A_tail)
                    T.copy(
                        W_down[d_tile * block_D, N_tiles_full * block_N], W_down_tail)
                    T.gemm(A_tail, W_down_tail, C_temp, transpose_B=True)
                    T.add(down_accum, down_accum, C_temp)

                # Cast and store output
                # Reuse A_shared (free after N-tile loop) as fp16 output buffer
                T.copy(down_accum, A_shared)
                T.copy(A_shared, Output[by * block_M, d_tile * block_D])

    return main_kernel_inner


def mlp_w8a16_fused_kernel_streaming_fullk(
    M, hidden, intermediate,
    block_M=32, block_N=128, block_K=128, block_D=128,
    num_stages=0,
    dtype="float16",
):
    """Generate a PPL-like streaming fused MLP kernel.

    Compared with ``mlp_w8a16_fused_kernel_v2``, this variant keeps the
    intermediate tile local: each intermediate tile computes gate/up/SiLU and
    immediately accumulates the down projection into a full hidden output tile.
    It also performs gate/up with full hidden K in one GEMM, matching the PPL
    single-core structure more closely than the K-tiled v2 implementation.

    The external signature still includes ``Act`` for runner compatibility, but
    this kernel does not read or write it.
    """
    if M % block_M != 0:
        block_M = _largest_supported_divisor((M,), block_M)

    # This streaming path handles the hidden dimension as full-K GEMM.  Only
    # intermediate tails need explicit handling because ppl.copy has no masking.
    if block_N <= 0:
        block_N = 128
    block_K = hidden
    block_D = hidden

    accum_dtype = "float32"
    M_tiles = T.ceildiv(M, block_M)
    N_tiles_full = intermediate // block_N
    N_tail = intermediate - N_tiles_full * block_N
    N_tail_alloc = N_tail if N_tail != 0 else 1

    @T.prim_func
    def main_kernel_inner(
        X: T.Tensor((M, hidden), dtype),
        W_gate: T.Tensor((intermediate, hidden), dtype),
        W_up: T.Tensor((intermediate, hidden), dtype),
        W_down: T.Tensor((hidden, intermediate), dtype),
        Act: T.Tensor((M, intermediate), dtype),
        Output: T.Tensor((M, hidden), dtype),
    ):
        with T.Kernel(M_tiles, is_cpu=True) as (by,):
            X_full = T.alloc_shared((block_M, hidden), dtype)
            W_gate_full = T.alloc_shared((block_N, hidden), dtype)
            W_up_full = T.alloc_shared((block_N, hidden), dtype)
            W_down_full = T.alloc_shared((hidden, block_N), dtype)

            gate_local = T.alloc_shared((block_M, block_N), dtype)
            up_local = T.alloc_shared((block_M, block_N), dtype)
            act_local = T.alloc_shared((block_M, block_N), dtype)
            sigmoid_local = T.alloc_shared((block_M, block_N), dtype)
            gate_orig = T.alloc_shared((block_M, block_N), accum_dtype)
            gate_f32 = T.alloc_shared((block_M, block_N), accum_dtype)
            work0 = T.alloc_shared((block_M, block_N), accum_dtype)

            down_accum = T.alloc_shared((block_M, hidden), accum_dtype)
            down_temp = T.alloc_shared((block_M, hidden), accum_dtype)
            out_local = T.alloc_shared((block_M, hidden), dtype)


            W_gate_tail = T.alloc_shared((N_tail_alloc, hidden), dtype)
            W_up_tail = T.alloc_shared((N_tail_alloc, hidden), dtype)
            W_down_tail = T.alloc_shared((hidden, N_tail_alloc), dtype)
            gate_tail = T.alloc_shared((block_M, N_tail_alloc), dtype)
            up_tail = T.alloc_shared((block_M, N_tail_alloc), dtype)
            act_tail = T.alloc_shared((block_M, N_tail_alloc), dtype)
            sigmoid_tail = T.alloc_shared((block_M, N_tail_alloc), dtype)
            gate_tail_orig = T.alloc_shared((block_M, N_tail_alloc), accum_dtype)
            gate_tail_f32 = T.alloc_shared((block_M, N_tail_alloc), accum_dtype)
            tail_work0 = T.alloc_shared((block_M, N_tail_alloc), accum_dtype)
            tail_work1 = T.alloc_shared((block_M, N_tail_alloc), accum_dtype)

            T.fill(down_accum, T.float32(0))

            if num_stages == 0:
                for n_tile in T.serial(N_tiles_full):
                    T.copy(X[by * block_M, 0], X_full)
                    T.copy(W_gate[n_tile * block_N, 0], W_gate_full)
                    T.copy(W_up[n_tile * block_N, 0], W_up_full)

                    T.gemm(X_full, W_gate_full, gate_local, transpose_B=True)
                    T.gemm(X_full, W_up_full, up_local, transpose_B=True)

                    # gate_f32 = silu(gate_local)
                    T.copy(gate_local, gate_orig)
                    T.sigmoid(work0, gate_orig)
                    T.copy(work0, sigmoid_local)
                    T.mul(gate_local, gate_local, sigmoid_local)
                    T.mul(act_local, gate_local, up_local)

                    T.copy(W_down[0, n_tile * block_N], W_down_full)
                    T.gemm(act_local, W_down_full, down_temp, transpose_B=True)
                    T.add(down_accum, down_accum, down_temp)
            else:
                for n_tile in T.Pipelined(N_tiles_full, num_stages=num_stages):
                    T.copy(X[by * block_M, 0], X_full)
                    T.copy(W_gate[n_tile * block_N, 0], W_gate_full)
                    T.copy(W_up[n_tile * block_N, 0], W_up_full)

                    T.gemm(X_full, W_gate_full, gate_local, transpose_B=True)
                    T.gemm(X_full, W_up_full, up_local, transpose_B=True)

                    # gate_f32 = silu(gate_local)
                    T.copy(gate_local, gate_orig)
                    T.sigmoid(work0, gate_orig)
                    T.copy(work0, sigmoid_local)
                    T.mul(gate_local, gate_local, sigmoid_local)
                    T.mul(act_local, gate_local, up_local)

                    T.copy(W_down[0, n_tile * block_N], W_down_full)
                    T.gemm(act_local, W_down_full, down_temp, transpose_B=True)
                    T.add(down_accum, down_accum, down_temp)

            if N_tail != 0:
                T.copy(X[by * block_M, 0], X_full)
                T.copy(W_gate[N_tiles_full * block_N, 0], W_gate_tail)
                T.copy(W_up[N_tiles_full * block_N, 0], W_up_tail)
                T.gemm(X_full, W_gate_tail, gate_tail, transpose_B=True)
                T.gemm(X_full, W_up_tail, up_tail, transpose_B=True)

                T.copy(gate_tail, gate_tail_orig)
                T.sigmoid(tail_work0, gate_tail_orig)
                T.copy(tail_work0, sigmoid_tail)
                T.mul(gate_tail, gate_tail, sigmoid_tail)
                T.mul(act_tail, gate_tail, up_tail)

                T.copy(W_down[0, N_tiles_full * block_N], W_down_tail)
                T.gemm(act_tail, W_down_tail, down_temp, transpose_B=True)
                T.add(down_accum, down_accum, down_temp)

            T.copy(down_accum, out_local)
            T.copy(out_local, Output[by * block_M, 0])

    return main_kernel_inner


def mlp_w8a16_fused_kernel_ppl_layout_streaming(
    M, hidden, intermediate,
    block_M=32, block_N=128, block_K=128, block_D=128,
    num_stages=0,
    dtype="float16",
):
    """Streaming MLP using PPL-style weight layout and non-transpose GEMM.

    The logical math is identical to ``mlp_w8a16_fused_kernel_streaming_fullk``,
    but the weight tensors use the same contiguous layout as the PPL baseline:
    ``W_gate_col/W_up_col`` are ``[hidden, intermediate]`` and
    ``W_down_col`` is ``[intermediate, hidden]``.  This lets the down projection
    use the existing non-transpose TPU GEMM accumulate path directly.
    """
    if M % block_M != 0:
        block_M = _largest_supported_divisor((M,), block_M)

    if block_N <= 0:
        block_N = 128
    block_K = hidden
    block_D = hidden

    accum_dtype = "float32"
    M_tiles = T.ceildiv(M, block_M)
    N_tiles_full = intermediate // block_N
    N_tail = intermediate - N_tiles_full * block_N
    N_tail_alloc = N_tail if N_tail != 0 else 1

    @T.prim_func
    def main_kernel_inner(
        X: T.Tensor((M, hidden), dtype),
        W_gate_col: T.Tensor((hidden, intermediate), dtype),
        W_up_col: T.Tensor((hidden, intermediate), dtype),
        W_down_col: T.Tensor((intermediate, hidden), dtype),
        Act: T.Tensor((M, intermediate), dtype),
        Output: T.Tensor((M, hidden), dtype),
    ):
        with T.Kernel(M_tiles, is_cpu=True) as (by,):
            X_full = T.alloc_shared((block_M, hidden), dtype)
            W_gate_full = T.alloc_shared((hidden, block_N), dtype)
            W_up_full = T.alloc_shared((hidden, block_N), dtype)
            W_down_full = T.alloc_shared((block_N, hidden), dtype)

            gate_local = T.alloc_shared((block_M, block_N), dtype)
            up_local = T.alloc_shared((block_M, block_N), dtype)
            act_local = T.alloc_shared((block_M, block_N), dtype)
            sigmoid_local = T.alloc_shared((block_M, block_N), dtype)
            gate_f32 = T.alloc_shared((block_M, block_N), accum_dtype)
            up_f32 = T.alloc_shared((block_M, block_N), accum_dtype)
            work0 = T.alloc_shared((block_M, block_N), accum_dtype)

            down_accum = T.alloc_shared((block_M, hidden), accum_dtype)
            out_local = T.alloc_shared((block_M, hidden), dtype)


            W_gate_tail = T.alloc_shared((hidden, N_tail_alloc), dtype)
            W_up_tail = T.alloc_shared((hidden, N_tail_alloc), dtype)
            W_down_tail = T.alloc_shared((N_tail_alloc, hidden), dtype)
            gate_tail = T.alloc_shared((block_M, N_tail_alloc), dtype)
            up_tail = T.alloc_shared((block_M, N_tail_alloc), dtype)
            act_tail = T.alloc_shared((block_M, N_tail_alloc), dtype)
            sigmoid_tail = T.alloc_shared((block_M, N_tail_alloc), dtype)
            gate_tail_f32 = T.alloc_shared((block_M, N_tail_alloc), accum_dtype)
            up_tail_f32 = T.alloc_shared((block_M, N_tail_alloc), accum_dtype)
            tail_work0 = T.alloc_shared((block_M, N_tail_alloc), accum_dtype)
            tail_work1 = T.alloc_shared((block_M, N_tail_alloc), accum_dtype)

            T.fill(down_accum, T.float32(0))

            if num_stages == 0:
                for n_tile in T.serial(N_tiles_full):
                    T.copy(X[by * block_M, 0], X_full)
                    T.copy(W_gate_col[0, n_tile * block_N], W_gate_full)
                    T.copy(W_up_col[0, n_tile * block_N], W_up_full)

                    T.fill(gate_f32, T.float32(0))
                    T.gemm(X_full, W_gate_full, gate_f32)
                    T.copy(gate_f32, gate_local)

                    T.fill(up_f32, T.float32(0))
                    T.gemm(X_full, W_up_full, up_f32)
                    T.copy(up_f32, up_local)

                    T.sigmoid(work0, gate_f32)
                    T.copy(work0, sigmoid_local)
                    T.mul(gate_local, gate_local, sigmoid_local)
                    T.mul(act_local, gate_local, up_local)

                    T.copy(W_down_col[n_tile * block_N, 0], W_down_full)
                    T.gemm(act_local, W_down_full, down_accum)
            else:
                for n_tile in T.Pipelined(N_tiles_full, num_stages=num_stages):
                    T.copy(X[by * block_M, 0], X_full)
                    T.copy(W_gate_col[0, n_tile * block_N], W_gate_full)
                    T.copy(W_up_col[0, n_tile * block_N], W_up_full)

                    T.fill(gate_f32, T.float32(0))
                    T.gemm(X_full, W_gate_full, gate_f32)
                    T.copy(gate_f32, gate_local)

                    T.fill(up_f32, T.float32(0))
                    T.gemm(X_full, W_up_full, up_f32)
                    T.copy(up_f32, up_local)

                    T.sigmoid(work0, gate_f32)
                    T.copy(work0, sigmoid_local)
                    T.mul(gate_local, gate_local, sigmoid_local)
                    T.mul(act_local, gate_local, up_local)

                    T.copy(W_down_col[n_tile * block_N, 0], W_down_full)
                    T.gemm(act_local, W_down_full, down_accum)

            if N_tail != 0:
                T.copy(X[by * block_M, 0], X_full)
                T.copy(W_gate_col[0, N_tiles_full * block_N], W_gate_tail)
                T.copy(W_up_col[0, N_tiles_full * block_N], W_up_tail)

                T.fill(gate_tail_f32, T.float32(0))
                T.gemm(X_full, W_gate_tail, gate_tail_f32)
                T.copy(gate_tail_f32, gate_tail)

                T.fill(up_tail_f32, T.float32(0))
                T.gemm(X_full, W_up_tail, up_tail_f32)
                T.copy(up_tail_f32, up_tail)

                T.sigmoid(tail_work0, gate_tail_f32)
                T.copy(tail_work0, sigmoid_tail)
                T.mul(gate_tail, gate_tail, sigmoid_tail)
                T.mul(act_tail, gate_tail, up_tail)

                T.copy(W_down_col[N_tiles_full * block_N, 0], W_down_tail)
                T.gemm(act_tail, W_down_tail, down_accum)

            T.copy(down_accum, out_local)
            T.copy(out_local, Output[by * block_M, 0])

    return main_kernel_inner
