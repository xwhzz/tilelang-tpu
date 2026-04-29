"""
TileLang-TPU AutoTune
======================
Enumerate candidate block configs, compile+run each, return the fastest.

Usage
-----
    from autotune import autotune_matmul, autotune_rmsnorm, autotune_flashattn

Each function returns:
    best_cfg  : dict  — best block config found
    best_ms   : float — average execution time in ms
    all_results: list of (cfg, ms) sorted ascending
"""

import time
import math
import torch
import tilelang
import tilelang.language as T

T.copy = T.ppl_copy

# BM1690 local SRAM per NPU lane
_SRAM_LIMIT = 262144  # 256 KB

_DTYPE_BYTES = {
    "float32": 4, "float": 4,
    "float16": 2, "bfloat16": 2,
}

WARMUP = 3
ITERS  = 5


def _time_kernel(kernel_fn, *args):
    """Run kernel WARMUP+ITERS times, return average ms over ITERS."""
    for _ in range(WARMUP):
        kernel_fn(*args)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        kernel_fn(*args)
    return (time.perf_counter() - t0) / ITERS * 1000


# ─────────────────────────────────────────────────────────────────────────────
# Matmul
# ─────────────────────────────────────────────────────────────────────────────

def _matmul_sram(bM, bN, bK, dtype):
    db = _DTYPE_BYTES[dtype]
    if dtype == "float32":
        # A_fp32 + B_fp32 + A_fp16 + B_fp16 + C_fp32
        return bM*bK*4 + bK*bN*4 + bM*bK*2 + bK*bN*2 + bM*bN*4
    else:
        # A + B (lowp) + C_fp32 + C_out (lowp)
        return bM*bK*db + bK*bN*db + bM*bN*4 + bM*bN*db


def _matmul_candidates(M, N, K, dtype):
    """
    Return a curated list of (bM, bN, bK) that:
    - divide M, N, K respectively (or at least ≤ dim)
    - fit in SRAM
    - bM, bN, bK ≥ 8 (minimum meaningful tile)
    - ≤ 256 per axis
    Focuses on larger tiles where BDC utilisation matters.
    """
    POW2 = [8, 16, 32, 64, 128, 256]
    out = []
    for bM in POW2:
        if bM > M: break
        for bN in POW2:
            if bN > N: break
            for bK in POW2:
                if bK > K: break
                if _matmul_sram(bM, bN, bK, dtype) < _SRAM_LIMIT:
                    out.append((bM, bN, bK))
    # deduplicate and keep those with large tile area
    out = sorted(set(out), key=lambda x: -(x[0]*x[1]*x[2]))
    # keep top-N unique area groups to avoid combinatorial explosion
    seen_area = set()
    filtered = []
    for c in out:
        area = c[0] * c[1] * c[2]
        if area not in seen_area:
            seen_area.add(area)
            filtered.append(c)
        if len(filtered) >= 12:
            break
    return filtered


def _build_matmul_fp32(M, N, K, bM, bN, bK):
    @T.prim_func
    def main_kernel_inner(
        A: T.Tensor((M, K), "float32"),
        B: T.Tensor((K, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), is_cpu=True) as (bx, by):
            A_shared = T.alloc_shared((bM, bK), "float32")
            B_shared = T.alloc_shared((bK, bN), "float32")
            A_tmp    = T.alloc_shared((bM, bK), "float16")
            B_tmp    = T.alloc_shared((bK, bN), "float16")
            C_shared = T.alloc_shared((bM, bN), "float32")
            T.ppl_fill(C_shared, T.float32(0))
            for k in T.Pipelined(T.ceildiv(K, bK), num_stages=1):
                T.ppl_copy(A[by * bM, k * bK], A_shared)
                T.ppl_copy(B[k * bK, bx * bN], B_shared)
                T.ppl_copy(A_shared, A_tmp)
                T.ppl_copy(B_shared, B_tmp)
                T.ppl_gemm(A_tmp, B_tmp, C_shared)
            T.ppl_copy(C_shared, C[by * bM, bx * bN])
    return main_kernel_inner


def _build_matmul_lowp(M, N, K, bM, bN, bK, dtype):
    @T.prim_func
    def main_kernel_inner(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), is_cpu=True) as (bx, by):
            A_shared     = T.alloc_shared((bM, bK), dtype)
            B_shared     = T.alloc_shared((bK, bN), dtype)
            C_shared     = T.alloc_shared((bM, bN), "float32")
            C_shared_ori = T.alloc_shared((bM, bN), dtype)
            T.ppl_fill(C_shared, T.float32(0))
            for k in T.Pipelined(T.ceildiv(K, bK), num_stages=1):
                T.ppl_copy(A[by * bM, k * bK], A_shared)
                T.ppl_copy(B[k * bK, bx * bN], B_shared)
                T.ppl_gemm(A_shared, B_shared, C_shared)
            T.ppl_copy(C_shared, C_shared_ori)
            T.ppl_copy(C_shared_ori, C[by * bM, bx * bN])
    return main_kernel_inner


def autotune_matmul(M, N, K, dtype="float32", atol=1e-1, rtol=1e-1, verbose=True):
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    a = torch.randn(M, K, dtype=torch_dtype)
    b = torch.randn(K, N, dtype=torch_dtype)
    if dtype == "float32":
        ref = torch.matmul(a, b)
    else:
        ref = torch.matmul(a.float(), b.float()).to(torch_dtype)

    candidates = _matmul_candidates(M, N, K, dtype)
    if verbose:
        print(f"  AutoTune Matmul {dtype} {M}x{N}x{K}: {len(candidates)} candidates")

    results = []
    for (bM, bN, bK) in candidates:
        try:
            if dtype == "float32":
                prog = _build_matmul_fp32(M, N, K, bM, bN, bK)
            else:
                prog = _build_matmul_lowp(M, N, K, bM, bN, bK, dtype)
            kernel = tilelang.compile(prog, out_idx=-1, target="tpu")
            c = torch.zeros(M, N, dtype=torch_dtype)
            kernel(a, b, c)
            if not torch.allclose(c.float(), ref.float(), atol=atol, rtol=rtol):
                if verbose:
                    print(f"    bM={bM} bN={bN} bK={bK}  WRONG")
                continue
            ms = _time_kernel(kernel, a, b, c)
            results.append(({"bM": bM, "bN": bN, "bK": bK}, ms))
            if verbose:
                print(f"    bM={bM} bN={bN} bK={bK}  {ms:.3f} ms")
        except Exception as e:
            if verbose:
                print(f"    bM={bM} bN={bN} bK={bK}  ERROR: {e}")

    results.sort(key=lambda x: x[1])
    best_cfg, best_ms = results[0]
    print(f"  >>> Best: bM={best_cfg['bM']} bN={best_cfg['bN']} bK={best_cfg['bK']}  {best_ms:.3f} ms")
    return best_cfg, best_ms, results


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm (single-tile)
# ─────────────────────────────────────────────────────────────────────────────

def _rmsnorm_sram(bM, N, dtype):
    db = _DTYPE_BYTES[dtype]
    if dtype == "float32":
        # A_shared[bM,N]*4 + A_pow2[bM,N]*4 + A_powsum[bM,1]*4
        return bM * N * 4 * 2 + bM * 4
    else:
        # A_shared[bM,N]*db + A_fp32[bM,N]*4 + A_pow2[bM,N]*4 + A_powsum[bM,1]*4
        return bM * N * db + bM * N * 4 * 2 + bM * 4


def _rmsnorm_candidates(M, N, dtype):
    POW2 = [4, 8, 16, 32, 64]
    out = []
    for bM in POW2:
        if bM > M: break
        if M % bM != 0: continue
        if _rmsnorm_sram(bM, N, dtype) < _SRAM_LIMIT:
            out.append(bM)
    return out


def _build_rmsnorm_fp32(M, N, bM):
    @T.prim_func
    def main_kernel_inner(
        A: T.Tensor((M, N), "float32"),
        B: T.Tensor((M, N), "float32"),
    ):
        reciprocal_N = T.float32(1.0 / N)
        with T.Kernel(T.ceildiv(M, bM), is_cpu=True) as (bx,):
            A_shared  = T.alloc_shared((bM, N), "float32")
            A_pow2    = T.alloc_shared((bM, N), "float32")
            A_powsum  = T.alloc_shared((bM, 1), "float32")
            T.ppl_copy(A[bx * bM, 0], A_shared)
            T.ppl_mul(A_pow2, A_shared, A_shared)
            T.ppl_reduce_sum(A_pow2, A_powsum, dim=1)
            T.ppl_mul_C(A_powsum, A_powsum, reciprocal_N)
            T.ppl_add_C(A_powsum, A_powsum, T.float32(1e-12))
            T.ppl_rsqrt(A_powsum, A_powsum)
            T.ppl_mul(A_shared, A_shared, A_powsum)
            T.ppl_copy(A_shared, B[bx * bM, 0])
    return main_kernel_inner


def _build_rmsnorm_lowp(M, N, bM, dtype):
    @T.prim_func
    def main_kernel_inner(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        reciprocal_N = T.float32(1.0 / N)
        with T.Kernel(T.ceildiv(M, bM), is_cpu=True) as (bx,):
            A_shared     = T.alloc_shared((bM, N), dtype)
            A_shared_fp32= T.alloc_shared((bM, N), "float32")
            A_pow2       = T.alloc_shared((bM, N), "float32")
            A_powsum     = T.alloc_shared((bM, 1), "float32")
            T.ppl_copy(A[bx * bM, 0], A_shared)
            T.ppl_copy(A_shared, A_shared_fp32)
            T.ppl_mul(A_pow2, A_shared_fp32, A_shared_fp32)
            T.ppl_reduce_sum(A_pow2, A_powsum, dim=1)
            T.ppl_mul_C(A_powsum, A_powsum, reciprocal_N)
            T.ppl_add_C(A_powsum, A_powsum, T.float32(1e-12))
            T.ppl_rsqrt(A_powsum, A_powsum)
            T.ppl_mul(A_shared_fp32, A_shared_fp32, A_powsum)
            T.ppl_copy(A_shared_fp32, A_shared)
            T.ppl_copy(A_shared, B[bx * bM, 0])
    return main_kernel_inner


def autotune_rmsnorm(M, N, dtype="float32", atol=5e-2, rtol=5e-2, verbose=True):
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    a = torch.randn(M, N, dtype=torch_dtype)
    if dtype == "float32":
        ref = a / torch.sqrt(torch.mean(a * a, dim=1, keepdim=True) + 1e-12)
    else:
        af = a.float()
        ref = (af / torch.sqrt(torch.mean(af * af, dim=1, keepdim=True) + 1e-12)).to(torch_dtype)

    candidates = _rmsnorm_candidates(M, N, dtype)
    if verbose:
        print(f"  AutoTune RMSNorm {dtype} {M}x{N}: blk_m candidates = {candidates}")

    results = []
    for bM in candidates:
        try:
            if dtype == "float32":
                prog = _build_rmsnorm_fp32(M, N, bM)
            else:
                prog = _build_rmsnorm_lowp(M, N, bM, dtype)
            kernel = tilelang.compile(prog, out_idx=-1, target="tpu")
            b = torch.zeros(M, N, dtype=torch_dtype)
            kernel(a, b)
            if not torch.allclose(b.float(), ref.float(), atol=atol, rtol=rtol):
                if verbose:
                    print(f"    blk_m={bM}  WRONG")
                continue
            ms = _time_kernel(kernel, a, b)
            results.append(({"blk_m": bM}, ms))
            if verbose:
                print(f"    blk_m={bM}  {ms:.3f} ms")
        except Exception as e:
            if verbose:
                print(f"    blk_m={bM}  ERROR: {e}")

    results.sort(key=lambda x: x[1])
    best_cfg, best_ms = results[0]
    print(f"  >>> Best: blk_m={best_cfg['blk_m']}  {best_ms:.3f} ms")
    return best_cfg, best_ms, results


# ─────────────────────────────────────────────────────────────────────────────
# FlashAttention
# ─────────────────────────────────────────────────────────────────────────────

def _fa_sram(bM, bN, D, dtype):
    db = _DTYPE_BYTES[dtype]
    # Q_shared[bM,D] K_shared[bN,D] V_shared[bN,D] O_shared[bM,D] = 4*bM*D*db  (lowp)
    # acc_s[bM,bN]*4  acc_s_cast[bM,bN]*db  acc_o[bM,D]*4
    # scores_max/prev/scale/sum/logsum * [bM,1]*4 = 5*bM*4
    # work0/work1 [bM,bN]*4  coeff[64,32]*4  table[64,192]*4
    qkvo  = 4 * bM * D * db
    accs  = bM * bN * 4 + bM * bN * db
    acco  = bM * D * 4
    state = 5 * bM * 1 * 4
    work  = 2 * bM * bN * 4
    lut   = (64*32 + 64*192) * 4
    # fp32: Q/K/V/O in fp32, plus Q_gemm/K_gemm/V_gemm in fp16
    if dtype == "float32":
        qkvo_fp32 = 4 * bM * D * 4
        gemm_bufs = 3 * bM * D * 2  # Q_gemm K_gemm V_gemm fp16 (K/V are bN not bM for K/V)
        return qkvo_fp32 + bN*D*4*2 + bN*D*2*2 + accs + acco + state + work + lut
    return qkvo + accs + acco + state + work + lut


def _fa_candidates(seq_len, D, dtype):
    """Candidate (bM, bN) pairs that fit in SRAM and divide seq_len."""
    POW2 = [2, 4, 8, 16, 32, 64]
    out = []
    for bM in POW2:
        if bM > seq_len: break
        for bN in POW2:
            if bN > seq_len: break
            if _fa_sram(bM, bN, D, dtype) < _SRAM_LIMIT:
                out.append((bM, bN))
    # Keep only interesting tile sizes (not too tiny)
    min_area = max(4, min(seq_len, 8) * min(seq_len, 8))
    out = [(bM, bN) for bM, bN in out if bM * bN >= min_area]
    # Sort by descending tile area
    out.sort(key=lambda x: -(x[0] * x[1]))
    # Deduplicate by area, keep top 8
    seen = set()
    filtered = []
    for c in out:
        a = c[0] * c[1]
        if a not in seen:
            seen.add(a)
            filtered.append(c)
        if len(filtered) >= 8:
            break
    return filtered


def _build_fa_lowp(batch, heads, seq_len, dim, bM, bN, dtype):
    """Build FA kernel with all ops inlined (no @T.macro) to avoid global name collisions."""
    scale = (1.0 / dim) ** 0.5
    shape = [batch, seq_len, heads, dim]
    accum = "float"

    @T.prim_func
    def main_kernel_inner(
        Q: T.Tensor(shape, dtype), K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype), Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, bM), heads, batch, is_cpu=True) as (bx, by, bz):
            Q_shared        = T.alloc_shared([bM, dim], dtype)
            K_shared        = T.alloc_shared([bN, dim], dtype)
            V_shared        = T.alloc_shared([bN, dim], dtype)
            O_shared        = T.alloc_shared([bM, dim], dtype)
            acc_s           = T.alloc_shared([bM, bN], accum)
            acc_s_cast      = T.alloc_shared([bM, bN], dtype)
            acc_o           = T.alloc_shared([bM, dim], accum)
            scores_max      = T.alloc_shared([bM, 1], accum)
            scores_max_prev = T.alloc_shared([bM, 1], accum)
            scores_scale    = T.alloc_shared([bM, 1], accum)
            scores_sum      = T.alloc_shared([bM, 1], accum)
            logsum          = T.alloc_shared([bM, 1], accum)
            work0           = T.alloc_shared([bM, bN], accum)
            work1           = T.alloc_shared([bM, bN], accum)
            coeff           = T.alloc_shared([64, 32], accum)
            table           = T.alloc_shared([64, 192], accum)

            T.copy(Q[bz, bx * bM:(bx + 1) * bM, by, :], Q_shared)
            T.ppl_mul_C(Q_shared, Q_shared, T.float32(scale))
            T.ppl_fill(acc_o, T.float32(0))
            T.ppl_fill(logsum, T.float32(0))
            T.ppl_fill(scores_max, -T.infinity(accum))
            T.ppl_exp_load_coeff(coeff, table)

            for k in T.Pipelined(T.ceildiv(seq_len, bN), num_stages=1):
                # MMA0: load K tile, compute Q@K^T
                T.copy(K[bz, k * bN:(k + 1) * bN, by, :], K_shared)
                T.ppl_fill(acc_s, T.float32(0))
                T.ppl_gemm(Q_shared, K_shared, acc_s, transpose_B=True)
                # Softmax: online softmax update
                T.copy(scores_max, scores_max_prev)
                T.ppl_fill(scores_max, -T.infinity(accum))
                T.ppl_reduce_max(acc_s, scores_max, dim=1, clear=False)
                T.ppl_subtract(scores_scale, scores_max_prev, scores_max)
                T.ppl_exp_compute(scores_scale, work0, work1, coeff, table)
                T.ppl_subtract(acc_s, acc_s, scores_max)
                T.ppl_exp_compute(acc_s, work0, work1, coeff, table)
                T.ppl_reduce_sum(acc_s, scores_sum, dim=1)
                T.ppl_mul(logsum, logsum, scores_scale)
                T.ppl_add(logsum, logsum, scores_sum)
                T.copy(acc_s, acc_s_cast)
                # Rescale: scale acc_o by exp(max_prev - max_new)
                T.ppl_mul(acc_o, acc_o, scores_scale)
                # MMA1: load V tile, accumulate P@V
                T.copy(V[bz, k * bN:(k + 1) * bN, by, :], V_shared)
                T.ppl_gemm(acc_s_cast, V_shared, acc_o)

            T.ppl_div(acc_o, acc_o, logsum)
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * bM:(bx + 1) * bM, by, :])
    return main_kernel_inner


def autotune_flashattn(batch, heads, seq_len, dim, dtype="bfloat16",
                       atol=5e-2, rtol=5e-2, verbose=True):
    import torch.nn.functional as F
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    shape = (batch, seq_len, heads, dim)
    q = torch.randn(*shape, dtype=torch_dtype)
    k = torch.randn(*shape, dtype=torch_dtype)
    v = torch.randn(*shape, dtype=torch_dtype)
    scores = torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / math.sqrt(dim)
    ref = torch.einsum('bhqk,bkhd->bqhd', F.softmax(scores, dim=-1), v.float()).to(torch_dtype)

    candidates = _fa_candidates(seq_len, dim, dtype)
    if verbose:
        print(f"  AutoTune FlashAttn {dtype} B{batch}H{heads}S{seq_len}D{dim}: {len(candidates)} candidates")

    results = []
    for (bM, bN) in candidates:
        try:
            prog   = _build_fa_lowp(batch, heads, seq_len, dim, bM, bN, dtype)
            kernel = tilelang.compile(prog, out_idx=-1, target="tpu")
            out = torch.zeros(*shape, dtype=torch_dtype)
            kernel(q, k, v, out)
            if not torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol):
                if verbose:
                    print(f"    bM={bM} bN={bN}  WRONG")
                continue
            ms = _time_kernel(kernel, q, k, v, out)
            results.append(({"bM": bM, "bN": bN}, ms))
            if verbose:
                print(f"    bM={bM} bN={bN}  {ms:.3f} ms")
        except Exception as e:
            if verbose:
                print(f"    bM={bM} bN={bN}  ERROR: {e}")

    if not results:
        raise RuntimeError("No valid configs found")
    results.sort(key=lambda x: x[1])
    best_cfg, best_ms = results[0]
    print(f"  >>> Best: bM={best_cfg['bM']} bN={best_cfg['bN']}  {best_ms:.3f} ms")
    return best_cfg, best_ms, results
