"""Clean bf16+fp8 non-paged latent attention front-end.

This file is the submit-oriented version of the V49 decode path. It keeps the
front-end self-contained so users do not need the historical fp16/Vxx experiment
files at runtime.

* decode: V49 HEAD_BLOCK=2 multicore fp8 kernel;
* prefill: stable row-tiled fused-dequant fp8 kernel;
* interface: Sophgo non-paged latent_attention_fp8 argument order.
"""

import math
from dataclasses import dataclass

import tilelang
import tilelang.language as T
import torch


@dataclass(frozen=True)
class MLADims:
    dq: int
    dnope: int
    dpe: int
    dkv: int
    dv: int
    seq: int
    heads: int
    softmax_scale: float

    @property
    def dqu(self):
        return self.dnope + self.dpe

    @property
    def dkvu(self):
        return self.dnope + self.dv


FIXED_DIMS = MLADims(
    dq=1536,
    dnope=128,
    dpe=64,
    dkv=512,
    dv=128,
    seq=192,
    heads=16,
    softmax_scale=0.07216878364870322,
)

SMOKE_DIMS = MLADims(
    dq=64,
    dnope=16,
    dpe=32,
    dkv=32,
    dv=16,
    seq=32,
    heads=1,
    softmax_scale=1.0 / math.sqrt(32),
)

NORMAL_ATTENTION_PREFILL = 0
NORMAL_ATTENTION_DECODE = 1
BF16_FP8_QBLOCK = 128
FP8_DTYPE = "e4m3_float8"
BF16_DTYPE = "bfloat16"
MASK_VALUE = -10000.0
FP8_E4M3_MAX = 448.0


def fresh(array):
    return array.clone().contiguous()


def _normalize_weights(weight, heads, rows, cols):
    if weight.dim() == 4 and weight.shape[0] == 1 and weight.shape[2] == 1:
        weight = weight[0, :, 0, :]
    if weight.dim() == 2:
        return weight.reshape(heads, rows, cols).contiguous()
    if weight.dim() == 3:
        if tuple(weight.shape) != (heads, rows, cols):
            raise ValueError(f"weight shape mismatch: got {tuple(weight.shape)}, expected {(heads, rows, cols)}")
        return weight.contiguous()
    raise ValueError(f"unsupported weight shape {tuple(weight.shape)}")


def _make_decode_mask(length, seq, dtype=torch.float32):
    mask = torch.zeros((1, seq), dtype=dtype)
    if length < seq:
        mask[:, int(length):] = MASK_VALUE
    return mask.contiguous()


def _check_finite_range(name, value, low, high):
    if value < low or value > high:
        raise ValueError(f"{name}={value} out of range [{low}, {high}]")
    return value


def _model_matrix(name, tensor):
    if tensor.dim() == 1:
        return tensor.unsqueeze(0).contiguous()
    if tensor.dim() == 3 and tensor.shape[1] == 1:
        return tensor[:, 0, :].contiguous()
    if tensor.dim() == 2:
        return tensor.contiguous()
    raise ValueError(f"{name} expects 1D, 2D, or (rows, 1, dim); got {tuple(tensor.shape)}")


def rope_interleaved(x, cos, sin):
    even = x[..., 0::2]
    odd = x[..., 1::2]
    c = cos[..., 0::2]
    s = sin[..., 0::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * c - odd * s
    out[..., 1::2] = odd * c + even * s
    return out.contiguous()


def expand_scale_for_head(scale, head, rows, cols, quant_block_size=BF16_FP8_QBLOCK):
    row_base = int(head) * int(rows)
    expanded = torch.empty((rows, cols), dtype=torch.bfloat16)
    for r in range(rows):
        sr = (row_base + r) // quant_block_size
        for c0 in range(0, cols, quant_block_size):
            c1 = min(c0 + quant_block_size, cols)
            sc = c0 // quant_block_size
            expanded[r, c0:c1] = scale[sr, sc]
    return expanded.contiguous()

def mla_prefill_bf16_fp8_fused_dequant_cache(
    B,
    S,
    DQ,
    DNOPE,
    DPE,
    DKV,
    DV,
    softmax_scale,
    dtype=BF16_DTYPE,
    accum_dtype="float32",
):
    DKVU = DNOPE + DV
    DQ_CHUNK = min(32, DQ)
    SCORE_CHUNK = min(32, S)
    DKV_CHUNK = min(32, DKV)
    DV_CHUNK = min(32, DV)
    SCORE_WITH_PREV = ((SCORE_CHUNK + 1 + 31) // 32) * 32
    assert DQ % DQ_CHUNK == 0
    assert S % SCORE_CHUNK == 0
    assert DKV % DKV_CHUNK == 0
    assert DV % DV_CHUNK == 0

    @T.prim_func
    def main_kernel_inner(
        Q: T.Tensor((B, DQ), dtype),
        KV: T.Tensor((S, DKV), dtype),
        PE_RAW: T.Tensor((S, DPE), dtype),
        WUQ: T.Tensor((DNOPE + DPE, DQ), FP8_DTYPE),
        WUKV: T.Tensor((DKVU, DKV), FP8_DTYPE),
        WUQ_SCALE: T.Tensor((DNOPE + DPE, DQ), dtype),
        WUKV_SCALE: T.Tensor((DKVU, DKV), dtype),
        KVcache_IN: T.Tensor((S, DKV), dtype),
        PEcache_IN: T.Tensor((S, DPE), dtype),
        COS_Q: T.Tensor((B, DPE), dtype),
        SIN_Q: T.Tensor((B, DPE), dtype),
        COS_ALL: T.Tensor((S, DPE), dtype),
        SIN_ALL: T.Tensor((S, DPE), dtype),
        MASK: T.Tensor((B, S), accum_dtype),
        OUT: T.Tensor((B, DV), dtype),
        KVcache_OUT: T.Tensor((S, DKV), dtype),
        PEcache_OUT: T.Tensor((S, DPE), dtype),
    ):
        with T.Kernel(1, is_cpu=True):
            q_tile = T.alloc_shared((B, DQ_CHUNK), dtype)
            wuq_nope_fp8 = T.alloc_shared((DNOPE, DQ_CHUNK), FP8_DTYPE)
            wuq_nope_scale = T.alloc_shared((DNOPE, DQ_CHUNK), dtype)
            wuq_nope_tile = T.alloc_shared((DNOPE, DQ_CHUNK), dtype)
            wuq_rope_fp8 = T.alloc_shared((DPE, DQ_CHUNK), FP8_DTYPE)
            wuq_rope_scale = T.alloc_shared((DPE, DQ_CHUNK), dtype)
            wuq_rope_tile = T.alloc_shared((DPE, DQ_CHUNK), dtype)
            q_nope_acc = T.alloc_shared((B, DNOPE), accum_dtype)
            q_rope_src_acc = T.alloc_shared((B, DPE), accum_dtype)
            q_nope_part = T.alloc_shared((B, DNOPE), accum_dtype)
            q_rope_part = T.alloc_shared((B, DPE), accum_dtype)
            q_nope = T.alloc_shared((B, DNOPE), dtype)
            q_rope_src = T.alloc_shared((B, DPE), dtype)
            q_rope = T.alloc_shared((B, DPE), dtype)
            cos_q_shared = T.alloc_shared((B, DPE), dtype)
            sin_q_shared = T.alloc_shared((B, DPE), dtype)
            q_rope_cos = T.alloc_shared((B, DPE), dtype)
            q_rope_sin = T.alloc_shared((B, DPE), dtype)
            q_rope_neg = T.alloc_shared((B, DPE), dtype)
            q_rope_neg_sin = T.alloc_shared((B, DPE), dtype)

            cache_kv_tile = T.alloc_shared((SCORE_CHUNK, DKV_CHUNK), dtype)
            pe_raw_tile = T.alloc_shared((SCORE_CHUNK, DPE), dtype)
            cos_all_tile = T.alloc_shared((SCORE_CHUNK, DPE), dtype)
            sin_all_tile = T.alloc_shared((SCORE_CHUNK, DPE), dtype)
            pe_rope_cos = T.alloc_shared((SCORE_CHUNK, DPE), dtype)
            pe_rope_sin = T.alloc_shared((SCORE_CHUNK, DPE), dtype)
            pe_rope_neg = T.alloc_shared((SCORE_CHUNK, DPE), dtype)
            pe_rope_neg_sin = T.alloc_shared((SCORE_CHUNK, DPE), dtype)
            pe_rope_tile = T.alloc_shared((SCORE_CHUNK, DPE), dtype)

            wukv_k_fp8 = T.alloc_shared((DNOPE, DKV_CHUNK), FP8_DTYPE)
            wukv_k_scale = T.alloc_shared((DNOPE, DKV_CHUNK), dtype)
            wukv_k_tile = T.alloc_shared((DNOPE, DKV_CHUNK), dtype)
            q_wukv_part = T.alloc_shared((B, DKV_CHUNK), accum_dtype)
            q_wukv_cast = T.alloc_shared((B, DKV_CHUNK), dtype)
            kv_k_tile = T.alloc_shared((SCORE_CHUNK, DKV_CHUNK), dtype)
            kv_value_tile = T.alloc_shared((SCORE_CHUNK, DKV), dtype)
            pe_score_tile = T.alloc_shared((SCORE_CHUNK, DPE), dtype)
            score_latent_tile = T.alloc_shared((B, SCORE_CHUNK), accum_dtype)
            score_latent_part = T.alloc_shared((B, SCORE_CHUNK), accum_dtype)
            score_rope_tile = T.alloc_shared((B, SCORE_CHUNK), accum_dtype)
            score_tile = T.alloc_shared((B, SCORE_CHUNK), accum_dtype)
            score_tile_with_prev = T.alloc_shared((B, SCORE_WITH_PREV), accum_dtype)
            mask_tile = T.alloc_shared((B, SCORE_CHUNK), accum_dtype)
            prob_tile = T.alloc_shared((B, SCORE_CHUNK), dtype)

            ctx_acc = T.alloc_shared((B, DKV), accum_dtype)
            ctx_cast = T.alloc_shared((B, DKV), dtype)
            scores_max = T.alloc_shared((B, 1), accum_dtype)
            scores_max_prev = T.alloc_shared((B, 1), accum_dtype)
            scores_scale = T.alloc_shared((B, 1), accum_dtype)
            scores_sum = T.alloc_shared((B, 1), accum_dtype)
            logsum = T.alloc_shared((B, 1), accum_dtype)
            prev_scaled = T.alloc_shared((B, 1), accum_dtype)
            curr_scaled = T.alloc_shared((B, 1), accum_dtype)
            work_scale0 = T.alloc_shared((B, 1), accum_dtype)
            work_scale1 = T.alloc_shared((B, 1), accum_dtype)
            work_score0 = T.alloc_shared((B, SCORE_CHUNK), accum_dtype)
            work_score1 = T.alloc_shared((B, SCORE_CHUNK), accum_dtype)
            coeff = T.alloc_shared((64, 32), accum_dtype)
            table = T.alloc_shared((64, 192), accum_dtype)

            ctx_chunk = T.alloc_shared((B, DKV_CHUNK), dtype)
            wukv_v_fp8 = T.alloc_shared((DV_CHUNK, DKV_CHUNK), FP8_DTYPE)
            wukv_v_scale = T.alloc_shared((DV_CHUNK, DKV_CHUNK), dtype)
            wukv_v_tile = T.alloc_shared((DV_CHUNK, DKV_CHUNK), dtype)
            out_part = T.alloc_shared((B, DV_CHUNK), accum_dtype)
            out_sub = T.alloc_shared((B, DV_CHUNK), accum_dtype)
            out_cast = T.alloc_shared((B, DV_CHUNK), dtype)

            for n in T.Pipelined(T.ceildiv(S, SCORE_CHUNK), num_stages=0):
                for d in T.Pipelined(T.ceildiv(DKV, DKV_CHUNK), num_stages=0):
                    T.ppl_copy(KV[n * SCORE_CHUNK, d * DKV_CHUNK], cache_kv_tile)
                    T.ppl_copy(cache_kv_tile, KVcache_OUT[n * SCORE_CHUNK, d * DKV_CHUNK])

            T.ppl_copy(COS_Q, cos_q_shared)
            T.ppl_copy(SIN_Q, sin_q_shared)
            T.ppl_fill(q_nope_acc, T.float32(0))
            T.ppl_fill(q_rope_src_acc, T.float32(0))
            for k in T.Pipelined(T.ceildiv(DQ, DQ_CHUNK), num_stages=0):
                T.ppl_copy(Q[0, k * DQ_CHUNK], q_tile)
                T.ppl_copy(WUQ[0, k * DQ_CHUNK], wuq_nope_fp8)
                T.ppl_copy(wuq_nope_fp8, wuq_nope_tile)
                T.ppl_copy(WUQ_SCALE[0, k * DQ_CHUNK], wuq_nope_scale)
                T.ppl_mul(wuq_nope_tile, wuq_nope_tile, wuq_nope_scale)
                T.ppl_fill(q_nope_part, T.float32(0))
                T.ppl_gemm(q_tile, wuq_nope_tile, q_nope_part, transpose_B=True)
                T.ppl_add(q_nope_acc, q_nope_acc, q_nope_part)

                T.ppl_copy(WUQ[DNOPE, k * DQ_CHUNK], wuq_rope_fp8)
                T.ppl_copy(wuq_rope_fp8, wuq_rope_tile)
                T.ppl_copy(WUQ_SCALE[DNOPE, k * DQ_CHUNK], wuq_rope_scale)
                T.ppl_mul(wuq_rope_tile, wuq_rope_tile, wuq_rope_scale)
                T.ppl_fill(q_rope_part, T.float32(0))
                T.ppl_gemm(q_tile, wuq_rope_tile, q_rope_part, transpose_B=True)
                T.ppl_add(q_rope_src_acc, q_rope_src_acc, q_rope_part)

            T.ppl_copy(q_nope_acc, q_nope)
            T.ppl_copy(q_rope_src_acc, q_rope_src)
            T.ppl_mul(q_rope_cos, q_rope_src, cos_q_shared)
            T.ppl_mul(q_rope_sin, q_rope_src, sin_q_shared)
            T.ppl_mul_C(q_rope_neg, q_rope_src, T.float16(-1.0))
            T.ppl_mul(q_rope_neg_sin, q_rope_neg, sin_q_shared)
            T.ppl_rope_add(q_rope, q_rope_cos, q_rope_neg_sin, q_rope_cos, q_rope_sin)

            T.ppl_fill(ctx_acc, T.float32(0))
            T.ppl_fill(logsum, T.float32(0))
            T.ppl_fill(scores_max, -T.infinity(accum_dtype))
            for n in T.Pipelined(T.ceildiv(S, SCORE_CHUNK), num_stages=0):
                T.ppl_copy(PE_RAW[n * SCORE_CHUNK, 0], pe_raw_tile)
                T.ppl_copy(COS_ALL[n * SCORE_CHUNK, 0], cos_all_tile)
                T.ppl_copy(SIN_ALL[n * SCORE_CHUNK, 0], sin_all_tile)
                T.ppl_mul(pe_rope_cos, pe_raw_tile, cos_all_tile)
                T.ppl_mul(pe_rope_sin, pe_raw_tile, sin_all_tile)
                T.ppl_mul_C(pe_rope_neg, pe_raw_tile, T.float16(-1.0))
                T.ppl_mul(pe_rope_neg_sin, pe_rope_neg, sin_all_tile)
                T.ppl_rope_add(pe_rope_tile, pe_rope_cos, pe_rope_neg_sin, pe_rope_cos, pe_rope_sin)
                T.ppl_copy(pe_rope_tile, PEcache_OUT[n * SCORE_CHUNK, 0])
                T.ppl_fill(score_latent_tile, T.float32(0))
                for d in T.Pipelined(T.ceildiv(DKV, DKV_CHUNK), num_stages=0):
                    T.ppl_copy(WUKV[0, d * DKV_CHUNK], wukv_k_fp8)
                    T.ppl_copy(wukv_k_fp8, wukv_k_tile)
                    T.ppl_copy(WUKV_SCALE[0, d * DKV_CHUNK], wukv_k_scale)
                    T.ppl_mul(wukv_k_tile, wukv_k_tile, wukv_k_scale)
                    T.ppl_fill(q_wukv_part, T.float32(0))
                    T.ppl_gemm(q_nope, wukv_k_tile, q_wukv_part)
                    T.ppl_copy(q_wukv_part, q_wukv_cast)
                    T.ppl_copy(KVcache_OUT[n * SCORE_CHUNK, d * DKV_CHUNK], kv_k_tile)
                    T.ppl_fill(score_latent_part, T.float32(0))
                    T.ppl_gemm(q_wukv_cast, kv_k_tile, score_latent_part, transpose_B=True)
                    T.ppl_add(score_latent_tile, score_latent_tile, score_latent_part)

                T.ppl_copy(pe_rope_tile, pe_score_tile)
                T.ppl_fill(score_rope_tile, T.float32(0))
                T.ppl_gemm(q_rope, pe_score_tile, score_rope_tile, transpose_B=True)
                T.ppl_add(score_tile, score_latent_tile, score_rope_tile)
                T.ppl_copy(MASK[0, n * SCORE_CHUNK], mask_tile)
                T.ppl_add(score_tile, score_tile, mask_tile)

                T.ppl_copy(scores_max, scores_max_prev)
                T.ppl_fill(score_tile_with_prev, -T.infinity(accum_dtype))
                T.ppl_copy_region(scores_max_prev[0:B, 0:1], score_tile_with_prev[0:B, 0:1], [B, 1])
                T.ppl_copy_region(
                    score_tile[0:B, 0:SCORE_CHUNK],
                    score_tile_with_prev[0:B, 1 : SCORE_CHUNK + 1],
                    [B, SCORE_CHUNK],
                )
                T.ppl_reduce_max(score_tile_with_prev, scores_max, dim=1)
                T.ppl_mul_C(prev_scaled, scores_max_prev, T.float32(softmax_scale))
                T.ppl_mul_C(curr_scaled, scores_max, T.float32(softmax_scale))
                T.ppl_subtract(scores_scale, prev_scaled, curr_scaled)
                T.ppl_exp2(scores_scale, work_scale0, work_scale1, coeff, table)
                T.ppl_subtract(score_tile, score_tile, scores_max)
                T.ppl_mul_C(score_tile, score_tile, T.float32(softmax_scale))
                T.ppl_exp2(score_tile, work_score0, work_score1, coeff, table)
                T.ppl_reduce_sum(score_tile, scores_sum, dim=1)
                T.ppl_mul(logsum, logsum, scores_scale)
                T.ppl_add(logsum, logsum, scores_sum)
                T.ppl_mul(ctx_acc, ctx_acc, scores_scale)
                T.ppl_copy(score_tile, prob_tile)
                T.ppl_copy(KVcache_OUT[n * SCORE_CHUNK, 0], kv_value_tile)
                T.ppl_gemm(prob_tile, kv_value_tile, ctx_acc)

            T.ppl_div(ctx_acc, ctx_acc, logsum)
            T.ppl_copy(ctx_acc, ctx_cast)
            for n in T.Pipelined(T.ceildiv(DV, DV_CHUNK), num_stages=0):
                T.ppl_fill(out_part, T.float32(0))
                for d in T.Pipelined(T.ceildiv(DKV, DKV_CHUNK), num_stages=0):
                    T.ppl_copy(ctx_cast[0, d * DKV_CHUNK], ctx_chunk)
                    T.ppl_copy(WUKV[DNOPE + n * DV_CHUNK, d * DKV_CHUNK], wukv_v_fp8)
                    T.ppl_copy(wukv_v_fp8, wukv_v_tile)
                    T.ppl_copy(WUKV_SCALE[DNOPE + n * DV_CHUNK, d * DKV_CHUNK], wukv_v_scale)
                    T.ppl_mul(wukv_v_tile, wukv_v_tile, wukv_v_scale)
                    T.ppl_fill(out_sub, T.float32(0))
                    T.ppl_gemm(ctx_chunk, wukv_v_tile, out_sub, transpose_B=True)
                    T.ppl_add(out_part, out_part, out_sub)
                T.ppl_copy(out_part, out_cast)
                T.ppl_copy(out_cast, OUT[0, n * DV_CHUNK])

    return main_kernel_inner

def _ppl_exp_inplace(out, work0, work1, coeff, table):
    """V28-local exp lowering; unlike the shared helper, preserve the extern call."""
    return T.call_extern(
        "handle",
        "ppl.exp",
        out.access_ptr("rw"),
        work0.access_ptr("rw"),
        work1.access_ptr("rw"),
        coeff.access_ptr("rw"),
        table.access_ptr("rw"),
    )


def mla_decode_bf16_fp8_batch_head_fused_dequant_cache(
    BATCH,
    HEADS,
    S,
    DQ,
    DNOPE,
    DPE,
    DKV,
    DV,
    softmax_scale,
    quant_block_size=BF16_FP8_QBLOCK,
    total_heads=None,
    head_start=0,
    dtype=BF16_DTYPE,
    accum_dtype="float32",
    head_block=2,
):
    DKVU = DNOPE + DV
    B = 1
    QBLOCK = int(quant_block_size)
    HEAD_BLOCK = min(int(head_block), int(HEADS))
    assert HEADS % HEAD_BLOCK == 0, "V34 requires HEADS divisible by HEAD_BLOCK"
    HEAD_GROUPS = HEADS // HEAD_BLOCK
    DQ_CHUNK = min(QBLOCK, DQ)
    # Sophgo decode materializes the complete valid QK row before softmax.
    # Keep the static TileLang shape, but use one full-sequence score tile.
    SCORE_CHUNK = S
    DKV_CHUNK = min(QBLOCK, DKV)
    assert DQ % DQ_CHUNK == 0
    assert S % SCORE_CHUNK == 0
    assert DKV % DKV_CHUNK == 0
    assert DNOPE == 128 and DPE == 64 and DV == 128
    QNOPE_HALF = 64
    TOTAL_HEADS = HEADS if total_heads is None else int(total_heads)
    HEAD_START = int(head_start)
    DQU = DNOPE + DPE
    WUQ_SCALE_ROWS = (TOTAL_HEADS * DQU + QBLOCK - 1) // QBLOCK
    WUQ_SCALE_COLS = (DQ + QBLOCK - 1) // QBLOCK
    WUKV_SCALE_ROWS = (TOTAL_HEADS * DKVU + QBLOCK - 1) // QBLOCK
    WUKV_SCALE_COLS = (DKV + QBLOCK - 1) // QBLOCK
    DEBUG_STAGE = "out"

    @T.prim_func
    def main_kernel_inner(
        Q: T.Tensor((BATCH, DQ), dtype),
        KV_CUR: T.Tensor((BATCH, DKV), dtype),
        PE_CUR: T.Tensor((BATCH, DPE), dtype),
        WUQ: T.Tensor((HEADS * DQU, DQ), FP8_DTYPE),
        WUKV: T.Tensor((HEADS * DKVU, DKV), FP8_DTYPE),
        WUQ_SCALE: T.Tensor((WUQ_SCALE_ROWS, WUQ_SCALE_COLS), dtype),
        WUKV_SCALE: T.Tensor((WUKV_SCALE_ROWS, WUKV_SCALE_COLS), dtype),
        KVcache_IN: T.Tensor((BATCH * S, DKV), dtype),
        PEcache_IN: T.Tensor((BATCH * S, DPE), dtype),
        COS: T.Tensor((BATCH, DPE), dtype),
        SIN: T.Tensor((BATCH, DPE), dtype),
        DECODE_MASK: T.Tensor((BATCH, S), accum_dtype),
        OUT: T.Tensor((BATCH, HEADS * DV), dtype),
        KVcache_OUT: T.Tensor((BATCH * S, DKV), dtype),
        PEcache_OUT: T.Tensor((BATCH * S, DPE), dtype),
        slot: T.int32,
    ):
        with T.Kernel(BATCH, 1, is_cpu=True) as (bb, _):
            core = T.ppl_workitem_index()
            for hg in T.unroll(HEAD_GROUPS):
                if core == hg:
                    # Shared once per head block.
                    q_tile = T.alloc_shared((B, DQ_CHUNK), dtype)
                    wuq_fp8 = T.alloc_shared((HEAD_BLOCK * DQU, DQ_CHUNK), FP8_DTYPE)
                    wuq_scale_scalar = T.alloc_shared((1, 1), dtype)
                    wuq_scale_stage = T.alloc_shared((HEAD_BLOCK * DQU, 1), dtype)
                    wuq_tile = T.alloc_shared((HEAD_BLOCK * DQU, DQ_CHUNK), dtype)
                    q_upper_acc_h0 = T.alloc_shared((B, DQU), accum_dtype)
                    q_upper_h0 = T.alloc_shared((B, DQU), dtype)
                    q_upper_acc_h1 = T.alloc_shared((B, DQU), accum_dtype)
                    q_upper_h1 = T.alloc_shared((B, DQU), dtype)
                    q_nope_head = T.alloc_shared((B, DNOPE), dtype)
                    # Keep each head's RoPE source/work/output in distinct local
                    # buffers. Reusing one buffer set across the unrolled HB2 loop
                    # produced corrupted head-1 values on TPU even though q_nope was
                    # correct.
                    q_rope_src_h0 = T.alloc_shared((B, DPE), dtype)
                    q_rope_h0 = T.alloc_shared((B, DPE), dtype)
                    q_rope_tmp0_h0 = T.alloc_shared((B, DPE), dtype)
                    q_rope_tmp1_h0 = T.alloc_shared((B, DPE), dtype)
                    q_rope_tmp2_h0 = T.alloc_shared((B, DPE), dtype)
                    q_rope_tmp3_h0 = T.alloc_shared((B, DPE), dtype)
                    q_rope_src_h1 = T.alloc_shared((B, DPE), dtype)
                    q_rope_h1 = T.alloc_shared((B, DPE), dtype)
                    q_rope_tmp0_h1 = T.alloc_shared((B, DPE), dtype)
                    q_rope_tmp1_h1 = T.alloc_shared((B, DPE), dtype)
                    q_rope_tmp2_h1 = T.alloc_shared((B, DPE), dtype)
                    q_rope_tmp3_h1 = T.alloc_shared((B, DPE), dtype)
                    cos_shared = T.alloc_shared((B, DPE), dtype)
                    sin_shared = T.alloc_shared((B, DPE), dtype)

                    # Head-independent cache update workspace. Keep the same
                    # interleaved RoPE contract validated by V46, but execute it once
                    # inside the fused decode kernel.
                    kv_cur_tile = T.alloc_shared((B, DKV_CHUNK), dtype)
                    pe_cur = T.alloc_shared((B, DPE), dtype)
                    pe_cur_cos = T.alloc_shared((B, DPE), dtype)
                    pe_cur_sin = T.alloc_shared((B, DPE), dtype)
                    pe_cur_neg = T.alloc_shared((B, DPE), dtype)
                    pe_cur_neg_sin = T.alloc_shared((B, DPE), dtype)
                    pe_cur_rope = T.alloc_shared((B, DPE), dtype)

                    wukv_fp8 = T.alloc_shared((HEAD_BLOCK * DKVU, DKV_CHUNK), FP8_DTYPE)
                    wukv_scale_scalar = T.alloc_shared((1, 1), dtype)
                    wukv_scale_stage = T.alloc_shared((HEAD_BLOCK * DKVU, 1), dtype)
                    wukv_tile = T.alloc_shared((HEAD_BLOCK * DKVU, DKV_CHUNK), dtype)
                    wukv_full_tile = T.alloc_shared((HEAD_BLOCK * DKVU, DKV), dtype)
                    q_wukv_acc = T.alloc_shared((B, DKV), accum_dtype)
                    q_wukv_head = T.alloc_shared((B, DKV), dtype)

                    # Cache, PE cache, and mask are loaded once per head block. The
                    # per-head attention chain reuses these tiles sequentially, which
                    # avoids unsupported local cross-NPU lane subviews.
                    kv_cache_full = T.alloc_shared((SCORE_CHUNK, DKV), dtype)
                    pe_score_tile = T.alloc_shared((SCORE_CHUNK, DPE), dtype)
                    mask_row = T.alloc_shared((B, SCORE_CHUNK), accum_dtype)
                    score_latent_h0 = T.alloc_shared((B, SCORE_CHUNK), accum_dtype)
                    score_rope_h0 = T.alloc_shared((B, SCORE_CHUNK), accum_dtype)
                    score_latent_h1 = T.alloc_shared((B, SCORE_CHUNK), accum_dtype)
                    score_rope_h1 = T.alloc_shared((B, SCORE_CHUNK), accum_dtype)
                    score_tile = T.alloc_shared((B, SCORE_CHUNK), accum_dtype)
                    prob_tile = T.alloc_shared((B, SCORE_CHUNK), dtype)
                    scores_max = T.alloc_shared((B, 1), accum_dtype)
                    scores_sum = T.alloc_shared((B, 1), accum_dtype)
                    work_score0 = T.alloc_shared((B, SCORE_CHUNK), accum_dtype)
                    work_score1 = T.alloc_shared((B, SCORE_CHUNK), accum_dtype)
                    coeff = T.alloc_shared((64, 32), accum_dtype)
                    table = T.alloc_shared((64, 192), accum_dtype)

                    ctx_acc = T.alloc_shared((B, DKV), accum_dtype)
                    ctx_head = T.alloc_shared((B, DKV), dtype)
                    out_part = T.alloc_shared((B, DV), accum_dtype)
                    out_cast = T.alloc_shared((B, DV), dtype)

                    T.ppl_copy(COS[bb, 0], cos_shared)
                    T.ppl_copy(SIN[bb, 0], sin_shared)

                    # Decode cache update is head-independent. Every head block writes
                    # the same slot so the multicore path keeps KV/PE cache semantics
                    # correct without relying on cross-core conditional side effects.
                    for d in T.Pipelined(T.ceildiv(DKV, DKV_CHUNK), num_stages=0):
                        T.ppl_copy(KV_CUR[bb, d * DKV_CHUNK], kv_cur_tile)
                        T.ppl_copy(kv_cur_tile, KVcache_OUT[bb * S + slot, d * DKV_CHUNK])
                    T.ppl_copy(PE_CUR[bb, 0], pe_cur)
                    T.ppl_mul(pe_cur_cos, pe_cur, cos_shared)
                    T.ppl_mul(pe_cur_sin, pe_cur, sin_shared)
                    T.ppl_mul_C(pe_cur_neg, pe_cur, T.float32(-1.0))
                    T.ppl_mul(pe_cur_neg_sin, pe_cur_neg, sin_shared)
                    T.ppl_rope_add(pe_cur_rope, pe_cur_cos, pe_cur_neg_sin, pe_cur_cos, pe_cur_sin)
                    T.ppl_copy(pe_cur_rope, PEcache_OUT[bb * S + slot, 0])
                    T.ppl_sync_all()

                    # One Q load and one wide WUQ GEMM serve every head in this block.
                    T.ppl_fill(q_upper_acc_h0, T.float32(0))
                    T.ppl_fill(q_upper_acc_h1, T.float32(0))
                    for k in T.Pipelined(T.ceildiv(DQ, DQ_CHUNK), num_stages=0):
                        T.ppl_copy(Q[bb, k * DQ_CHUNK], q_tile)
                        T.ppl_copy_region(
                            WUQ[hg * HEAD_BLOCK * DQU, k * DQ_CHUNK],
                            wuq_fp8,
                            [HEAD_BLOCK * DQU, DQ_CHUNK],
                        )
                        for hbi in T.unroll(HEAD_BLOCK):
                            head = hg * HEAD_BLOCK + hbi
                            for qpart in T.unroll(3):
                                row_start = hbi * DQU + qpart * QNOPE_HALF
                                T.ppl_copy(
                                    WUQ_SCALE[
                                        T.truncdiv((HEAD_START + head) * DQU + qpart * QNOPE_HALF, QBLOCK),
                                        (k * DQ_CHUNK) // QBLOCK,
                                    ],
                                    wuq_scale_scalar,
                                )
                                T.ppl_npu_bcast(
                                    wuq_scale_scalar,
                                    wuq_scale_stage[row_start:row_start + QNOPE_HALF, 0:1],
                                )
                        T.ppl_dequant_fp8_block_scale(wuq_fp8, wuq_scale_stage, wuq_tile)
                        for hbi in T.unroll(HEAD_BLOCK):
                            if hbi == 0:
                                T.ppl_gemm_region(
                                    q_tile,
                                    wuq_tile[0:DQU, 0:DQ_CHUNK],
                                    q_upper_acc_h0,
                                    transpose_B=True,
                                    A_extent=[1, DQ_CHUNK],
                                    B_extent=[DQU, DQ_CHUNK],
                                    C_extent=[1, DQU],
                                    accumulate=True,
                                )
                            else:
                                T.ppl_gemm_region(
                                    q_tile,
                                    wuq_tile[DQU:2 * DQU, 0:DQ_CHUNK],
                                    q_upper_acc_h1,
                                    transpose_B=True,
                                    A_extent=[1, DQ_CHUNK],
                                    B_extent=[DQU, DQ_CHUNK],
                                    C_extent=[1, DQU],
                                    accumulate=True,
                                )

                    T.ppl_copy(q_upper_acc_h0, q_upper_h0)
                    T.ppl_copy(q_upper_acc_h1, q_upper_h1)

                    # Load/dequant WUKV once for the head block. The row order matches
                    # Sophgo flat global layout: [h0.key, h0.value, h1.key, h1.value, ...].
                    for d in T.Pipelined(T.ceildiv(DKV, DKV_CHUNK), num_stages=0):
                        T.ppl_copy_region(
                            WUKV[hg * HEAD_BLOCK * DKVU, d * DKV_CHUNK],
                            wukv_fp8,
                            [HEAD_BLOCK * DKVU, DKV_CHUNK],
                        )
                        for hbi in T.unroll(HEAD_BLOCK):
                            head = hg * HEAD_BLOCK + hbi
                            T.ppl_copy(
                                WUKV_SCALE[(HEAD_START + head) * 2, (d * DKV_CHUNK) // QBLOCK],
                                wukv_scale_scalar,
                            )
                            T.ppl_npu_bcast(
                                wukv_scale_scalar,
                                wukv_scale_stage[hbi * DKVU:hbi * DKVU + DNOPE, 0:1],
                            )
                            T.ppl_copy(
                                WUKV_SCALE[(HEAD_START + head) * 2 + 1, (d * DKV_CHUNK) // QBLOCK],
                                wukv_scale_scalar,
                            )
                            T.ppl_npu_bcast(
                                wukv_scale_scalar,
                                wukv_scale_stage[hbi * DKVU + DNOPE:(hbi + 1) * DKVU, 0:1],
                            )
                        T.ppl_dequant_fp8_block_scale(
                            wukv_fp8,
                            wukv_scale_stage,
                            wukv_full_tile[0:HEAD_BLOCK * DKVU, d * DKV_CHUNK:(d + 1) * DKV_CHUNK],
                        )

                    if DEBUG_STAGE == "q_upper_after_wukv":
                        T.ppl_fill(out_cast, T.float32(0))
                        T.ppl_copy_region(q_upper_h0[0:B, 0:DV], out_cast[0:B, 0:DV], [1, DV])
                        T.ppl_copy(out_cast, OUT[bb, hg * HEAD_BLOCK * DV])
                        T.ppl_fill(out_cast, T.float32(0))
                        T.ppl_copy_region(q_upper_h1[0:B, 0:DV], out_cast[0:B, 0:DV], [1, DV])
                        T.ppl_copy(out_cast, OUT[bb, (hg * HEAD_BLOCK + 1) * DV])

                    T.ppl_copy(KVcache_OUT[bb * S, 0], kv_cache_full)
                    T.ppl_copy(PEcache_OUT[bb * S, 0], pe_score_tile)
                    T.ppl_copy(DECODE_MASK[bb, 0], mask_row)
                    T.ppl_fill(score_latent_h0, T.float32(0))
                    T.ppl_fill(score_rope_h0, T.float32(0))
                    T.ppl_fill(score_latent_h1, T.float32(0))
                    T.ppl_fill(score_rope_h1, T.float32(0))

                    if DEBUG_STAGE == "q_upper_after_cache":
                        T.ppl_fill(out_cast, T.float32(0))
                        T.ppl_copy_region(q_upper_h0[0:B, 0:DV], out_cast[0:B, 0:DV], [1, DV])
                        T.ppl_copy(out_cast, OUT[bb, hg * HEAD_BLOCK * DV])
                        T.ppl_fill(out_cast, T.float32(0))
                        T.ppl_copy_region(q_upper_h1[0:B, 0:DV], out_cast[0:B, 0:DV], [1, DV])
                        T.ppl_copy(out_cast, OUT[bb, (hg * HEAD_BLOCK + 1) * DV])

                    # Compute both head RoPE vectors before the sequential per-head
                    # attention chain. This keeps HB2 weight/cache reuse while making
                    # the two RoPE lifetimes explicit to lowering/address assignment.
                    T.ppl_copy_region(q_upper_h0[0:B, DNOPE:DQU], q_rope_src_h0, [1, DPE])
                    T.ppl_mul(q_rope_tmp0_h0, q_rope_src_h0, cos_shared)
                    T.ppl_mul(q_rope_tmp1_h0, q_rope_src_h0, sin_shared)
                    T.ppl_mul_C(q_rope_tmp2_h0, q_rope_src_h0, T.float32(-1.0))
                    T.ppl_mul(q_rope_tmp3_h0, q_rope_tmp2_h0, sin_shared)
                    T.ppl_rope_add(
                        q_rope_h0, q_rope_tmp0_h0, q_rope_tmp3_h0, q_rope_tmp0_h0, q_rope_tmp1_h0
                    )
                    T.ppl_copy_region(q_upper_h1[0:B, DNOPE:DQU], q_rope_src_h1, [1, DPE])
                    T.ppl_mul(q_rope_tmp0_h1, q_rope_src_h1, cos_shared)
                    T.ppl_mul(q_rope_tmp1_h1, q_rope_src_h1, sin_shared)
                    T.ppl_mul_C(q_rope_tmp2_h1, q_rope_src_h1, T.float32(-1.0))
                    T.ppl_mul(q_rope_tmp3_h1, q_rope_tmp2_h1, sin_shared)
                    T.ppl_rope_add(
                        q_rope_h1, q_rope_tmp0_h1, q_rope_tmp3_h1, q_rope_tmp0_h1, q_rope_tmp1_h1
                    )

                    # Sophgo performs several per-head GEMMs after block-level weight
                    # preparation. Keep that legal structure here while reusing all
                    # head-independent local tiles across the block.
                    for hbi in T.unroll(HEAD_BLOCK):
                        if hbi == 0:
                            T.ppl_copy_region(
                                q_upper_h0[0:B, 0:DNOPE],
                                q_nope_head,
                                [1, DNOPE],
                            )
                        else:
                            T.ppl_copy_region(
                                q_upper_h1[0:B, 0:DNOPE],
                                q_nope_head,
                                [1, DNOPE],
                            )
                        if DEBUG_STAGE == "q_nope":
                            T.ppl_fill(out_cast, T.float32(0))
                            T.ppl_copy_region(q_nope_head[0:B, 0:DV], out_cast[0:B, 0:DV], [1, DV])
                            T.ppl_copy(out_cast, OUT[bb, (hg * HEAD_BLOCK + hbi) * DV])
                        if DEBUG_STAGE == "q_rope":
                            T.ppl_fill(out_cast, T.float32(0))
                            if hbi == 0:
                                T.ppl_copy_region(q_rope_h0[0:B, 0:DPE], out_cast[0:B, 0:DPE], [1, DPE])
                            else:
                                T.ppl_copy_region(q_rope_h1[0:B, 0:DPE], out_cast[0:B, 0:DPE], [1, DPE])
                            T.ppl_copy(out_cast, OUT[bb, (hg * HEAD_BLOCK + hbi) * DV])

                        T.ppl_fill(q_wukv_acc, T.float32(0))
                        T.ppl_gemm_region(
                            q_nope_head,
                            wukv_full_tile[hbi * DKVU:hbi * DKVU + DNOPE, 0:DKV],
                            q_wukv_acc,
                            A_extent=[1, DNOPE],
                            B_extent=[DNOPE, DKV],
                            C_extent=[1, DKV],
                        )
                        T.ppl_copy(q_wukv_acc, q_wukv_head)
                        if DEBUG_STAGE == "q_wukv":
                            T.ppl_fill(out_cast, T.float32(0))
                            T.ppl_copy_region(q_wukv_head[0:B, 0:DV], out_cast[0:B, 0:DV], [1, DV])
                            T.ppl_copy(out_cast, OUT[bb, (hg * HEAD_BLOCK + hbi) * DV])
                        if DEBUG_STAGE == "q_rope_after_qwukv":
                            T.ppl_fill(out_cast, T.float32(0))
                            if hbi == 0:
                                T.ppl_copy_region(q_rope_h0[0:B, 0:DPE], out_cast[0:B, 0:DPE], [1, DPE])
                            else:
                                T.ppl_copy_region(q_rope_h1[0:B, 0:DPE], out_cast[0:B, 0:DPE], [1, DPE])
                            T.ppl_copy(out_cast, OUT[bb, (hg * HEAD_BLOCK + hbi) * DV])

                        if hbi == 0:
                            T.ppl_gemm_region(
                                q_wukv_head,
                                kv_cache_full,
                                score_latent_h0,
                                transpose_B=True,
                                A_extent=[1, DKV],
                                B_extent=[SCORE_CHUNK, DKV],
                                C_extent=[1, SCORE_CHUNK],
                            )
                            T.ppl_gemm_region(
                                q_rope_h0,
                                pe_score_tile,
                                score_rope_h0,
                                transpose_B=True,
                                A_extent=[1, DPE],
                                B_extent=[SCORE_CHUNK, DPE],
                                C_extent=[1, SCORE_CHUNK],
                            )
                            T.ppl_add(score_tile, score_latent_h0, score_rope_h0)
                        else:
                            T.ppl_gemm_region(
                                q_wukv_head,
                                kv_cache_full,
                                score_latent_h0,
                                transpose_B=True,
                                A_extent=[1, DKV],
                                B_extent=[SCORE_CHUNK, DKV],
                                C_extent=[1, SCORE_CHUNK],
                            )
                            T.ppl_gemm_region(
                                q_rope_h1,
                                pe_score_tile,
                                score_rope_h0,
                                transpose_B=True,
                                A_extent=[1, DPE],
                                B_extent=[SCORE_CHUNK, DPE],
                                C_extent=[1, SCORE_CHUNK],
                            )
                            T.ppl_add(score_tile, score_latent_h0, score_rope_h0)
                        if DEBUG_STAGE == "score_latent":
                            T.ppl_fill(out_cast, T.float32(0))
                            if hbi == 0:
                                T.ppl_copy(score_latent_h0, prob_tile)
                            else:
                                T.ppl_copy(score_latent_h0, prob_tile)
                            T.ppl_copy_region(prob_tile[0:B, 0:DV], out_cast[0:B, 0:DV], [1, DV])
                            T.ppl_copy(out_cast, OUT[bb, (hg * HEAD_BLOCK + hbi) * DV])
                        if DEBUG_STAGE == "score_rope":
                            T.ppl_fill(out_cast, T.float32(0))
                            if hbi == 0:
                                T.ppl_copy(score_rope_h0, prob_tile)
                            else:
                                T.ppl_copy(score_rope_h0, prob_tile)
                            T.ppl_copy_region(prob_tile[0:B, 0:DV], out_cast[0:B, 0:DV], [1, DV])
                            T.ppl_copy(out_cast, OUT[bb, (hg * HEAD_BLOCK + hbi) * DV])
                        T.ppl_add(score_tile, score_tile, mask_row)
                        T.ppl_mul_C(score_tile, score_tile, T.float32(softmax_scale))
                        if DEBUG_STAGE == "score_scaled":
                            T.ppl_fill(out_cast, T.float32(0))
                            T.ppl_copy(score_tile, prob_tile)
                        if DEBUG_STAGE == "prob":
                            T.ppl_fill(out_cast, T.float32(0))
                            T.ppl_copy_region(prob_tile[0:B, 0:DV], out_cast[0:B, 0:DV], [1, DV])
                            T.ppl_copy(out_cast, OUT[bb, (hg * HEAD_BLOCK + hbi) * DV])
                            T.ppl_copy_region(prob_tile[0:B, 0:DV], out_cast[0:B, 0:DV], [1, DV])
                            T.ppl_copy(out_cast, OUT[bb, (hg * HEAD_BLOCK + hbi) * DV])
                        T.ppl_fill(scores_max, -T.infinity(accum_dtype))
                        T.ppl_reduce_max(score_tile, scores_max, dim=1, clear=False)
                        T.ppl_subtract(score_tile, score_tile, scores_max)
                        _ppl_exp_inplace(score_tile, work_score0, work_score1, coeff, table)
                        T.ppl_reduce_sum(score_tile, scores_sum, dim=1)
                        T.ppl_copy(score_tile, prob_tile)
                        T.ppl_fill(ctx_acc, T.float32(0))
                        T.ppl_gemm(prob_tile, kv_cache_full, ctx_acc)
                        T.ppl_div(ctx_acc, ctx_acc, scores_sum)
                        T.ppl_copy(ctx_acc, ctx_head)
                        if DEBUG_STAGE == "ctx":
                            T.ppl_fill(out_cast, T.float32(0))
                            T.ppl_copy_region(ctx_head[0:B, 0:DV], out_cast[0:B, 0:DV], [1, DV])
                            T.ppl_copy(out_cast, OUT[bb, (hg * HEAD_BLOCK + hbi) * DV])

                        T.ppl_fill(out_part, T.float32(0))
                        T.ppl_gemm_region(
                            ctx_head,
                            wukv_full_tile[hbi * DKVU + DNOPE:(hbi + 1) * DKVU, 0:DKV],
                            out_part,
                            transpose_B=True,
                            A_extent=[1, DKV],
                            B_extent=[DV, DKV],
                            C_extent=[1, DV],
                        )
                        T.ppl_copy(out_part, out_cast)
                        if DEBUG_STAGE == "out":
                            T.ppl_copy(out_cast, OUT[bb, (hg * HEAD_BLOCK + hbi) * DV])

    return main_kernel_inner


class InterfaceMLABf16Fp8:
    """Self-contained non-paged bf16+fp8 MLA front-end."""

    def __init__(self, dims: MLADims = FIXED_DIMS, mode="cmodel", cache_mode="inplace", quant_block_size=BF16_FP8_QBLOCK):
        self.dims = dims
        self.mode = mode
        self.cache_mode = cache_mode
        self.managed_outputs = cache_mode == "explicit"
        self.quant_block_size = int(quant_block_size)
        self._kernels = {}

    def _compile(self, key, builder, out_idx):
        kernel = self._kernels.get(key)
        if kernel is None:
            kernel = tilelang.compile(builder(), out_idx=out_idx, target="tpu", mode=self.mode)
            self._kernels[key] = kernel
        return kernel

    def _validate_static_params(self, head=None, q_lora_rank=None, kv_lora_rank=None, qk_nope_head_dim=None, qk_rope_head_dim=None, v_head_dim=None, max_cache_size=None, C=None):
        dims = self.dims
        expected = {"head": dims.heads, "q_lora_rank": dims.dq, "kv_lora_rank": dims.dkv, "qk_nope_head_dim": dims.dnope, "qk_rope_head_dim": dims.dpe, "v_head_dim": dims.dv, "max_cache_size": dims.seq}
        provided = {"head": head, "q_lora_rank": q_lora_rank, "kv_lora_rank": kv_lora_rank, "qk_nope_head_dim": qk_nope_head_dim, "qk_rope_head_dim": qk_rope_head_dim, "v_head_dim": v_head_dim, "max_cache_size": max_cache_size}
        for name, value in provided.items():
            if value is not None and int(value) != expected[name]:
                raise ValueError(f"{name}={value} does not match static MLA dim {expected[name]}")
        if C is not None and abs(float(C) - float(dims.softmax_scale)) > 1e-8:
            raise ValueError(f"C={C} does not match static MLA scale {dims.softmax_scale}")

    @staticmethod
    def _expect_shape(name, value, shape):
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} shape mismatch: got {tuple(value.shape)}, expected {tuple(shape)}")

    def _expanded_scales(self, WUQ_scale, WUKV_scale, head):
        dims = self.dims
        return (
            expand_scale_for_head(WUQ_scale, head, dims.dqu, dims.dq, self.quant_block_size),
            expand_scale_for_head(WUKV_scale, head, dims.dkvu, dims.dkv, self.quant_block_size),
        )

    def prefill_kernel(self, rows):
        dims = self.dims
        return self._compile(("clean_prefill_bf16_fp8_fused_dequant_cache", rows, dims), lambda: mla_prefill_bf16_fp8_fused_dequant_cache(rows, dims.seq, dims.dq, dims.dnope, dims.dpe, dims.dkv, dims.dv, dims.softmax_scale), [14, 15, 16])

    def decode_batch_head_kernel(self, batch, heads, head_start=0):
        dims = self.dims
        return self._compile(("clean_v49_multicore_head_block2_dynamic_slot", int(batch), int(heads), int(head_start), dims), lambda: mla_decode_bf16_fp8_batch_head_fused_dequant_cache(int(batch), int(heads), dims.seq, dims.dq, dims.dnope, dims.dpe, dims.dkv, dims.dv, dims.softmax_scale, self.quant_block_size, dims.heads, int(head_start), head_block=2), [12, 13, 14])

    def run_decode_batch_head(self, q, kv_cur, pe_cur, wuq, wukv, wuq_scale, wukv_scale, kv_cache, pe_cache, cos, sin, slot, decode_mask, head_start=0):
        if self.managed_outputs:
            raise ValueError("V49 head-block2 currently supports cache-mode=inplace only")
        batch = q.shape[0]
        heads = wuq.shape[0]
        slots = [int(v) for v in slot] if isinstance(slot, (list, tuple)) else [int(slot)] * batch
        for slot_b in slots:
            if slot_b < 0 or slot_b >= self.dims.seq:
                raise ValueError(f"decode slot {slot_b} out of range [0, {self.dims.seq})")
        if len(set(slots)) != 1:
            out_parts = []
            for bi, slot_b in enumerate(slots):
                head_out, _, _ = self.run_decode_batch_head(fresh(q[bi:bi + 1]), fresh(kv_cur[bi:bi + 1]), fresh(pe_cur[bi:bi + 1]), wuq, wukv, wuq_scale, wukv_scale, kv_cache[bi:bi + 1], pe_cache[bi:bi + 1], fresh(cos[bi:bi + 1]), fresh(sin[bi:bi + 1]), slot_b, fresh(decode_mask[bi:bi + 1]), head_start=head_start)
                out_parts.append(head_out[0])
            return torch.stack(out_parts, dim=0).contiguous(), fresh(kv_cache), fresh(pe_cache)
        slot0 = slots[0]
        kv_flat = kv_cache.reshape(batch * self.dims.seq, self.dims.dkv)
        pe_flat = pe_cache.reshape(batch * self.dims.seq, self.dims.dpe)
        wuq_flat = wuq.reshape(heads * self.dims.dqu, self.dims.dq).contiguous()
        wukv_flat = wukv.reshape(heads * self.dims.dkvu, self.dims.dkv).contiguous()
        out_flat = torch.empty((batch, heads * self.dims.dv), dtype=torch.bfloat16)
        self.decode_batch_head_kernel(batch, heads, head_start=head_start)(fresh(q), fresh(kv_cur), fresh(pe_cur), fresh(wuq_flat), fresh(wukv_flat), fresh(wuq_scale.contiguous()), fresh(wukv_scale.contiguous()), kv_flat, pe_flat, fresh(cos), fresh(sin), fresh(decode_mask), out_flat, kv_flat, pe_flat, slot0)
        return fresh(out_flat).reshape(batch, heads, self.dims.dv), fresh(kv_cache), fresh(pe_cache)

    def run_prefill_head(self, q, kv, pe_raw, wuq, wukv, wuq_scale, wukv_scale, kv_cache, pe_cache, cos, sin, mask, row_tile=8):
        row_tile = max(1, min(row_tile, q.shape[0]))
        if self.managed_outputs:
            kv_in = fresh(kv_cache)
            pe_in = fresh(pe_cache)
            kv_out = torch.empty_like(kv)
            pe_out = torch.empty_like(pe_raw)
        else:
            kv_in = kv_cache
            pe_in = pe_cache
            kv_out = kv_cache
            pe_out = pe_cache
        parts = []
        for row_start in range(0, q.shape[0], row_tile):
            row_end = min(row_start + row_tile, q.shape[0])
            rows = row_end - row_start
            out_tile = torch.empty((rows, self.dims.dv), dtype=torch.bfloat16)
            self.prefill_kernel(rows)(fresh(q[row_start:row_end]), fresh(kv), fresh(pe_raw), fresh(wuq), fresh(wukv), fresh(wuq_scale), fresh(wukv_scale), kv_in, pe_in, fresh(cos[row_start:row_end]), fresh(sin[row_start:row_end]), fresh(cos), fresh(sin), fresh(mask[row_start:row_end]), out_tile, kv_out, pe_out)
            parts.append(fresh(out_tile))
            if not self.managed_outputs:
                kv_in = kv_out
                pe_in = pe_out
        return torch.cat(parts, dim=0).contiguous(), fresh(kv_out), fresh(pe_out)

    def latent_attention_fp8(self, OUT, Q, KV, PE, WUQ, WUKV, KVcache, PEcache, cos, sin, WUQ_scale, WUKV_scale, KVU=None, mask=None, input_lengths=None, head=None, generate_token=1, q_lora_rank=None, kv_lora_rank=None, qk_nope_head_dim=None, qk_rope_head_dim=None, v_head_dim=None, mask_size=None, quant_block_size=BF16_FP8_QBLOCK, max_cache_size=None, C=None, attention_mode=NORMAL_ATTENTION_DECODE, row_tile=8, head_start=0, heads=None):
        del KVU
        if int(quant_block_size) != self.quant_block_size:
            raise ValueError(f"quant_block_size={quant_block_size} does not match runner {self.quant_block_size}")
        dims = self.dims
        self._validate_static_params(head=head, q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank, qk_nope_head_dim=qk_nope_head_dim, qk_rope_head_dim=qk_rope_head_dim, v_head_dim=v_head_dim, max_cache_size=max_cache_size, C=C)
        wuq = _normalize_weights(WUQ, dims.heads, dims.dqu, dims.dq)
        wukv = _normalize_weights(WUKV, dims.heads, dims.dkvu, dims.dkv)
        head_end = dims.heads if heads is None else min(head_start + heads, dims.heads)
        if int(attention_mode) == NORMAL_ATTENTION_DECODE:
            return self._run_decode_fp8_original(OUT, Q, KV, PE, wuq, wukv, WUQ_scale, WUKV_scale, KVcache, PEcache, cos, sin, input_lengths, int(generate_token), head_start, head_end)
        if int(attention_mode) == NORMAL_ATTENTION_PREFILL:
            return self._run_prefill_fp8_original(OUT, Q, KV, PE, wuq, wukv, WUQ_scale, WUKV_scale, KVcache, PEcache, cos, sin, mask, row_tile, head_start, head_end)
        raise ValueError(f"unsupported attention_mode={attention_mode}")

    def _run_decode_fp8_original(self, OUT, Q, KV, PE, WUQ, WUKV, WUQ_scale, WUKV_scale, KVcache, PEcache, cos, sin, input_lengths, generate_token, head_start, head_end):
        dims = self.dims
        Q = _model_matrix("decode Q", Q)
        KV = _model_matrix("decode KV", KV)
        PE = _model_matrix("decode PE", PE)
        cos = _model_matrix("decode cos", cos)
        sin = _model_matrix("decode sin", sin)
        if input_lengths is None:
            raise ValueError("decode requires input_lengths")
        if int(generate_token) != 1:
            raise ValueError("bf16+fp8 decode currently supports generate_token=1")
        batch = Q.shape[0]
        self._expect_shape("decode Q", Q, (batch, dims.dq))
        self._expect_shape("decode KV", KV, (batch, dims.dkv))
        self._expect_shape("decode PE", PE, (batch, dims.dpe))
        self._expect_shape("decode KVcache", KVcache, (batch, dims.seq, dims.dkv))
        self._expect_shape("decode PEcache", PEcache, (batch, dims.seq, dims.dpe))
        self._expect_shape("decode cos", cos, (batch, dims.dpe))
        self._expect_shape("decode sin", sin, (batch, dims.dpe))
        self._expect_shape("decode OUT", OUT, (batch, dims.heads, dims.dv))
        self._expect_shape("decode input_lengths", input_lengths, (batch,))
        slots = []
        masks = []
        for bi in range(batch):
            length = _check_finite_range("input_lengths", int(input_lengths[bi].item()), 1, dims.seq)
            slots.append(length - int(generate_token))
            masks.append(_make_decode_mask(length, dims.seq))
        decode_mask = torch.cat(masks, dim=0).contiguous()
        head_out, kv_out, pe_out = self.run_decode_batch_head(fresh(Q), fresh(KV), fresh(PE), fresh(WUQ[head_start:head_end]), fresh(WUKV[head_start:head_end]), fresh(WUQ_scale), fresh(WUKV_scale), KVcache, PEcache, fresh(cos), fresh(sin), slots, decode_mask, head_start=head_start)
        OUT[:, head_start:head_end, :] = head_out
        return {"OUT": OUT, "KVcache": kv_out, "PEcache": pe_out}

    def _run_prefill_fp8_original(self, OUT, Q, KV, PE, WUQ, WUKV, WUQ_scale, WUKV_scale, KVcache, PEcache, cos, sin, mask, row_tile, head_start, head_end):
        dims = self.dims
        Q = _model_matrix("prefill Q", Q)
        KV = _model_matrix("prefill KV", KV)
        PE = _model_matrix("prefill PE", PE)
        cos = _model_matrix("prefill cos", cos)
        sin = _model_matrix("prefill sin", sin)
        rows = Q.shape[0]
        if rows != dims.seq:
            raise ValueError(f"static correctness prefill requires M=S={dims.seq}; got M={rows}")
        if mask is None:
            q_pos = torch.arange(rows, dtype=torch.int64).unsqueeze(1)
            k_pos = torch.arange(dims.seq, dtype=torch.int64).unsqueeze(0)
            mask = torch.zeros((rows, dims.seq), dtype=torch.float32)
            mask[k_pos > q_pos] = MASK_VALUE
        elif mask.dim() == 4 and mask.shape[0] == 1 and mask.shape[2] == 1:
            mask = mask[0, :, 0, :].contiguous()
        elif mask.dim() != 2:
            raise ValueError(f"prefill mask expects 2D or (1, rows, 1, S); got {tuple(mask.shape)}")
        self._expect_shape("prefill Q", Q, (rows, dims.dq))
        self._expect_shape("prefill KV", KV, (dims.seq, dims.dkv))
        self._expect_shape("prefill PE", PE, (dims.seq, dims.dpe))
        self._expect_shape("prefill KVcache", KVcache, (1, dims.seq, dims.dkv))
        self._expect_shape("prefill PEcache", PEcache, (1, dims.seq, dims.dpe))
        self._expect_shape("prefill cos", cos, (dims.seq, dims.dpe))
        self._expect_shape("prefill sin", sin, (dims.seq, dims.dpe))
        self._expect_shape("prefill mask", mask, (rows, dims.seq))
        self._expect_shape("prefill OUT", OUT, (rows, dims.heads, dims.dv))
        kernel_mask = torch.clamp(mask, min=-1000.0)
        final_kv = KVcache
        final_pe = PEcache
        for head in range(head_start, head_end):
            wuq_scale_h, wukv_scale_h = self._expanded_scales(WUQ_scale, WUKV_scale, head)
            head_out, kv_out, pe_out = self.run_prefill_head(fresh(Q), fresh(KV), fresh(PE), fresh(WUQ[head]), fresh(WUKV[head]), wuq_scale_h, wukv_scale_h, KVcache[0], PEcache[0], fresh(cos), fresh(sin), fresh(kernel_mask), row_tile=row_tile)
            OUT[:, head, :] = head_out
            if self.managed_outputs:
                final_kv = final_kv.clone() if final_kv.data_ptr() == KVcache.data_ptr() else final_kv
                final_pe = final_pe.clone() if final_pe.data_ptr() == PEcache.data_ptr() else final_pe
                final_kv[0] = kv_out
                final_pe[0] = pe_out
        return {"OUT": OUT, "KVcache": final_kv, "PEcache": final_pe}


InterfaceMLABf16Fp8Decode = InterfaceMLABf16Fp8

__all__ = ["BF16_FP8_QBLOCK", "FIXED_DIMS", "SMOKE_DIMS", "InterfaceMLABf16Fp8", "InterfaceMLABf16Fp8Decode", "MLADims", "NORMAL_ATTENTION_DECODE", "NORMAL_ATTENTION_PREFILL", "mla_decode_bf16_fp8_batch_head_fused_dequant_cache", "mla_prefill_bf16_fp8_fused_dequant_cache"]
