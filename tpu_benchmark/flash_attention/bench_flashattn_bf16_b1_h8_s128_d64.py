"""
FlashAttention BF16 B1H8S128D64 — tilelang vs PPL

Optimization vs v1:
  1. Pre-scale Q by 1/sqrt(d) before the K-loop (matches PPL).
  2. Use T.exp so BM1690 exp work/coeff/table buffers stay hidden.
  3. Keep Softmax frontend code free of backend scratch-buffer plumbing.
"""

import math
import os, sys, torch
import torch.nn.functional as F

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_ROOT = os.path.dirname(BENCH_DIR)
sys.path.insert(0, BENCHMARK_ROOT)

import tilelang
import tilelang.language as T
BATCH, HEADS, SEQ_LEN, DIM = 1, 8, 128, 64
BLOCK_M, BLOCK_N = 16, 16
IS_CAUSAL = False
NUM_STAGES = 1
ATOL, RTOL = 5e-2, 5e-2


def tl_flashattn_bf16_v2(batch, heads, seq_len, dim, block_M, block_N,
                          is_causal, num_stages):
    scale = (1.0 / dim) ** 0.5
    shape = [batch, seq_len, heads, dim]
    dtype = "bfloat16"
    accum_dtype = "float"

    @T.macro
    def MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz):
        T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
        T.fill(acc_s, T.float32(0))
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)

    @T.macro
    def MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz):
        T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
        T.gemm(acc_s_cast, V_shared, acc_o)

    @T.macro
    def Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, block_max,
                scores_scale, scores_sum, logsum):
        T.copy(scores_max, scores_max_prev)
        T.reduce_max(acc_s, block_max, dim=1)
        T.max(scores_max, scores_max_prev, block_max)
        T.subtract(scores_scale, scores_max_prev, scores_max)
        T.exp(scores_scale)
        T.subtract(acc_s, acc_s, scores_max)
        T.exp(acc_s)
        T.reduce_sum(acc_s, scores_sum, dim=1)
        T.mul(logsum, logsum, scores_scale)
        T.add(logsum, logsum, scores_sum)
        T.copy(acc_s, acc_s_cast)

    @T.macro
    def Rescale(acc_o, scores_scale):
        T.mul(acc_o, acc_o, scores_scale)

    @T.prim_func
    def main_kernel_inner(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, is_cpu=True) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_shared([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_shared([block_M, block_N], dtype)
            acc_o = T.alloc_shared([block_M, dim], accum_dtype)
            scores_max = T.alloc_shared([block_M, 1], accum_dtype)
            scores_max_prev = T.alloc_shared([block_M, 1], accum_dtype)
            block_max = T.alloc_shared([block_M, 1], accum_dtype)
            scores_scale = T.alloc_shared([block_M, 1], accum_dtype)
            scores_sum = T.alloc_shared([block_M, 1], accum_dtype)
            logsum = T.alloc_shared([block_M, 1], accum_dtype)

            T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
            T.mul_C(Q_shared, Q_shared, T.float32(scale))
            T.fill(acc_o, T.float32(0))
            T.fill(logsum, T.float32(0))
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_len, block_N), T.ceildiv(
                    (bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N))

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, block_max, scores_scale,
                        scores_sum, logsum)
                Rescale(acc_o, scores_scale)
                MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
            T.div(acc_o, acc_o, logsum)
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])

    return main_kernel_inner


def ref_program(Q, K, V, is_causal):
    dim = Q.size(-1)
    scores = torch.einsum('bqhd,bkhd->bhqk', Q.float(), K.float()) / math.sqrt(dim)
    if is_causal:
        sl = Q.size(1)
        mask = torch.tril(torch.ones(sl, sl, device=scores.device))
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
    return torch.einsum('bhqk,bkhd->bqhd', F.softmax(scores, dim=-1), V.float()).bfloat16()


def run_and_check(name, kernel_func, q, k, v, ref):
    out = torch.zeros(BATCH, SEQ_LEN, HEADS, DIM, dtype=torch.bfloat16)
    kernel_func(q, k, v, out)
    correct = torch.allclose(out.float(), ref.float(), atol=ATOL, rtol=RTOL)
    max_diff = (out.float() - ref.float()).abs().max().item()
    avg_diff = (out.float() - ref.float()).abs().mean().item()
    print(f"  [{name}] correct={correct}  max_diff={max_diff:.6f}  avg_diff={avg_diff:.8f}")
    return correct


def main():
    q = torch.randn(BATCH, SEQ_LEN, HEADS, DIM, dtype=torch.bfloat16)
    k = torch.randn(BATCH, SEQ_LEN, HEADS, DIM, dtype=torch.bfloat16)
    v = torch.randn(BATCH, SEQ_LEN, HEADS, DIM, dtype=torch.bfloat16)
    ref = ref_program(q, k, v, IS_CAUSAL)

    print("=" * 60)
    print(f"FlashAttention BF16 v2  B={BATCH} H={HEADS} S={SEQ_LEN} D={DIM}")
    print("=" * 60)

    print("\n--- tilelang ---")
    tl_kernel = tilelang.compile(
        tl_flashattn_bf16_v2(BATCH, HEADS, SEQ_LEN, DIM, BLOCK_M, BLOCK_N,
                             IS_CAUSAL, NUM_STAGES),
        out_idx=-1, target="tpu")
    run_and_check("tilelang", tl_kernel, q, k, v, ref)

    print("\n--- PPL ---")
    try:
        from ppl_utils import compile_ppl_kernel
        pl_path = os.path.join(BENCH_DIR, "pl", "flashattn_bf16_b1_h8_s128_d64.pl")
        qkv_shape = (BATCH, SEQ_LEN, HEADS, DIM)
        arg_specs = [(qkv_shape, torch.bfloat16)] * 4
        ppl_forward = compile_ppl_kernel(pl_path, arg_specs, result_idx=[3])
        run_and_check("PPL", ppl_forward, q, k, v, ref)
    except Exception as e:
        print(f"  PPL failed: {e}")
        import traceback; traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
