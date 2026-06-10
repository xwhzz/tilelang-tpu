"""
Benchmark: optimized FlashAttention fp16_b1_h8_s128_d64 vs PPL.

Fixed TL tile: `128 x 128`.
Timing uses the TPU template default: 5 warmup runs, then 10 measured runs.
"""

import json
import math
import os
import sys

import torch
import torch.nn.functional as F

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_ROOT = os.path.dirname(BENCH_DIR)
FLASH_DIR = os.path.join(BENCHMARK_ROOT, "flash_attention")
sys.path.insert(0, BENCHMARK_ROOT)

import tilelang
import tilelang.language as T
IS_CAUSAL = False
NUM_STAGES = 1
DTYPE_LABEL = "fp16"
TL_DTYPE = "float16"
TORCH_DTYPE = torch.float16
BATCH = 1
HEADS = 8
SEQ_LEN = 128
DIM = 64
BLOCK_M = 128
BLOCK_N = 128
ATOL = 1e-2
RTOL = 1e-2


def tl_flashattn_opt():
    batch = BATCH
    heads = HEADS
    seq_len = SEQ_LEN
    dim = DIM
    block_M = BLOCK_M
    block_N = BLOCK_N
    dtype = TL_DTYPE
    accum_dtype = "float"
    scale = (1.0 / dim) ** 0.5
    shape = [batch, seq_len, heads, dim]

    @T.macro
    def LoadQ(Q, Q_shared, bx, bz):
        for h in T.serial(heads):
            T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, h, :], Q_shared[h, :, :])

    @T.macro
    def LoadK(K, K_shared, k, bz):
        for h in T.serial(heads):
            T.copy(K[bz, k * block_N:(k + 1) * block_N, h, :], K_shared[h, :, :])

    @T.macro
    def LoadV(V, V_shared, k, bz):
        for h in T.serial(heads):
            T.copy(V[bz, k * block_N:(k + 1) * block_N, h, :], V_shared[h, :, :])

    @T.macro
    def StoreO(O_shared, Output, bx, bz):
        for h in T.serial(heads):
            T.copy(O_shared[h, :, :], Output[bz, bx * block_M:(bx + 1) * block_M, h, :])

    @T.macro
    def MMA0(K, K_shared, Q_shared, acc_s, k, bz):
        LoadK(K, K_shared, k, bz)
        T.fill(acc_s, T.float32(0))
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)

    @T.macro
    def MMA1(V, V_shared, acc_s_cast, acc_o, k, bz):
        LoadV(V, V_shared, k, bz)
        T.gemm(acc_s_cast, V_shared, acc_o)

    @T.macro
    def Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, block_max,
                scores_scale, scores_sum, logsum):
        T.copy(scores_max, scores_max_prev)
        T.reduce_max(acc_s, block_max, dim=2)
        T.max(scores_max, scores_max_prev, block_max)
        T.subtract(scores_scale, scores_max_prev, scores_max)
        T.exp(scores_scale)
        T.subtract(acc_s, acc_s, scores_max)
        T.exp(acc_s)
        T.reduce_sum(acc_s, scores_sum, dim=2)
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
        with T.Kernel(T.ceildiv(seq_len, block_M), batch, is_cpu=True) as (bx, bz):
            Q_shared = T.alloc_shared([heads, block_M, dim], dtype)
            K_shared = T.alloc_shared([heads, block_N, dim], dtype)
            V_shared = T.alloc_shared([heads, block_N, dim], dtype)
            O_shared = T.alloc_shared([heads, block_M, dim], dtype)
            acc_s = T.alloc_shared([heads, block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_shared([heads, block_M, block_N], dtype)
            acc_o = T.alloc_shared([heads, block_M, dim], accum_dtype)
            scores_max = T.alloc_shared([heads, block_M, 1], accum_dtype)
            scores_max_prev = T.alloc_shared([heads, block_M, 1], accum_dtype)
            block_max = T.alloc_shared([heads, block_M, 1], accum_dtype)
            scores_scale = T.alloc_shared([heads, block_M, 1], accum_dtype)
            scores_sum = T.alloc_shared([heads, block_M, 1], accum_dtype)
            logsum = T.alloc_shared([heads, block_M, 1], accum_dtype)

            LoadQ(Q, Q_shared, bx, bz)
            T.mul_C(Q_shared, Q_shared, T.float32(scale))
            T.fill(acc_o, T.float32(0))
            T.fill(logsum, T.float32(0))
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_len, block_N), T.ceildiv(
                    (bx + 1) * block_M, block_N)) if IS_CAUSAL else T.ceildiv(seq_len, block_N))

            for k in T.Pipelined(loop_range, num_stages=NUM_STAGES):
                MMA0(K, K_shared, Q_shared, acc_s, k, bz)
                Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, block_max, scores_scale,
                        scores_sum, logsum)
                Rescale(acc_o, scores_scale)
                MMA1(V, V_shared, acc_s_cast, acc_o, k, bz)
            T.div(acc_o, acc_o, logsum)
            T.copy(acc_o, O_shared)
            StoreO(O_shared, Output, bx, bz)

    return main_kernel_inner


def ref_program(Q, K, V, out_dtype):
    scores = torch.einsum("bqhd,bkhd->bhqk", Q.float(), K.float()) / math.sqrt(DIM)
    if IS_CAUSAL:
        mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, device=scores.device))
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", F.softmax(scores, dim=-1), V.float()).to(out_dtype)


def make_inputs(seed):
    torch.manual_seed(seed)
    shape = (BATCH, SEQ_LEN, HEADS, DIM)
    q = torch.randn(shape, dtype=TORCH_DTYPE)
    k = torch.randn(shape, dtype=TORCH_DTYPE)
    v = torch.randn(shape, dtype=TORCH_DTYPE)
    ref = ref_program(q, k, v, TORCH_DTYPE)
    return q, k, v, ref


def check_output(name, out, ref):
    correct = torch.allclose(out.float(), ref.float(), atol=ATOL, rtol=RTOL)
    max_diff = (out.float() - ref.float()).abs().max().item()
    avg_diff = (out.float() - ref.float()).abs().mean().item()
    result = {
        "name": name,
        "correct": bool(correct),
        "max_diff": max_diff,
        "avg_diff": avg_diff,
    }
    print(f"  [{name}] correct={correct}  max_diff={max_diff:.6f}  avg_diff={avg_diff:.8f}")
    print("RESULT_JSON:" + json.dumps(result, sort_keys=True), file=sys.stderr, flush=True)
    return correct


def run_tilelang(q, k, v, ref):
    print("\n--- tilelang optimized ---", flush=True)
    print(f"  tile={BLOCK_M}x{BLOCK_N} dtype={TL_DTYPE}", flush=True)
    kernel = tilelang.compile(tl_flashattn_opt(), out_idx=-1, target="tpu")
    out = torch.zeros_like(ref)
    kernel(q, k, v, out)
    return check_output("tilelang-opt", out, ref)


def run_ppl(q, k, v, ref):
    print("\n--- PPL ---", flush=True)
    from ppl_utils import compile_ppl_kernel

    pl_name = f"flashattn_{DTYPE_LABEL}_b{BATCH}_h{HEADS}_s{SEQ_LEN}_d{DIM}.pl"
    pl_path = os.path.join(FLASH_DIR, "pl", pl_name)
    qkv_shape = (BATCH, SEQ_LEN, HEADS, DIM)
    arg_specs = [(qkv_shape, TORCH_DTYPE)] * 4
    kernel = compile_ppl_kernel(pl_path, arg_specs, result_idx=[3])
    out = torch.zeros_like(ref)
    kernel(q, k, v, out)
    return check_output("PPL", out, ref)


def main():
    q, k, v, ref = make_inputs(seed=0)

    print("=" * 72)
    print(
        f"FlashAttention optimized fp16_b1_h8_s128_d64 B={BATCH} H={HEADS} S={SEQ_LEN} D={DIM} "
        f"tile={BLOCK_M}x{BLOCK_N}"
    )
    print("=" * 72, flush=True)

    run_tilelang(q, k, v, ref)
    run_ppl(q, k, v, ref)
    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
