# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import math

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
def flashattn(batch, heads, seq_len, dim, is_causal):
    scale = (1.0 / dim)**0.5
    shape = [batch, seq_len, heads, dim]
    dtype = "float16"
    accum_dtype = "float"

    def kernel_func(block_M, block_N, num_stages, threads):

        @T.macro
        def MMA0(
            K: T.Tensor(shape, dtype),
            Q_shared: T.Tensor([block_M, dim], dtype),
            K_shared: T.Tensor([block_N, dim], dtype),
            acc_s: T.Tensor([block_M, block_N], accum_dtype),
            k: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
            T.fill(acc_s, T.float32(0))
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)

        @T.macro
        def MMA1(
                V: T.Tensor(shape, dtype),
                V_shared: T.Tensor([block_M, dim], dtype),
                acc_s_cast: T.Tensor([block_M, block_N], dtype),
                acc_o: T.Tensor([block_M, dim], accum_dtype),
                k: T.int32,
                by: T.int32,
                bz: T.int32,
        ):
            T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
            T.gemm(acc_s_cast, V_shared, acc_o)

        @T.macro
        def Softmax(
                acc_s: T.Tensor([block_M, block_N], accum_dtype),
                acc_s_cast: T.Tensor([block_M, block_N], dtype),
                scores_max: T.Tensor([block_M, 1], accum_dtype),
                scores_max_prev: T.Tensor([block_M, 1], accum_dtype),
                scores_scale: T.Tensor([block_M, 1], accum_dtype),
                scores_sum: T.Tensor([block_M, 1], accum_dtype),
                logsum: T.Tensor([block_M, 1], accum_dtype),
        ):
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            T.subtract(scores_scale, scores_max_prev, scores_max)
            T.mul_C(scores_scale, scores_scale, scale)
            T.exp(scores_scale)
            T.subtract(acc_s, acc_s, scores_max)
            T.mul_C(acc_s, acc_s, scale)
            T.exp(acc_s)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            T.mul(logsum, logsum, scores_scale)
            T.add(logsum, logsum, scores_sum)
            T.copy(acc_s, acc_s_cast)

        @T.macro
        def Rescale(
                acc_o: T.Tensor([block_M, dim], accum_dtype),
                scores_scale: T.Tensor([block_M, 1], accum_dtype),
        ):
            T.mul(acc_o, acc_o, scores_scale)

        @T.prim_func
        def main_kernel_inner(
                Q: T.Tensor(shape, dtype),
                K: T.Tensor(shape, dtype),
                V: T.Tensor(shape, dtype),
                Output: T.Tensor(shape, dtype),
        ):
            with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, is_cpu=True) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)  # 1, block_m, 1, dim
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)
                # to alloc_shared
                acc_s = T.alloc_shared([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_shared([block_M, block_N], dtype)
                acc_o = T.alloc_shared([block_M, dim], accum_dtype)
                scores_max = T.alloc_shared([block_M, 1], accum_dtype)
                scores_max_prev = T.alloc_shared([block_M, 1], accum_dtype)
                scores_scale = T.alloc_shared([block_M, 1], accum_dtype)
                scores_sum = T.alloc_shared([block_M, 1], accum_dtype)
                logsum = T.alloc_shared([block_M, 1], accum_dtype)
                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)

                T.fill(acc_o, T.float32(0))
                T.fill(logsum, T.float32(0))
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.min(T.ceildiv(seq_len, block_N), T.ceildiv(
                        (bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N))

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                    Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale,
                            scores_sum, logsum)
                    Rescale(acc_o, scores_scale)
                    MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
                T.div(acc_o, acc_o, logsum)
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])

        return main_kernel_inner

    def kernel(block_M, block_N, num_stages, threads):
        return kernel_func(block_M, block_N, num_stages, threads)

    return kernel


batch = 1
heads = 2
seq_len = 4
dim = 8
block_M = 2
block_N = 2
is_causal = False
num_stages = 1
threads = 16

kernel = tilelang.compile(
    flashattn(
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        dim=dim,
        is_causal=is_causal,
    )(block_M, block_N, num_stages, threads),
    out_idx=-1,
    target="tpu",
    mode="cmodel",
)

q = torch.randn(batch, seq_len, heads, dim).half()
k = torch.randn(batch, seq_len, heads, dim).half()
v = torch.randn(batch, seq_len, heads, dim).half()
out = torch.zeros(batch, seq_len, heads, dim).half()

res = kernel(q, k, v, out)
print(res)
print("output:")
print(out)


def ref_program(Q, K, V, is_causal):
    dim = Q.size(-1)
    scores = torch.einsum('bqhd,bkhd->bhqk', Q.float(), K.float())
    scores = scores / math.sqrt(dim)
    if is_causal:
        seq_len = Q.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V.float())
    return output.half()


ref = ref_program(q, k, v, is_causal)
print("ref:")
print(ref)

diff = ref.float() - out.float()
max_diff = torch.max(torch.abs(diff))
avg_diff = torch.mean(torch.abs(diff))

print(f"\n=== 差异分析 ===")
print(f"最大差异: {max_diff}")
print(f"平均差异: {avg_diff}")
print("check close:")
print(torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2))
