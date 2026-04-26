# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import math
import os

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
from tilelang.jit.adapter.utils import get_tpu_template_dir
from tilelang.jit.adapter.wrapper import TLWrapper
from tilelang.utils.target import determine_target

T.copy = T.ppl_copy


def flashattn(batch, heads, seq_len, dim, is_causal):
    scale = (1.0 / dim)**0.5
    shape = [batch, seq_len, heads, dim]
    dtype = "float32"
    gemm_dtype = "float16"
    accum_dtype = "float32"

    def kernel_func(block_M, block_N, num_stages, threads):

        @T.macro
        def MMA0(
            K: T.Tensor(shape, dtype),
            Q_shared: T.Tensor([block_M, dim], dtype),
            K_shared: T.Tensor([block_N, dim], dtype),
            Q_gemm: T.Tensor([block_M, dim], gemm_dtype),
            K_gemm: T.Tensor([block_N, dim], gemm_dtype),
            acc_s: T.Tensor([block_M, block_N], accum_dtype),
            k: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
            T.copy(Q_shared, Q_gemm)
            T.copy(K_shared, K_gemm)
            T.ppl_fill(acc_s, T.float32(0))
            T.ppl_gemm(Q_gemm, K_gemm, acc_s, transpose_B=True)

        @T.macro
        def MMA1(
                V: T.Tensor(shape, dtype),
                V_shared: T.Tensor([block_M, dim], dtype),
                V_gemm: T.Tensor([block_N, dim], gemm_dtype),
                acc_s_cast: T.Tensor([block_M, block_N], gemm_dtype),
                acc_o: T.Tensor([block_M, dim], accum_dtype),
                k: T.int32,
                by: T.int32,
                bz: T.int32,
        ):
            T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
            T.copy(V_shared, V_gemm)
            T.ppl_gemm(acc_s_cast, V_gemm, acc_o)

        @T.macro
        def Softmax(
                acc_s: T.Tensor([block_M, block_N], accum_dtype),
                acc_s_cast: T.Tensor([block_M, block_N], gemm_dtype),
                scores_max: T.Tensor([block_M, 1], accum_dtype),
                scores_max_prev: T.Tensor([block_M, 1], accum_dtype),
                scores_scale: T.Tensor([block_M, 1], accum_dtype),
                scores_sum: T.Tensor([block_M, 1], accum_dtype),
                logsum: T.Tensor([block_M, 1], accum_dtype),
        ):
            T.copy(scores_max, scores_max_prev)
            T.ppl_fill(scores_max, -T.infinity(accum_dtype))
            T.ppl_reduce_max(acc_s, scores_max, dim=1, clear=False)
            T.ppl_subtract(scores_scale, scores_max_prev, scores_max)
            T.ppl_mul_C(scores_scale, scores_scale, scale)
            work0 = T.alloc_shared([block_M, 1], accum_dtype)
            work1 = T.alloc_shared([block_M, 1], accum_dtype)
            coeff = T.alloc_shared([64, 32], accum_dtype)  # npu number is 64
            table = T.alloc_shared([64, 192], accum_dtype)  # npu number is 64
            T.ppl_exp2(scores_scale, work0, work1, coeff, table)
            T.ppl_subtract(acc_s, acc_s, scores_max)
            T.ppl_mul_C(acc_s, acc_s, scale)
            work0_1 = T.alloc_shared([block_M, block_N], accum_dtype)
            work1_1 = T.alloc_shared([block_M, block_N], accum_dtype)
            coeff_1 = T.alloc_shared([64, 32], accum_dtype)  # npu number is 64
            table_1 = T.alloc_shared([64, 192], accum_dtype)  # npu number is 64
            T.ppl_exp2(acc_s, work0_1, work1_1, coeff_1, table_1)
            T.ppl_reduce_sum(acc_s, scores_sum, dim=1)
            T.ppl_mul(logsum, logsum, scores_scale)
            T.ppl_add(logsum, logsum, scores_sum)
            T.copy(acc_s, acc_s_cast)

        @T.macro
        def Rescale(
                acc_o: T.Tensor([block_M, dim], accum_dtype),
                scores_scale: T.Tensor([block_M, 1], accum_dtype),
        ):
            # need bdcast
            # for i, j in T.Parallel(block_M, dim):
            #     acc_o[i, j] *= scores_scale[i]
            # bdcast
            T.ppl_mul(acc_o, acc_o, scores_scale)

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
                Q_gemm = T.alloc_shared([block_M, dim], gemm_dtype)
                K_gemm = T.alloc_shared([block_N, dim], gemm_dtype)
                V_gemm = T.alloc_shared([block_N, dim], gemm_dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)
                # to alloc_shared
                acc_s = T.alloc_shared([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_shared([block_M, block_N], gemm_dtype)
                acc_o = T.alloc_shared([block_M, dim], accum_dtype)
                scores_max = T.alloc_shared([block_M, 1], accum_dtype)
                scores_max_prev = T.alloc_shared([block_M, 1], accum_dtype)
                scores_scale = T.alloc_shared([block_M, 1], accum_dtype)
                scores_sum = T.alloc_shared([block_M, 1], accum_dtype)
                logsum = T.alloc_shared([block_M, 1], accum_dtype)
                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)

                T.ppl_fill(acc_o, T.float32(0))
                T.ppl_fill(logsum, T.float32(0))
                T.ppl_fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.min(T.ceildiv(seq_len, block_N), T.ceildiv(
                        (bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N))

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    MMA0(K, Q_shared, K_shared, Q_gemm, K_gemm, acc_s, k, bx, by, bz)
                    Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale,
                            scores_sum, logsum)
                    Rescale(acc_o, scores_scale)
                    MMA1(V, V_shared, V_gemm, acc_s_cast, acc_o, k, by, bz)
                T.ppl_div(acc_o, acc_o, logsum)
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

prim_func = flashattn(
    batch=batch,
    heads=heads,
    seq_len=seq_len,
    dim=dim,
    is_causal=is_causal,
)(block_M, block_N, num_stages, threads)

if os.environ.get("TILELANG_TPU_GENERATE_ONLY") == "1":
    target = tilelang.tvm.target.Target(determine_target("tpu"))
    with tilelang.tvm.transform.PassContext(opt_level=3):
        artifact = tilelang.lower(
            prim_func,
            target=target,
            enable_host_codegen=False,
            enable_device_compile=False,
        )

    wrapper = TLWrapper(target)
    wrapper.assign_optimized_module(tilelang.tvm.IRModule({prim_func.attrs["global_symbol"]: prim_func}))
    wrapper.assign_host_module(artifact.host_mod)
    wrapper.assign_device_module(artifact.device_mod)
    wrapper.assign_output_indices([3])
    wrapper.assign_pass_configs(None)
    wrapper.wrap(artifact.kernel_source)

    tpu_template_dir = get_tpu_template_dir()
    print(f"generated: {tpu_template_dir}/kernel.c")
    print(f"generated: {tpu_template_dir}/kernel.cpp")
    print(f"generated: {tpu_template_dir}/main.cpp")
    raise SystemExit(0)

kernel = tilelang.compile(
    prim_func,
    out_idx=-1,
    target="tpu",
)

q = torch.randn(batch, seq_len, heads, dim).float()
k = torch.randn(batch, seq_len, heads, dim).float()
v = torch.randn(batch, seq_len, heads, dim).float()
out = torch.zeros(batch, seq_len, heads, dim).float()

res = kernel(q, k, v, out)
print(res)
print("output:")
print(out)


def ref_program(Q, K, V, is_causal):
    dim = Q.size(-1)
    scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)
    scores = scores / math.sqrt(dim)
    if is_causal:
        seq_len = Q.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)
    return output


ref = ref_program(q, k, v, is_causal)
print("ref:")
print(ref)

diff = ref - out
max_diff = torch.max(torch.abs(diff))
avg_diff = torch.mean(torch.abs(diff))

print(f"\n=== 差异分析 ===")
print(f"最大差异: {max_diff}")
print(f"平均差异: {avg_diff}")
print("check close:")
print(torch.allclose(out, ref, atol=1e-2, rtol=1e-2))
