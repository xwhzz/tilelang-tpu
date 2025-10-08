# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T

T.copy = T.ppl_copy


def llama2_fa_tpu(batch_size, heads_num, seq_len, dim, is_causal=False):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch_size, seq_len, heads_num, dim]
    dtype = "float16"      # TPU PPL path supports FP16/FP32
    accum_dtype = "float32"

    def kernel_func(block_M, block_N, num_stages, threads):

        @T.macro
        def MMA_QK(
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
            T.ppl_fill(acc_s, T.float32(0))
            T.ppl_gemm(Q_shared, K_shared, acc_s, transpose_B=True)

        @T.macro
        def MMA_SV(
            V: T.Tensor(shape, dtype),
            V_shared: T.Tensor([block_N, dim], dtype),
            acc_s_cast: T.Tensor([block_M, block_N], dtype),
            acc_o: T.Tensor([block_M, dim], accum_dtype),
            k: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
            T.ppl_gemm(acc_s_cast, V_shared, acc_o)

        @T.macro
        def Softmax_Tile(
            acc_s: T.Tensor([block_M, block_N], accum_dtype),
            acc_s_cast: T.Tensor([block_M, block_N], dtype),
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
            T.ppl_mul_C(scores_scale, scores_scale, T.float32(scale))

            work0 = T.alloc_shared([block_M, 1], accum_dtype)
            work1 = T.alloc_shared([block_M, 1], accum_dtype)
            coeff = T.alloc_shared([64, 32], accum_dtype)   # EU tables
            table = T.alloc_shared([64, 192], accum_dtype)  # npu number is 64
            T.ppl_exp2(scores_scale, work0, work1, coeff, table)
            
            T.ppl_subtract(acc_s, acc_s, scores_max)
            T.ppl_mul_C(acc_s, acc_s, T.float32(scale))

            work0_1 = T.alloc_shared([block_M, block_N], accum_dtype)
            work1_1 = T.alloc_shared([block_M, block_N], accum_dtype)
            coeff_1 = T.alloc_shared([64, 32], accum_dtype)
            table_1 = T.alloc_shared([64, 192], accum_dtype)
            T.ppl_exp2(acc_s, work0_1, work1_1, coeff_1, table_1)

            T.ppl_reduce_sum(acc_s, scores_sum, dim=1)
            T.ppl_mul(logsum, logsum, scores_scale)
            T.ppl_add(logsum, logsum, scores_sum)

            T.copy(acc_s, acc_s_cast)

        @T.prim_func
        def main(
            Q: T.Tensor(shape, dtype),
            K: T.Tensor(shape, dtype),
            V: T.Tensor(shape, dtype),
            Output: T.Tensor(shape, dtype),
        ):
            with T.Kernel(T.ceildiv(seq_len, block_M), heads_num, batch_size, is_cpu=True) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)

                acc_s = T.alloc_shared([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_shared([block_M, block_N], dtype)
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
                    T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N)) if is_causal
                    else T.ceildiv(seq_len, block_N)
                )

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    MMA_QK(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                    # Optional is_causal mask: mask future positions
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            if bx * block_M + i < k * block_N + j:
                                acc_s[i, j] = -T.infinity(accum_dtype)
                    Softmax_Tile(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                    # rescale accumulated output
                    T.ppl_mul(acc_o, acc_o, scores_scale)
                    MMA_SV(V, V_shared, acc_s_cast, acc_o, k, by, bz)

                T.ppl_div(acc_o, acc_o, logsum)
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])

        return main

    return kernel_func


batch_size, heads_num, seq_len, dim = 32, 8, 2048, 128
block_M, block_N, num_stages, threads = 128, 64, 2, 128
func = llama2_fa_tpu(batch_size, heads_num, seq_len, dim, is_causal=False)(block_M, block_N, num_stages, threads)
mod = tilelang.lower(func)
