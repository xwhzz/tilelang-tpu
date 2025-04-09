# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T

def flashattn(batch, heads, seq_len, dim, is_causal):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = "float16"
    accum_dtype = "float"

    def kernel_func(block_M, block_N, num_stages, threads):

        @T.macro
        def MMA0(
            K: T.Buffer(shape, dtype),
            Q_shared: T.Buffer([block_M, dim], dtype),
            K_shared: T.Buffer([block_N, dim], dtype),
            acc_s: T.Buffer([block_M, block_N], accum_dtype),
            k: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.ppl_copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
            T.ppl_fill(acc_s, T.float32(0))
            T.ppl_gemm(Q_shared, K_shared, acc_s, transpose_B=True)

        @T.macro
        def MMA1(
                V: T.Buffer(shape, dtype),
                V_shared: T.Buffer([block_M, dim], dtype),
                acc_s_cast: T.Buffer([block_M, block_N], dtype),
                acc_o: T.Buffer([block_M, dim], accum_dtype),
                k: T.int32,
                by: T.int32,
                bz: T.int32,
        ):
            T.ppl_copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
            T.ppl_gemm(acc_s_cast, V_shared, acc_o)

        @T.macro
        def Softmax(
                acc_s: T.Buffer([block_M, block_N], accum_dtype),
                acc_s_cast: T.Buffer([block_M, block_N], dtype),
                scores_max: T.Buffer([block_M, 1], accum_dtype),
                scores_max_prev: T.Buffer([block_M, 1], accum_dtype),
                scores_scale: T.Buffer([block_M, 1], accum_dtype),
                scores_sum: T.Buffer([block_M, 1], accum_dtype),
                logsum: T.Buffer([block_M, 1], accum_dtype),
        ):
            T.ppl_copy(scores_max, scores_max_prev)
            T.ppl_fill(scores_max, -T.infinity(accum_dtype))
            T.ppl_reduce_max(acc_s, scores_max, dim=1, clear=False)
            T.ppl_subtract(scores_scale, scores_max_prev, scores_max)
            T.ppl_mul_C(scores_scale, scores_scale, scale)
            T.ppl_exp2(scores_scale)
            T.ppl_subtract(acc_s, acc_s, scores_max)
            T.ppl_mul_C(acc_s, acc_s, scale)
            T.ppl_exp2(acc_s)
            T.ppl_reduce_sum(acc_s, scores_sum, dim=1)
            T.ppl_mul(logsum, logsum, scores_scale)
            T.ppl_add(logsum, logsum, scores_sum)
            T.ppl_copy(acc_s, acc_s_cast)

        @T.macro
        def Rescale(
                acc_o: T.Buffer([block_M, dim], accum_dtype),
                scores_scale: T.Buffer([block_M, 1], accum_dtype),
        ):
            # need bdcast
            # for i, j in T.Parallel(block_M, dim):
            #     acc_o[i, j] *= scores_scale[i]
            # bdcast
            T.ppl_mul(acc_o, acc_o,scores_scale)
            
        @T.prim_func
        def main(
                Q: T.Buffer(shape, dtype),
                K: T.Buffer(shape, dtype),
                V: T.Buffer(shape, dtype),
                Output: T.Buffer(shape, dtype),
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), heads, batch, is_cpu=True) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype) # 1, block_m, 1, dim
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

                T.ppl_copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                T.ppl_fill(acc_o, T.float32(0))
                T.ppl_fill(logsum,T.float32(0))
                T.ppl_fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.min(T.ceildiv(seq_len, block_N), T.ceildiv(
                        (bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N))

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                    Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale,
                            scores_sum, logsum)
                    Rescale(acc_o, scores_scale)
                    MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
                T.ppl_div(acc_o, acc_o,logsum)
                T.ppl_copy(acc_o, O_shared)
                T.ppl_copy(O_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])

        return main

    def kernel(block_M, block_N, num_stages, threads):
        return kernel_func(block_M, block_N, num_stages, threads)
    return kernel


func =  flashattn(1, 8, 256, 80, False)(64, 64, 2, 128)
mod = tilelang.lower(func)