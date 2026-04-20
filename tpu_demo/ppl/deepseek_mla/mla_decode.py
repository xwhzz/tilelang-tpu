# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T

T.copy = T.ppl_copy

def flashattn(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, block_N, block_H):
    scale = (1.0 / (dim + pe_dim))**0.5
    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = heads // kv_head_num
    VALID_BLOCK_H = min(block_H, kv_group_num)
    assert kv_head_num == 1, "kv_head_num must be 1"

    @T.prim_func
    def main(
        Q: T.Tensor([batch, heads, 1, dim], dtype),
        Q_pe: T.Tensor([batch, heads, 1, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        Output: T.Tensor([batch, heads, 1, dim], dtype),
    ):
        with T.Kernel(batch, T.ceildiv(heads, VALID_BLOCK_H), is_cpu=True) as (bx, by):
            # --- allocate buffers ---
            Q_shared    = T.alloc_shared([VALID_BLOCK_H, dim], dtype)
            Q_pe_shared = T.alloc_shared([VALID_BLOCK_H, pe_dim], dtype)
            KV_shared   = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            S_shared    = T.alloc_shared([VALID_BLOCK_H, block_N], dtype)
            O_shared    = T.alloc_shared([VALID_BLOCK_H, dim], dtype)

            acc_s           = T.alloc_shared([VALID_BLOCK_H, block_N], accum_dtype)
            acc_s_pe        = T.alloc_shared([VALID_BLOCK_H, block_N], accum_dtype)
            acc_o           = T.alloc_shared([VALID_BLOCK_H, dim], accum_dtype)
            scores_max      = T.alloc_shared([VALID_BLOCK_H, 1], accum_dtype)
            scores_max_prev = T.alloc_shared([VALID_BLOCK_H, 1], accum_dtype)
            scores_scale    = T.alloc_shared([VALID_BLOCK_H, 1], accum_dtype)
            scores_sum      = T.alloc_shared([VALID_BLOCK_H, 1], accum_dtype)
            logsum          = T.alloc_shared([VALID_BLOCK_H, 1], accum_dtype)

            # --- head mapping ---
            cur_kv_head = by // (kv_group_num // VALID_BLOCK_H)

            # --- preload Q / Q_pe tile for these heads ---
            T.copy(Q[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :, :], Q_shared)
            T.copy(Q_pe[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :, :], Q_pe_shared)

            # --- init accumulators (online softmax state) ---
            T.ppl_fill(acc_o, T.float32(0))
            T.ppl_fill(logsum, T.float32(0))
            T.ppl_fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(seqlen_kv, block_N)
            for k in T.Pipelined(loop_range, num_stages=2): # num_stages=2, k<126
                # 1) load K / K_pe tile
                T.copy(KV [bx, k * block_N:(k + 1) * block_N, cur_kv_head, :], KV_shared)
                T.copy(K_pe[bx, k * block_N:(k + 1) * block_N, cur_kv_head, :], K_pe_shared)

                # 2) scores tile = Q@K^T + Q_pe@K_pe^T
                T.ppl_clear(acc_s)
                T.ppl_gemm(Q_shared, KV_shared, acc_s, transpose_B=True)
                T.ppl_clear(acc_s_pe)
                T.ppl_gemm(Q_pe_shared, K_pe_shared, acc_s_pe, transpose_B=True)
                T.ppl_add(acc_s, acc_s, acc_s_pe)

                # 3) online softmax (row-wise)
                T.copy(scores_max, scores_max_prev)
                T.ppl_fill(scores_max, -T.infinity(accum_dtype))
                T.ppl_reduce_max(acc_s, scores_max, dim=1, clear=False)

                # for i in T.Parallel(VALID_BLOCK_H):
                #     scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                prev_scaled = T.alloc_shared([VALID_BLOCK_H, 1], accum_dtype)
                curr_scaled = T.alloc_shared([VALID_BLOCK_H, 1], accum_dtype)
                T.ppl_mul_C(prev_scaled, scores_max_prev, scale)  # prev_scaled = scores_max_prev * scale
                T.ppl_mul_C(curr_scaled, scores_max, scale)  # curr_scaled = scores_max * scale
                T.ppl_subtract(scores_scale, prev_scaled, curr_scaled)
                work0 = T.alloc_shared([VALID_BLOCK_H, 1], accum_dtype)
                work1 = T.alloc_shared([VALID_BLOCK_H, 1], accum_dtype)
                coeff = T.alloc_shared([64, 32], accum_dtype)  # npu number is 64
                table = T.alloc_shared([64, 192], accum_dtype)  # npu number is 64
                T.ppl_exp2(scores_scale, work0, work1, coeff, table)

                # for i, j in T.Parallel(VALID_BLOCK_H, block_N):
                #     acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                scores_scaled  = T.alloc_shared([VALID_BLOCK_H, 1], accum_dtype)
                T.ppl_mul_C(acc_s, acc_s, scale)  # acc_s *= scale
                T.ppl_mul_C(scores_scaled, scores_max, scale)  # scores_scaled = scores_max * scale
                T.ppl_subtract(acc_s, acc_s, scores_scaled)
                work0_1 = T.alloc_shared([VALID_BLOCK_H, block_N], accum_dtype)
                work1_1 = T.alloc_shared([VALID_BLOCK_H, block_N], accum_dtype)
                T.ppl_exp2(acc_s, work0_1, work1_1, coeff, table)

                T.ppl_reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)

                # for i in T.Parallel(VALID_BLOCK_H):
                #     logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.ppl_mul(logsum, logsum, scores_scale)
                T.ppl_add(logsum, logsum, scores_sum)


                # for i, j in T.Parallel(VALID_BLOCK_H, dim):
                #     acc_o[i, j] *= scores_scale[i]
                T.ppl_mul(acc_o, acc_o, scores_scale)

                # 4) output accumulate: O += S @ V
                T.ppl_gemm(S_shared, KV_shared, acc_o)

            # 5) final normalize: O /= logsum
            # for i, j in T.Parallel(VALID_BLOCK_H, dim):
            #     acc_o[i, j] /= logsum[i]
            T.ppl_div(acc_o, acc_o, logsum)

            # 6) store
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :, :])

    return main


func = flashattn(4, 8, 1, 512, 32, 4, 4, 4)
mod = tilelang.lower(func)
