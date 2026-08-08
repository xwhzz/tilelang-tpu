# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def llama_mlp_gptq(
    M: int,
    K: int,
    group_size: int = 128,
    block_M: int = 128,
    block_K: int = 128,
    block_B: int = 1,
    num_stages: int = 2,
    threads: int = 128,
    batch_size: int = 1,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    del threads

    num_bits = 4
    num_elems_per_byte = 8 // num_bits
    packed_m = M // num_elems_per_byte
    packed_k = K // num_elems_per_byte
    block_packed_m = block_M // num_elems_per_byte
    block_packed_k = block_K // num_elems_per_byte

    groups_m = M // group_size
    groups_k = K // group_size
    packed_groups_m = _ceil_div(groups_m, 2)
    packed_groups_k = _ceil_div(groups_k, 2)
    block_groups_m = block_M // group_size
    block_groups_k = block_K // group_size
    block_packed_groups_m = _ceil_div(block_groups_m, 2)
    block_packed_groups_k = _ceil_div(block_groups_k, 2)

    assert M % group_size == 0 and K % group_size == 0
    assert M % block_M == 0 and K % block_K == 0
    assert block_M % group_size == 0 and block_K % group_size == 0
    assert batch_size % block_B == 0

    input_shape = (batch_size, M)
    output_shape = (batch_size, M)

    gate_up_weight_shape = (K, packed_m)
    gate_up_zp_shape = (K, packed_groups_m)
    gate_up_scale_shape = (K, groups_m)
    down_weight_shape = (M, packed_k)
    down_zp_shape = (M, packed_groups_k)
    down_scale_shape = (M, groups_k)

    input_tile_shape = (block_B, block_M)
    gate_up_weight_tile_shape = (block_K, block_packed_m)
    gate_up_zp_tile_shape = (block_K, block_packed_groups_m)
    gate_up_scale_tile_shape = (block_K, block_groups_m)
    gate_up_dequant_tile_shape = (block_K, block_M)
    down_weight_tile_shape = (block_M, block_packed_k)
    down_zp_tile_shape = (block_M, block_packed_groups_k)
    down_scale_tile_shape = (block_M, block_groups_k)

    activation_tile_shape = (block_B, block_K)
    output_acc_tile_shape = (block_B, block_M)

    @T.macro
    def SiLU_Gated(
        gate_in: T.Tensor(activation_tile_shape, accum_dtype),
        up_in: T.Tensor(activation_tile_shape, accum_dtype),
        gated_out: T.Tensor(activation_tile_shape, accum_dtype),
        neg_exp: T.Tensor(activation_tile_shape, accum_dtype),
        denom: T.Tensor(activation_tile_shape, accum_dtype),
        exp_work0: T.Tensor(activation_tile_shape, accum_dtype),
        exp_work1: T.Tensor(activation_tile_shape, accum_dtype),
        exp_coeff: T.Tensor((64, 32), accum_dtype),
        exp_table: T.Tensor((64, 192), accum_dtype),
    ):
        T.ppl_mul_C(neg_exp, gate_in, T.float32(-1.0))
        T.ppl_exp2(neg_exp, exp_work0, exp_work1, exp_coeff, exp_table)
        T.ppl_add_C(denom, neg_exp, T.float32(1.0))
        T.ppl_div(neg_exp, gate_in, denom)
        T.ppl_mul(gated_out, up_in, neg_exp)

    @T.prim_func
    def main_kernel_inner(
        input_tensor: T.Tensor(input_shape, dtype),
        up_w: T.Tensor(gate_up_weight_shape, "uint8"),
        up_zp: T.Tensor(gate_up_zp_shape, "uint8"),
        up_scale: T.Tensor(gate_up_scale_shape, dtype),
        gate_w: T.Tensor(gate_up_weight_shape, "uint8"),
        gate_zp: T.Tensor(gate_up_zp_shape, "uint8"),
        gate_scale: T.Tensor(gate_up_scale_shape, dtype),
        down_w: T.Tensor(down_weight_shape, "uint8"),
        down_zp: T.Tensor(down_zp_shape, "uint8"),
        down_scale: T.Tensor(down_scale_shape, dtype),
        output_tensor: T.Tensor(output_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(batch_size, block_B), is_cpu=True) as (bx, by):
            input_shared = T.alloc_shared(input_tile_shape, dtype)

            # Separate gate/up buffers for pipelining
            gw_shared = T.alloc_shared(gate_up_weight_tile_shape, "uint8")
            gz_shared = T.alloc_shared(gate_up_zp_tile_shape, "uint8")
            gs_shared = T.alloc_shared(gate_up_scale_tile_shape, dtype)
            g_dequant = T.alloc_shared(gate_up_dequant_tile_shape, dtype)
            gw_work = T.alloc_shared((block_K, max(block_groups_m, 2)), dtype)

            uw_shared = T.alloc_shared(gate_up_weight_tile_shape, "uint8")
            uz_shared = T.alloc_shared(gate_up_zp_tile_shape, "uint8")
            us_shared = T.alloc_shared(gate_up_scale_tile_shape, dtype)
            u_dequant = T.alloc_shared(gate_up_dequant_tile_shape, dtype)
            uw_work = T.alloc_shared((block_K, max(block_groups_m, 2)), dtype)

            # Down buffers (outside pipelined loop, can reuse)
            w_shared = T.alloc_shared(down_weight_tile_shape, "uint8")
            zp_shared = T.alloc_shared(down_zp_tile_shape, "uint8")
            scale_shared = T.alloc_shared(down_scale_tile_shape, dtype)
            dequant = T.alloc_shared(gate_up_dequant_tile_shape, dtype)
            dequant_work = T.alloc_shared((block_M, max(block_groups_k, 2)), dtype)

            gate_out = T.alloc_shared(activation_tile_shape, accum_dtype)
            up_out = T.alloc_shared(activation_tile_shape, accum_dtype)
            gate_exp = T.alloc_shared(activation_tile_shape, accum_dtype)
            gate_denom = T.alloc_shared(activation_tile_shape, accum_dtype)
            swiglu_out = T.alloc_shared(activation_tile_shape, accum_dtype)
            work0 = T.alloc_shared(activation_tile_shape, accum_dtype)
            work1 = T.alloc_shared(activation_tile_shape, accum_dtype)
            coeff = T.alloc_shared((64, 32), accum_dtype)
            table = T.alloc_shared((64, 192), accum_dtype)
            output_acc = T.alloc_shared(output_acc_tile_shape, accum_dtype)
            output_cast = T.alloc_shared(output_acc_tile_shape, dtype)
            gemm_temp = T.alloc_shared(activation_tile_shape, accum_dtype)

            T.ppl_fill(output_acc, T.float32(0.0))

            for ko in range(T.ceildiv(K, block_K)):
                T.ppl_fill(gate_out, T.float32(0.0))
                T.ppl_fill(up_out, T.float32(0.0))

                # Pipelined inner loop
                for ki in T.Pipelined(T.ceildiv(M, block_M), num_stages=num_stages):
                    # Load input (same for gate and up)
                    T.ppl_copy(input_tensor[by * block_B, ki * block_M], input_shared)

                    # Gate: load + dequant + gemm
                    T.ppl_copy(gate_w[ko * block_K, ki * block_packed_m], gw_shared)
                    T.ppl_copy(gate_zp[ko * block_K, (ki * block_groups_m) // 2], gz_shared)
                    T.ppl_copy(gate_scale[ko * block_K, ki * block_groups_m], gs_shared)
                    T.ppl_dequant_gptq(g_dequant, gw_shared, gz_shared, gs_shared,
                                       gw_work, group_size=128, weight_bits=4,
                                       group_offset=ki * block_groups_m)
                    T.ppl_gemm(input_shared, g_dequant, gemm_temp, transpose_B=True)
                    T.ppl_add(gate_out, gate_out, gemm_temp)

                    # Up: load + dequant + gemm
                    T.ppl_copy(up_w[ko * block_K, ki * block_packed_m], uw_shared)
                    T.ppl_copy(up_zp[ko * block_K, (ki * block_groups_m) // 2], uz_shared)
                    T.ppl_copy(up_scale[ko * block_K, ki * block_groups_m], us_shared)
                    T.ppl_dequant_gptq(u_dequant, uw_shared, uz_shared, us_shared,
                                       uw_work, group_size=128, weight_bits=4,
                                       group_offset=ki * block_groups_m)
                    T.ppl_gemm(input_shared, u_dequant, gemm_temp, transpose_B=True)
                    T.ppl_add(up_out, up_out, gemm_temp)

                SiLU_Gated(gate_out, up_out, swiglu_out, gate_exp, gate_denom,
                           work0, work1, coeff, table)

                # Down path (outside pipelined loop)
                T.ppl_copy(down_w[bx * block_M, ko * block_packed_k], w_shared)
                T.ppl_copy(down_zp[bx * block_M, (ko * block_groups_k) // 2], zp_shared)
                T.ppl_copy(down_scale[bx * block_M, ko * block_groups_k], scale_shared)
                T.ppl_dequant_gptq(dequant, w_shared, zp_shared, scale_shared,
                                   dequant_work, group_size=128, weight_bits=4,
                                   group_offset=ko * block_groups_k)
                T.ppl_copy(swiglu_out, input_shared)
                T.ppl_gemm(input_shared, dequant, gemm_temp, transpose_B=True)
                T.ppl_add(output_acc, output_acc, gemm_temp)

            T.ppl_copy(output_acc, output_cast)
            T.ppl_copy(output_cast, output_tensor[by * block_B, bx * block_M])

    return main_kernel_inner


def reference_llama_mlp_gptq(
    input_tensor, gate_w, gate_zp, gate_scale,
    up_w, up_zp, up_scale, down_w, down_zp, down_scale,
    group_size: int = 128,
):
    import torch
    import torch.nn.functional as F

    def dequantize(packed_w, packed_zp_uint8, scale, groups):
        rows, packed_cols = packed_w.shape
        cols = packed_cols * 2
        weights = torch.empty((rows, cols), dtype=torch.float16, device=packed_w.device)
        for col in range(cols):
            weights[:, col] = ((packed_w[:, col // 2] >> (4 * (col % 2))) & 0xF).to(torch.float16)
        num_groups = scale.shape[1]
        zp = torch.empty((rows, num_groups), dtype=torch.float16, device=packed_zp_uint8.device)
        for g in range(num_groups):
            zp[:, g] = ((packed_zp_uint8[:, g // 2] >> (4 * (g % 2))) & 0xF).to(torch.float16)
        group_size_val = cols // num_groups
        zp_expanded = torch.empty((rows, cols), dtype=torch.float16, device=packed_w.device)
        scale_expanded = torch.empty((rows, cols), dtype=torch.float16, device=packed_w.device)
        for g in range(num_groups):
            start = g * group_size_val
            end = start + group_size_val
            zp_expanded[:, start:end] = zp[:, g:g+1]
            scale_expanded[:, start:end] = scale[:, g:g+1]
        return (weights - zp_expanded) * scale_expanded

    gate = dequantize(gate_w, gate_zp, gate_scale, groups=gate_scale.shape[1])
    up = dequantize(up_w, up_zp, up_scale, groups=up_scale.shape[1])
    down = dequantize(down_w, down_zp, down_scale, groups=down_scale.shape[1])
    gate_proj = F.linear(input_tensor, gate)
    up_proj = F.linear(input_tensor, up)
    return F.linear(F.silu(gate_proj) * up_proj, down)


if __name__ == "__main__":
    kernel = llama_mlp_gptq(M=4096, K=11008, group_size=128, block_M=128, block_K=128, batch_size=1, num_stages=2)
    print(tilelang.lower(kernel))
