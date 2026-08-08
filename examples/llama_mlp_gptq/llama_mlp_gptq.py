# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""llama_mlp_gptq with tensor-wrapped constants (following xwhzz's approach).

This version wraps group_size and weight_bits as T.Tensor((1,), "int64")
parameters so they survive the TIR pipeline (all params are tensors → all
params have buffer_map entries → no params stripped by Simplify pass).

The actual group_size/weight_bits values used in computation are still
compile-time Python constants captured by closure (follows tilelang-tpu
convention). The tensor wrappers exist purely for ABI compatibility.
"""

import tilelang
import tilelang.language as T


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def llama_mlp_gptq(
    M: int,
    K: int,
    group_size: int = 128,
    weight_bits: int = 4,
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

    assert weight_bits == 4, "Current kernel only supports GPTQ 4-bit weights"
    assert M % group_size == 0, "M must be divisible by group_size"
    assert K % group_size == 0, "K must be divisible by group_size"
    assert M % block_M == 0, "M must be divisible by block_M"
    assert K % block_K == 0, "K must be divisible by block_K"
    assert block_M % group_size == 0, "block_M must align to group_size"
    assert block_K % group_size == 0, "block_K must align to group_size"
    assert batch_size % block_B == 0, "batch_size must be divisible by block_B"

    num_elems_per_byte = 8 // weight_bits
    packed_m = M // num_elems_per_byte
    packed_k = K // num_elems_per_byte
    groups_m = M // group_size
    groups_k = K // group_size
    packed_groups_m = _ceil_div(groups_m, num_elems_per_byte)
    packed_groups_k = _ceil_div(groups_k, num_elems_per_byte)
    block_groups_m = block_M // group_size
    block_groups_k = block_K // group_size
    block_packed_m = block_M // num_elems_per_byte
    block_packed_k = block_K // num_elems_per_byte
    block_packed_groups_m = _ceil_div(block_groups_m, num_elems_per_byte)
    block_packed_groups_k = _ceil_div(block_groups_k, num_elems_per_byte)

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
    down_dequant_tile_shape = (block_M, block_K)

    activation_tile_shape = (block_B, block_K)
    output_acc_tile_shape = (block_B, block_M)

    scalar_shape = (1,)  # shape for tensor-wrapped scalars

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
        group_size_tensor: T.Tensor(scalar_shape, "int32"),   # ← tensor wrapper
        weight_bits_tensor: T.Tensor(scalar_shape, "int32"),  # ← tensor wrapper
        output_tensor: T.Tensor(output_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(batch_size, block_B), is_cpu=True) as (bx, by):
            # --- shared memory allocations (same as original) ---
            input_shared = T.alloc_shared(input_tile_shape, dtype)
            w_shared = T.alloc_shared(gate_up_weight_tile_shape, "uint8")
            zp_shared = T.alloc_shared(gate_up_zp_tile_shape, "uint8")
            scale_shared = T.alloc_shared(gate_up_scale_tile_shape, dtype)
            dequant = T.alloc_shared(gate_up_dequant_tile_shape, dtype)
            dequant_work = T.alloc_shared((block_K, block_groups_m), dtype)

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
            T.ppl_fill(output_acc, T.float32(0.0))

            for ko in range(T.ceildiv(K, block_K)):
                T.ppl_fill(gate_out, T.float32(0.0))
                T.ppl_fill(up_out, T.float32(0.0))

                for ki in range(T.ceildiv(M, block_M)):
                    T.ppl_copy(input_tensor[by * block_B, ki * block_M], input_shared)

                    T.ppl_copy(gate_w[ko * block_K, ki * block_packed_m], w_shared)
                    T.ppl_copy(gate_zp[ko * block_K, ki * block_packed_groups_m], zp_shared)
                    T.ppl_copy(gate_scale[ko * block_K, ki * block_groups_m], scale_shared)

                    T.ppl_dequant_gptq(dequant, w_shared, zp_shared, scale_shared,
                                       dequant_work, group_size=group_size, weight_bits=weight_bits)
                    T.ppl_gemm(input_shared, dequant, gate_out, transpose_B=True)

                    T.ppl_copy(up_w[ko * block_K, ki * block_packed_m], w_shared)
                    T.ppl_copy(up_zp[ko * block_K, ki * block_packed_groups_m], zp_shared)
                    T.ppl_copy(up_scale[ko * block_K, ki * block_groups_m], scale_shared)

                    T.ppl_dequant_gptq(dequant, w_shared, zp_shared, scale_shared,
                                       dequant_work, group_size=group_size, weight_bits=weight_bits)
                    T.ppl_gemm(input_shared, dequant, up_out, transpose_B=True)

                SiLU_Gated(gate_out, up_out, swiglu_out, gate_exp, gate_denom,
                           work0, work1, coeff, table)

                T.ppl_copy(down_w[bx * block_M, ko * block_packed_k], w_shared)
                T.ppl_copy(down_zp[bx * block_M, ko * block_packed_groups_k], zp_shared)
                T.ppl_copy(down_scale[bx * block_M, ko * block_groups_k], scale_shared)

                T.ppl_dequant_gptq(dequant, w_shared, zp_shared, scale_shared,
                                   dequant_work, group_size=group_size, weight_bits=weight_bits)
                T.ppl_copy(swiglu_out, input_shared)
                T.ppl_gemm(input_shared, dequant, output_acc, transpose_B=True)

            T.ppl_copy(output_acc, output_cast)
            T.ppl_copy(output_cast, output_tensor[by * block_B, bx * block_M])

    return main_kernel_inner


def reference_llama_mlp_gptq(
    input_tensor,
    up_w, up_zp, up_scale,
    gate_w, gate_zp, gate_scale,
    down_w, down_zp, down_scale,
    group_size: int = 128,
):
    import torch
    import torch.nn.functional as F

    def dequantize(packed_w, packed_zp_uint8, scale):
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
            zp_expanded[:, start:end] = zp[:, g:g + 1]
            scale_expanded[:, start:end] = scale[:, g:g + 1]

        return (weights - zp_expanded) * scale_expanded

    gate = dequantize(gate_w, gate_zp, gate_scale)
    up = dequantize(up_w, up_zp, up_scale)
    down = dequantize(down_w, down_zp, down_scale)
    gate_proj = F.linear(input_tensor, gate)
    up_proj = F.linear(input_tensor, up)
    swiglu = (F.silu(gate_proj.float()) * up_proj.float()).to(torch.float16)
    return F.linear(swiglu, down)


if __name__ == "__main__":
    kernel = llama_mlp_gptq(
        M=4096,
        K=11008,
        group_size=128,
        weight_bits=4,
        block_M=128,
        block_K=128,
        block_B=1,
        num_stages=2,
        batch_size=1,
    )
    print(tilelang.lower(kernel))
