import tilelang
import tilelang.language as T
import os

T.copy = T.ppl_copy

def deepseek_v3_mlp_fp32(
    Block_bs: int,
    Block_h: int,
    Block_i: int,
    batch_seq: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: str = "float32",        # 全局张量类型
    accum_dtype: str = "float32"   # 内部计算类型
):
    """
    内部 FP32，GEMM 输入 BF16。
    """
    assert batch_seq % Block_bs == 0
    assert hidden_size % Block_h == 0
    assert intermediate_size % Block_i == 0

    x_shape = (batch_seq, hidden_size)
    gate_weight_shape = (intermediate_size, hidden_size)
    up_weight_shape = (intermediate_size, hidden_size)
    down_weight_shape = (hidden_size, intermediate_size)
    output_shape = (batch_seq, hidden_size)

    @T.macro
    def SiLU(
        gate_in: T.Tensor((Block_bs, Block_i), accum_dtype),
        silu_out: T.Tensor((Block_bs, Block_i), accum_dtype),
        x_neg: T.Tensor((Block_bs, Block_i), accum_dtype),
        ones: T.Tensor((Block_bs, Block_i), accum_dtype),
        x_neg_exp_1: T.Tensor((Block_bs, Block_i), accum_dtype),
        work0: T.Tensor((Block_bs, Block_i), accum_dtype),
        work1: T.Tensor((Block_bs, Block_i), accum_dtype),
        coeff: T.Tensor((64, 32), accum_dtype),
        table: T.Tensor((64, 192), accum_dtype)
    ):
        T.ppl_mul_C(x_neg, gate_in, T.float32(-1.0))
        T.ppl_exp2(x_neg, work0, work1, coeff, table)
        T.ppl_add(x_neg_exp_1, x_neg, ones)
        T.ppl_div(silu_out, gate_in, x_neg_exp_1)

    @T.prim_func
    def main(
        output: T.Tensor(output_shape, dtype),
        x: T.Tensor(x_shape, dtype),
        gate_weight: T.Tensor(gate_weight_shape, dtype),
        up_weight: T.Tensor(up_weight_shape, dtype),
        down_weight: T.Tensor(down_weight_shape, dtype)
    ):
        num_blocks_bs = T.ceildiv(batch_seq, Block_bs)
        num_blocks_h = T.ceildiv(hidden_size, Block_h)
        num_blocks_i = T.ceildiv(intermediate_size, Block_i)

        with T.Kernel(num_blocks_bs, num_blocks_h, num_blocks_i, is_cpu=True) as (bx, by, bz):
            # -------------------- FP32 buffers --------------------
            x_block = T.alloc_shared((Block_bs, Block_h), accum_dtype)
            gate_weight_block = T.alloc_shared((Block_i, Block_h), accum_dtype)
            up_weight_block = T.alloc_shared((Block_i, Block_h), accum_dtype)
            down_weight_block = T.alloc_shared((Block_h, Block_i), accum_dtype)

            gate_out_block = T.alloc_shared((Block_bs, Block_i), accum_dtype)
            up_out_block = T.alloc_shared((Block_bs, Block_i), accum_dtype)
            down_out_block = T.alloc_shared((Block_bs, Block_h), accum_dtype)

            silu_out_block = T.alloc_shared((Block_bs, Block_i), accum_dtype)
            gated_up_block = T.alloc_shared((Block_bs, Block_i), accum_dtype)

            # SiLU 工作区
            x_neg = T.alloc_shared((Block_bs, Block_i), accum_dtype)
            ones = T.alloc_shared((Block_bs, Block_i), accum_dtype)
            x_neg_exp_1 = T.alloc_shared((Block_bs, Block_i), accum_dtype)
            exp_work0 = T.alloc_shared((Block_bs, Block_i), accum_dtype)
            exp_work1 = T.alloc_shared((Block_bs, Block_i), accum_dtype)
            exp_coeff = T.alloc_shared((64, 32), accum_dtype)
            exp_table = T.alloc_shared((64, 192), accum_dtype)
            T.ppl_fill(ones, T.float32(1.0))

            # load global -> local (FP32)
            T.copy(x[bx*Block_bs:(bx+1)*Block_bs, by*Block_h:(by+1)*Block_h], x_block)
            T.copy(gate_weight[bz*Block_i:(bz+1)*Block_i, by*Block_h:(by+1)*Block_h], gate_weight_block)
            T.copy(up_weight[bz*Block_i:(bz+1)*Block_i, by*Block_h:(by+1)*Block_h], up_weight_block)
            T.copy(down_weight[by*Block_h:(by+1)*Block_h, bz*Block_i:(bz+1)*Block_i], down_weight_block)

            # -------------------- GEMM buffers (BF16) --------------------
            x_block_bf16 = T.alloc_shared((Block_bs, Block_h), "bfloat16")
            gate_weight_block_bf16 = T.alloc_shared((Block_i, Block_h), "bfloat16")
            up_weight_block_bf16 = T.alloc_shared((Block_i, Block_h), "bfloat16")
            down_weight_block_bf16 = T.alloc_shared((Block_h, Block_i), "bfloat16")
            gate_out_block_bf16 = T.alloc_shared((Block_bs, Block_i), "bfloat16")
            up_out_block_bf16 = T.alloc_shared((Block_bs, Block_i), "bfloat16")
            gated_up_block_bf16 = T.alloc_shared((Block_bs, Block_i), "bfloat16")
            down_out_block_bf16 = T.alloc_shared((Block_bs, Block_h), "bfloat16")

            # FP32 -> BF16
            T.copy(x_block, x_block_bf16)
            T.copy(gate_weight_block, gate_weight_block_bf16)
            T.copy(up_weight_block, up_weight_block_bf16)
            T.copy(down_weight_block, down_weight_block_bf16)

            # ---------- GATE GEMM ----------
            T.ppl_gemm(x_block_bf16, gate_weight_block_bf16, gate_out_block_bf16, transpose_B=True)
            T.copy(gate_out_block_bf16, gate_out_block)  # BF16 -> FP32
            SiLU(gate_out_block, silu_out_block, x_neg, ones, x_neg_exp_1, exp_work0, exp_work1, exp_coeff, exp_table)

            # ---------- UP GEMM ----------
            T.ppl_gemm(x_block_bf16, up_weight_block_bf16, up_out_block_bf16, transpose_B=True)
            T.copy(up_out_block_bf16, up_out_block)
            T.ppl_mul(gated_up_block, silu_out_block, up_out_block)
            T.copy(gated_up_block, gated_up_block_bf16)  # FP32 -> BF16

            # ---------- DOWN GEMM ----------
            T.ppl_gemm(gated_up_block_bf16, down_weight_block_bf16, down_out_block_bf16, transpose_B=True)
            T.copy(down_out_block_bf16, down_out_block)

            # 写回 global output
            T.copy(down_out_block, output[bx*Block_bs:(bx+1)*Block_bs, by*Block_h:(by+1)*Block_h])

    return main


if __name__ == "__main__":
    Block_bs = 8
    Block_h = 8
    Block_i = 8
    batch_seq = 16
    hidden_size = 16
    intermediate_size = 16
    dtype = "float32"
    accum_dtype = "float32"

    func_fixed = deepseek_v3_mlp_fp32(
        Block_bs, Block_h, Block_i,
        batch_seq, hidden_size, intermediate_size,
        dtype=dtype, accum_dtype=accum_dtype
    )

    mod_fixed = tilelang.lower(func_fixed, target='c')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mlp_ir_file_fixed = os.path.join(script_dir, "deepseek_v3_mlp_fp32_ir.py")
    mlp_kernel_file_fixed = os.path.join(script_dir, "deepseek_v3_mlp_fp32_kernel.c")

    with open(mlp_ir_file_fixed, "w") as f:
        f.write(str(func_fixed))
    with open(mlp_kernel_file_fixed, "w") as f:
        f.write(mod_fixed)

