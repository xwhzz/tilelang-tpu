import tilelang
import tilelang.language as T
import os

T.copy = T.ppl_copy

def get_llama_mlp_forward_kernel(
    batch_seq: int,
    hidden_size: int,
    intermediate_size: int,
    Block_bs: int = 64,
    Block_h: int = 64,
    Block_i: int = 64,
    dtype: str = "bfloat16",
    accum_dtype: str = "float32"
):
    """
    完整的 Llama MLP Forward 算子：
    X -> gate_proj + up_proj -> SiLU -> SwiGLU -> down_proj -> output

    Inputs, aligned with torch.ops.my_ops.llama_mlp_forward:
      X:        (batch_seq, hidden_size)
      up_weight:   (intermediate_size, hidden_size)
      gate_weight: (intermediate_size, hidden_size)
      down_weight: (hidden_size, intermediate_size)
      output:      (batch_seq, hidden_size)
    """
    assert batch_seq % Block_bs == 0, "batch_seq must be divisible by Block_bs"
    assert hidden_size % Block_h == 0, "hidden_size must be divisible by Block_h"
    assert intermediate_size % Block_i == 0, "intermediate_size must be divisible by Block_i"

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
    def llama_mlp_forward(
        x: T.Tensor(x_shape, dtype),
        up_weight: T.Tensor(up_weight_shape, dtype),
        gate_weight: T.Tensor(gate_weight_shape, dtype),
        down_weight: T.Tensor(down_weight_shape, dtype),
        output: T.Tensor(output_shape, dtype)
    ):
        num_blocks_bs = T.ceildiv(batch_seq, Block_bs)
        num_blocks_h = T.ceildiv(hidden_size, Block_h)
        num_blocks_i = T.ceildiv(intermediate_size, Block_i)

        with T.Kernel(num_blocks_bs, num_blocks_h, is_cpu=True) as (bx, by):
            # X 和权重分块缓存
            x_block = T.alloc_shared((Block_bs, Block_h), dtype)
            gate_weight_block = T.alloc_shared((Block_i, Block_h), dtype)
            up_weight_block = T.alloc_shared((Block_i, Block_h), dtype)
            down_weight_block = T.alloc_shared((Block_h, Block_i), dtype)

            gated_up_block = T.alloc_shared((Block_bs, Block_i), dtype)

            # Stage2 输出累加
            down_out_acc = T.alloc_shared((Block_bs, Block_h), "float32")
            down_out_part_fp32 = T.alloc_shared((Block_bs, Block_h), "float32")
            down_out_block = T.alloc_shared((Block_bs, Block_h), dtype)

            # FP32 工作区，用于 SiLU + fusion
            gate_out_block_fp32 = T.alloc_shared((Block_bs, Block_i), "float32")
            up_out_block_fp32 = T.alloc_shared((Block_bs, Block_i), "float32")
            proj_part_fp32 = T.alloc_shared((Block_bs, Block_i), "float32")
            silu_out_block = T.alloc_shared((Block_bs, Block_i), accum_dtype)
            gated_up_fp32 = T.alloc_shared((Block_bs, Block_i), accum_dtype)

            # SiLU 工作区
            x_neg = T.alloc_shared((Block_bs, Block_i), accum_dtype)
            ones = T.alloc_shared((Block_bs, Block_i), accum_dtype)
            x_neg_exp_1 = T.alloc_shared((Block_bs, Block_i), accum_dtype)
            exp_work0 = T.alloc_shared((Block_bs, Block_i), accum_dtype)
            exp_work1 = T.alloc_shared((Block_bs, Block_i), accum_dtype)
            exp_coeff = T.alloc_shared((64, 32), accum_dtype)
            exp_table = T.alloc_shared((64, 192), accum_dtype)

            T.ppl_fill(ones, T.float32(1.0))
            T.ppl_fill(down_out_acc, T.float32(0.0))

            for bz in range(num_blocks_i):
                T.ppl_fill(gate_out_block_fp32, T.float32(0.0))
                T.ppl_fill(up_out_block_fp32, T.float32(0.0))

                for kk in range(T.ceildiv(hidden_size, Block_h)):
                    T.copy(
                        x[bx * Block_bs, kk * Block_h],
                        x_block
                    )
                    T.copy(
                        gate_weight[bz * Block_i, kk * Block_h],
                        gate_weight_block
                    )
                    T.copy(
                        up_weight[bz * Block_i, kk * Block_h],
                        up_weight_block
                    )
                    T.ppl_fill(proj_part_fp32, T.float32(0.0))
                    T.ppl_gemm(x_block, gate_weight_block, proj_part_fp32, transpose_B=True)
                    T.ppl_add(gate_out_block_fp32, gate_out_block_fp32, proj_part_fp32)
                    T.ppl_fill(proj_part_fp32, T.float32(0.0))
                    T.ppl_gemm(x_block, up_weight_block, proj_part_fp32, transpose_B=True)
                    T.ppl_add(up_out_block_fp32, up_out_block_fp32, proj_part_fp32)

                SiLU(
                    gate_out_block_fp32,
                    silu_out_block,
                    x_neg,
                    ones,
                    x_neg_exp_1,
                    exp_work0,
                    exp_work1,
                    exp_coeff,
                    exp_table
                )
                T.ppl_mul(gated_up_fp32, silu_out_block, up_out_block_fp32)
                T.copy(gated_up_fp32, gated_up_block)

                T.copy(
                    down_weight[by * Block_h, bz * Block_i],
                    down_weight_block
                )
                T.ppl_fill(down_out_part_fp32, T.float32(0.0))
                T.ppl_gemm(gated_up_block, down_weight_block, down_out_part_fp32, transpose_B=True)
                T.ppl_add(down_out_acc, down_out_acc, down_out_part_fp32)

            T.copy(down_out_acc, down_out_block)
            T.copy(down_out_block, output[bx * Block_bs, by * Block_h])

    return llama_mlp_forward


if __name__ == "__main__":
    batch_seq = 16
    hidden_size = 16
    intermediate_size = 16
    func = get_llama_mlp_forward_kernel(
        batch_seq, hidden_size, intermediate_size,
        Block_bs=8, Block_h=8, Block_i=8,
        dtype="bfloat16", accum_dtype="float32"
    )
    mod = tilelang.lower(func, target='c')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "llama_mlp_forward_ir.py"), "w") as f:
        f.write(str(func))
    with open(os.path.join(script_dir, "llama_mlp_forward_kernel.c"), "w") as f:
        f.write(str(mod))
