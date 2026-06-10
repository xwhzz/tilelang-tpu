import tilelang
import tilelang.language as T
import os


def _kernel_source(artifact):
    source = getattr(artifact, "kernel_source", artifact)
    if hasattr(source, "get_source"):
        return source.get_source()
    return str(source)


def deepseek_v3_mlp_bf16(
    Block_bs: int,          # 批次维度分块大小
    Block_h: int,           # 隐藏层维度分块大小
    Block_i: int,           # 中间层维度分块大小
    batch_seq: int,         # 输入序列长度 (batch_size * seq_len)
    hidden_size: int,       # 隐藏层维度
    intermediate_size: int, # 中间层维度
    dtype: str = "bfloat16",
    accum_dtype: str = "float32"
):
    """

      - gate_gemm(bf16) -> gate_out_bf16 -> copy-> gate_out_fp32 -> SiLU(fp32)
      - up_gemm(bf16)   -> up_out_bf16   -> copy-> up_out_fp32
      - gated_up_fp32 = SiLU_fp32 * up_out_fp32
      - gated_up_fp32 -> copy -> gated_up_bf16
      - down_gemm(bf16) using (gated_up_bf16, down_weight_bf16) -> down_out_bf16
      - write down_out_bf16 -> output (bf16)

    """
    assert batch_seq % Block_bs == 0, "batch_seq must be divisible by Block_bs for this version."
    assert hidden_size % Block_h == 0, "hidden_size must be divisible by Block_h for this version."
    assert intermediate_size % Block_i == 0, "intermediate_size must be divisible by Block_i for this version."

    x_shape = (batch_seq, hidden_size)
    gate_weight_shape = (intermediate_size, hidden_size)
    up_weight_shape = (intermediate_size, hidden_size)
    down_weight_shape = (hidden_size, intermediate_size)
    output_shape = (batch_seq, hidden_size)  # 全局输出为 bfloat16

    @T.macro
    def SiLU(
        gate_in: T.Tensor((Block_bs, Block_i), accum_dtype),
        silu_out: T.Tensor((Block_bs, Block_i), accum_dtype)
    ):
        T.silu(silu_out, gate_in)

    @T.prim_func
    def main_kernel_inner(
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
            # ---------- shared buffers (严格区分 bf16 / fp32) ----------
            # 从 global 加载的 bf16 段
            x_block_bf16 = T.alloc_shared((Block_bs, Block_h), dtype)
            gate_weight_block_bf16 = T.alloc_shared((Block_i, Block_h), dtype)
            up_weight_block_bf16 = T.alloc_shared((Block_i, Block_h), dtype)
            down_weight_block_bf16 = T.alloc_shared((Block_h, Block_i), dtype)

            # GEMM 输出（bf16）
            gate_out_block_bf16 = T.alloc_shared((Block_bs, Block_i), dtype)
            up_out_block_bf16 = T.alloc_shared((Block_bs, Block_i), dtype)
            # down gemm 直接产出 bf16
            down_out_block_bf16 = T.alloc_shared((Block_bs, Block_h), dtype)

            # 用于高精度计算（fp32）
            gate_out_block_fp32 = T.alloc_shared((Block_bs, Block_i), accum_dtype)
            up_out_block_fp32 = T.alloc_shared((Block_bs, Block_i), accum_dtype)
            silu_out_block_fp32 = T.alloc_shared((Block_bs, Block_i), accum_dtype)
            gated_up_block_fp32 = T.alloc_shared((Block_bs, Block_i), accum_dtype)

            # 把 gated_up 从 fp32 转为 bf16 以供 down GEMM 使用
            gated_up_block_bf16 = T.alloc_shared((Block_bs, Block_i), dtype)

            # ---------- loads: global -> shared (bf16) ----------
            T.copy(x[bx*Block_bs : (bx+1)*Block_bs, by*Block_h : (by+1)*Block_h], x_block_bf16)
            T.copy(gate_weight[bz*Block_i : (bz+1)*Block_i, by*Block_h : (by+1)*Block_h], gate_weight_block_bf16)
            T.copy(up_weight[bz*Block_i : (bz+1)*Block_i, by*Block_h : (by+1)*Block_h], up_weight_block_bf16)
            T.copy(down_weight[by*Block_h : (by+1)*Block_h, bz*Block_i : (bz+1)*Block_i], down_weight_block_bf16)

            # ---------- GATE_PROJ: bf16 GEMM ----------
            T.gemm(x_block_bf16, gate_weight_block_bf16, gate_out_block_bf16, transpose_B=True)
            # 转 fp32 做 SiLU
            T.copy(gate_out_block_bf16, gate_out_block_fp32)
            SiLU(gate_out_block_fp32, silu_out_block_fp32)

            # ---------- UP_PROJ: bf16 GEMM ----------
            T.gemm(x_block_bf16, up_weight_block_bf16, up_out_block_bf16, transpose_B=True)
            # 转 fp32 做逐元素乘
            T.copy(up_out_block_bf16, up_out_block_fp32)

            # ---------- 元素乘（fp32）: gated_up = SiLU(fp32) * up_out_fp32 ----------
            T.mul(gated_up_block_fp32, silu_out_block_fp32, up_out_block_fp32)

            # ---------- 把 gated_up 从 fp32 转为 bf16（以保持 down GEMM 的 bf16 要求） ----------
            T.copy(gated_up_block_fp32, gated_up_block_bf16)

            # ---------- DOWN_PROJ: bf16 GEMM ----------
            # gated_up_block_bf16: (Block_bs, Block_i)
            # down_weight_block_bf16: (Block_h, Block_i)  -> transpose_B=True to match (Block_i, Block_h)
            T.gemm(gated_up_block_bf16, down_weight_block_bf16, down_out_block_bf16, transpose_B=True)

            # ---------- 写回 global output (bf16 -> global) ----------
            T.copy(down_out_block_bf16, output[bx*Block_bs : (bx+1)*Block_bs, by*Block_h : (by+1)*Block_h])

    return main_kernel_inner

# --- 使用示例 ---
if __name__ == "__main__":
    Block_bs = 8
    Block_h = 8
    Block_i = 8
    batch_seq = 16
    hidden_size = 16
    intermediate_size = 16  # 通常是 hidden_size 的 4 倍
    dtype = "bfloat16"
    accum_dtype = "float32"

    # 生成修复后的 TileLang 函数
    func_fixed = deepseek_v3_mlp_bf16(
        Block_bs=Block_bs, Block_h=Block_h, Block_i=Block_i,
        batch_seq=batch_seq, hidden_size=hidden_size, intermediate_size=intermediate_size,
        dtype=dtype, accum_dtype=accum_dtype
    )

    # 生成 IR 和 C Kernel
    mod_fixed = tilelang.lower(func_fixed, target='c')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mlp_ir_file_fixed = os.path.join(script_dir, "deepseek_v3_mlp_bf16_ir.py")
    mlp_kernel_file_fixed = os.path.join(script_dir, "deepseek_v3_mlp_bf16_kernel.c")

    with open(mlp_ir_file_fixed, "w") as f:
        f.write(str(func_fixed))

    with open(mlp_kernel_file_fixed, "w") as f:
        f.write(_kernel_source(mod_fixed))
