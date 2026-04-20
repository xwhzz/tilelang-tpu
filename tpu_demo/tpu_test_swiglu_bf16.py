import tilelang
import tilelang.language as T
import torch    

def swi_glu(Block_w, Block_c, C, W, dtype="bfloat16", accum_dtype="float32"):
    
    global_shape = (C, W)
    @T.prim_func
    def main_kernel_inner(
        G_in: T.Tensor(global_shape,dtype),
        G_right: T.Tensor(global_shape, dtype),
        G_out: T.Tensor(global_shape,dtype)
    ):
        with T.Kernel(T.ceildiv(C, Block_c), T.ceildiv(W, Block_w), is_cpu=True) as (bx, by):
            # scale = T.float32(1.44269504)  # log2(e)
            
            block_shape = (Block_c, Block_w)

            x_ori = T.alloc_shared(block_shape, dtype)
            right_ori = T.alloc_shared(block_shape, dtype)

            x = T.alloc_shared(block_shape, accum_dtype)
            right = T.alloc_shared(block_shape, accum_dtype)
            # x_neg = T.alloc_shared(block_shape, accum_dtype)
            x_neg_exp = T.alloc_shared(block_shape, accum_dtype)
            ones = T.alloc_shared(block_shape, accum_dtype)
            T.ppl_fill(ones, T.float32(1.0))
            x_neg_exp_1 = T.alloc_shared(block_shape, accum_dtype)
            x_neg_exp_1_div = T.alloc_shared(block_shape, accum_dtype)
            out = T.alloc_shared(block_shape, accum_dtype)
            out_ori = T.alloc_shared(block_shape, dtype)
            # * - +  /
            T.ppl_copy(G_in[bx * Block_c, by * Block_w], x_ori)
            T.ppl_copy(G_right[bx * Block_c, by * Block_w], right_ori)


            T.ppl_copy(x_ori, x)
            T.ppl_copy(right_ori, right)

            T.ppl_mul_C(x_neg_exp, x, T.float32(-1.0))
            # T.ppl_copy(x_neg[:, :], x_neg_exp[:, :])
            work0_1 = T.alloc_shared([Block_c, Block_w], accum_dtype)
            work1_1 = T.alloc_shared([Block_c, Block_w], accum_dtype)
            coeff_1 = T.alloc_shared([64, 32], accum_dtype)  # npu number is 64
            table_1 = T.alloc_shared([64, 192], accum_dtype)  # npu number is 64
            T.ppl_exp2(x_neg_exp, work0_1, work1_1, coeff_1, table_1) # exp(-x)
            T.ppl_add(x_neg_exp_1, x_neg_exp, ones) # exp(-x) + 1
            T.ppl_div(x_neg_exp_1_div, x, x_neg_exp_1) # x / (exp(-x) + 1)
            T.ppl_mul(out, right, x_neg_exp_1_div) # right * x / (exp(-x) + 1)

            T.ppl_copy(out, out_ori)
            T.ppl_copy(out_ori, G_out[bx * Block_c, by * Block_w])

    return main_kernel_inner



kernel = tilelang.compile(swi_glu(32, 32, 64, 64), out_idx=-1, target="tpu") #,pass_config={"disable_storage_rewrite": True})

a = torch.randn(64, 64, dtype=torch.bfloat16)
b = torch.randn(64, 64, dtype=torch.bfloat16)
c = torch.zeros(64, 64, dtype=torch.bfloat16)
res = kernel(a, b, c)
print(res)
print("output:")
print(c)
import torch.nn.functional as F
ref = b * F.silu(a)
print("ref")
print(ref)
diff = ref-c
max_diff = torch.max(diff)
avg_diff = torch.mean(diff)

print(f"\n=== 差异分析 ===")
print(f"最大差异: {max_diff}")
print(f"平均差异: {avg_diff}")
print("check close:")
print(torch.allclose(c, ref, atol=1e-2, rtol=1e-2))

