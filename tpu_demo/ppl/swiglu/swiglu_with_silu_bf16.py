import tilelang
import tilelang.language as T

def swi_glu(Block_w, Block_c, C, W, dtype="bfloat16", accum_dtype="float32"):
    
    global_shape = (C, W)
    @T.prim_func
    def main(
        G_out: T.Tensor(global_shape,dtype),
        G_in: T.Tensor(global_shape,dtype),
        G_right: T.Tensor(global_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(C, Block_c), T.ceildiv(W, Block_w), is_cpu=True) as (bx, by):
            block_shape = (Block_c, Block_w)

            x = T.alloc_shared(block_shape, dtype)
            right = T.alloc_shared(block_shape, dtype)


            x_neg_exp_1_div_work0 = T.alloc_shared(block_shape, dtype)
            x_neg_exp_1_div_work1 = T.alloc_shared(block_shape, dtype)
            x_neg_exp_1_div_coeff = T.alloc_shared((64, 32), dtype)
            x_neg_exp_1_div_table = T.alloc_shared((64, 192), dtype)
            x_neg_exp_1_div = T.alloc_shared(block_shape, dtype)# x / (exp(-x) + 1)
            
            out = T.alloc_shared(block_shape, dtype)

            T.ppl_copy(G_in[bx * Block_c, by * Block_w], x)
            T.ppl_copy(G_right[bx * Block_c, by * Block_w], right)

            T.ppl_silu(x_neg_exp_1_div, x, x_neg_exp_1_div_work0, x_neg_exp_1_div_work1, x_neg_exp_1_div_coeff, x_neg_exp_1_div_table)
            T.ppl_mul(out, right, x_neg_exp_1_div, ) # right * x / (exp(-x) + 1)

            T.ppl_copy(out, G_out[bx * Block_c, by * Block_w])

    return main

func =  swi_glu(64, 64, 1, 32)
mod = tilelang.lower(func)

print("\n\n\n",mod)