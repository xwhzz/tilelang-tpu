import tilelang
import tilelang.language as T


def swi_glu(Block_w, Block_c, C, W, dtype="bfloat16", accum_dtype="float32"):
    
    global_shape = (C, W)
    @T.prim_func
    def main_kernel_inner(
        G_out: T.Tensor(global_shape,dtype),
        G_in: T.Tensor(global_shape,dtype),
        G_right: T.Tensor(global_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(C, Block_c), T.ceildiv(W, Block_w), is_cpu=True) as (bx, by):
            # scale = T.float32(1.44269504)  # log2(e)
            
            block_shape = (Block_c, Block_w)

            x_ori = T.alloc_shared(block_shape, dtype)
            right_ori = T.alloc_shared(block_shape, dtype)

            x = T.alloc_shared(block_shape, accum_dtype)
            right = T.alloc_shared(block_shape, accum_dtype)
            x_neg_exp = T.alloc_shared(block_shape, accum_dtype)
            ones = T.alloc_shared(block_shape, accum_dtype)
            T.fill(ones, T.float32(1.0))
            x_neg_exp_1 = T.alloc_shared(block_shape, accum_dtype)
            x_neg_exp_1_div = T.alloc_shared(block_shape, accum_dtype)
            out = T.alloc_shared(block_shape, accum_dtype)
            out_ori = T.alloc_shared(block_shape, dtype)
            # * - +  /
            T.copy(G_in[bx * Block_c, by * Block_w], x_ori)
            T.copy(G_right[bx * Block_c, by * Block_w], right_ori)


            T.copy(x_ori, x)
            T.copy(right_ori, right)

            T.mul_C(x_neg_exp, x, T.float32(-1.0))
            T.exp(x_neg_exp) # exp(-x)
            T.add(x_neg_exp_1, x_neg_exp, ones) # exp(-x) + 1
            T.div(x_neg_exp_1_div, x, x_neg_exp_1) # x / (exp(-x) + 1)
            T.mul(out, right, x_neg_exp_1_div) # right * x / (exp(-x) + 1)

            T.copy(out, out_ori)
            T.copy(out_ori, G_out[bx * Block_c, by * Block_w])

    return main_kernel_inner

func =  swi_glu(64, 64, 64, 64)
mod = tilelang.lower(func)

print("\n\n\n",mod)
