import tilelang
import tilelang.language as T

import tilelang
import tilelang.language as T

def swi_glu(Block_w, Block_c, C, W, dtype="float32", accum_dtype="float32"):
    
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

            x = T.alloc_shared(block_shape, accum_dtype)
            right = T.alloc_shared(block_shape, accum_dtype)
            # x_neg = T.alloc_shared(block_shape, accum_dtype)
            x_neg_exp = T.alloc_shared(block_shape, accum_dtype)
            ones = T.alloc_shared(block_shape, accum_dtype)
            T.fill(ones, T.float32(1.0))
            x_neg_exp_1 = T.alloc_shared(block_shape, accum_dtype)
            x_neg_exp_1_div = T.alloc_shared(block_shape, accum_dtype)
            out = T.alloc_shared(block_shape, accum_dtype)
            # * - +  /
            T.copy(G_in[bx * Block_c, by * Block_w], x)
            T.copy(G_right[bx * Block_c, by * Block_w], right)

            T.mul_C(x_neg_exp, x, T.float32(-1.0))
            # T.copy(x_neg[:, :], x_neg_exp[:, :])
            T.exp(x_neg_exp) # exp(-x)
            # T.copy(x_neg, x_neg_exp)
            T.add(x_neg_exp_1, x_neg_exp, ones) # exp(-x) + 1
            T.div(x_neg_exp_1_div, x, x_neg_exp_1) # x / (exp(-x) + 1)
            T.mul(out, right, x_neg_exp_1_div) # right * x / (exp(-x) + 1)

            T.copy(out, G_out[bx * Block_c, by * Block_w])

    return main_kernel_inner

func =  swi_glu(32, 32, 64, 64)
mod = tilelang.lower(func)

print("\n\n\n",mod)