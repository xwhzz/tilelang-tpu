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
            block_shape = (Block_c, Block_w)

            x = T.alloc_shared(block_shape, dtype)
            right = T.alloc_shared(block_shape, dtype)

            x_neg_exp_1_div = T.alloc_shared(block_shape, dtype)# x / (exp(-x) + 1)
            
            out = T.alloc_shared(block_shape, dtype)

            T.copy(G_in[bx * Block_c, by * Block_w], x)
            T.copy(G_right[bx * Block_c, by * Block_w], right)

            T.silu(x_neg_exp_1_div, x)
            T.mul(out, right, x_neg_exp_1_div, ) # right * x / (exp(-x) + 1)

            T.copy(out, G_out[bx * Block_c, by * Block_w])

    return main_kernel_inner

func = swi_glu(64, 64, 64, 64)
mod = tilelang.lower(func)

print("\n\n\n",mod)
