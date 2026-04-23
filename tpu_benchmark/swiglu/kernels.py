"""SwiGLU tilelang kernel builders, parameterized by dtype/shape."""

import tilelang.language as T


def build_tl_swiglu(dtype, C, W, block_C, block_W):
    is_lowp = dtype in ("bfloat16", "float16")
    accum_dtype = "float32"
    global_shape = (C, W)

    if not is_lowp:
        @T.prim_func
        def main_kernel_inner(
            G_in: T.Tensor(global_shape, dtype),
            G_right: T.Tensor(global_shape, dtype),
            G_out: T.Tensor(global_shape, dtype),
        ):
            with T.Kernel(T.ceildiv(C, block_C), T.ceildiv(W, block_W), is_cpu=True) as (bx, by):
                bs = (block_C, block_W)
                x = T.alloc_shared(bs, accum_dtype)
                right = T.alloc_shared(bs, accum_dtype)
                x_neg_exp = T.alloc_shared(bs, accum_dtype)
                ones = T.alloc_shared(bs, accum_dtype)
                T.ppl_fill(ones, T.float32(1.0))
                x_neg_exp_1 = T.alloc_shared(bs, accum_dtype)
                x_neg_exp_1_div = T.alloc_shared(bs, accum_dtype)
                out = T.alloc_shared(bs, accum_dtype)
                T.ppl_copy(G_in[bx * block_C, by * block_W], x)
                T.ppl_copy(G_right[bx * block_C, by * block_W], right)
                T.ppl_mul_C(x_neg_exp, x, T.float32(-1.0))
                w0 = T.alloc_shared([block_C, block_W], accum_dtype)
                w1 = T.alloc_shared([block_C, block_W], accum_dtype)
                coeff = T.alloc_shared([64, 32], accum_dtype)
                table = T.alloc_shared([64, 192], accum_dtype)
                T.ppl_exp2(x_neg_exp, w0, w1, coeff, table)
                T.ppl_add(x_neg_exp_1, x_neg_exp, ones)
                T.ppl_div(x_neg_exp_1_div, x, x_neg_exp_1)
                T.ppl_mul(out, right, x_neg_exp_1_div)
                T.ppl_copy(out, G_out[bx * block_C, by * block_W])
    else:
        @T.prim_func
        def main_kernel_inner(
            G_in: T.Tensor(global_shape, dtype),
            G_right: T.Tensor(global_shape, dtype),
            G_out: T.Tensor(global_shape, dtype),
        ):
            with T.Kernel(T.ceildiv(C, block_C), T.ceildiv(W, block_W), is_cpu=True) as (bx, by):
                bs = (block_C, block_W)
                x_ori = T.alloc_shared(bs, dtype)
                right_ori = T.alloc_shared(bs, dtype)
                x = T.alloc_shared(bs, accum_dtype)
                right = T.alloc_shared(bs, accum_dtype)
                x_neg_exp = T.alloc_shared(bs, accum_dtype)
                ones = T.alloc_shared(bs, accum_dtype)
                T.ppl_fill(ones, T.float32(1.0))
                x_neg_exp_1 = T.alloc_shared(bs, accum_dtype)
                x_neg_exp_1_div = T.alloc_shared(bs, accum_dtype)
                out = T.alloc_shared(bs, accum_dtype)
                out_ori = T.alloc_shared(bs, dtype)
                T.ppl_copy(G_in[bx * block_C, by * block_W], x_ori)
                T.ppl_copy(G_right[bx * block_C, by * block_W], right_ori)
                T.ppl_copy(x_ori, x)
                T.ppl_copy(right_ori, right)
                T.ppl_mul_C(x_neg_exp, x, T.float32(-1.0))
                w0 = T.alloc_shared([block_C, block_W], accum_dtype)
                w1 = T.alloc_shared([block_C, block_W], accum_dtype)
                coeff = T.alloc_shared([64, 32], accum_dtype)
                table = T.alloc_shared([64, 192], accum_dtype)
                T.ppl_exp2(x_neg_exp, w0, w1, coeff, table)
                T.ppl_add(x_neg_exp_1, x_neg_exp, ones)
                T.ppl_div(x_neg_exp_1_div, x, x_neg_exp_1)
                T.ppl_mul(out, right, x_neg_exp_1_div)
                T.ppl_copy(out, out_ori)
                T.ppl_copy(out_ori, G_out[bx * block_C, by * block_W])

    return main_kernel_inner
