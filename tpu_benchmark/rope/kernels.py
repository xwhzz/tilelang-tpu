"""RoPE tilelang kernel builders, parameterized by dtype/shape."""

import tilelang.language as T


def build_tl_rope(dtype, C, W, block_C, block_W):
    accum_dtype = dtype
    global_shape = (C, W)

    @T.prim_func
    def main_kernel_inner(
        G_in: T.Tensor(global_shape, dtype),
        G_cos: T.Tensor(global_shape, dtype),
        G_sin: T.Tensor(global_shape, dtype),
        G_out: T.Tensor(global_shape, accum_dtype),
    ):
        with T.Kernel(T.ceildiv(C, block_C), T.ceildiv(W, block_W), is_cpu=True) as (bx, by):
            bs = (block_C, block_W)
            in_x = T.alloc_shared(bs, dtype)
            in_cos = T.alloc_shared(bs, dtype)
            in_sin = T.alloc_shared(bs, dtype)
            T.ppl_copy(G_in[bx * block_C, by * block_W], in_x)
            T.ppl_copy(G_cos[bx * block_C, by * block_W], in_cos)
            T.ppl_copy(G_sin[bx * block_C, by * block_W], in_sin)
            x_cos = T.alloc_shared(bs, dtype)
            T.ppl_mul(x_cos, in_x, in_cos)
            x_sin = T.alloc_shared(bs, dtype)
            T.ppl_mul(x_sin, in_x, in_sin)
            x_neg = T.alloc_shared(bs, dtype)
            T.ppl_mul_C(x_neg, in_x, T.float16(-1.0))
            x_neg_sin = T.alloc_shared(bs, dtype)
            T.ppl_mul(x_neg_sin, x_neg, in_sin)
            out = T.alloc_shared(bs, accum_dtype)
            T.ppl_rope_add(out, x_cos, x_neg_sin, x_cos, x_sin)
            T.ppl_copy(out, G_out[bx * block_C, by * block_W])

    return main_kernel_inner
