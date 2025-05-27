import tilelang
import tilelang.language as T


def test_exp(M, N, block_M, block_N, accum_dtype):

    @T.prim_func
    def main(scores_scale: T.Tensor([M, N], accum_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
            scores_scale_shared = T.alloc_shared([block_M, block_N], accum_dtype)
            T.ppl_copy(scores_scale[by * block_M, bx * block_N], scores_scale_shared)
            T.ppl_exp2(scores_scale_shared, block_M, block_N, accum_dtype)
            T.ppl_copy(scores_scale_shared, scores_scale[by * block_M, bx * block_N])

    return main


accum_dtype = "float32"
func = test_exp(10, 2, 5, 1, accum_dtype)
mod = tilelang.lower(func)
