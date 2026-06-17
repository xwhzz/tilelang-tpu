import tilelang
import tilelang.language as T

M = T.symbolic("M")
N = 128

@T.prim_func
def main_kernel_inner(
    X: T.Tensor((M, N), "float16"),
    Y: T.Tensor((M, N), "float16"),
):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        local = T.alloc_shared((1, N), "float16")
        T.ppl_copy(X[0, 0], local)
        T.ppl_copy(local, Y[0, 0])

mod = tilelang.lower(main_kernel_inner)
print("=== C source ===")
print(mod.kernel_source)
