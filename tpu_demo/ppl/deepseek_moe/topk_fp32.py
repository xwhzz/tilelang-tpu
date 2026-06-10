import tilelang
import tilelang.language as T


def kernel(length, K, descended=True, dtype="float32"):
    descended_int = 1 if descended else 0

    @T.prim_func
    def main_kernel_inner(
        Input: T.Tensor((length,), dtype),
        Output: T.Tensor((length,), dtype),
        Indices: T.Tensor((length,), "int32"),
    ):
        with T.Kernel(1, 1, is_cpu=True) as (bx, by):
            T.topk(Output, Indices, Input, K, descended_int, length)

    return main_kernel_inner


func = kernel(length=16, K=4, descended=True, dtype="float32")
artifact = tilelang.lower(func)
print("\n\n\n")
print(artifact.kernel_source)
