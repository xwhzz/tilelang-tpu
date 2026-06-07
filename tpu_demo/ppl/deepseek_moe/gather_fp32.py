import tilelang
import tilelang.language as T


def gather_kernel(num_rows, hidden_size, num_indices, dtype="float32"):

    @T.prim_func
    def main_kernel_inner(
        Param: T.Tensor((num_rows, hidden_size), dtype),
        Index: T.Tensor((num_indices, 1), "uint32"),
        Output: T.Tensor((num_indices, hidden_size), dtype),
    ):
        with T.Kernel(1, 1, is_cpu=True) as (bx, by):
            T.ppl_gather(Output, Param, Index, num_rows)

    return main_kernel_inner


NUM_ROWS    = 128
HIDDEN_SIZE = 64
NUM_INDICES = 8

func = gather_kernel(NUM_ROWS, HIDDEN_SIZE, NUM_INDICES)
mod = tilelang.lower(func)

print("\n\n\n")
print(mod.kernel_source)
