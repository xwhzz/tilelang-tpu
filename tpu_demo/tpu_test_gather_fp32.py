# 依赖 tpu_hau_line_gather (DDR→DDR)

import tilelang
import tilelang.language as T
import torch


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

kernel = tilelang.compile(
    gather_kernel(NUM_ROWS, HIDDEN_SIZE, NUM_INDICES),
    out_idx=-1,
    target="tpu",
)

param  = torch.randn(NUM_ROWS, HIDDEN_SIZE).float()
idx_vals = torch.randint(0, NUM_ROWS, (NUM_INDICES, 1), dtype=torch.int32)
index    = idx_vals.view(torch.uint32)   # reinterpret as uint32
output   = torch.zeros(NUM_INDICES, HIDDEN_SIZE).float()

res = kernel(param, index, output)

ref = param[idx_vals.view(-1)]   # shape (NUM_INDICES, HIDDEN_SIZE)


diff     = (output - ref).abs()
max_diff = diff.max().item()
avg_diff = diff.mean().item()

print(f"output[:2]:\n{output[:2]}")
print(f"ref[:2]:\n{ref[:2]}")
print(f"\n=== 差异分析 ===")
print(f"最大差异: {max_diff:.6f}")
print(f"平均差异: {avg_diff:.6f}")
print(f"allclose (atol=1e-5): {torch.allclose(output, ref, atol=1e-5, rtol=1e-5)}")
