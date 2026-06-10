import tilelang
import tilelang.language as T
import torch


def topk_kernel(length, K, descended=True, dtype="float32"):
    descended_int = 1 if descended else 0

    @T.prim_func
    def main_kernel_inner(
        Input: T.Tensor((length,), dtype),
        Output: T.Tensor((length,), dtype),     # HAU 写入 length 个；前 K 个是 top-K value
        Indices: T.Tensor((length,), "int32"),  # HAU 写入 length 个；前 K 个是 top-K index
    ):
        with T.Kernel(1, 1, is_cpu=True) as (bx, by):
            T.topk(Output, Indices, Input, K, descended_int, length)

    return main_kernel_inner


LENGTH = 1024
K      = 8

kernel = tilelang.compile(
    topk_kernel(LENGTH, K, descended=True),
    out_idx=[1, 2],
    target="tpu",
)

src     = torch.randn(LENGTH).float()
output  = torch.zeros(LENGTH).float()
indices = torch.zeros(LENGTH, dtype=torch.int32)
res = kernel(src, output, indices)

# 取前 K 个有效结果
topk_vals  = output[:K]
topk_idx   = indices[:K]

ref_vals, ref_idx = torch.topk(src, K, largest=True, sorted=True)

val_diff = (topk_vals - ref_vals).abs()
max_diff = val_diff.max().item()
avg_diff = val_diff.mean().item()
print(f"top-{K} values  (TPU): {topk_vals.tolist()}")
print(f"top-{K} values  (ref): {ref_vals.tolist()}")
print(f"top-{K} indices (TPU): {topk_idx.tolist()}")
print(f"top-{K} indices (ref): {ref_idx.tolist()}")
print(f"\n=== 差异分析 ===")
print(f"value 最大差异: {max_diff:.6f}")
print(f"value 平均差异: {avg_diff:.6f}")

tpu_vals_via_idx = src[topk_idx.long()]
print(f"index 反查值 allclose: {torch.allclose(tpu_vals_via_idx, ref_vals, atol=1e-5)}")
print(f"value allclose (atol=1e-5): {torch.allclose(topk_vals, ref_vals, atol=1e-5)}")
