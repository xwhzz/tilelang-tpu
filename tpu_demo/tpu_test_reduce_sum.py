import tilelang
import tilelang.language as T
import torch


def reduce_sum(M, N, block_M, block_N, dtype="float16", accum_dtype="float16"):
    @T.prim_func
    def main_kernel_inner(
            X: T.Tensor((M, N), dtype),
            Y: T.Tensor((M, 1), accum_dtype),
    ):
        with T.Kernel(1, T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
            # 分配共享内存
            X_shared = T.alloc_shared((block_M, block_N), dtype)
            Y_shared = T.alloc_shared((block_M, 1), accum_dtype)
            # 初始化 Y_shared 为最小值
            T.fill(Y_shared, T.float32(float(0)))
            # 复制输入数据到共享内存
            T.copy(X[by * block_M, 0], X_shared)
            # 执行 reduce_max 操作
            # 参数说明：输入张量，输出张量，沿哪个维度执行(1表示列方向)
            T.reduce_sum(X_shared, Y_shared, 1)
            # 将结果从共享内存复制回全局内存
            T.copy(Y_shared, Y[by * block_M, 0])
    
    return main_kernel_inner


M, N, block_M, block_N = 64, 64, 32, 64
kernel = tilelang.compile(reduce_sum(M, N, block_M, block_N), out_idx=-1, target="tpu")

x = torch.randn(M, N).half()
y = torch.zeros(M, 1).half()

res = kernel(x, y)
y_torch = torch.sum(x, dim=1, keepdim=True)

print("=" * 5, "TileLang-TPU Output", "=" * 5)
print(y)
print("=" * 5, "PyTorch Reference Output", "=" * 5)
print(y_torch)

print("=" * 5, "Comparison:", "=" * 5)
print(f"Max absolute difference: {torch.max(torch.abs(y - y_torch)).item():.6e}")
print(f"Mean absolute difference: {torch.mean(torch.abs(y - y_torch)).item():.6e}")
is_close = torch.allclose(y, y_torch, rtol=1e-2, atol=1e-2)
print(f"Results match (rtol=1e-2, atol=1e-2): {is_close}")
