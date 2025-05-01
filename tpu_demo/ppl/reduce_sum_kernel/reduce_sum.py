import tilelang
import tilelang.language as T
import os

def reduce_sum(M, N, block_M, block_N, dtype="float32", accum_dtype="float32"):
    @T.prim_func
    def main(
            X: T.Tensor((M, N), dtype),
            Y: T.Tensor((M, 1), accum_dtype),
    ):
        with T.Kernel(1, T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
            # 分配共享内存
            X_shared = T.alloc_shared((block_M, block_N), dtype)
            Y_shared = T.alloc_shared((block_M, 1), accum_dtype)
            # 初始化 Y_shared 为最小值
            T.ppl_fill(Y_shared, T.float32(float(0)))
            # 复制输入数据到共享内存
            T.ppl_copy(X[by * block_M, 0], X_shared)
            # 执行 reduce_max 操作
            # 参数说明：输入张量，输出张量，沿哪个维度执行(1表示列方向)
            T.ppl_reduce_sum(X_shared, Y_shared, 1)
            # 将结果从共享内存复制回全局内存
            T.ppl_copy(Y_shared, Y[by * block_M, 0])
    
    return main

# 使用示例配置测试函数
func = reduce_sum(8192, 8192, 512, 8192)
mod = tilelang.lower(func)
script_dir = os.path.dirname(os.path.abspath(__file__))
reduce_sum_ir = os.path.join(script_dir, "reduce_sum_ir.py")
reduce_max_kernel = os.path.join(script_dir, "reduce_sum_kernel.c")
with open(reduce_sum_ir, "w") as f:
    f.write(str(func))
with open(reduce_max_kernel, "w") as f:
    modified_code = mod.replace("v1", "ptr_input_v1").replace("v2", "ptr_output_v2")
    f.write(modified_code)