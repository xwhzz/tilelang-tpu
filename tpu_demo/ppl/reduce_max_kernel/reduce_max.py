import tilelang
import tilelang.language as T
import os

def reduce_max(M, N, block_M, block_N, dtype="float32", accum_dtype="float32"):
    @T.prim_func
    def main(
            X: T.Tensor((M, N), dtype),
            Y: T.Tensor((M, 1), accum_dtype),
    ):
        with T.Kernel(1, T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
            # 分配共享内存
            X_shared = T.alloc_shared((block_M, block_N), dtype)
            Y_shared = T.alloc_shared((block_M, 1), accum_dtype)
            # 用于临时计算的共享内存
            # temp_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            # 初始化 Y_shared 为最小值
            # T.ppl_fill(Y_shared, T.float16(float('-inf')))
            # 复制输入数据到共享内存
            T.ppl_copy(X[by * block_M, 0], X_shared)
            # 执行 reduce_max 操作
            # 参数说明：输入张量，输出张量，沿哪个维度执行(1表示列方向)，是否保持维度，临时空间
            T.ppl_reduce_max(X_shared, Y_shared, 1, True)
            # 将结果从共享内存复制回全局内存
            T.ppl_copy(Y_shared, Y[by * block_M, 0])
    
    return main

# 使用示例配置测试函数
func = reduce_max(8192, 1020, 512, 1020)
mod = tilelang.lower(func)
script_dir = os.path.dirname(os.path.abspath(__file__))
reduce_max_ir = os.path.join(script_dir, "reduce_max_ir.py")
reduce_max_kernel = os.path.join(script_dir, "reduce_max_kernel.c")
with open(reduce_max_ir, "w") as f:
    f.write(str(func))
with open(reduce_max_kernel, "w") as f:
    modified_code = mod.replace("v1", "ptr_input_v1").replace("v2", "ptr_output_v2")
    f.write(modified_code)