import tilelang
import tilelang.language as T
import os

def embedding_test(select_num, inner_num, index_num, dtype_0="float16", dtype_1 = 'uint16'):
    @T.prim_func
    def main(
            output: T.Tensor((index_num, inner_num), dtype_0),
            params: T.Tensor((select_num, inner_num), dtype_0),
            indices: T.Tensor((index_num, 1), dtype_1)
    ):
        params_shared = T.alloc_shared((select_num, inner_num), dtype_0)
        indices_shared = T.alloc_shared((index_num, 1), dtype_1)
        output_shared = T.alloc_shared((index_num, inner_num), dtype_0)
        T.ppl_copy(params, params_shared)
        T.ppl_copy(indices, indices_shared)
        T.ppl_fill(output_shared, T.float32(float(0.0)))
        T.ppl_embedding(output_shared, params_shared, indices_shared,
                        T.int32(1), T.int32(inner_num), T.int32(select_num), T.int32(index_num))
        T.ppl_copy(output_shared, output)
    return main

# 使用示例配置测试函数 - embedding测试
# 参数: index_num=32(要查询的索引数量), inner_num=128(向量维度), select_num=1024(词表大小)
func = embedding_test(65, 17, 7, "float16")
mod = tilelang.lower(func)
script_dir = os.path.dirname(os.path.abspath(__file__))
embedding_ir = os.path.join(script_dir, "embedding_ir.py")
embedding_kernel = os.path.join(script_dir, "embedding_kernel.c")
with open(embedding_ir, "w") as f:
    f.write(str(func))
with open(embedding_kernel, "w") as f:
    modified_code = mod.replace("v1", "ptr_output_v1").replace("v2", "ptr_param_v2").replace("v3", "ptr_index_v3")
    f.write(modified_code)
