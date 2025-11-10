# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import tilelang
import tilelang.language as T
import torch

def matmul(M, N, K, block_M, block_N, block_K,dtype="bfloat16", accum_dtype="float32"):

    @T.prim_func
    def main_kernel_inner(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
      with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            C_shared_ori = T.alloc_shared((block_M, block_N), dtype)

            T.ppl_fill(C_shared, T.float32(0))
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.ppl_copy(A[by * block_M, k * block_K], A_shared)
                T.ppl_copy(B[k * block_K, bx * block_N], B_shared)
                T.ppl_gemm(A_shared, B_shared, C_shared)
            T.ppl_copy(C_shared, C_shared_ori)
            T.ppl_copy(C_shared_ori, C[by * block_M, bx * block_N])
    return main_kernel_inner



kernel = tilelang.compile(matmul(64,64,64,32,32,32), out_idx=-1, target="tpu")

a = torch.randn(64, 64).bfloat16()
b = torch.randn(64, 64).bfloat16()
c = torch.zeros(64, 64).bfloat16()

res = kernel(a, b, c)
print(res)
print("output:")
print(c)
import torch.nn.functional as F
ref = torch.matmul(a, b).bfloat16()
print("ref")
print(ref)
diff = ref-c
max_diff = torch.max(diff)
avg_diff = torch.mean(diff)

print(f"\n=== 差异分析 ===")
print(f"最大差异: {max_diff}")
print(f"平均差异: {avg_diff}")
print("check close:")
print(torch.allclose(c, ref, atol=1e-2, rtol=1e-2))

