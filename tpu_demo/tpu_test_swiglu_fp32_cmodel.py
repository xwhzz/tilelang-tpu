# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T
import torch


def swi_glu(Block_w, Block_c, C, W, dtype="float32", accum_dtype="float32"):
    
    global_shape = (C, W)
    @T.prim_func
    def main_kernel_inner(
        G_in: T.Tensor(global_shape,dtype),
        G_right: T.Tensor(global_shape, dtype),
        G_out: T.Tensor(global_shape,accum_dtype)
    ):
        with T.Kernel(T.ceildiv(C, Block_c), T.ceildiv(W, Block_w), is_cpu=True) as (bx, by):
            # scale = T.float32(1.44269504)  # log2(e)
            
            block_shape = (Block_c, Block_w)

            x = T.alloc_shared(block_shape, accum_dtype)
            right = T.alloc_shared(block_shape, accum_dtype)
            # x_neg = T.alloc_shared(block_shape, accum_dtype)
            x_neg_exp = T.alloc_shared(block_shape, accum_dtype)
            ones = T.alloc_shared(block_shape, accum_dtype)
            T.fill(ones, T.float32(1.0))
            x_neg_exp_1 = T.alloc_shared(block_shape, accum_dtype)
            x_neg_exp_1_div = T.alloc_shared(block_shape, accum_dtype)
            out = T.alloc_shared(block_shape, accum_dtype)
            # * - +  /
            T.copy(G_in[bx * Block_c, by * Block_w], x)
            T.copy(G_right[bx * Block_c, by * Block_w], right)

            T.mul_C(x_neg_exp, x, T.float32(-1.0))
            # T.copy(x_neg[:, :], x_neg_exp[:, :])
            T.exp(x_neg_exp) # exp(-x)
            # T.copy(x_neg, x_neg_exp)
            T.add(x_neg_exp_1, x_neg_exp, ones) # exp(-x) + 1
            T.div(x_neg_exp_1_div, x, x_neg_exp_1) # x / (exp(-x) + 1)
            T.mul(out, right, x_neg_exp_1_div) # right * x / (exp(-x) + 1)

            T.copy(out, G_out[bx * Block_c, by * Block_w])

    return main_kernel_inner


C = 64
W = 64
block_C = 32
block_W = 32
kernel = tilelang.compile(swi_glu(Block_c=block_C, Block_w=block_W, C=C, W=W), out_idx=-1, target="tpu", mode="cmodel") #,pass_config={"disable_storage_rewrite": True})

a = torch.randn(C, W).float()
b = torch.randn(C, W).float()
c = torch.zeros(C, W).float()
res = kernel(a, b, c)
print(res)
print("output:")
print(c)
import torch.nn.functional as F
ref = b * F.silu(a)
ref = ref.float()
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

