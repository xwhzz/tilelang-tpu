import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T

C, W = 64, 64
BLOCK_C, BLOCK_W = 32, 32
ATOL, RTOL = 1e-2, 1e-2


def tl_swiglu_fp16(Block_w, Block_c, C, W):
    dtype = "float16"
    accum_dtype = "float32"
    global_shape = (C, W)

    @T.prim_func
    def main_kernel_inner(
        G_in: T.Tensor(global_shape, dtype),
        G_right: T.Tensor(global_shape, dtype),
        G_out: T.Tensor(global_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(C, Block_c), T.ceildiv(W, Block_w), is_cpu=True) as (bx, by):
            block_shape = (Block_c, Block_w)
            x_ori = T.alloc_shared(block_shape, dtype)
            right_ori = T.alloc_shared(block_shape, dtype)
            x = T.alloc_shared(block_shape, accum_dtype)
            right = T.alloc_shared(block_shape, accum_dtype)
            x_neg_exp = T.alloc_shared(block_shape, accum_dtype)
            ones = T.alloc_shared(block_shape, accum_dtype)
            T.fill(ones, T.float32(1.0))
            x_neg_exp_1 = T.alloc_shared(block_shape, accum_dtype)
            x_neg_exp_1_div = T.alloc_shared(block_shape, accum_dtype)
            out = T.alloc_shared(block_shape, accum_dtype)
            out_ori = T.alloc_shared(block_shape, dtype)

            T.copy(G_in[bx * Block_c, by * Block_w], x_ori)
            T.copy(G_right[bx * Block_c, by * Block_w], right_ori)
            T.copy(x_ori, x)
            T.copy(right_ori, right)
            T.mul_C(x_neg_exp, x, T.float32(-1.0))
            T.exp(x_neg_exp)
            T.add(x_neg_exp_1, x_neg_exp, ones)
            T.div(x_neg_exp_1_div, x, x_neg_exp_1)
            T.mul(out, right, x_neg_exp_1_div)
            T.copy(out, out_ori)
            T.copy(out_ori, G_out[bx * Block_c, by * Block_w])

    return main_kernel_inner


def torch_ref(a, b):
    return (b.float() * F.silu(a.float())).half()


def run_and_check(name, kernel_func, a, b, ref):
    c = torch.zeros(C, W, dtype=torch.float16)
    kernel_func(a, b, c)
    correct = torch.allclose(c, ref, atol=ATOL, rtol=RTOL)
    max_diff = (c.float() - ref.float()).abs().max().item()
    avg_diff = (c.float() - ref.float()).abs().mean().item()
    print(f"  [{name}] correct={correct}  max_diff={max_diff:.6f}  avg_diff={avg_diff:.8f}")


def main():
    a = torch.randn(C, W, dtype=torch.float16)
    b = torch.randn(C, W, dtype=torch.float16)
    ref = torch_ref(a, b)

    tl_kernel = tilelang.compile(
        tl_swiglu_fp16(BLOCK_W, BLOCK_C, C, W), out_idx=-1, target="tpu")
    run_and_check("tilelang", tl_kernel, a, b, ref)



if __name__ == "__main__":
    main()
