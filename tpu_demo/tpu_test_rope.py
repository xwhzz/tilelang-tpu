import tilelang
import tilelang.language as T
import torch

def rope(Block_c, Block_w, C, W, dtype="float32", accum_dtype="float32"):

    global_shape = (C, W)
    
    @T.prim_func
    def main_kernel_inner(
        G_in: T.Tensor(global_shape,dtype),
        G_cos: T.Tensor(global_shape, dtype),
        G_sin: T.Tensor(global_shape, dtype),    
        G_out: T.Tensor(global_shape,accum_dtype),
    ):
        with T.Kernel(T.ceildiv(C, Block_c), T.ceildiv(W, Block_w), is_cpu=True) as (bx, by):
            block_shape = (Block_c, Block_w)
            block_half_shape = (Block_c, Block_w // 2)

            in_x = T.alloc_shared(block_shape, dtype)
            in_cos = T.alloc_shared(block_shape, dtype)
            in_sin = T.alloc_shared(block_shape, dtype)

            T.ppl_copy(G_in[bx * Block_c, by * Block_w], in_x)
            T.ppl_copy(G_cos[bx * Block_c, by * Block_w], in_cos)
            T.ppl_copy(G_sin[bx * Block_c, by * Block_w], in_sin)

            # x_cos = x * cos
            x_cos = T.alloc_shared(block_shape, dtype)
            T.ppl_mul(x_cos, in_x, in_cos)

            # x_sin = x * sin
            x_sin = T.alloc_shared(block_shape, dtype)
            T.ppl_mul(x_sin, in_x, in_sin)

            # -x[:,:,:, 1::2]
            x_neg = T.alloc_shared(block_shape, dtype)
            T.ppl_mul_C(x_neg, in_x, T.float32(-1.0))
            x_neg_sin = T.alloc_shared(block_shape, dtype)
            T.ppl_mul(x_neg_sin, x_neg, in_sin)

            out = T.alloc_shared(block_shape, accum_dtype)

            # x[i] = x[i] * cos[i] + (- x[i+1] * sin[i])
            # x[i+1] = x[i+1] * cos[i] + x[i] * sin[i]
            T.ppl_rope_add(out, x_cos, x_neg_sin, x_cos, x_sin)

            # write back to global
            T.ppl_copy(out, G_out[bx * Block_c, by * Block_w])

    return main_kernel_inner


def make_rope_cos_sin(C, W, base=10000.0, pos_start=0, device="cpu", dtype=torch.float32):
    assert W % 2 == 0
    half = W // 2
    k = torch.arange(half, device=device, dtype=dtype)
    inv_freq = (base ** (-2 * k / W)).to(dtype)
    pos = torch.arange(pos_start, pos_start + C, device=device, dtype=dtype).unsqueeze(1)
    theta = pos * inv_freq
    c = torch.cos(theta)
    s = torch.sin(theta)

    cos = torch.empty(C, W, device=device, dtype=dtype)
    sin = torch.empty(C, W, device=device, dtype=dtype)
    cos[:, 0::2] = c
    cos[:, 1::2] = c
    sin[:, 0::2] = s
    sin[:, 1::2] = s
    return cos.contiguous(), sin.contiguous()


def rope_ref_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.shape == cos.shape == sin.shape
    assert x.shape[1] % 2 == 0

    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]
    c = cos[:, 0::2]
    s = sin[:, 0::2]

    out = torch.empty_like(x)
    out[:, 0::2] = x_even * c + (-x_odd) * s
    out[:, 1::2] = x_odd * c + x_even * s
    return out



func =  rope(64, 16, 128, 64)
kernel = tilelang.compile(rope(64, 16, 128, 64), out_idx=-1, target="tpu")

C, W = 128, 64

x = torch.randn(C, W, dtype=torch.float32)
cos, sin = make_rope_cos_sin(C, W, dtype=x.dtype)
output = torch.zeros(C, W, dtype=torch.float32)

res = kernel(x, cos, sin, output)
print(res)
ref = rope_ref_torch(x, cos, sin).float()
diff = ref - output
print("ref:")
print(ref)
print("output:")
print(output)
print("\n=== Difference analysis ===")
print("max diff:", diff.abs().max().item())
print("mean diff:", diff.abs().mean().item())
print("check close:", torch.allclose(output, ref, atol=1e-2, rtol=1e-2))
