import tilelang
import tilelang.language as T
import torch

def rope(Block_c, Block_w, C, W, dtype="bfloat16", accum_dtype="bfloat16"):

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

            T.copy(G_in[bx * Block_c, by * Block_w], in_x)
            T.copy(G_cos[bx * Block_c, by * Block_w], in_cos)
            T.copy(G_sin[bx * Block_c, by * Block_w], in_sin)

            # x_cos = x * cos
            x_cos = T.alloc_shared(block_shape, dtype)
            T.mul(x_cos, in_x, in_cos)

            # x_sin = x * sin
            x_sin = T.alloc_shared(block_shape, dtype)
            T.mul(x_sin, in_x, in_sin)

            # -x[:,:,:, 1::2]
            x_neg = T.alloc_shared(block_shape, dtype)
            T.mul_C(x_neg, in_x, T.float16(-1.0))
            x_neg_sin = T.alloc_shared(block_shape, dtype)
            T.mul(x_neg_sin, x_neg, in_sin)

            out = T.alloc_shared(block_shape, accum_dtype)

            # x[i] = x[i] * cos[i] + (- x[i+1] * sin[i])
            # x[i+1] = x[i+1] * cos[i] + x[i] * sin[i]
            T.rope_add(out, x_cos, x_neg_sin, x_cos, x_sin)

            # write back to global
            T.copy(out, G_out[bx * Block_c, by * Block_w])

    return main_kernel_inner

def make_rope_cos_sin(C, W, base=10000.0, pos_start=0, device="cpu", dtype=torch.bfloat16):
    assert W % 2 == 0, "W 必须为偶数"
    half = W // 2
    # 频率向量：shape (half,)
    k = torch.arange(half, device=device, dtype=dtype)
    inv_freq = (base ** (-2 * k / W)).to(dtype)

    # 位置：shape (C, 1)
    pos = (torch.arange(pos_start, pos_start + C, device=device, dtype=dtype)
           .unsqueeze(1))  # (C,1)

    # 角度：shape (C, half)
    theta = pos * inv_freq  # 广播

    c = torch.cos(theta)
    s = torch.sin(theta)

    cos = torch.empty(C, W, device=device, dtype=dtype)
    sin = torch.empty(C, W, device=device, dtype=dtype)
    cos[:, 0::2] = c;  cos[:, 1::2] = c
    sin[:, 0::2] = s;  sin[:, 1::2] = s
    return cos.contiguous(), sin.contiguous()

def rope_ref_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.shape == cos.shape == sin.shape, "x/cos/sin 的形状必须一致"
    C, W = x.shape
    assert W % 2 == 0, "W 必须为偶数"

    x_even = x[:, 0::2]
    x_odd  = x[:, 1::2]
    c = cos[:, 0::2]
    s = sin[:, 0::2]

    out_even = x_even * c + (-x_odd) * s
    out_odd  = x_odd  * c + ( x_even) * s

    out = torch.empty_like(x)
    out[:, 0::2] = out_even
    out[:, 1::2] = out_odd
    return out

func =  rope(64, 16, 128, 64)
kernel = tilelang.compile(rope(64, 16, 128, 64), out_idx=-1, target="tpu")


def read_txt_per_line(path: str, C: int, W: int, dtype=torch.bfloat16, device="cpu"):
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            vals.append(float(line))
    need = C * W
    if len(vals) != need:
        raise ValueError(f"{path}: 读到 {len(vals)} 个数，但期望 {need} 个。")
    return torch.tensor(vals, dtype=dtype, device=device).view(C, W).contiguous()

C, W = 128, 64

x = torch.randn(C, W, dtype=torch.bfloat16)
# cos = read_txt_per_line("tpu_demo/cos.txt", C, W, dtype=torch.bfloat16)
# sin = read_txt_per_line("tpu_demo/sin.txt", C, W, dtype=torch.bfloat16)
cos, sin = make_rope_cos_sin(C, W, device=x.device, dtype=x.dtype)
out_kernel = torch.zeros(C, W, dtype=torch.bfloat16)

# 运行 kernel
res = kernel(x, cos, sin, out_kernel)

print(res)
out_ref = rope_ref_torch(x, cos, sin).bfloat16()

print("参考实现输出:")
print(out_ref)
print("Kernel 输出:")
print(out_kernel)

# 对比
diff = out_ref - out_kernel
print("\n=== 差异分析 ===")
print("最大差异:", diff.abs().max().item())
print("平均差异:", diff.abs().mean().item())
print("check close:", torch.allclose(out_kernel, out_ref, atol=1e-2, rtol=1e-2))

