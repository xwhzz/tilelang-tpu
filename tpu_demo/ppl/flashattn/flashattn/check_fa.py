import torch
import torch.nn.functional as F


# 定义文件路径和期望的形状
files_and_shapes = {
    "Q_tensor": ("./test/flashattn_q0.in", (1, 4, 2, 8)),
    "K_tensor": ("./test/flashattn_k1.in", (1, 4, 2, 8)),
    "V_tensor": ("./test/flashattn_v2.in", (1, 4, 2, 8)),
    "res": ("./test/flashattn_output3.out", (1, 4, 2, 8))
}

# 存储读取后的Tensor
tensors = {}

# 遍历文件并读取
for name, (path, shape) in files_and_shapes.items():
    with open(path, "rb") as f:
        data = f.read()

    tensor = torch.frombuffer(data, dtype=torch.float32) # 按数据类型读取为Tensor

    if tensor.numel() != torch.prod(torch.tensor(shape)): # 检查大小是否匹配
        raise ValueError(f"{name} 的大小和期望的 {shape} 不匹配，"
                         f"实际元素数：{tensor.numel()}，期望元素数：{torch.prod(torch.tensor(shape)).item()}")

    tensors[name] = tensor.reshape(shape) # reshape 成目标形状

# 从字典中提取 Tensor
Q_tensor = tensors["Q_tensor"]
K_tensor = tensors["K_tensor"]
V_tensor = tensors["V_tensor"]
res = tensors["res"]

print("Q_tensor:\n", Q_tensor)
print("K_tensor:\n", K_tensor)
print("V_tensor:\n", V_tensor)


def ref_program(Q, K, V, is_causal):
    dim = Q.size(-1)
    scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_len = Q.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)
    return output

# 运行 refs
result = ref_program(Q_tensor, K_tensor, V_tensor, is_causal=False)


print("reference result:\n", result)
print("kernel output:\n", res)
print("allclose:", torch.allclose(result, res, rtol=0.05, atol=0.05))
