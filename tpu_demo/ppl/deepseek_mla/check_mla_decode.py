import torch
import torch.nn.functional as F


# 定义文件路径和期望的形状
batch = 4
heads = 8
kv_head_num = 1
seqlen_kv = 512
dim = 32
pe_dim = 4

files_and_shapes = {
    "q_tensor": ("./test/flashattn_q0.in", (batch, 1, heads, dim)),
    "qpe_tensor": ("./test/flashattn_qpe1.in", (batch, 1, heads, pe_dim)),
    "kv_tensor": ("./test/flashattn_kv2.in", (batch, seqlen_kv, kv_head_num, dim)),
    "kpe_tensor": ("./test/flashattn_kpe3.in", (batch, seqlen_kv, kv_head_num, pe_dim)),
    "output_tensor": ("./test/flashattn_output4.out", (batch, 1, heads, dim))
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
q_tensor = tensors["q_tensor"]
qpe_tensor = tensors["qpe_tensor"]
kv_tensor = tensors["kv_tensor"]
kpe_tensor = tensors["kpe_tensor"]
output_tensor = tensors["output_tensor"]

print("q_tensor:\n", q_tensor)
print("qpe_tensor:\n", qpe_tensor)
print("kv_tensor:\n", kv_tensor)
print("kpe_tensor:\n", kpe_tensor)


def ref_program(q, qpe, kv, kpe, is_causal):
    """
    q:      [B, 1, H_q, D]
    qpe:    [B, 1, H_q, D_pe]
    kv:     [B, S_kv, H_kv, D]
    kpe:    [B, S_kv, H_kv, D_pe]
    """
    # 1) 分数：Q·K^T + Q_pe·K_pe^T
    scores_nope = torch.einsum('bqhd,bkhd->bhqk', q, kv)  # Q @ K^T (不带位置)
    scores_pe   = torch.einsum('bqhd,bkhd->bhqk', qpe, kpe)  # Q_pe @ K_pe^T (位置)
    scores = scores_nope + scores_pe

    scale = 1.0 / torch.sqrt(torch.tensor(dim + pe_dim, dtype=scores.dtype))
    scores = scores * scale  # 缩放

    if is_causal:
        seq_len = q.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S_q, S_kv]
        scores = scores.masked_fill(mask == 0, float('-inf'))  # 对下三角mask为负无穷

    attention_weights = F.softmax(scores, dim=-1)  # 对每个 seq 的 S_kv 做 softmax

    output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, kv)  # O = Attn @ V

    return output

# 运行 refs
result = ref_program(q_tensor, qpe_tensor, kv_tensor, kpe_tensor, is_causal=False)


print("reference result:\n", result)
print("kernel output:\n", output_tensor)
print("allclose:", torch.allclose(result, output_tensor, rtol=0.05, atol=0.05))
