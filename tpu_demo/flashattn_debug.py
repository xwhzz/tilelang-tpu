import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T

T.copy = T.ppl_copy

batch, heads, seq_len, dim = 1, 4, 4, 8
is_causal = False

Q = torch.randn(batch, seq_len, heads, dim, dtype=torch.float16)
K = torch.randn(batch, seq_len, heads, dim, dtype=torch.float16)
V = torch.randn(batch, seq_len, heads, dim, dtype=torch.float16)

from tilelang.jit.adapter.flashattn import tpu_test_flashattn_fp16
kernel = tpu_test_flashattn_fp16.flashattn(batch, heads, seq_len, dim, is_causal)

block_M, block_N = 2, 2
func = kernel(block_M, block_N, num_stages=1, threads=1)

out = torch.zeros_like(Q)
func(Q, K, V, out)

print("Output shape:", out.shape)
print("Output contains NaN:", torch.isnan(out).any())
print("Output contains Inf:", torch.isinf(out).any())
print("Output sample:", out[0, 0, 0, :])
