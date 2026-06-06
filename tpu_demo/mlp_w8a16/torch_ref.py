# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""PyTorch reference implementation for W8A16 dequantized MLP.

This provides the ground-truth computation for verifying TileLang-TPU kernels.
"""

import torch
import torch.nn.functional as F


def dequant_weight(w_int8: torch.Tensor, scale: torch.Tensor, blocksize: int = 128):
    """Dequantize int8 weight to fp16 using per-block scales.

    Matches the weight_dequant logic from:
    /nfs/CloudService/SDK_v1.9/AI-Toolchain/LLMs/.../layers/fp8.py

    w_int8: [M, N] int8
    scale:  [ceil(M/128), ceil(N/128)] fp16
    returns: [M, N] fp16
    """
    assert w_int8.dim() == 2 and scale.dim() == 2, "Input tensors must have 2 dimensions"
    M, N = w_int8.shape
    m, n = (M + blocksize - 1) // blocksize, (N + blocksize - 1) // blocksize

    # Pad to multiples of blocksize
    pad_M = m * blocksize - M
    pad_N = n * blocksize - N
    if pad_M or pad_N:
        w_int8 = F.pad(w_int8.float(), (0, pad_N, 0, pad_M)).to(w_int8.dtype)

    # Reshape to 4D: [m, blocksize, n, blocksize]
    w_4d = w_int8.float().view(m, blocksize, n, blocksize)
    s_4d = scale.float().view(m, 1, n, 1)

    # Broadcast multiply: each block * its scale
    dequant = (w_4d * s_4d).reshape(m * blocksize, n * blocksize)

    # Remove padding
    if pad_M or pad_N:
        dequant = dequant[:M, :N]
    return dequant.half()


def dequant_matmul_ref(x, w_int8, w_scale, blocksize=128):
    """Reference: x @ dequant(w)^T

    x:       [M, K] fp16
    w_int8:  [N, K] int8
    w_scale: [ceil(N/128), ceil(K/128)] fp16
    returns: [M, N] fp16
    """
    w_deq = dequant_weight(w_int8, w_scale, blocksize)
    return torch.matmul(x.float(), w_deq.T.float()).half()


def mlp_w8a16_dq_forward_ref(x, gate_w, up_w, down_w,
                              gate_s, up_s, down_s, blocksize=128):
    """Full MLP reference: down(silu(gate(x)) * up(x))

    x:       [M, hidden] fp16
    gate_w:  [intermediate, hidden] int8
    up_w:    [intermediate, hidden] int8
    down_w:  [hidden, intermediate] int8  (vllm layout)
    gate_s:  [ceil(intermediate/128), ceil(hidden/128)] fp16
    up_s:    [ceil(intermediate/128), ceil(hidden/128)] fp16
    down_s:  [ceil(hidden/128), ceil(intermediate/128)] fp16
    returns: [M, hidden] fp16
    """
    gate = dequant_matmul_ref(x, gate_w, gate_s, blocksize)
    up = dequant_matmul_ref(x, up_w, up_s, blocksize)
    act = F.silu(gate.float()) * up.float()
    out = dequant_matmul_ref(act.half(), down_w, down_s, blocksize)
    return out


def make_scale(weight_shape, blocksize=128):
    """Create a random fp16 scale tensor matching the weight shape.

    weight_shape: (M, N)
    returns: [ceil(M/128), ceil(N/128)] fp16
    """
    m = (weight_shape[0] + blocksize - 1) // blocksize
    n = (weight_shape[1] + blocksize - 1) // blocksize
    return (torch.rand(m, n).half() + 0.5).contiguous()
