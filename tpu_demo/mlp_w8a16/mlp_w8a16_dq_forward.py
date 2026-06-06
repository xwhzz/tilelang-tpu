# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Python wrapper matching torch.ops.my_ops.mlp_w8a16_dq_forward interface.

Usage:
    output = mlp_w8a16_dq_forward(x, gate_w, up_w, down_w,
                                   gate_s, up_s, down_s, output, blocksize=128)
"""

import torch
import torch.nn.functional as F

from tpu_demo.mlp_w8a16.torch_ref import dequant_weight
from tpu_demo.mlp_w8a16.tpu_mlp_w8a16_kernel import make_matmul_kernel


# Cache compiled kernels keyed by (M, N, K)
_kernel_cache = {}


def _get_or_compile_kernel(M, N, K, target="tpu", mode="cmodel"):
    """Get or compile a fp16 matmul kernel for dimensions (M, N, K)."""
    cache_key = (M, N, K, target, mode)
    if cache_key not in _kernel_cache:
        block_M = min(32, M)
        block_N = min(128, N)
        block_K = min(128, K)
        _kernel_cache[cache_key] = make_matmul_kernel(
            M, N, K, block_M=block_M, block_N=block_N, block_K=block_K,
            target=target, mode=mode,
        )
    return _kernel_cache[cache_key]


def mlp_w8a16_dq_forward(x, gate_weight, up_weight, down_weight,
                          gate_scale, up_scale, down_scale,
                          output, blocksize=128,
                          target="tpu", mode="cmodel"):
    """W8A16 dequantized MLP forward.

    Computes: output = down_proj(silu(gate_proj(x)) * up_proj(x))

    All three projections are w8a16 dequant matmuls. Dequantization
    is done on CPU before calling TPU matmul kernels.

    Parameters
    ----------
    x : torch.Tensor
        fp16 activation [M, hidden_size]
    gate_weight : torch.Tensor
        int8 weight [intermediate_size, hidden_size]
    up_weight : torch.Tensor
        int8 weight [intermediate_size, hidden_size]
    down_weight : torch.Tensor
        int8 weight [hidden_size, intermediate_size] (vllm layout)
    gate_scale : torch.Tensor
        fp16 per-block scale [ceil(intermediate/128), ceil(hidden/128)]
    up_scale : torch.Tensor
        fp16 per-block scale
    down_scale : torch.Tensor
        fp16 per-block scale [ceil(hidden/128), ceil(intermediate/128)]
    output : torch.Tensor
        fp16 output buffer [M, hidden_size] (written in-place)
    blocksize : int
        Quantization block size (default 128)
    target : str
        Compilation target ("tpu")
    mode : str
        Execution mode ("cmodel" or "pcie")

    Returns
    -------
    torch.Tensor
        The output tensor (same as the output parameter)
    """
    M, hidden = x.shape
    intermediate = gate_weight.shape[0]
    dtype = x.dtype
    device = x.device

    # --- CPU-side weight dequantization ---
    gate_w_fp16 = dequant_weight(gate_weight, gate_scale, blocksize)
    up_w_fp16 = dequant_weight(up_weight, up_scale, blocksize)
    down_w_fp16 = dequant_weight(down_weight, down_scale, blocksize)

    # --- Kernel 1: gate = x @ gate_w_fp16^T ---
    gate_kernel = _get_or_compile_kernel(M, intermediate, hidden, target, mode)
    gate_out = torch.empty(M, intermediate, dtype=dtype, device=device)
    gate_kernel(x, gate_w_fp16, gate_out)

    # --- Kernel 2: up = x @ up_w_fp16^T ---
    up_kernel = _get_or_compile_kernel(M, intermediate, hidden, target, mode)
    up_out = torch.empty(M, intermediate, dtype=dtype, device=device)
    up_kernel(x, up_w_fp16, up_out)

    # --- CPU: SiLU activation (keep in fp32, clamp to normal fp16 range) ---
    act = F.silu(gate_out.float()) * up_out.float()
    # Clamp to normal fp16 range to avoid subnormals and overflow
    fp16_min_normal = 6.0e-5  # smallest normal fp16
    fp16_max = 65500.0
    act = act.clamp(-fp16_max, fp16_max)
    # Set very small values to 0 to avoid subnormals
    act = torch.where(act.abs() < fp16_min_normal, torch.zeros_like(act), act)
    act = act.to(dtype)

    # --- Kernel 3: output = act @ down_w_fp16^T ---
    down_kernel = _get_or_compile_kernel(M, hidden, intermediate, target, mode)
    down_kernel(act, down_w_fp16, output)

    return output
