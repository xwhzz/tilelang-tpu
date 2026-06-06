# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Python wrapper matching torch.ops.my_ops.mlp_w8a16_dq_forward interface.

Uses the fused MLP kernel that performs gate, up, SiLU, and down
projections in a single TPU launch.

Usage:
    output = mlp_w8a16_dq_forward(x, gate_w, up_w, down_w,
                                   gate_s, up_s, down_s, output, blocksize=128)
"""

import torch

from tpu_demo.mlp_w8a16.torch_ref import dequant_weight
from tpu_demo.mlp_w8a16.mlp_w8a16_fused_kernel import mlp_w8a16_fused_kernel
import tilelang


# Cache compiled kernel keyed by (M, hidden, intermediate, block_M, block_N, block_K, block_D, mode)
_kernel_cache = {}


def _get_or_compile_kernel(M, hidden, intermediate,
                            block_M, block_N, block_K, block_D,
                            target="tpu", mode="cmodel"):
    """Get or compile the fused MLP kernel."""
    cache_key = (M, hidden, intermediate, block_M, block_N, block_K, block_D, target, mode)
    if cache_key not in _kernel_cache:
        _kernel_cache[cache_key] = tilelang.compile(
            mlp_w8a16_fused_kernel(
                M, hidden, intermediate,
                block_M=block_M, block_N=block_N,
                block_K=block_K, block_D=block_D,
            ),
            out_idx=-1,
            target=target,
            mode=mode,
        )
    return _kernel_cache[cache_key]


def mlp_w8a16_dq_forward(x, gate_weight, up_weight, down_weight,
                          gate_scale, up_scale, down_scale,
                          output, blocksize=128,
                          target="tpu", mode="cmodel"):
    """W8A16 dequantized MLP forward — single fused TPU kernel.

    Computes: output = down_proj(silu(gate_proj(x)) * up_proj(x))

    All three projections are w8a16 dequant matmuls. Dequantization
    is done on CPU before invoking the fused TPU kernel.

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
    """
    M, hidden = x.shape
    intermediate = gate_weight.shape[0]
    dtype = x.dtype
    device = x.device

    # CPU-side weight dequantization (one-time cost per weight)
    gate_w_fp16 = dequant_weight(gate_weight, gate_scale, blocksize)
    up_w_fp16 = dequant_weight(up_weight, up_scale, blocksize)
    down_w_fp16 = dequant_weight(down_weight, down_scale, blocksize)

    # Tile size selection
    block_M = min(32, M)
    block_N = min(128, intermediate)
    block_K = min(128, hidden)
    block_D = min(128, hidden)

    # Single fused kernel: gate + up + SiLU + down
    kernel = _get_or_compile_kernel(
        M, hidden, intermediate,
        block_M, block_N, block_K, block_D,
        target, mode,
    )
    kernel(x, gate_w_fp16, up_w_fp16, down_w_fp16, output)

    return output
