# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T
import math

T.copy = T.ppl_copy

def rope_tpu(seq_len, dim, theta=10000.0):
    """
    RoPE (Rotary Position Embedding) implementation for TPU
    
    Args:
        seq_len: sequence length
        dim: embedding dimension (must be even)
        theta: base frequency for rotation
    """
    dtype = "float16"
    accum_dtype = "float32"
    
    def kernel_func(block_size, num_stages, threads):
        
        @T.macro
        def apply_rope(
            x: T.Tensor([block_size, dim], dtype),
            cos: T.Tensor([block_size, dim//2], dtype),
            sin: T.Tensor([block_size, dim//2], dtype),
            output: T.Tensor([block_size, dim], dtype)
        ):
            # Split x into even and odd parts for rotation
            x_even = T.alloc_shared([block_size, dim//2], dtype)
            x_odd = T.alloc_shared([block_size, dim//2], dtype)
            
            # Extract even and odd elements
            for i, j in T.Parallel(block_size, dim//2):
                x_even[i, j] = x[i, 2*j]
                x_odd[i, j] = x[i, 2*j + 1]
            
            # Apply rotation: [cos*even - sin*odd, cos*odd + sin*even]
            even_out = T.alloc_shared([block_size, dim//2], dtype)
            odd_out = T.alloc_shared([block_size, dim//2], dtype)
            tmp = T.alloc_shared([block_size, dim//2], dtype)
            
            # even_out = cos * x_even - sin * x_odd
            T.ppl_mul(even_out, cos, x_even)
            T.ppl_mul(tmp,      sin, x_odd)
            T.ppl_subtract(even_out, even_out, tmp)
            
            # odd_out = cos * x_odd + sin * x_even
            T.ppl_mul(odd_out, cos, x_odd)
            T.ppl_mul(tmp,     sin, x_even)
            T.ppl_add(odd_out, odd_out, tmp)
            
            # Interleave back to output
            for i, j in T.Parallel(block_size, dim//2):
                output[i, 2*j]     = even_out[i, j]
                output[i, 2*j + 1] = odd_out[i, j]
        
        @T.prim_func
        def main(
            x: T.Tensor([seq_len, dim], dtype),
            cos_cache: T.Tensor([seq_len, dim//2], dtype),
            sin_cache: T.Tensor([seq_len, dim//2], dtype),
            output: T.Tensor([seq_len, dim], dtype)
        ):
            with T.Kernel(T.ceildiv(seq_len, block_size), 1, 1, is_cpu=True) as (bx, by, bz):
                # Shared memory for current block
                x_shared = T.alloc_shared([block_size, dim], dtype)
                cos_shared = T.alloc_shared([block_size, dim//2], dtype)
                sin_shared = T.alloc_shared([block_size, dim//2], dtype)
                output_shared = T.alloc_shared([block_size, dim], dtype)
                
                # Load current block
                start_idx = bx * block_size
                end_idx = T.min(start_idx + block_size, seq_len)
                
                # Copy input block (handle tail block with region slices)
                valid = end_idx - start_idx
                T.copy(x[start_idx:end_idx, :], x_shared[0:valid, :])
                T.copy(cos_cache[start_idx:end_idx, :], cos_shared[0:valid, :])
                T.copy(sin_cache[start_idx:end_idx, :], sin_shared[0:valid, :])
                
                # Apply RoPE
                apply_rope(x_shared, cos_shared, sin_shared, output_shared)
                
                # Store result (only valid rows)
                T.copy(output_shared[0:valid, :], output[start_idx:end_idx, :])
           
        return main
    
    return kernel_func


def generate_rope_cache(seq_len, dim, theta=10000.0):
    """
    Generate cos and sin cache for RoPE
    
    Args:
        seq_len: sequence length
        dim: embedding dimension (must be even)
        theta: base frequency for rotation
    
    Returns:
        cos_cache, sin_cache: precomputed rotation matrices
    """
    cos_cache = []
    sin_cache = []
    
    for pos in range(seq_len):
        cos_row = []
        sin_row = []
        
        for i in range(dim // 2):
            # Calculate frequency
            freq = 1.0 / (theta ** (2 * i / dim))
            angle = pos * freq

            cos_row.append(math.cos(angle))
            sin_row.append(math.sin(angle))
        
        cos_cache.append(cos_row)
        sin_cache.append(sin_row)
    
    return cos_cache, sin_cache


if __name__ == "__main__":

    seq_len, dim, block_size, num_stages, threads = 2048, 128, 64, 2, 128
    cos_cache, sin_cache = generate_rope_cache(seq_len, dim)
    func = rope_tpu(seq_len, dim)(block_size, num_stages, threads)
    
    mod = tilelang.lower(func)
    print(mod)
    print("RoPE kernel generated successfully!")
