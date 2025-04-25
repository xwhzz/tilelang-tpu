# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

import tilelang.language as T
from tvm.tir import PrimExpr, Buffer, BufferRegion
from typing import List, Union
from .copy import buffer_to_tile_region, buffer_region_to_tile_region, buffer_load_to_tile_region


def atomic_add(dst: Buffer, value: PrimExpr) -> PrimExpr:
    """Perform an atomic addition operation.

    Args:
        dst (Buffer): Destination buffer where the atomic addition will be performed
        value (PrimExpr): Value to be atomically added

    Returns:
        PrimExpr: Handle to the atomic addition operation
    """
    return T.call_extern("handle", "AtomicAdd", T.address_of(dst), value)


def atomic_addx2(dst: Buffer, value: PrimExpr) -> PrimExpr:
    """Perform an atomic addition operation with double-width operands.

    Args:
        dst (Buffer): Destination buffer where the atomic addition will be performed
        value (PrimExpr): Value to be atomically added (double-width)

    Returns:
        PrimExpr: Handle to the double-width atomic addition operation
    """
    return T.call_extern("handle", "AtomicAddx2", T.address_of(dst), T.address_of(value))


def dp4a(A: Buffer, B: Buffer, C: Buffer) -> PrimExpr:
    """Perform a 4-element dot product with accumulation (DP4A).

    Args:
        A (Buffer): First input buffer
        B (Buffer): Second input buffer
        C (Buffer): Accumulation buffer

    Returns:
        PrimExpr: Handle to the DP4A operation
    """
    return T.call_extern("handle", "DP4A", T.address_of(A), T.address_of(B), T.address_of(C))


def clamp(dst: PrimExpr, min_val: PrimExpr, max_val: PrimExpr) -> PrimExpr:
    """Clamps the input value dst between [min_val, max_val]
    
    Args:
        dst: Input value to be clamped
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Value clamped to the specified range
    """
    dst = T.max(dst, min_val)  # Ensure value is not less than minimum
    dst = T.min(dst, max_val)  # Ensure value is not greater than maximum
    return dst


def reshape(src: Buffer, shape: List[PrimExpr]) -> Buffer:
    """Reshapes the input buffer to the specified shape.
    
    Args:
        src (Buffer): Input buffer to be reshaped
        shape (List[PrimExpr]): New shape for the buffer

    Returns:
        Buffer: A new buffer view with the specified shape
    """
    return T.Buffer(shape, src.dtype, src.data)


def view(src: Buffer,
         shape: Union[List[PrimExpr], None] = None,
         dtype: Union[str, None] = None) -> Buffer:
    """Views the input buffer with optionally modified shape and dtype.
    
    Args:
        src (Buffer): Input buffer to be viewed
        shape (Union[List[PrimExpr], None], optional): New shape for the buffer. Defaults to None.
        dtype (Union[str, None], optional): New dtype for the buffer. Defaults to None.

    Returns:
        Buffer: A new buffer view with the specified shape and dtype
    """
    if shape is None:
        shape = src.shape
    if dtype is None:
        dtype = src.dtype
    return T.Buffer(shape, dtype, src.data)



def ppl_gemm(A, B, C, transpose_A=False, transpose_B=False):
    Aptr = A.access_ptr("r")
    Bptr = B.access_ptr("r")
    Cptr = C.access_ptr("rw")
    M = C.shape[0]
    N = C.shape[1]
    K = A.shape[0] if transpose_A else A.shape[1]
    K_B = B.shape[1] if transpose_B else B.shape[0]
    assert K == K_B, "gemm K shape check failed"
    return T.call_extern("handle", "ppl.gemm", Aptr, Bptr, Cptr, transpose_A, transpose_B, M, N, K)


def ppl_copy(
    src,
    dst,
):

    def get_extent(data):
        if isinstance(data, Buffer):
            return data.shape
        elif isinstance(data, BufferRegion):
            return [x.extent for x in data.region]
        else:
            return None

    src_extent = get_extent(src)
    dst_extent = get_extent(dst)

    if src_extent:
        extent = src_extent
    elif dst_extent:
        extent = dst_extent
    else:
        raise TypeError("Can't deduce copy extents from args")

    def _to_region(data, access_type):
        if isinstance(data, Buffer):
            return buffer_to_tile_region(data, access_type)
        elif isinstance(data, BufferRegion):
            return buffer_region_to_tile_region(data, access_type)
        else:
            return buffer_load_to_tile_region(data, access_type, extent)

    src = _to_region(src, "r")
    dst = _to_region(dst, "w")
    return T.call_extern("handle", "ppl.copy", src, dst)


def ppl_fill(buffer, value):
    buffer = buffer.access_ptr("w")
    return T.call_extern("handle", "ppl.fill", buffer, value)

def ppl_subtract(out, inp1, inp2):
    outptr = out.access_ptr("w")
    inpptr1 = inp1.access_ptr("r")
    inpptr2 = inp2.access_ptr("r")
    return T.call_extern("handle", "ppl.sub", outptr, inpptr1, inpptr2)

def ppl_mul_C(out, inp1, value):
    outptr = out.access_ptr("w")
    inpptr1 = inp1.access_ptr("r")
    return T.call_extern("handle", "ppl.mul_C", outptr, inpptr1, value)

def ppl_mul(out, inp1, inp2):
    outptr = out.access_ptr("w")
    inpptr1 = inp1.access_ptr("r")
    inpptr2 = inp2.access_ptr("r")
    return T.call_extern("handle", "ppl.mul", outptr, inpptr1, inpptr2)

def ppl_exp2(out):
    buffer = out.access_ptr("rw")
    return T.call_extern("handle", "ppl.exp", buffer)

def ppl_add(out, inp1, inp2):
    outptr = out.access_ptr("w")
    inpptr1 = inp1.access_ptr("r")
    inpptr2 = inp2.access_ptr("r")
    return T.call_extern("handle", "ppl.add", outptr, inpptr1, inpptr2)

def ppl_div(out, inp1, inp2):
    outptr = out.access_ptr("w")
    inpptr1 = inp1.access_ptr("r")
    inpptr2 = inp2.access_ptr("r")
    return T.call_extern("handle", "ppl.div", outptr, inpptr1, inpptr2)

@T.macro
def ppl_reduce_sum_safe(inp, out, dim):
    inpptr = inp.access_ptr("rw")
    outptr = out.access_ptr("rw")
    with T.block("reduce_sum"):
        tmp_shape = [inp.shape[0], 32]  # EU数量为32
        tmp_buffer = T.alloc_shared(tmp_shape, inp.dtype)
        tmp_ptr = tmp_buffer.access_ptr("rw")
        eu_num = T.int32(32)
        channel = T.int32(64)
        align_w = T.ceildiv(inp.shape[1], eu_num) * eu_num 
        stride = T.ceildiv(inp.shape[0], channel) * align_w
        # 调用底层reduce_max实现a
        T.call_extern("handle", "ppl.reduce_sum", inpptr, outptr, tmp_ptr, eu_num, align_w, stride)

def ppl_reduce_sum(inp, out, dim):
    assert dim == 1, "Only dim=1 is supported for reduction"
    return ppl_reduce_sum_safe(inp, out, dim)


@T.macro
def ppl_reduce_max_safe(inp, out, dim, clear=True):
    inpptr = inp.access_ptr("rw")
    outptr = out.access_ptr("rw")
    if clear:
        T.call_extern("handle", "ppl.fill", outptr, T.float16(float('-inf')))
    # 仅支持2D张量和dim=1
    # assert len(shape) == 2, "Only 2D tensors are supported"
    # 如果没有提供临时缓冲区，则创建一个
        # 创建一个临时缓冲区用于中间结果
        # 注意：这里的32是EU数量，可能需要根据实际情况调整
    with T.block("reduce_max"):
        tmp_shape = [inp.shape[0], 32]  # EU数量为32
        tmp_buffer = T.alloc_shared(tmp_shape, inp.dtype)
        tmp_ptr = tmp_buffer.access_ptr("rw")
        eu_num = T.int32(32)  
        channel = T.int32(64)
        align_w = T.ceildiv(inp.shape[1], eu_num) * eu_num 
        stride = T.ceildiv(inp.shape[0], channel) * align_w
    # 调用底层reduce_max实现a
        T.call_extern("handle", "ppl.reduce_max", inpptr, outptr, tmp_ptr, eu_num, align_w, stride)

def ppl_reduce_max(inp, out, dim, clear=True):
       # 在函数外部进行检查
    assert dim == 1, "Only dim=1 is supported"
       # 调用不含断言的宏函数
    return ppl_reduce_max_safe(inp, out, dim, clear)


def ppl_embedding(out, param, index, outer_num, inner_num, select_num, index_num):
    # 先判断
    assert outer_num == 1, "Only outer_num=1 is supported for embedding"
    return ppl_embedding_safe(out, param, index, outer_num, inner_num, select_num, index_num)

@T.macro
def ppl_embedding_safe(out, param, index, outer_num, inner_num, select_num, index_num):
    outptr = out.access_ptr("rw")
    paramptr = param.access_ptr("r")
    indexptr = index.access_ptr("r")
    with T.block("embedding"):
        params_tmp_buffer = T.alloc_shared([inner_num, select_num], param.dtype)
        params_tmp_ptr = params_tmp_buffer.access_ptr("rw")
        output_tmp_buffer = T.alloc_shared([inner_num, index_num], out.dtype)
        output_tmp_ptr = output_tmp_buffer.access_ptr("rw")
        # 这里的const_val是为了避免在调用时传入None，该参数为预留参数，暂未利用。
        const_val = T.float32(0.0)
        T.call_extern("handle", "ppl.embedding", outptr, paramptr, indexptr, params_tmp_ptr, output_tmp_ptr, outer_num, inner_num, select_num, index_num, const_val)