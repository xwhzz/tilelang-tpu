# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

import tilelang.language as T
from tvm.tir import PrimExpr, Buffer, BufferRegion, BufferLoad
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
    """Launch a TPU GEMM on local/shared tiles.

    Args:
        A: Left-hand input tile.
        B: Right-hand input tile.
        C: Output/accumulation tile with shape `(M, N)`.
        transpose_A: Whether `A` should be treated as transposed.
            This option is not recommended in current TPU usage.
        transpose_B: Whether `B` should be treated as transposed.

    Returns:
        PrimExpr: Handle to the emitted GEMM extern call.

    Example:
        `T.ppl_gemm(Q_shared, K_shared, acc_s, transpose_B=True)`

    Notes:
        `K` is inferred from `A` and `B`, and must match.
        In current TPU usage, `C` is typically initialized first, then reused
        as the accumulation tile across one or more `ppl_gemm` calls.
        `transpose_A` is not recommended in the current TPU path; prefer using
        `transpose_B=True` when a transpose form is needed.
    """
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
    """Copy a tile/region between global and local memory, with optional cast.

    Args:
        src: Source buffer, `BufferRegion`, or `BufferLoad`.
        dst: Destination buffer, `BufferRegion`, or `BufferLoad`.

    Returns:
        PrimExpr: Handle to the emitted copy extern call.

    Example:
        `T.ppl_copy(X[by * block_M, 0], X_shared)`
        `T.ppl_copy(A_shared_fp32, A_shared)`

    Notes:
        This op is commonly used for global-to-shared loads, shared-to-global
        stores, and shared-to-shared copies between temporary tiles.
        When source and destination dtypes differ, this op can also be used as
        a convenient copy-and-convert step.
        The most common TPU usage is copying 2D tiles or simple row/column
        slices.
    """

    def get_extent(data):
        if isinstance(data, Buffer):
            return data.shape
        elif isinstance(data, BufferRegion):
            return [x.extent for x in data.region]
        elif isinstance(data, BufferLoad):
            print(data.indices)
        else:
            return None

    print(type(src))
    src_extent = get_extent(src)
    dst_extent = get_extent(dst)

    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)
    extent = max(src_extent, dst_extent)

    def _to_region(data, access_type):
        if isinstance(data, Buffer):
            return buffer_to_tile_region(data, access_type)
        elif isinstance(data, BufferRegion):
            return buffer_region_to_tile_region(data, access_type)
        else:
            return buffer_load_to_tile_region(data, access_type, extent)

    src = _to_region(src, "r")
    dst = _to_region(dst, "w")
    print(src)
    print(dst)
    return T.call_extern("handle", "ppl.copy", src, dst)


def ppl_fill(buffer, value):
    """Fill a local/shared tile with a scalar constant.

    Args:
        buffer: Destination tile to be written.
        value: Scalar literal to broadcast to every element.

    Returns:
        PrimExpr: Handle to the emitted fill extern call.

    Example:
        `T.ppl_fill(C_shared, T.float32(0))`
        `T.ppl_fill(scores_max, -T.infinity(accum_dtype))`

    Notes:
        This is typically used to initialize accumulation buffers, masks,
        or temporary outputs before later elementwise or reduction ops.
        The common supported destination dtypes are `float16`, `bfloat16`,
        and `float32`.
    """
    buffer = buffer.access_ptr("w")
    return T.call_extern("handle", "ppl.fill", buffer, value)


def ppl_clear(buffer):
    return T.ppl_fill(buffer, T.float32(0))


def ppl_subtract(out, inp1, inp2):
    """Compute elementwise subtraction `out = inp1 - inp2`.

    Args:
        out: Output tile.
        inp1: Left-hand input tile.
        inp2: Right-hand input tile.

    Returns:
        PrimExpr: Handle to the emitted subtraction extern call.

    Example:
        `T.ppl_subtract(scores_scale, scores_max_prev, scores_max)`

    Notes:
        The usual usage is that all tiles have the same shape.
        A limited broadcast-style usage is also supported in common cases when
        the second input has shape `(M, 1)`.
    """
    outptr = out.access_ptr("w")
    inpptr1 = inp1.access_ptr("r")
    inpptr2 = inp2.access_ptr("r")
    return T.call_extern("handle", "ppl.sub", outptr, inpptr1, inpptr2)


def ppl_mul_C(out, inp1, value):
    """Compute elementwise scalar multiply `out = inp1 * value`.

    Args:
        out: Output tile.
        inp1: Input tile.
        value: Scalar multiplier.

    Returns:
        PrimExpr: Handle to the emitted multiply-by-constant extern call.

    Example:
        `T.ppl_mul_C(scores_scale, scores_scale, scale)`
        `T.ppl_mul_C(x_neg, in_x, T.float32(-1.0))`

    Notes:
        This is commonly used for scaling, sign flip, and normalization-style
        updates on a local tile.
    """
    outptr = out.access_ptr("w")
    inpptr1 = inp1.access_ptr("r")
    return T.call_extern("handle", "ppl.mul_C", outptr, inpptr1, value)


def ppl_mul(out, inp1, inp2):
    """Compute elementwise multiplication `out = inp1 * inp2`.

    Args:
        out: Output tile.
        inp1: Left-hand input tile.
        inp2: Right-hand input tile.

    Returns:
        PrimExpr: Handle to the emitted multiply extern call.

    Example:
        `T.ppl_mul(A_pow2, A_shared, A_shared)`
        `T.ppl_mul(out, right, x_neg_exp_1_div)`

    Notes:
        The usual usage is that all tiles have the same shape.
        A limited broadcast-style usage is also supported in common cases when
        the second input has shape `(M, 1)`.
    """
    outptr = out.access_ptr("w")
    inpptr1 = inp1.access_ptr("r")
    inpptr2 = inp2.access_ptr("r")
    return T.call_extern("handle", "ppl.mul", outptr, inpptr1, inpptr2)


@T.macro
def ppl_exp2(out, work0, work1, coeff, table):  # only support FP32
    """Compute `exp(out)` in place.

    Args:
        out: Input/output tile. The result overwrites this buffer.
        work0: Scratch tile with the same shape as `out`.
        work1: Scratch tile with the same shape as `out`.
        coeff: FP32 coefficient buffer, typically shaped like `(64, 32)`.
        table: FP32 lookup-table buffer, typically shaped like `(64, 192)`.

    Example:
        `T.ppl_exp2(scores_scale, work0, work1, coeff, table)`

    Notes:
        Despite the name `ppl_exp2`, this op computes natural exponential
        `exp(x)`, not `2^x`.
        `out`, `work0`, `work1`, `coeff`, and `table` should all be allocated
        by the caller before invoking this macro.
        The current usage requires FP32 buffers.
    """
    buffer = out.access_ptr("rw")
    work0ptr = work0.access_ptr("rw")
    work1ptr = work1.access_ptr("rw")
    coeffptr = coeff.access_ptr("rw")
    tableptr = table.access_ptr("rw")
    T.call_extern("handle", "ppl.exp", buffer, work0ptr, work1ptr, coeffptr, tableptr)


@T.macro
def ppl_sigmoid(out, inp, work0, work1, coeff, table):  # only support FP32
    outptr = out.access_ptr("rw")
    inpptr = inp.access_ptr("rw")
    work0ptr = work0.access_ptr("rw")
    work1ptr = work1.access_ptr("rw")
    coeffptr = coeff.access_ptr("rw")
    tableptr = table.access_ptr("rw")
    T.call_extern("handle", "ppl.sigmoid", outptr, inpptr, work0ptr, work1ptr, coeffptr, tableptr)

def ppl_gather(output, param, index, param_h):
    outptr = output.access_ptr("w")
    paramptr = param.access_ptr("r")
    indexptr = index.access_ptr("r")
    return T.call_extern("handle", "ppl.gather", outptr, paramptr, indexptr, param_h)
    
def ppl_topk(dst_data, dst_idx, src, K, descended, length):
    dst_data_ptr = dst_data.access_ptr("w")
    dst_idx_ptr = dst_idx.access_ptr("w")
    srcptr = src.access_ptr("r")
    return T.call_extern("handle", "ppl.topk", dst_data_ptr, dst_idx_ptr, srcptr, K, descended, length)

# def ppl_exp2(out, block_M, block_N, dtype): # only support FP32
#     buffer = out.access_ptr("rw")
#     work0 = T.alloc_shared([block_M, block_N], dtype)
#     work1 = T.alloc_shared([block_M, block_N], dtype)
#     coeff = T.alloc_shared([64, 32], dtype) # npu number is 64
#     table = T.alloc_shared([64, 192], dtype) # npu number is 64
#     work0ptr = work0.access_ptr("rw")
#     work1ptr = work1.access_ptr("rw")
#     coeffptr = coeff.access_ptr("rw")
#     tableptr = table.access_ptr("rw")
#     T.call_extern("handle", "ppl.exp", buffer, work0ptr, work1ptr, coeffptr, tableptr)


def ppl_rsqrt(out, inp):
    """Compute reciprocal square root `out = rsqrt(inp)`.

    Args:
        out: Output tile.
        inp: Input tile.

    Returns:
        PrimExpr: Handle to the emitted rsqrt extern call.

    Example:
        `T.ppl_rsqrt(A_powsum, A_powsum)`

    Notes:
        The current usage requires both `out` and `inp` to be FP32.
        For FP16/BF16 workflows, first copy into an FP32 temporary buffer,
        apply `ppl_rsqrt`, then copy back if needed.
    """
    inpptr = inp.access_ptr("r")
    outptr = out.access_ptr("w")
    return T.call_extern("handle", "ppl.rsqrt", outptr, inpptr)


def ppl_add_C(out, inp1, value):
    """Compute elementwise scalar add `out = inp1 + value`.

    Args:
        out: Output tile.
        inp1: Input tile.
        value: Scalar bias to add.

    Returns:
        PrimExpr: Handle to the emitted add-by-constant extern call.

    Example:
        `T.ppl_add_C(A_powsum, A_powsum, T.float32(1e-12))`

    Notes:
        This is commonly used to add epsilon, bias, or other scalar offsets to
        a local tile.
    """
    outptr = out.access_ptr("w")
    inpptr1 = inp1.access_ptr("r")
    return T.call_extern("handle", "ppl.add_C", outptr, inpptr1, value)


def ppl_add(out, inp1, inp2):
    """Compute elementwise addition `out = inp1 + inp2`.

    Args:
        out: Output tile.
        inp1: Left-hand input tile.
        inp2: Right-hand input tile.

    Returns:
        PrimExpr: Handle to the emitted add extern call.

    Example:
        `T.ppl_add(logsum, logsum, scores_sum)`
        `T.ppl_add(x_neg_exp_1, x_neg_exp, ones)`

    Notes:
        The usual usage is that all tiles have the same shape.
        A limited broadcast-style usage is also supported in common cases when
        the second input has shape `(M, 1)`.
    """
    outptr = out.access_ptr("w")
    inpptr1 = inp1.access_ptr("r")
    inpptr2 = inp2.access_ptr("r")
    return T.call_extern("handle", "ppl.add", outptr, inpptr1, inpptr2)


def ppl_div(out, inp1, inp2):
    """Compute elementwise division `out = inp1 / inp2`.

    Args:
        out: Output tile.
        inp1: Numerator tile.
        inp2: Denominator tile.

    Returns:
        PrimExpr: Handle to the emitted division extern call.

    Example:
        `T.ppl_div(acc_o, acc_o, logsum)`
        `T.ppl_div(x_neg_exp_1_div, x, x_neg_exp_1)`

    Notes:
        The usual usage is that all tiles have the same shape.
        A limited broadcast-style usage is also supported in common cases when
        the second input has shape `(M, 1)`.
    """
    outptr = out.access_ptr("w")
    inpptr1 = inp1.access_ptr("r")
    inpptr2 = inp2.access_ptr("r")
    return T.call_extern("handle", "ppl.div", outptr, inpptr1, inpptr2)


def ppl_min(out, inp1, inp2):
    """Compute elementwise minimum `out = min(inp1, inp2)`.

    Args:
        out: Output tile.
        inp1: Left-hand input tile.
        inp2: Right-hand input tile.

    Returns:
        PrimExpr: Handle to the emitted minimum extern call.

    Example:
        `T.ppl_min(clamped, act, max_val)`

    Notes:
        The usual usage is that all tiles have the same shape.
    """
    outptr = out.access_ptr("w")
    inpptr1 = inp1.access_ptr("r")
    inpptr2 = inp2.access_ptr("r")
    return T.call_extern("handle", "ppl.min", outptr, inpptr1, inpptr2)


def ppl_max(out, inp1, inp2):
    """Compute elementwise maximum `out = max(inp1, inp2)`.

    Args:
        out: Output tile.
        inp1: Left-hand input tile.
        inp2: Right-hand input tile.

    Returns:
        PrimExpr: Handle to the emitted maximum extern call.

    Example:
        `T.ppl_max(clamped, act, min_val)`

    Notes:
        The usual usage is that all tiles have the same shape.
    """
    outptr = out.access_ptr("w")
    inpptr1 = inp1.access_ptr("r")
    inpptr2 = inp2.access_ptr("r")
    return T.call_extern("handle", "ppl.max", outptr, inpptr1, inpptr2)


@T.macro
def ppl_reduce_sum_safe(inp, out, dim):
    """Internal macro backing `ppl_reduce_sum`.

    Prefer calling `ppl_reduce_sum(...)` directly in user kernels.
    """
    inpptr = inp.access_ptr("rw")
    outptr = out.access_ptr("rw")
    with T.block("reduce_sum"):
        tmp_shape = [inp.shape[0], 32]
        tmp_buffer_sum = T.alloc_shared(tmp_shape, inp.dtype)
        tmp_ptr = tmp_buffer_sum.access_ptr("rw")
        eu_num = T.int32(32)
        channel = T.int32(64)
        align_w = T.ceildiv(inp.shape[1], eu_num) * eu_num
        stride = T.ceildiv(inp.shape[0], channel) * align_w
        # 调用底层reduce_max实现a
        T.call_extern("handle", "ppl.reduce_sum", inpptr, outptr, tmp_ptr, eu_num, align_w, stride)


@T.macro
def ppl_reduce_sum_safe_4d(inp, out, dim):
    inpptr = inp.access_ptr("rw")
    outptr = out.access_ptr("rw")
    with T.block("reduce_sum"):
        tmp_shape = [
            inp.shape[0],
            inp.shape[1],
            inp.shape[2],
            32,
        ]
        tmp_buffer_sum = T.alloc_shared(tmp_shape, inp.dtype)
        tmp_ptr = tmp_buffer_sum.access_ptr("rw")
        eu_num = T.int32(32)
        align_w = T.ceildiv(inp.shape[3], eu_num) * eu_num
        stride = inp.shape[1] * inp.shape[2] * align_w
        T.call_extern(
            "handle",
            "ppl.reduce_sum",
            inpptr,
            outptr,
            tmp_ptr,
            eu_num,
            align_w,
            stride,
        )


def ppl_reduce_sum(inp, out, dim):
    """Reduce a tile along its last dimension with summation.

    Args:
        inp: Input tile, shaped `(M, N)` or `(N, C, H, W)`.
        out: Output tile with a final dimension of one.
        dim: Reduction selector. The current TPU API uses `dim=1`.

    Returns:
        PrimExpr: Handle to the emitted reduction macro call.

    Example:
        `T.ppl_reduce_sum(acc_s, scores_sum, dim=1)`
        `T.ppl_reduce_sum(X_shared, Y_shared, dim=1)`

    Notes:
        Four-dimensional tiles reduce their final `W` dimension while
        preserving `N`, `C`, and `H`.
    """
    assert dim == 1, "Only dim=1 is supported for reduction"
    if len(inp.shape) == 2:
        return ppl_reduce_sum_safe(inp, out, dim)
    assert len(inp.shape) == 4, "Only 2D and 4D tensors are supported"
    return ppl_reduce_sum_safe_4d(inp, out, dim)


@T.macro
def ppl_reduce_max_safe(inp, out, dim, clear=True):
    """Internal macro backing `ppl_reduce_max`.

    Prefer calling `ppl_reduce_max(...)` directly in user kernels.
    """
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
        tmp_buffer_max = T.alloc_shared(tmp_shape, inp.dtype)
        tmp_ptr = tmp_buffer_max.access_ptr("rw")
        eu_num = T.int32(32)
        channel = T.int32(64)
        align_w = T.ceildiv(inp.shape[1], eu_num) * eu_num
        stride = T.ceildiv(inp.shape[0], channel) * align_w
        # 调用底层reduce_max实现a
        T.call_extern("handle", "ppl.reduce_max", inpptr, outptr, tmp_ptr, eu_num, align_w, stride)


def ppl_reduce_max(inp, out, dim, clear=True):
    """Reduce a 2D tile along its second dimension with max.

    Args:
        inp: Input tile, typically shaped `(M, N)`.
        out: Output tile, typically shaped `(M, 1)`.
        dim: Reduction axis. The current TPU path only supports `dim=1`.
        clear: Whether to initialize `out` to `-inf` before reduction.
            Set this to `False` when you intentionally accumulate across tiles.

    Returns:
        PrimExpr: Handle to the emitted reduction macro call.

    Example:
        `T.ppl_reduce_max(acc_s, scores_max, dim=1, clear=False)`
        `T.ppl_reduce_max(X_shared, Y_shared, dim=1, clear=True)`

    Notes:
        This op is intended for 2D tiles and currently only supports
        reduction along `dim=1`.
        The usual output shape is `(inp.shape[0], 1)`.
        Set `clear=False` only when you intentionally want to keep and update
        the previous contents of `out`.
    """
    # 在函数外部进行检查
    assert dim == 1, "Only dim=1 is supported"
    # 调用不含断言的宏函数
    return ppl_reduce_max_safe(inp, out, dim, clear)


def ppl_rope_add(out, even_inp1, even_inp2, odd_inp1, odd_inp2):
    """Assemble interleaved RoPE output from precomputed even/odd terms.

    Args:
        out: Output tile. Its last dimension must be even.
        even_inp1: Tensor contributing the even-lane base term.
        even_inp2: Tensor contributing the even-lane cross term.
        odd_inp1: Tensor contributing the odd-lane base term.
        odd_inp2: Tensor contributing the odd-lane cross term.

    Returns:
        PrimExpr: Handle to the emitted RoPE extern call.

    Example:
        `T.ppl_rope_add(out, x_cos, x_neg_sin, x_cos, x_sin)`

    Notes:
        This helper is intended for the common RoPE pattern where the caller
        has already prepared the even/odd terms, such as `x * cos(theta)`,
        `x * sin(theta)`, and `-x * sin(theta)`.
        The last dimension of `out` should be even.
    """
    outptr = out.access_ptr("w")
    even_inpptr1 = even_inp1.access_ptr("r")
    even_inpptr2 = even_inp2.access_ptr("r")
    odd_inpptr1 = odd_inp1.access_ptr("r")
    odd_inpptr2 = odd_inp2.access_ptr("r")
    return T.call_extern("handle", "ppl.rope_add", outptr, even_inpptr1, even_inpptr2, odd_inpptr1, odd_inpptr2)
