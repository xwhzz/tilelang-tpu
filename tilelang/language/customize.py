# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tvm import ir
from tvm.tir import PrimExpr, Buffer, BufferRegion, BufferLoad
from typing import List, Union
import builtins

import tilelang.language as T
from .copy import (
    copy as _tl_copy,
    buffer_to_tile_region,
    buffer_region_to_tile_region,
    buffer_load_to_tile_region,
)
from .gemm import gemm as _tl_gemm
from .tir.ir import exp as _tir_exp
from .tir.ir import rsqrt as _tir_rsqrt
from .tir.ir import sigmoid as _tir_sigmoid
from tvm.script.ir_builder.tir.ir import max as _tir_max


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


def gemm(
    A,
    B,
    C,
    transpose_A=False,
    transpose_B=False,
    policy=None,
    clear_accum=False,
    k_pack=1,
    wg_wait=0,
):
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
        `T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)`

    Notes:
        `K` is inferred from `A` and `B`, and must match.
        In current TPU usage, `C` is typically initialized first, then reused
        as the accumulation tile across one or more `T.gemm` calls.
        `transpose_A` is not recommended in the current TPU path; prefer using
        `transpose_B=True` when a transpose form is needed.
    """
    if policy is not None or clear_accum or k_pack != 1 or wg_wait != 0:
        kwargs = {
            "transpose_A": transpose_A,
            "transpose_B": transpose_B,
            "clear_accum": clear_accum,
            "k_pack": k_pack,
            "wg_wait": wg_wait,
        }
        if policy is not None:
            kwargs["policy"] = policy
        return _tl_gemm(A, B, C, **kwargs)

    def _extent(data):
        if isinstance(data, Buffer):
            return list(data.shape)
        if isinstance(data, BufferRegion):
            return [x.extent for x in data.region]
        if isinstance(data, BufferLoad):
            return [getattr(index, "lanes", 1) for index in data.indices]
        raise TypeError(f"Unsupported T.gemm operand type: {type(data)}")

    def _matrix_shape(data):
        shape = _extent(data)
        assert len(shape) in (2, 3), f"T.gemm expects rank-2 or rank-3 tiles, got {shape}"
        return shape[-2:]

    def _to_region(data, access_type):
        if isinstance(data, Buffer):
            return buffer_to_tile_region(data, access_type)
        if isinstance(data, BufferRegion):
            return buffer_region_to_tile_region(data, access_type)
        return buffer_load_to_tile_region(data, access_type, _extent(data))

    A_extent = _extent(A)
    B_extent = _extent(B)
    C_extent = _extent(C)
    assert len(A_extent) == len(B_extent) == len(
        C_extent), "T.gemm expects A/B/C to have the same rank"
    for A_dim, B_dim, C_dim in zip(A_extent[:-2], B_extent[:-2], C_extent[:-2]):
        ir.assert_structural_equal(A_dim, B_dim)
        ir.assert_structural_equal(A_dim, C_dim)
    A_matrix = _matrix_shape(A)
    B_matrix = _matrix_shape(B)
    C_matrix = _matrix_shape(C)
    M = C_matrix[0]
    N = C_matrix[1]
    K = A_matrix[0] if transpose_A else A_matrix[1]
    K_B = B_matrix[1] if transpose_B else B_matrix[0]
    A_M = A_matrix[1] if transpose_A else A_matrix[0]
    B_N = B_matrix[0] if transpose_B else B_matrix[1]
    ir.assert_structural_equal(A_M, M)
    ir.assert_structural_equal(B_N, N)
    ir.assert_structural_equal(K, K_B)
    return T.call_extern(
        "handle",
        "ppl.gemm",
        _to_region(A, "r"),
        _to_region(B, "r"),
        _to_region(C, "rw"),
        transpose_A,
        transpose_B,
        M,
        N,
        K,
    )


def copy(
    src,
    dst,
    coalesced_width=None,
):
    """Copy a tile/region between global and local memory, with optional cast.

    Args:
        src: Source buffer, `BufferRegion`, or `BufferLoad`.
        dst: Destination buffer, `BufferRegion`, or `BufferLoad`.

    Returns:
        PrimExpr: Handle to the emitted copy extern call.

    Example:
        `T.copy(X[by * block_M, 0], X_shared)`
        `T.copy(A_shared_fp32, A_shared)`

    Notes:
        This op is commonly used for global-to-shared loads, shared-to-global
        stores, and shared-to-shared copies between temporary tiles.
        When source and destination dtypes differ, this op can also be used as
        a convenient copy-and-convert step.
        The most common TPU usage is copying 2D tiles or simple row/column
        slices.
    """

    if coalesced_width is not None:
        return _tl_copy(src, dst, coalesced_width=coalesced_width)

    def _is_one(value):
        return isinstance(value, int) and value == 1 or (
            hasattr(value, "value") and value.value == 1)

    def _merge_extent(src_value, dst_value):
        if _is_one(src_value):
            return dst_value
        if _is_one(dst_value):
            return src_value
        ir.assert_structural_equal(src_value, dst_value)
        return src_value

    def get_extent(data):
        if isinstance(data, Buffer):
            return data.shape
        elif isinstance(data, BufferRegion):
            return [x.extent for x in data.region]
        elif isinstance(data, BufferLoad):
            return [getattr(index, "lanes", 1) for index in data.indices]
        else:
            return None

    src_extent = get_extent(src)
    dst_extent = get_extent(dst)

    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)
    rank = builtins.max(len(src_extent), len(dst_extent))
    src_extent = [1] * (rank - len(src_extent)) + src_extent
    dst_extent = [1] * (rank - len(dst_extent)) + dst_extent
    extent = [
        _merge_extent(src_value, dst_value)
        for src_value, dst_value in zip(src_extent, dst_extent)
    ]

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


def fill(buffer, value):
    """Fill a local/shared tile with a scalar constant.

    Args:
        buffer: Destination tile to be written.
        value: Scalar literal to broadcast to every element.

    Returns:
        PrimExpr: Handle to the emitted fill extern call.

    Example:
        `T.fill(C_shared, T.float32(0))`
        `T.fill(scores_max, -T.infinity(accum_dtype))`

    Notes:
        This is typically used to initialize accumulation buffers, masks,
        or temporary outputs before later elementwise or reduction ops.
        The common supported destination dtypes are `float16`, `bfloat16`,
        and `float32`.
    """
    buffer = buffer.access_ptr("w")
    return T.call_extern("handle", "ppl.fill", buffer, value)


def clear(buffer):
    return fill(buffer, T.float32(0))


def subtract(out, inp1, inp2):
    """Compute elementwise subtraction `out = inp1 - inp2`.

    Args:
        out: Output tile.
        inp1: Left-hand input tile.
        inp2: Right-hand input tile.

    Returns:
        PrimExpr: Handle to the emitted subtraction extern call.

    Example:
        `T.subtract(scores_scale, scores_max_prev, scores_max)`

    Notes:
        The usual usage is that all tiles have the same shape.
        A limited broadcast-style usage is also supported in common cases when
        the second input has shape `(M, 1)`.
    """
    outptr = out.access_ptr("w")
    inpptr1 = inp1.access_ptr("r")
    inpptr2 = inp2.access_ptr("r")
    return T.call_extern("handle", "ppl.sub", outptr, inpptr1, inpptr2)


def mul_C(out, inp1, value):
    """Compute elementwise scalar multiply `out = inp1 * value`.

    Args:
        out: Output tile.
        inp1: Input tile.
        value: Scalar multiplier.

    Returns:
        PrimExpr: Handle to the emitted multiply-by-constant extern call.

    Example:
        `T.mul_C(scores_scale, scores_scale, scale)`
        `T.mul_C(x_neg, in_x, T.float32(-1.0))`

    Notes:
        This is commonly used for scaling, sign flip, and normalization-style
        updates on a local tile.
    """
    outptr = out.access_ptr("w")
    inpptr1 = inp1.access_ptr("r")
    return T.call_extern("handle", "ppl.mul_C", outptr, inpptr1, value)


def mul(out, inp1, inp2):
    """Compute elementwise multiplication `out = inp1 * inp2`.

    Args:
        out: Output tile.
        inp1: Left-hand input tile.
        inp2: Right-hand input tile.

    Returns:
        PrimExpr: Handle to the emitted multiply extern call.

    Example:
        `T.mul(A_pow2, A_shared, A_shared)`
        `T.mul(out, right, x_neg_exp_1_div)`

    Notes:
        The usual usage is that all tiles have the same shape.
        A limited broadcast-style usage is also supported in common cases when
        the second input has shape `(M, 1)`.
    """
    outptr = out.access_ptr("w")
    inpptr1 = inp1.access_ptr("r")
    inpptr2 = inp2.access_ptr("r")
    return T.call_extern("handle", "ppl.mul", outptr, inpptr1, inpptr2)


def _max_tile(out, inp1, inp2):
    """Compute elementwise maximum `out = max(inp1, inp2)`.

    Args:
        out: Output tile.
        inp1: Left-hand input tile.
        inp2: Right-hand input tile.

    Returns:
        PrimExpr: Handle to the emitted max extern call.

    Example:
        `T.max(scores_max, scores_max_prev, block_max)`

    Notes:
        The usual usage is that all tiles have the same shape.
    """
    outptr = out.access_ptr("w")
    inpptr1 = inp1.access_ptr("r")
    inpptr2 = inp2.access_ptr("r")
    return T.call_extern("handle", "ppl.max", outptr, inpptr1, inpptr2)


@T.macro
def _exp_fp32(out):
    """Compute `exp(out)` in place and hide BM1690 work buffers."""
    work0 = T.alloc_shared(out.shape, "float32")
    work1 = T.alloc_shared(out.shape, "float32")
    coeff = T.alloc_shared([64, 32], "float32")
    table = T.alloc_shared([64, 192], "float32")
    buffer = out.access_ptr("rw")
    work0ptr = work0.access_ptr("rw")
    work1ptr = work1.access_ptr("rw")
    coeffptr = coeff.access_ptr("rw")
    tableptr = table.access_ptr("rw")
    T.call_extern("handle", "ppl.exp", buffer, work0ptr, work1ptr, coeffptr, tableptr)


@T.macro
def _exp_cast(out):
    tmp = T.alloc_shared(out.shape, "float32")
    copy(out, tmp)
    _exp_fp32(tmp)
    copy(tmp, out)


def exp(out):
    """Compute tile-wise natural exponential, or scalar TIR exp for scalars."""
    if isinstance(out, Buffer):
        if str(out.dtype) == "float32":
            return _exp_fp32(out)
        return _exp_cast(out)
    return _tir_exp(out)


@T.macro
def _exp_load_coeff(coeff, table):
    """Load exp coeff+table into SRAM for the following exp compute.

    On BM1690, `tpu_bdc_fp32_exp` uses the coeff/table buffers as scratch.
    Reload them immediately before each `_exp_compute`; hoisting this load
    across an exp compute can corrupt later exp results.

    Args:
        coeff: FP32 coefficient buffer shaped `(64, 32)`.
        table: FP32 lookup-table buffer shaped `(64, 192)`.
    """
    coeffptr = coeff.access_ptr("rw")
    tableptr = table.access_ptr("rw")
    T.call_extern("handle", "ppl.exp_load_coeff", coeffptr, tableptr)


@T.macro
def _exp_compute(out, work0, work1, coeff, table):
    """Compute `exp(out)` in place after `_exp_load_coeff`.

    The coeff/table buffers should be reloaded immediately before this call on
    BM1690 because the backend exp routine may modify them.
    """
    buffer = out.access_ptr("rw")

    work0ptr = work0.access_ptr("rw")
    work1ptr = work1.access_ptr("rw")
    coeffptr = coeff.access_ptr("rw")
    tableptr = table.access_ptr("rw")
    T.call_extern("handle", "ppl.exp_compute", buffer, work0ptr, work1ptr, coeffptr, tableptr)


@T.macro
def _sigmoid_fp32(out, inp):
    work0 = T.alloc_shared(inp.shape, "float32")
    work1 = T.alloc_shared(inp.shape, "float32")
    coeff = T.alloc_shared([64, 32], "float32")
    table = T.alloc_shared([64, 192], "float32")
    outptr = out.access_ptr("rw")
    inpptr = inp.access_ptr("rw")
    work0ptr = work0.access_ptr("rw")
    work1ptr = work1.access_ptr("rw")
    coeffptr = coeff.access_ptr("rw")
    tableptr = table.access_ptr("rw")
    T.call_extern("handle", "ppl.sigmoid", outptr, inpptr, work0ptr, work1ptr, coeffptr, tableptr)


@T.macro
def _sigmoid_cast(out, inp):
    inp_fp32 = T.alloc_shared(inp.shape, "float32")
    out_fp32 = T.alloc_shared(out.shape, "float32")
    copy(inp, inp_fp32)
    _sigmoid_fp32(out_fp32, inp_fp32)
    copy(out_fp32, out)


def sigmoid(out, inp=None):
    """Compute tile-wise sigmoid, or scalar TIR sigmoid for scalars."""
    if inp is None:
        return _tir_sigmoid(out)
    if str(out.dtype) == "float32" and str(inp.dtype) == "float32":
        return _sigmoid_fp32(out, inp)
    return _sigmoid_cast(out, inp)


@T.macro
def _silu_tile(out, inp):
    sigmoid_value = T.alloc_shared(inp.shape, inp.dtype)
    sigmoid(sigmoid_value, inp)
    mul(out, inp, sigmoid_value)


def silu(out, inp):
    return _silu_tile(out, inp)


def gather(output, param, index, param_h):
    outptr = output.access_ptr("w")
    paramptr = param.access_ptr("r")
    indexptr = index.access_ptr("r")
    return T.call_extern("handle", "ppl.gather", outptr, paramptr, indexptr, param_h)


def topk(dst_data, dst_idx, src, K, descended, length):
    dst_data_ptr = dst_data.access_ptr("w")
    dst_idx_ptr = dst_idx.access_ptr("w")
    srcptr = src.access_ptr("r")
    return T.call_extern("handle", "ppl.topk", dst_data_ptr, dst_idx_ptr, srcptr, K, descended, length)


def _rsqrt_tile(out, inp):
    """Compute reciprocal square root `out = rsqrt(inp)`.

    Args:
        out: Output tile.
        inp: Input tile.

    Returns:
        PrimExpr: Handle to the emitted rsqrt extern call.

    Example:
        `T.rsqrt(A_powsum, A_powsum)`

    Notes:
        The current usage requires both `out` and `inp` to be FP32.
        For FP16/BF16 workflows, first copy into an FP32 temporary buffer,
        apply `T.rsqrt`, then copy back if needed.
    """
    inpptr = inp.access_ptr("r")
    outptr = out.access_ptr("w")
    return T.call_extern("handle", "ppl.rsqrt", outptr, inpptr)


def rsqrt(out, inp=None):
    if inp is None:
        return _tir_rsqrt(out)
    return _rsqrt_tile(out, inp)


def add_C(out, inp1, value):
    """Compute elementwise scalar add `out = inp1 + value`.

    Args:
        out: Output tile.
        inp1: Input tile.
        value: Scalar bias to add.

    Returns:
        PrimExpr: Handle to the emitted add-by-constant extern call.

    Example:
        `T.add_C(A_powsum, A_powsum, T.float32(1e-12))`

    Notes:
        This is commonly used to add epsilon, bias, or other scalar offsets to
        a local tile.
    """
    outptr = out.access_ptr("w")
    inpptr1 = inp1.access_ptr("r")
    return T.call_extern("handle", "ppl.add_C", outptr, inpptr1, value)


def add(out, inp1, inp2):
    """Compute elementwise addition `out = inp1 + inp2`.

    Args:
        out: Output tile.
        inp1: Left-hand input tile.
        inp2: Right-hand input tile.

    Returns:
        PrimExpr: Handle to the emitted add extern call.

    Example:
        `T.add(logsum, logsum, scores_sum)`
        `T.add(x_neg_exp_1, x_neg_exp, ones)`

    Notes:
        The usual usage is that all tiles have the same shape.
        A limited broadcast-style usage is also supported in common cases when
        the second input has shape `(M, 1)`.
    """
    outptr = out.access_ptr("w")
    inpptr1 = inp1.access_ptr("r")
    inpptr2 = inp2.access_ptr("r")
    return T.call_extern("handle", "ppl.add", outptr, inpptr1, inpptr2)


def max(a, b, c=None):  # pylint: disable=redefined-builtin
    if c is None:
        return _tir_max(a, b)
    return _max_tile(a, b, c)


def div(out, inp1, inp2):
    """Compute elementwise division `out = inp1 / inp2`.

    Args:
        out: Output tile.
        inp1: Numerator tile.
        inp2: Denominator tile.

    Returns:
        PrimExpr: Handle to the emitted division extern call.

    Example:
        `T.div(acc_o, acc_o, logsum)`
        `T.div(x_neg_exp_1_div, x, x_neg_exp_1)`

    Notes:
        The usual usage is that all tiles have the same shape.
        A limited broadcast-style usage is also supported in common cases when
        the second input has shape `(M, 1)`.
    """
    outptr = out.access_ptr("w")
    inpptr1 = inp1.access_ptr("r")
    inpptr2 = inp2.access_ptr("r")
    return T.call_extern("handle", "ppl.div", outptr, inpptr1, inpptr2)


@T.macro
def _reduce_sum_safe(inp, out, dim):
    """Internal macro backing `reduce_sum`.

    Prefer calling `reduce_sum(...)` directly in user kernels.
    """
    inpptr = inp.access_ptr("rw")
    outptr = out.access_ptr("rw")
    with T.block("reduce_sum"):
        tmp_shape = [inp.shape[0], 32]  # EU数量为32
        tmp_buffer_sum = T.alloc_shared(tmp_shape, inp.dtype)
        tmp_ptr = tmp_buffer_sum.access_ptr("rw")
        eu_num = T.int32(32)
        channel = T.int32(64)
        align_w = T.ceildiv(inp.shape[1], eu_num) * eu_num
        stride = T.ceildiv(inp.shape[0], channel) * align_w
        # 调用底层reduce_max实现a
        T.call_extern("handle", "ppl.reduce_sum", inpptr, outptr, tmp_ptr, eu_num, align_w, stride)


@T.macro
def _reduce_sum_safe_3d(inp, out, dim):
    inpptr = inp.access_ptr("rw")
    outptr = out.access_ptr("rw")
    with T.block("reduce_sum"):
        tmp_shape = [inp.shape[0], inp.shape[1], 32]
        tmp_buffer_sum = T.alloc_shared(tmp_shape, inp.dtype)
        tmp_ptr = tmp_buffer_sum.access_ptr("rw")
        eu_num = T.int32(32)
        channel = T.int32(64)
        align_w = T.ceildiv(inp.shape[2], eu_num) * eu_num
        stride = T.ceildiv(inp.shape[1], channel) * align_w
        T.call_extern("handle", "ppl.reduce_sum", inpptr, outptr, tmp_ptr, eu_num, align_w, stride)


def reduce_sum(inp, out, dim):
    """Reduce a rank-2/rank-3 tile along its last dimension with summation.

    Args:
        inp: Input tile, typically shaped `(M, N)` or `(H, M, N)`.
        out: Output tile, typically shaped `(M, 1)` or `(H, M, 1)`.
        dim: Reduction axis. The current TPU path supports the last dimension.

    Returns:
        PrimExpr: Handle to the emitted reduction macro call.

    Example:
        `T.reduce_sum(acc_s, scores_sum, dim=1)`
        `T.reduce_sum(X_shared, Y_shared, dim=1)`

    Notes:
        This op is intended for 2D or 3D local tiles and currently only
        supports reduction along the last dimension.
    """
    rank = len(inp.shape)
    assert dim == rank - 1, "Only reduction along the last dim is supported"
    if rank == 2:
        return _reduce_sum_safe(inp, out, dim)
    assert rank == 3, "Only rank-2 and rank-3 tensors are supported for reduction"
    return _reduce_sum_safe_3d(inp, out, dim)


@T.macro
def _reduce_max_safe(inp, out, dim, clear=True):
    """Internal macro backing `reduce_max`.

    Prefer calling `reduce_max(...)` directly in user kernels.
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


@T.macro
def _reduce_max_safe_3d(inp, out, dim, clear=True):
    inpptr = inp.access_ptr("rw")
    outptr = out.access_ptr("rw")
    if clear:
        T.call_extern("handle", "ppl.fill", outptr, T.float16(float('-inf')))
    with T.block("reduce_max"):
        tmp_shape = [inp.shape[0], inp.shape[1], 32]
        tmp_buffer_max = T.alloc_shared(tmp_shape, inp.dtype)
        tmp_ptr = tmp_buffer_max.access_ptr("rw")
        eu_num = T.int32(32)
        channel = T.int32(64)
        align_w = T.ceildiv(inp.shape[2], eu_num) * eu_num
        stride = T.ceildiv(inp.shape[1], channel) * align_w
        T.call_extern("handle", "ppl.reduce_max", inpptr, outptr, tmp_ptr, eu_num, align_w, stride)


def reduce_max(inp, out, dim, clear=True):
    """Reduce a rank-2/rank-3 tile along its last dimension with max.

    Args:
        inp: Input tile, typically shaped `(M, N)` or `(H, M, N)`.
        out: Output tile, typically shaped `(M, 1)` or `(H, M, 1)`.
        dim: Reduction axis. The current TPU path supports the last dimension.
        clear: Whether to initialize `out` to `-inf` before reduction.
            Set this to `False` when you intentionally accumulate across tiles.

    Returns:
        PrimExpr: Handle to the emitted reduction macro call.

    Example:
        `T.reduce_max(acc_s, scores_max, dim=1, clear=False)`
        `T.reduce_max(X_shared, Y_shared, dim=1, clear=True)`

    Notes:
        This op is intended for 2D or 3D local tiles and currently only
        supports reduction along the last dimension.
        Set `clear=False` only when you intentionally want to keep and update
        the previous contents of `out`.
    """
    # 在函数外部进行检查
    rank = len(inp.shape)
    assert dim == rank - 1, "Only reduction along the last dim is supported"
    if rank == 2:
        return _reduce_max_safe(inp, out, dim, clear)
    assert rank == 3, "Only rank-2 and rank-3 tensors are supported for reduction"
    return _reduce_max_safe_3d(inp, out, dim, clear)


def rope_add(out, even_inp1, even_inp2, odd_inp1, odd_inp2):
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
        `T.rope_add(out, x_cos, x_neg_sin, x_cos, x_sin)`

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
