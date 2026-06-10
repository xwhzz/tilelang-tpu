# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import contextlib
import io

from tilelang import tvm
import tilelang.language as T
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget


BANK_SIZE = 16 * 1024


def _assigned_attrs(func):
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    target = tvm.target.Target("tpu")
    with contextlib.redirect_stdout(io.StringIO()):
        mod = LowerAndLegalize(mod, target)
        mod = OptimizeForTarget(mod, target)
    return mod["main"].attrs


def _addr(attrs, name):
    return int(attrs[name])


def _attr_keys(attrs):
    return [str(key) for key in attrs._dict().keys()]


def _single_attr_with_prefix(attrs, prefix):
    keys = [key for key in _attr_keys(attrs) if key.startswith(prefix)]
    assert len(keys) == 1, keys
    return keys[0]


def test_tpu_gemm_buffers_are_bank_separated():

    @T.prim_func
    def main():
        with T.Kernel(1, is_cpu=True) as _:
            a_shared = T.alloc_shared((64, 1024), "float32")
            b_shared = T.alloc_shared((1024, 64), "float32")
            c_shared = T.alloc_shared((64, 64), "float32")

            T.fill(c_shared, T.float32(0.0))
            T.gemm(a_shared, b_shared, c_shared)

    attrs = _assigned_attrs(main)
    a_addr = _addr(attrs, "a_shared")
    b_addr = _addr(attrs, "b_shared")
    c_addr = _addr(attrs, "c_shared")

    assert len({a_addr // BANK_SIZE, b_addr // BANK_SIZE, c_addr // BANK_SIZE}) == 3


def test_elementwise_reads_are_bank_separated_while_outputs_remain_flexible():

    @T.prim_func
    def main():
        with T.Kernel(1, is_cpu=True) as _:
            src0 = T.alloc_shared((64, 1024), "float32")
            src1 = T.alloc_shared((64, 1024), "float32")
            dst = T.alloc_shared((64, 64), "float32")

            T.add(dst, src0, src1)

    attrs = _assigned_attrs(main)
    src0_addr = _addr(attrs, "src0")
    src1_addr = _addr(attrs, "src1")
    dst_addr = _addr(attrs, "dst")

    assert src0_addr // BANK_SIZE != src1_addr // BANK_SIZE
    assert dst_addr // BANK_SIZE == src0_addr // BANK_SIZE


def test_reduce_tmp_is_separated_from_input_bank():

    @T.prim_func
    def main():
        with T.Kernel(1, is_cpu=True) as _:
            inp = T.alloc_shared((64, 1024), "float32")
            out = T.alloc_shared((64, 1), "float32")

            T.reduce_sum(inp, out, dim=1)

    attrs = _assigned_attrs(main)
    tmp_name = _single_attr_with_prefix(attrs, "tmp_buffer_sum")
    inp_addr = _addr(attrs, "inp")
    tmp_addr = _addr(attrs, tmp_name)

    assert inp_addr // BANK_SIZE != tmp_addr // BANK_SIZE


def test_exp_hidden_workspace_is_reused_for_multiple_calls():

    @T.prim_func
    def main():
        with T.Kernel(1, is_cpu=True) as _:
            out0 = T.alloc_shared((64, 1024), "float32")
            out1 = T.alloc_shared((64, 1024), "float32")

            T.exp(out0)
            T.exp(out1)

    attrs = _assigned_attrs(main)
    keys = _attr_keys(attrs)
    assert len([key for key in keys if key.startswith("work0")]) == 1, keys
    assert len([key for key in keys if key.startswith("work1")]) == 1, keys
    assert len([key for key in keys if key.startswith("coeff")]) == 1, keys
    assert len([key for key in keys if key.startswith("table")]) == 1, keys

    first_call_banks = {
        _addr(attrs, "out0") // BANK_SIZE,
        _addr(attrs, "work0") // BANK_SIZE,
        _addr(attrs, "work1") // BANK_SIZE,
        _addr(attrs, "coeff") // BANK_SIZE,
        _addr(attrs, "table") // BANK_SIZE,
    }
    second_call_banks = first_call_banks | {_addr(attrs, "out1") // BANK_SIZE}

    assert len(first_call_banks) == 5
    assert len(second_call_banks) >= 5, (keys, first_call_banks, second_call_banks)


if __name__ == "__main__":
    test_tpu_gemm_buffers_are_bank_separated()
    test_elementwise_reads_are_bank_separated_while_outputs_remain_flexible()
    test_reduce_tmp_is_separated_from_input_bank()
    test_exp_hidden_workspace_is_reused_for_multiple_calls()
