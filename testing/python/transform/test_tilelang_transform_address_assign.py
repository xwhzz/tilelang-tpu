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


def test_ppl_gemm_output_write_phase_can_share_input_bank():

    @T.prim_func
    def main():
        with T.Kernel(1, is_cpu=True) as _:
            a_shared = T.alloc_shared((64, 1024), "float32")
            b_shared = T.alloc_shared((1024, 64), "float32")
            c_shared = T.alloc_shared((64, 64), "float32")

            T.ppl_fill(c_shared, T.float32(0.0))
            T.ppl_gemm(a_shared, b_shared, c_shared)

    attrs = _assigned_attrs(main)
    a_addr = _addr(attrs, "a_shared")
    b_addr = _addr(attrs, "b_shared")
    c_addr = _addr(attrs, "c_shared")

    assert a_addr // BANK_SIZE != b_addr // BANK_SIZE
    assert c_addr // BANK_SIZE == a_addr // BANK_SIZE
    assert a_addr <= c_addr
    assert c_addr < a_addr + BANK_SIZE


def test_elementwise_reads_are_bank_separated_while_outputs_remain_flexible():

    @T.prim_func
    def main():
        with T.Kernel(1, is_cpu=True) as _:
            src0 = T.alloc_shared((64, 1024), "float32")
            src1 = T.alloc_shared((64, 1024), "float32")
            dst = T.alloc_shared((64, 64), "float32")

            T.ppl_add(dst, src0, src1)

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

            T.ppl_reduce_sum(inp, out, dim=1)

    attrs = _assigned_attrs(main)
    tmp_name = _single_attr_with_prefix(attrs, "tmp_buffer_sum")
    inp_addr = _addr(attrs, "inp")
    tmp_addr = _addr(attrs, tmp_name)

    assert inp_addr // BANK_SIZE != tmp_addr // BANK_SIZE


def test_exp_composite_operands_are_conservative_bank_clique():

    @T.prim_func
    def main():
        with T.Kernel(1, is_cpu=True) as _:
            out = T.alloc_shared((64, 1024), "float32")
            work0 = T.alloc_shared((64, 1024), "float32")
            work1 = T.alloc_shared((64, 1024), "float32")
            coeff = T.alloc_shared((64, 32), "float32")
            table = T.alloc_shared((64, 192), "float32")

            T.ppl_exp2(out, work0, work1, coeff, table)

    attrs = _assigned_attrs(main)
    banks = {
        _addr(attrs, "out") // BANK_SIZE,
        _addr(attrs, "work0") // BANK_SIZE,
        _addr(attrs, "work1") // BANK_SIZE,
        _addr(attrs, "coeff") // BANK_SIZE,
        _addr(attrs, "table") // BANK_SIZE,
    }

    assert len(banks) == 5


if __name__ == "__main__":
    test_ppl_gemm_output_write_phase_can_share_input_bank()
    test_elementwise_reads_are_bank_separated_while_outputs_remain_flexible()
    test_reduce_tmp_is_separated_from_input_bank()
    test_exp_composite_operands_are_conservative_bank_clique()
