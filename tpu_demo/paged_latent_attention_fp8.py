"""Unified model entry for FP8 paged latent attention.

This file contains the TileLang TPU kernels, the Soph DeepSeek V3 model ABI
adapter, and the ``torch.ops.my_ops.paged_latent_attention_fp8`` registration.
Like the original MLA implementation, the public operator dispatches by
``attention_mode`` while Prefill and Decode remain separate kernels.  The
current Prefill scope is batch 1, cache length 0, a fixed query length,
additive causal mask, BF16 KV/PE cache, and FP8 Q-B/KV-B weights.
"""

import ctypes
from dataclasses import dataclass
import os
import shutil
import threading

import torch
import tilelang
import tilelang.language as T


T.copy = T.ppl_copy


_KERNEL_ENV_LOCK = threading.Lock()


def _patch_generated_head_multicore(loop_var: str, head_count: int, block_num: int) -> None:
    """Patch the just-generated TPU kernel to split one head loop by block."""
    kernel_dir = os.environ["TPU_KERNEL_PATH"]
    kernel_c = os.path.join(kernel_dir, "kernel.c")
    kernel_cpp = os.path.join(kernel_dir, "kernel.cpp")

    with open(kernel_c, "r") as f:
        c_src = f.read()
    loop = f"for (int {loop_var} = 0; {loop_var} < {head_count}; ++{loop_var}) {{"
    patched_loop = (
        f"int __pla_block_num = tpu_core_num();\n"
        f"  if (__pla_block_num > {block_num}) {{\n"
        f"    __pla_block_num = {block_num};\n"
        f"  }}\n"
        f"  int __pla_block_idx = tpu_core_index();\n"
        f"  if (__pla_block_idx >= __pla_block_num) {{\n"
        f"    return;\n"
        f"  }}\n"
        f"  int __pla_h_per_block = ({head_count} + __pla_block_num - 1) / __pla_block_num;\n"
        f"  int __pla_h_begin = __pla_block_idx * __pla_h_per_block;\n"
        f"  int __pla_h_end = MIN(__pla_h_begin + __pla_h_per_block, {head_count});\n"
        f"  for (int {loop_var} = __pla_h_begin; {loop_var} < __pla_h_end; ++{loop_var}) {{"
    )
    if loop not in c_src:
        raise RuntimeError(f"expected generated head loop not found: {loop}")
    c_src = c_src.replace(loop, patched_loop, 1)
    with open(kernel_c, "w") as f:
        f.write(c_src)

    with open(kernel_cpp, "r") as f:
        cpp_src = f.read()
    cpp_src = cpp_src.replace("int core_num = 1;", f"int core_num = {block_num};", 1)
    cpp_src = cpp_src.replace("int block_num = 1;", "int block_num = core_num;", 1)
    with open(kernel_cpp, "w") as f:
        f.write(cpp_src)


def _patch_generated_attention_multicore(head_count: int, block_num: int) -> None:
    """Patch the attention kernel: core0 updates cache, all cores split heads."""
    kernel_dir = os.environ["TPU_KERNEL_PATH"]
    kernel_c = os.path.join(kernel_dir, "kernel.c")
    kernel_cpp = os.path.join(kernel_dir, "kernel.cpp")

    with open(kernel_c, "r") as f:
        c_src = f.read()

    loop = f"for (int by = 0; by < {head_count}; ++by) {{"
    patched_loop = (
        f"int __pla_block_num = tpu_core_num();\n"
        f"  if (__pla_block_num > {block_num}) {{\n"
        f"    __pla_block_num = {block_num};\n"
        f"  }}\n"
        f"  int __pla_block_idx = tpu_core_index();\n"
        f"  if (__pla_block_idx >= __pla_block_num) {{\n"
        f"    return;\n"
        f"  }}\n"
        f"  int __pla_h_per_block = ({head_count} + __pla_block_num - 1) / __pla_block_num;\n"
        f"  int __pla_h_begin = __pla_block_idx * __pla_h_per_block;\n"
        f"  int __pla_h_end = MIN(__pla_h_begin + __pla_h_per_block, {head_count});\n"
        f"  for (int by = __pla_h_begin; by < __pla_h_end; ++by) {{"
    )
    if loop not in c_src:
        raise RuntimeError(f"expected generated attention head loop not found: {loop}")
    c_src = c_src.replace(loop, patched_loop, 1)

    cache_start = "    __ppl_tensor_info pe_new = "
    cache_end = (
        "    {\n"
        "    dim4 __gather_block_shape = {1, 1, 2, 1024};\n"
        "    tpu_gdma_h_gather_S2S(v30.addr, v22.addr, v27.addr, false, (scalar_t){.u32 = 0}, &__gather_block_shape, 2, NULL, NULL, NULL, DT_BFP16);\n"
        "    }\n"
    )
    start = c_src.find(cache_start)
    end = c_src.find(cache_end, start)
    if start < 0 or end < 0:
        raise RuntimeError("expected attention cache/gather block not found")
    end += len("    }\n")
    cache_block = c_src[start:end]
    guarded = (
        "    if (__pla_block_idx == 0 && by == __pla_h_begin) {\n"
        + "".join("  " + line if line.strip() else line for line in cache_block.splitlines(True))
        + "    }\n"
        + "    if (by == __pla_h_begin) {\n"
        + "      tpu_sync_core();\n"
        + "    }\n"
    )
    c_src = c_src[:start] + guarded + c_src[end:]

    output_store = (
        "    tpu_gdma_cpy_L2S(output.addr, out_s_2.addr, &output.shape, "
        "(output.default_stride ? NULL : &output.stride), "
        "(out_s_2.default_stride ? NULL : &out_s_2.stride), DT_BFP16);\n"
    )
    if output_store not in c_src:
        raise RuntimeError("expected attention output store not found")
    with open(kernel_c, "w") as f:
        f.write(c_src)

    with open(kernel_cpp, "r") as f:
        cpp_src = f.read()
    cpp_src = cpp_src.replace("int core_num = 1;", f"int core_num = {block_num};", 1)
    cpp_src = cpp_src.replace("int block_num = 1;", "int block_num = core_num;", 1)
    with open(kernel_cpp, "w") as f:
        f.write(cpp_src)


def _patch_generated_fused_multicore(head_count: int, block_num: int) -> None:
    """Patch fused kernel: split heads, run cache update/gather once."""
    kernel_dir = os.environ["TPU_KERNEL_PATH"]
    kernel_c = os.path.join(kernel_dir, "kernel.c")
    kernel_cpp = os.path.join(kernel_dir, "kernel.cpp")

    with open(kernel_c, "r") as f:
        c_src = f.read()

    loop = f"for (int by = 0; by < {head_count}; ++by) {{"
    patched_loop = (
        f"int __pla_block_num = tpu_core_num();\n"
        f"  if (__pla_block_num > {block_num}) {{\n"
        f"    __pla_block_num = {block_num};\n"
        f"  }}\n"
        f"  int __pla_block_idx = tpu_core_index();\n"
        f"  if (__pla_block_idx >= __pla_block_num) {{\n"
        f"    return;\n"
        f"  }}\n"
        f"  int __pla_h_per_block = ({head_count} + __pla_block_num - 1) / __pla_block_num;\n"
        f"  int __pla_h_begin = __pla_block_idx * __pla_h_per_block;\n"
        f"  int __pla_h_end = MIN(__pla_h_begin + __pla_h_per_block, {head_count});\n"
        f"  for (int by = __pla_h_begin; by < __pla_h_end; ++by) {{"
    )
    if loop not in c_src:
        raise RuntimeError(f"expected generated fused head loop not found: {loop}")
    c_src = c_src.replace(loop, patched_loop, 1)

    rope_start = c_src.find("    __ppl_tensor_info pe_new = ")
    rope_end = c_src.find("    __ppl_tensor_info k_rotated_1 =", rope_start)
    if rope_start < 0 or rope_end < 0:
        raise RuntimeError("expected fused K-RoPE block not found")
    rope_block = c_src[rope_start:rope_end]
    rope_guarded = (
        "    if (by == __pla_h_begin) {\n"
        + "".join("  " + line if line.strip() else line for line in rope_block.splitlines(True))
        + "    }\n"
    )
    c_src = c_src[:rope_start] + rope_guarded + c_src[rope_end:]

    cache_start = "    tpu_gdma_cpy_L2S(pe_new_rope.addr, "
    start = c_src.find(cache_start)
    # The generated gather width depends on the selected tile/core schedule;
    # locate the PE-cache gather by its stable buffer operands instead of a
    # hard-coded dim4 (the old 2x1024 form becomes 8x1024 for block_n=64).
    gather = c_src.find(
        "    tpu_gdma_h_gather_S2S(v32.addr, v24.addr, v29.addr,", start
    )
    end = c_src.find("    }\n", gather) if gather >= 0 else -1
    if start < 0 or end < 0:
        raise RuntimeError("expected fused cache/gather block not found")
    end += len("    }\n")
    cache_block = c_src[start:end]
    guarded = (
        "    if (__pla_block_idx == 0 && by == __pla_h_begin) {\n"
        + "".join("  " + line if line.strip() else line for line in cache_block.splitlines(True))
        + "    }\n"
        + "    if (by == __pla_h_begin) {\n"
        + "      tpu_sync_core();\n"
        + "    }\n"
    )
    c_src = c_src[:start] + guarded + c_src[end:]

    query_start = c_src.find("    __ppl_tensor_info query = ", start)
    query_end = c_src.find("    for (int ib = 0;", query_start)
    if query_start < 0 or query_end < 0:
        raise RuntimeError("expected fused query load block not found")
    query_block = c_src[query_start:query_end]
    query_guarded = (
        "    if (by == __pla_h_begin) {\n"
        + "".join("  " + line if line.strip() else line for line in query_block.splitlines(True))
        + "    }\n"
    )
    c_src = c_src[:query_start] + query_guarded + c_src[query_end:]

    with open(kernel_c, "w") as f:
        f.write(c_src)

    with open(kernel_cpp, "r") as f:
        cpp_src = f.read()
    cpp_src = cpp_src.replace("int core_num = 1;", f"int core_num = {block_num};", 1)
    cpp_src = cpp_src.replace("int block_num = 1;", "int block_num = core_num;", 1)
    with open(kernel_cpp, "w") as f:
        f.write(cpp_src)


def _rebuild_current_tpu_artifacts() -> tuple[str, str]:
    from tilelang.jit.adapter.libgen import LibraryGenerator

    target = tilelang.tvm.target.Target("tpu")
    generator = LibraryGenerator(target)
    generator.compile_lib(timeout=180)
    return generator.libpath, os.environ["PPL_KERNEL_PATH"]


def _patch_generated_single_run() -> None:
    """Disable the generated launcher's built-in 5+10 benchmark repeats."""
    main_cpp = os.path.join(os.environ["TPU_KERNEL_PATH"], "main.cpp")
    with open(main_cpp, "r") as f:
        source = f.read()
    benchmark_start = source.find("  // 性能测试")
    copy_back_start = source.find("  // 拷贝输出数据回主机",
                                  benchmark_start)
    if benchmark_start < 0 or copy_back_start < 0:
        raise RuntimeError("expected generated benchmark section not found")
    source = source[:benchmark_start] + source[copy_back_start:]
    with open(main_cpp, "w") as f:
        f.write(source)


def _bind_unique_so(kernel, name: str, multicore_head_loop: tuple[str, int, int] | None = None, multicore_attention: tuple[int, int] | None = None, multicore_fused: tuple[int, int] | None = None):
    """Bind matching host and device libraries for one compiled TPU kernel."""
    kernel_dir = os.environ["TPU_KERNEL_PATH"]
    bound_dir = os.path.join(kernel_dir, "pla_bound")
    os.makedirs(bound_dir, exist_ok=True)
    rebuilt_artifacts = None
    if multicore_head_loop is not None:
        _patch_generated_head_multicore(*multicore_head_loop)
    if multicore_attention is not None:
        _patch_generated_attention_multicore(*multicore_attention)
    if multicore_fused is not None:
        _patch_generated_fused_multicore(*multicore_fused)
    if os.environ.get("PLA_TPU_SINGLE_RUN") == "1":
        _patch_generated_single_run()
    if (multicore_head_loop is not None or multicore_attention is not None
            or multicore_fused is not None
            or os.environ.get("PLA_TPU_SINGLE_RUN") == "1"):
        rebuilt_artifacts = _rebuild_current_tpu_artifacts()
    if rebuilt_artifacts is None:
        host_source = os.path.join(kernel_dir, "main.so")
        device_source = os.path.join(kernel_dir, "libkernel.so")
        # Some TPU compiler revisions only emit the numbered intermediate
        # libraries during ``tilelang.compile``.  Materialize the conventional
        # pair before binding when that happens.
        if not os.path.exists(host_source) or not os.path.exists(device_source):
            host_source, device_source = _rebuild_current_tpu_artifacts()
    else:
        host_source, device_source = rebuilt_artifacts
    # LibraryGenerator clears top-level ``*.so`` files before every TPU build.
    # Keep bound host/device pairs in a subdirectory so compiling another
    # kernel cannot invalidate an already-created callable.
    host_target = os.path.join(bound_dir, f"main_{name}.so")
    device_target = os.path.join(bound_dir, f"libkernel_{name}.so")
    shutil.copy(host_source, host_target)
    shutil.copy(device_source, device_target)
    for filename in ("kernel.c", "kernel.cpp", "kernel.h", "main.cpp"):
        source = os.path.join(kernel_dir, filename)
        stem, extension = os.path.splitext(filename)
        target = os.path.join(kernel_dir, f"{stem}_{name}{extension}")
        if os.path.exists(source):
            shutil.copy(source, target)
    kernel.adapter.lib = ctypes.CDLL(host_target)
    kernel.adapter.lib.tilelang_tpu_run.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    kernel.adapter.lib.tilelang_tpu_run.restype = ctypes.c_int
    original_forward = kernel.torch_function

    def isolated_forward(*args):
        with _KERNEL_ENV_LOCK:
            previous = os.environ.get("PPL_KERNEL_PATH")
            os.environ["PPL_KERNEL_PATH"] = device_target
            try:
                return original_forward(*args)
            finally:
                if previous is None:
                    os.environ.pop("PPL_KERNEL_PATH", None)
                else:
                    os.environ["PPL_KERNEL_PATH"] = previous

    kernel.adapter.func = isolated_forward
    kernel.torch_function = isolated_forward
    return kernel


def paged_latent_attention_fp8_kernel(
    batch: int,
    heads: int,
    total_rows: int,
    seqlen_kv: int,
    latent_dim: int,
    rope_dim: int,
    block_n: int = 32,
    block_h: int = 1,
    paged_block_size: int = 16,
    score_scale: float | None = None,
):
    """Build the verified BF16-cache TileLang decode kernel.

    ``fp8`` in the model operator name describes Q-B/KV-B weights. The model's
    paged KV/PE cache is BF16, matching the original PPL ABI.
    """
    dtype = "bfloat16"
    accum_dtype = "float"
    scale = ((latent_dim + rope_dim) ** -0.5
             if score_scale is None else float(score_scale))
    valid_block_h = min(block_h, heads)
    physical_blocks = total_rows // paged_block_size
    logical_blocks = seqlen_kv // paged_block_size
    assert batch == 1
    assert scale > 0.0
    assert total_rows % paged_block_size == 0
    assert seqlen_kv % paged_block_size == 0
    assert seqlen_kv % block_n == 0
    neg_one = T.Cast(dtype, -1.0)

    @T.prim_func
    def main_kernel_inner(
        q: T.Tensor([batch, heads, 1, latent_dim], dtype),
        q_rope: T.Tensor([batch, heads, 1, rope_dim], dtype),
        q_cos: T.Tensor([batch, heads, 1, rope_dim], dtype),
        q_sin: T.Tensor([batch, heads, 1, rope_dim], dtype),
        kv_cache: T.Tensor([total_rows, latent_dim], dtype),
        pe_cache: T.Tensor([total_rows, rope_dim], dtype),
        kv_new: T.Tensor([1, latent_dim], dtype),
        pe_new: T.Tensor([1, rope_dim], dtype),
        k_cos: T.Tensor([1, rope_dim], dtype),
        k_sin: T.Tensor([1, rope_dim], dtype),
        block_table: T.Tensor([logical_blocks, 1], "uint32"),
        save_slots: T.Tensor([1, 1], "uint32"),
        kv_gather: T.Tensor([seqlen_kv, latent_dim], dtype),
        pe_gather: T.Tensor([seqlen_kv, rope_dim], dtype),
        pe_new_rope: T.Tensor([1, rope_dim], dtype),
        output: T.Tensor([batch, heads, 1, latent_dim], dtype),
    ):
        with T.Kernel(batch, T.ceildiv(heads, valid_block_h), is_cpu=True) as (bx, by):
            k_src = T.alloc_shared([1, rope_dim], dtype)
            k_cos_s = T.alloc_shared([1, rope_dim], dtype)
            k_sin_s = T.alloc_shared([1, rope_dim], dtype)
            k_x_cos = T.alloc_shared([1, rope_dim], dtype)
            k_x_sin = T.alloc_shared([1, rope_dim], dtype)
            k_neg = T.alloc_shared([1, rope_dim], dtype)
            k_neg_sin = T.alloc_shared([1, rope_dim], dtype)
            k_rotated = T.alloc_shared([1, rope_dim], dtype)
            T.copy(pe_new, k_src)
            T.copy(k_cos, k_cos_s)
            T.copy(k_sin, k_sin_s)
            T.ppl_mul(k_x_cos, k_src, k_cos_s)
            T.ppl_mul(k_x_sin, k_src, k_sin_s)
            T.ppl_mul_C(k_neg, k_src, neg_one)
            T.ppl_mul(k_neg_sin, k_neg, k_sin_s)
            T.ppl_rope_add(k_rotated, k_x_cos, k_neg_sin, k_x_cos, k_x_sin)
            T.copy(k_rotated, pe_new_rope)

            T.ppl_scatter(kv_cache, kv_new, save_slots, 1)
            T.ppl_scatter(pe_cache, pe_new_rope, save_slots, 1)
            T.ppl_gather_block(kv_gather, kv_cache, block_table,
                               physical_blocks, logical_blocks,
                               paged_block_size * latent_dim)
            T.ppl_gather_block(pe_gather, pe_cache, block_table,
                               physical_blocks, logical_blocks,
                               paged_block_size * rope_dim)

            q_s = T.alloc_shared([valid_block_h, latent_dim], dtype)
            qr_s = T.alloc_shared([valid_block_h, rope_dim], dtype)
            qc_s = T.alloc_shared([valid_block_h, rope_dim], dtype)
            qs_s = T.alloc_shared([valid_block_h, rope_dim], dtype)
            q_x_cos = T.alloc_shared([valid_block_h, rope_dim], dtype)
            q_x_sin = T.alloc_shared([valid_block_h, rope_dim], dtype)
            q_neg = T.alloc_shared([valid_block_h, rope_dim], dtype)
            q_neg_sin = T.alloc_shared([valid_block_h, rope_dim], dtype)
            q_pe = T.alloc_shared([valid_block_h, rope_dim], dtype)
            kv_s = T.alloc_shared([block_n, latent_dim], dtype)
            pe_s = T.alloc_shared([block_n, rope_dim], dtype)
            prob_s = T.alloc_shared([valid_block_h, block_n], dtype)
            out_s = T.alloc_shared([valid_block_h, latent_dim], dtype)
            score = T.alloc_shared([valid_block_h, block_n], accum_dtype)
            score_pe = T.alloc_shared([valid_block_h, block_n], accum_dtype)
            acc_o = T.alloc_shared([valid_block_h, latent_dim], accum_dtype)
            max_v = T.alloc_shared([valid_block_h, 1], accum_dtype)
            max_prev = T.alloc_shared([valid_block_h, 1], accum_dtype)
            max_curr = T.alloc_shared([valid_block_h, 1], accum_dtype)
            max_pair = T.alloc_shared([valid_block_h, 2], accum_dtype)
            rescale = T.alloc_shared([valid_block_h, 1], accum_dtype)
            score_sum = T.alloc_shared([valid_block_h, 1], accum_dtype)
            logsum = T.alloc_shared([valid_block_h, 1], accum_dtype)

            T.copy(q[bx, by * valid_block_h:(by + 1) * valid_block_h, :, :], q_s)
            T.copy(q_rope[bx, by * valid_block_h:(by + 1) * valid_block_h, :, :], qr_s)
            T.copy(q_cos[bx, by * valid_block_h:(by + 1) * valid_block_h, :, :], qc_s)
            T.copy(q_sin[bx, by * valid_block_h:(by + 1) * valid_block_h, :, :], qs_s)
            T.ppl_mul(q_x_cos, qr_s, qc_s)
            T.ppl_mul(q_x_sin, qr_s, qs_s)
            T.ppl_mul_C(q_neg, qr_s, neg_one)
            T.ppl_mul(q_neg_sin, q_neg, qs_s)
            T.ppl_rope_add(q_pe, q_x_cos, q_neg_sin, q_x_cos, q_x_sin)
            T.ppl_fill(acc_o, T.float32(0))
            T.ppl_fill(logsum, T.float32(0))
            T.ppl_fill(max_v, -T.infinity(accum_dtype))

            for k in T.Pipelined(seqlen_kv // block_n, num_stages=2):
                T.copy(kv_gather[k * block_n:(k + 1) * block_n, :], kv_s)
                T.copy(pe_gather[k * block_n:(k + 1) * block_n, :], pe_s)
                T.ppl_clear(score)
                T.ppl_gemm(q_s, kv_s, score, transpose_B=True)
                T.ppl_clear(score_pe)
                T.ppl_gemm(q_pe, pe_s, score_pe, transpose_B=True)
                T.ppl_add(score, score, score_pe)
                T.copy(max_v, max_prev)
                T.ppl_fill(max_curr, -T.infinity(accum_dtype))
                T.ppl_reduce_max(score, max_curr, dim=1, clear=False)

                # Online softmax must normalize against the running maximum,
                # not just the maximum of the current KV tile.  Keeping only
                # the per-tile maximum makes ``exp(prev_max - curr_max)`` grow
                # above one when a later tile has a smaller maximum, which can
                # overflow for multi-tile (long-context) attention.
                T.copy(max_prev, max_pair[:, 0:1])
                T.copy(max_curr, max_pair[:, 1:2])
                T.ppl_fill(max_v, -T.infinity(accum_dtype))
                T.ppl_reduce_max(max_pair, max_v, dim=1, clear=False)

                prev_scaled = T.alloc_shared([valid_block_h, 1], accum_dtype)
                curr_scaled = T.alloc_shared([valid_block_h, 1], accum_dtype)
                T.ppl_mul_C(prev_scaled, max_prev, scale)
                T.ppl_mul_C(curr_scaled, max_v, scale)
                T.ppl_subtract(rescale, prev_scaled, curr_scaled)
                work0 = T.alloc_shared([valid_block_h, 1], accum_dtype)
                work1 = T.alloc_shared([valid_block_h, 1], accum_dtype)
                coeff = T.alloc_shared([64, 32], accum_dtype)
                table = T.alloc_shared([64, 192], accum_dtype)
                T.ppl_exp2(rescale, work0, work1, coeff, table)

                max_scaled = T.alloc_shared([valid_block_h, 1], accum_dtype)
                T.ppl_mul_C(score, score, scale)
                T.ppl_mul_C(max_scaled, max_v, scale)
                T.ppl_subtract(score, score, max_scaled)
                work2 = T.alloc_shared([valid_block_h, block_n], accum_dtype)
                work3 = T.alloc_shared([valid_block_h, block_n], accum_dtype)
                T.ppl_exp2(score, work2, work3, coeff, table)
                T.ppl_reduce_sum(score, score_sum, dim=1)
                T.copy(score, prob_s)
                T.ppl_mul(logsum, logsum, rescale)
                T.ppl_add(logsum, logsum, score_sum)
                T.ppl_mul(acc_o, acc_o, rescale)
                T.ppl_gemm(prob_s, kv_s, acc_o)

            T.ppl_div(acc_o, acc_o, logsum)
            T.copy(acc_o, out_s)
            T.copy(out_s, output[bx, by * valid_block_h:(by + 1) * valid_block_h, :, :])

    return main_kernel_inner


def fp8_block_dequant_kernel(
    out_features: int,
    in_features: int,
    block_size: int,
):
    """Expand compact block scales and dequantize one FP8 weight on TPU."""
    assert out_features % block_size == 0
    assert in_features % block_size == 0
    out_blocks = out_features // block_size
    in_blocks = in_features // block_size
    fp8_dtype = "e4m3_float8"
    dtype = "bfloat16"

    @T.prim_func
    def main_kernel_inner(
        weight: T.Tensor([out_features, in_features], fp8_dtype),
        expanded_scale: T.Tensor([out_features, in_blocks], dtype),
        output: T.Tensor([out_features, in_features], dtype),
    ):
        with T.Kernel(out_blocks, in_blocks, is_cpu=True) as (ob, ib):
            weight_fp8 = T.alloc_shared([block_size, block_size], fp8_dtype)
            weight_bf16 = T.alloc_shared([block_size, block_size], dtype)
            scale_rows = T.alloc_shared([block_size, 1], dtype)
            dequant = T.alloc_shared([block_size, block_size], dtype)
            T.copy(
                weight[
                    ob * block_size:(ob + 1) * block_size,
                    ib * block_size:(ib + 1) * block_size,
                ],
                weight_fp8,
            )
            T.copy(weight_fp8, weight_bf16)
            T.copy(
                expanded_scale[
                    ob * block_size:(ob + 1) * block_size,
                    ib:ib + 1,
                ],
                scale_rows,
            )
            T.ppl_mul(dequant, weight_bf16, scale_rows)
            T.copy(
                dequant,
                output[
                    ob * block_size:(ob + 1) * block_size,
                    ib * block_size:(ib + 1) * block_size,
                ],
            )

    return main_kernel_inner


def wuq_projection_kernel(
    q_lora_rank: int,
    q_out_features: int,
    block_size: int = 128,
):
    """Fused FP8 dequantization and token WUQ projection on TPU."""
    assert q_out_features % block_size == 0
    assert q_lora_rank % block_size == 0
    out_blocks = q_out_features // block_size
    in_blocks = q_lora_rank // block_size
    fp8_dtype = "e4m3_float8"
    dtype = "bfloat16"
    accum_dtype = "float"

    @T.prim_func
    def main_kernel_inner(
        query: T.Tensor([1, q_lora_rank], dtype),
        wuq: T.Tensor([q_out_features, q_lora_rank], fp8_dtype),
        expanded_scale: T.Tensor([q_out_features, in_blocks], dtype),
        q_full: T.Tensor([1, q_out_features], dtype),
    ):
        with T.Kernel(out_blocks, 1, is_cpu=True) as (ob, _):
            acc = T.alloc_shared([1, block_size], accum_dtype)
            out_shared = T.alloc_shared([1, block_size], dtype)
            T.ppl_clear(acc)
            for ib in T.serial(in_blocks):
                query_shared = T.alloc_shared([1, block_size], dtype)
                weight_fp8 = T.alloc_shared([block_size, block_size], fp8_dtype)
                weight_bf16 = T.alloc_shared([block_size, block_size], dtype)
                scale_rows = T.alloc_shared([block_size, 1], dtype)
                weight_dequant = T.alloc_shared([block_size, block_size], dtype)
                partial = T.alloc_shared([1, block_size], accum_dtype)
                T.copy(
                    query[:, ib * block_size:(ib + 1) * block_size],
                    query_shared,
                )
                T.copy(
                    wuq[
                        ob * block_size:(ob + 1) * block_size,
                        ib * block_size:(ib + 1) * block_size,
                    ],
                    weight_fp8,
                )
                T.copy(weight_fp8, weight_bf16)
                T.copy(
                    expanded_scale[
                        ob * block_size:(ob + 1) * block_size, ib:ib + 1
                    ],
                    scale_rows,
                )
                T.ppl_mul(weight_dequant, weight_bf16, scale_rows)
                T.ppl_clear(partial)
                T.ppl_gemm(
                    query_shared, weight_dequant, partial, transpose_B=True
                )
                T.ppl_add(acc, acc, partial)
            T.copy(acc, out_shared)
            T.copy(
                out_shared,
                q_full[:, ob * block_size:(ob + 1) * block_size],
            )

    return main_kernel_inner


def latent_absorb_kernel(
    heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    nope_dim: int,
    rope_dim: int,
    value_dim: int,
    block_size: int = 128,
):
    """Split Q-RoPE and fuse FP8 WUKV dequantization with absorption."""
    del q_lora_rank
    assert nope_dim == block_size
    assert kv_lora_rank % block_size == 0
    q_head_dim = nope_dim + rope_dim
    wukv_head_dim = nope_dim + value_dim
    in_blocks = kv_lora_rank // block_size
    fp8_dtype = "e4m3_float8"
    dtype = "bfloat16"
    accum_dtype = "float"

    @T.prim_func
    def main_kernel_inner(
        q_full: T.Tensor([1, heads * q_head_dim], dtype),
        wukv: T.Tensor([heads * wukv_head_dim, kv_lora_rank], fp8_dtype),
        expanded_scale: T.Tensor([heads * wukv_head_dim, in_blocks], dtype),
        q_abs: T.Tensor([1, heads, 1, kv_lora_rank], dtype),
        q_rope: T.Tensor([1, heads, 1, rope_dim], dtype),
    ):
        with T.Kernel(1, heads, is_cpu=True) as (_, h):
            q_nope = T.alloc_shared([1, nope_dim], dtype)
            q_rope_shared = T.alloc_shared([1, rope_dim], dtype)
            T.copy(
                q_full[:, h * q_head_dim:h * q_head_dim + nope_dim],
                q_nope,
            )
            T.copy(
                q_full[
                    :, h * q_head_dim + nope_dim:(h + 1) * q_head_dim
                ],
                q_rope_shared,
            )
            T.copy(q_rope_shared, q_rope[:, h:h + 1, :, :])
            for ib in T.serial(in_blocks):
                weight_fp8 = T.alloc_shared([block_size, block_size], fp8_dtype)
                weight_bf16 = T.alloc_shared([block_size, block_size], dtype)
                scale_rows = T.alloc_shared([block_size, 1], dtype)
                weight_dequant = T.alloc_shared([block_size, block_size], dtype)
                acc = T.alloc_shared([1, block_size], accum_dtype)
                out_shared = T.alloc_shared([1, block_size], dtype)
                T.copy(
                    wukv[
                        h * wukv_head_dim:h * wukv_head_dim + block_size,
                        ib * block_size:(ib + 1) * block_size,
                    ],
                    weight_fp8,
                )
                T.copy(weight_fp8, weight_bf16)
                T.copy(
                    expanded_scale[
                        h * wukv_head_dim:h * wukv_head_dim + block_size,
                        ib:ib + 1,
                    ],
                    scale_rows,
                )
                T.ppl_mul(weight_dequant, weight_bf16, scale_rows)
                T.ppl_clear(acc)
                T.ppl_gemm(q_nope, weight_dequant, acc)
                T.copy(acc, out_shared)
                T.copy(
                    out_shared,
                    q_abs[
                        :, h:h + 1, :,
                        ib * block_size:(ib + 1) * block_size,
                    ],
                )

    return main_kernel_inner


def wuv_projection_kernel(
    heads: int,
    kv_lora_rank: int,
    nope_dim: int,
    value_dim: int,
    block_size: int = 128,
):
    """Fuse FP8 WUV dequantization with the output projection on TPU."""
    assert value_dim == block_size
    assert kv_lora_rank % block_size == 0
    wukv_head_dim = nope_dim + value_dim
    in_blocks = kv_lora_rank // block_size
    fp8_dtype = "e4m3_float8"
    dtype = "bfloat16"
    accum_dtype = "float"

    @T.prim_func
    def main_kernel_inner(
        latent: T.Tensor([1, heads, 1, kv_lora_rank], dtype),
        wukv: T.Tensor([heads * wukv_head_dim, kv_lora_rank], fp8_dtype),
        expanded_scale: T.Tensor([heads * wukv_head_dim, in_blocks], dtype),
        output: T.Tensor([1, heads, 1, value_dim], dtype),
    ):
        with T.Kernel(1, heads, is_cpu=True) as (_, h):
            acc = T.alloc_shared([1, value_dim], accum_dtype)
            out_shared = T.alloc_shared([1, value_dim], dtype)
            T.ppl_clear(acc)
            for ib in T.serial(in_blocks):
                latent_shared = T.alloc_shared([1, block_size], dtype)
                weight_fp8 = T.alloc_shared([block_size, block_size], fp8_dtype)
                weight_bf16 = T.alloc_shared([block_size, block_size], dtype)
                scale_rows = T.alloc_shared([block_size, 1], dtype)
                weight_dequant = T.alloc_shared([block_size, block_size], dtype)
                partial = T.alloc_shared([1, value_dim], accum_dtype)
                T.copy(
                    latent[
                        :, h:h + 1, :,
                        ib * block_size:(ib + 1) * block_size,
                    ],
                    latent_shared,
                )
                T.copy(
                    wukv[
                        h * wukv_head_dim + nope_dim:(h + 1) * wukv_head_dim,
                        ib * block_size:(ib + 1) * block_size,
                    ],
                    weight_fp8,
                )
                T.copy(weight_fp8, weight_bf16)
                T.copy(
                    expanded_scale[
                        h * wukv_head_dim + nope_dim:(h + 1) * wukv_head_dim,
                        ib:ib + 1,
                    ],
                    scale_rows,
                )
                T.ppl_mul(weight_dequant, weight_bf16, scale_rows)
                T.ppl_clear(partial)
                T.ppl_gemm(
                    latent_shared, weight_dequant, partial, transpose_B=True
                )
                T.ppl_add(acc, acc, partial)
            T.copy(acc, out_shared)
            T.copy(out_shared, output[:, h:h + 1, :, :])

    return main_kernel_inner


def paged_latent_attention_fp8_fused_kernel(
    heads: int,
    total_rows: int,
    seqlen_kv: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    nope_dim: int,
    rope_dim: int,
    value_dim: int,
    block_size: int = 128,
    block_n: int = 32,
    paged_block_size: int = 16,
    score_scale: float | None = None,
):
    """Fuse WUQ, absorb, paged attention, and WUV into one TPU kernel."""
    assert nope_dim == block_size
    assert value_dim == block_size
    assert q_lora_rank % block_size == 0
    assert kv_lora_rank % block_size == 0
    assert total_rows % paged_block_size == 0
    assert seqlen_kv % paged_block_size == 0
    assert seqlen_kv % block_n == 0
    q_head_dim = nope_dim + rope_dim
    wukv_head_dim = nope_dim + value_dim
    q_in_blocks = q_lora_rank // block_size
    kv_in_blocks = kv_lora_rank // block_size
    q_out_features = heads * q_head_dim
    wukv_out_features = heads * wukv_head_dim
    physical_blocks = total_rows // paged_block_size
    logical_blocks = seqlen_kv // paged_block_size
    scale = ((kv_lora_rank + rope_dim) ** -0.5
             if score_scale is None else float(score_scale))
    fp8_dtype = "e4m3_float8"
    dtype = "bfloat16"
    accum_dtype = "float"
    neg_one = T.Cast(dtype, -1.0)

    @T.prim_func
    def main_kernel_inner(
        query: T.Tensor([1, q_lora_rank], dtype),
        wuq: T.Tensor([q_out_features, q_lora_rank], fp8_dtype),
        wuq_scale_expanded: T.Tensor([q_out_features, q_in_blocks], dtype),
        wukv: T.Tensor([wukv_out_features, kv_lora_rank], fp8_dtype),
        wukv_scale_expanded: T.Tensor([wukv_out_features, kv_in_blocks], dtype),
        kv_cache: T.Tensor([total_rows, kv_lora_rank], dtype),
        pe_cache: T.Tensor([total_rows, rope_dim], dtype),
        kv_new: T.Tensor([1, kv_lora_rank], dtype),
        pe_new: T.Tensor([1, rope_dim], dtype),
        cos: T.Tensor([1, rope_dim], dtype),
        sin: T.Tensor([1, rope_dim], dtype),
        block_table: T.Tensor([logical_blocks, 1], "uint32"),
        save_slots: T.Tensor([1, 1], "uint32"),
        kv_gather: T.Tensor([seqlen_kv, kv_lora_rank], dtype),
        pe_gather: T.Tensor([seqlen_kv, rope_dim], dtype),
        pe_new_rope: T.Tensor([1, rope_dim], dtype),
        output: T.Tensor([1, heads, 1, value_dim], dtype),
    ):
        with T.Kernel(1, heads, is_cpu=True) as (_, by):
            k_src = T.alloc_shared([1, rope_dim], dtype)
            k_cos_s = T.alloc_shared([1, rope_dim], dtype)
            k_sin_s = T.alloc_shared([1, rope_dim], dtype)
            k_x_cos = T.alloc_shared([1, rope_dim], dtype)
            k_x_sin = T.alloc_shared([1, rope_dim], dtype)
            k_neg = T.alloc_shared([1, rope_dim], dtype)
            k_neg_sin = T.alloc_shared([1, rope_dim], dtype)
            k_rotated = T.alloc_shared([1, rope_dim], dtype)
            T.copy(pe_new, k_src)
            T.copy(cos, k_cos_s)
            T.copy(sin, k_sin_s)
            T.ppl_mul(k_x_cos, k_src, k_cos_s)
            T.ppl_mul(k_x_sin, k_src, k_sin_s)
            T.ppl_mul_C(k_neg, k_src, neg_one)
            T.ppl_mul(k_neg_sin, k_neg, k_sin_s)
            T.ppl_rope_add(k_rotated, k_x_cos, k_neg_sin, k_x_cos, k_x_sin)
            T.copy(k_rotated, pe_new_rope)

            T.ppl_scatter(kv_cache, kv_new, save_slots, 1)
            T.ppl_scatter(pe_cache, pe_new_rope, save_slots, 1)
            T.ppl_gather_block(kv_gather, kv_cache, block_table,
                               physical_blocks, logical_blocks,
                               paged_block_size * kv_lora_rank)
            T.ppl_gather_block(pe_gather, pe_cache, block_table,
                               physical_blocks, logical_blocks,
                               paged_block_size * rope_dim)

            q_nope = T.alloc_shared([1, nope_dim], dtype)
            q_rope = T.alloc_shared([1, rope_dim], dtype)
            q_full_acc = T.alloc_shared([1, q_head_dim], accum_dtype)
            q_abs = T.alloc_shared([1, kv_lora_rank], dtype)
            T.ppl_clear(q_full_acc)

            query_shared = T.alloc_shared([1, q_lora_rank], dtype)
            T.copy(query, query_shared)
            for ib in T.serial(q_in_blocks):
                query_block = T.alloc_shared([1, block_size], dtype)
                weight_fp8 = T.alloc_shared([q_head_dim, block_size], fp8_dtype)
                weight_bf16 = T.alloc_shared([q_head_dim, block_size], dtype)
                scale_rows = T.alloc_shared([q_head_dim, 1], dtype)
                weight_dequant = T.alloc_shared([q_head_dim, block_size], dtype)
                T.copy(
                    query_shared[:, ib * block_size:(ib + 1) * block_size],
                    query_block,
                )
                T.copy(
                    wuq[
                        by * q_head_dim:(by + 1) * q_head_dim,
                        ib * block_size:(ib + 1) * block_size,
                    ],
                    weight_fp8,
                )
                T.copy(weight_fp8, weight_bf16)
                T.copy(
                    wuq_scale_expanded[
                        by * q_head_dim:(by + 1) * q_head_dim,
                        ib:ib + 1,
                    ],
                    scale_rows,
                )
                T.ppl_mul(weight_dequant, weight_bf16, scale_rows)
                # Compute the contiguous 128-dim nope and 64-dim rope rows in
                # one projection.  This removes one conversion, scale/mul,
                # and GEMM dispatch per input-rank block while preserving the
                # same FP32 accumulation order along the input dimension.
                T.ppl_gemm(query_block, weight_dequant, q_full_acc,
                           transpose_B=True)

            T.copy(q_full_acc[:, 0:nope_dim], q_nope)
            T.copy(q_full_acc[:, nope_dim:q_head_dim], q_rope)

            # Mirror the PPL decode kernel's WUKV preparation: load and
            # dequantize the K/V rows together, then keep the V half in
            # transient local storage across the attention calculation.
            wukv_v_local = T.alloc_shared([value_dim, kv_lora_rank], dtype)
            for ib in T.serial(kv_in_blocks):
                weight_fp8 = T.alloc_shared(
                    [wukv_head_dim, block_size], fp8_dtype
                )
                weight_bf16 = T.alloc_shared(
                    [wukv_head_dim, block_size], dtype
                )
                scale_rows = T.alloc_shared([wukv_head_dim, 1], dtype)
                k_weight_block = T.alloc_shared(
                    [block_size, block_size], dtype
                )
                acc = T.alloc_shared([1, block_size], accum_dtype)
                T.copy(
                    wukv[
                        by * wukv_head_dim:(by + 1) * wukv_head_dim,
                        ib * block_size:(ib + 1) * block_size,
                    ],
                    weight_fp8,
                )
                T.copy(weight_fp8, weight_bf16)
                T.copy(
                    wukv_scale_expanded[
                        by * wukv_head_dim:(by + 1) * wukv_head_dim,
                        ib:ib + 1,
                    ],
                    scale_rows,
                )
                T.ppl_mul(weight_bf16, weight_bf16, scale_rows)
                T.copy(weight_bf16[0:block_size, :], k_weight_block)
                T.copy(
                    weight_bf16[block_size:wukv_head_dim, :],
                    wukv_v_local[
                        :, ib * block_size:(ib + 1) * block_size
                    ],
                )
                T.ppl_clear(acc)
                T.ppl_gemm(q_nope, k_weight_block, acc)
                T.copy(
                    acc,
                    q_abs[:, ib * block_size:(ib + 1) * block_size],
                )

            q_x_cos = T.alloc_shared([1, rope_dim], dtype)
            q_x_sin = T.alloc_shared([1, rope_dim], dtype)
            q_neg = T.alloc_shared([1, rope_dim], dtype)
            q_neg_sin = T.alloc_shared([1, rope_dim], dtype)
            q_pe = T.alloc_shared([1, rope_dim], dtype)
            T.ppl_mul(q_x_cos, q_rope, k_cos_s)
            T.ppl_mul(q_x_sin, q_rope, k_sin_s)
            T.ppl_mul_C(q_neg, q_rope, neg_one)
            T.ppl_mul(q_neg_sin, q_neg, k_sin_s)
            T.ppl_rope_add(q_pe, q_x_cos, q_neg_sin, q_x_cos, q_x_sin)

            kv_s = T.alloc_shared([block_n, kv_lora_rank], dtype)
            pe_s = T.alloc_shared([block_n, rope_dim], dtype)
            prob_s = T.alloc_shared([1, block_n], dtype)
            score = T.alloc_shared([1, block_n], accum_dtype)
            score_pe = T.alloc_shared([1, block_n], accum_dtype)
            acc_o = T.alloc_shared([1, kv_lora_rank], accum_dtype)
            score_sum = T.alloc_shared([1, 1], accum_dtype)
            logsum = T.alloc_shared([1, 1], accum_dtype)
            if seqlen_kv == block_n:
                # A full-context tile does not need the online-softmax state
                # used to merge multiple KV tiles.  Removing the previous-max
                # rescale path saves one scalar exp plus several reductions,
                # copies and elementwise operations for every head.
                max_v = T.alloc_shared([1, 1], accum_dtype)
                work2 = T.alloc_shared([1, block_n], accum_dtype)
                work3 = T.alloc_shared([1, block_n], accum_dtype)
                coeff = T.alloc_shared([64, 32], accum_dtype)
                table = T.alloc_shared([64, 192], accum_dtype)
                T.copy(kv_gather, kv_s)
                T.copy(pe_gather, pe_s)
                T.ppl_clear(score)
                T.ppl_gemm(q_abs, kv_s, score, transpose_B=True)
                T.ppl_clear(score_pe)
                T.ppl_gemm(q_pe, pe_s, score_pe, transpose_B=True)
                T.ppl_add(score, score, score_pe)
                T.ppl_mul_C(score, score, scale)
                T.ppl_fill(max_v, -T.infinity(accum_dtype))
                T.ppl_reduce_max(score, max_v, dim=1, clear=False)
                T.ppl_subtract(score, score, max_v)
                T.ppl_exp2(score, work2, work3, coeff, table)
                T.ppl_reduce_sum(score, score_sum, dim=1)
                T.copy(score, prob_s)
                T.ppl_clear(acc_o)
                T.ppl_gemm(prob_s, kv_s, acc_o)
                T.ppl_div(acc_o, acc_o, score_sum)
            else:
                max_v = T.alloc_shared([1, 1], accum_dtype)
                max_prev = T.alloc_shared([1, 1], accum_dtype)
                max_curr = T.alloc_shared([1, 1], accum_dtype)
                max_pair = T.alloc_shared([1, 2], accum_dtype)
                rescale = T.alloc_shared([1, 1], accum_dtype)
                T.ppl_fill(acc_o, T.float32(0))
                T.ppl_fill(logsum, T.float32(0))
                T.ppl_fill(max_v, -T.infinity(accum_dtype))

                for k in T.Pipelined(seqlen_kv // block_n, num_stages=2):
                    T.copy(kv_gather[k * block_n:(k + 1) * block_n, :], kv_s)
                    T.copy(pe_gather[k * block_n:(k + 1) * block_n, :], pe_s)
                    T.ppl_clear(score)
                    T.ppl_gemm(q_abs, kv_s, score, transpose_B=True)
                    T.ppl_clear(score_pe)
                    T.ppl_gemm(q_pe, pe_s, score_pe, transpose_B=True)
                    T.ppl_add(score, score, score_pe)
                    T.copy(max_v, max_prev)
                    T.ppl_fill(max_curr, -T.infinity(accum_dtype))
                    T.ppl_reduce_max(score, max_curr, dim=1, clear=False)

                    # Preserve the global running maximum across KV tiles.
                    T.copy(max_prev, max_pair[:, 0:1])
                    T.copy(max_curr, max_pair[:, 1:2])
                    T.ppl_fill(max_v, -T.infinity(accum_dtype))
                    T.ppl_reduce_max(max_pair, max_v, dim=1, clear=False)

                    prev_scaled = T.alloc_shared([1, 1], accum_dtype)
                    curr_scaled = T.alloc_shared([1, 1], accum_dtype)
                    T.ppl_mul_C(prev_scaled, max_prev, scale)
                    T.ppl_mul_C(curr_scaled, max_v, scale)
                    T.ppl_subtract(rescale, prev_scaled, curr_scaled)
                    work0 = T.alloc_shared([1, 1], accum_dtype)
                    work1 = T.alloc_shared([1, 1], accum_dtype)
                    coeff = T.alloc_shared([64, 32], accum_dtype)
                    table = T.alloc_shared([64, 192], accum_dtype)
                    T.ppl_exp2(rescale, work0, work1, coeff, table)

                    max_scaled = T.alloc_shared([1, 1], accum_dtype)
                    T.ppl_mul_C(score, score, scale)
                    T.ppl_mul_C(max_scaled, max_v, scale)
                    T.ppl_subtract(score, score, max_scaled)
                    work2 = T.alloc_shared([1, block_n], accum_dtype)
                    work3 = T.alloc_shared([1, block_n], accum_dtype)
                    T.ppl_exp2(score, work2, work3, coeff, table)
                    T.ppl_reduce_sum(score, score_sum, dim=1)
                    T.copy(score, prob_s)
                    T.ppl_mul(logsum, logsum, rescale)
                    T.ppl_add(logsum, logsum, score_sum)
                    T.ppl_mul(acc_o, acc_o, rescale)
                    T.ppl_gemm(prob_s, kv_s, acc_o)

                T.ppl_div(acc_o, acc_o, logsum)

            value_acc = T.alloc_shared([1, value_dim], accum_dtype)
            value_out = T.alloc_shared([1, value_dim], dtype)
            T.ppl_clear(value_acc)
            for ib in T.serial(kv_in_blocks):
                latent_shared = T.alloc_shared([1, block_size], dtype)
                v_weight_block = T.alloc_shared(
                    [block_size, block_size], dtype
                )
                T.copy(acc_o[:, ib * block_size:(ib + 1) * block_size],
                       latent_shared)
                T.copy(
                    wukv_v_local[
                        :, ib * block_size:(ib + 1) * block_size
                    ],
                    v_weight_block,
                )
                # Accumulate each input-rank block directly into the final
                # output accumulator, mirroring the attention GEMM path.
                T.ppl_gemm(latent_shared, v_weight_block, value_acc,
                           transpose_B=True)
            T.copy(value_acc, value_out)
            T.copy(value_out, output[:, by:by + 1, :, :])

    return main_kernel_inner


def dequant_block_weight(weight: torch.Tensor, scale: torch.Tensor,
                         block_size: int) -> torch.Tensor:
    """Dequantize one rank-2 FP8 linear weight with block scales."""
    expected = ((weight.shape[0] + block_size - 1) // block_size,
                (weight.shape[1] + block_size - 1) // block_size)
    if weight.ndim != 2 or scale.ndim != 2 or tuple(scale.shape) != expected:
        raise ValueError(f"invalid FP8 weight/scale layout; expected scale {expected}")
    expanded = scale.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    return weight.to(torch.bfloat16) * expanded[:weight.shape[0], :weight.shape[1]].to(torch.bfloat16)


@dataclass(frozen=True)
class PagedLatentAttentionFp8Config:
    num_heads: int = 32
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    value_head_dim: int = 128
    quant_block_size: int = 128
    logical_blocks: int = 2
    paged_cache_block_size: int = 16
    softmax_scale: float = (128 + 64) ** -0.5
    # Long-context tuning: a wider KV/PE tile reduces copy and softmax-merge
    # iterations.  ``None`` selects the measured context-aware default.
    attention_block_n: int | None = None


class PagedLatentAttentionFp8:
    """Persistent model-facing adapter for paged decode mode 3."""

    def __init__(self, config: PagedLatentAttentionFp8Config, total_cache_rows: int):
        self.config = config
        self.total_cache_rows = total_cache_rows
        self.context_rows = config.logical_blocks * config.paged_cache_block_size
        q_out_features = config.num_heads * (
            config.qk_nope_head_dim + config.qk_rope_head_dim
        )
        wukv_out_features = config.num_heads * (
            config.qk_nope_head_dim + config.value_head_dim
        )
        suffix = (
            f"h{config.num_heads}_qr{config.q_lora_rank}_"
            f"kr{config.kv_lora_rank}"
        )
        block_n = (
            self.config.attention_block_n
            if self.config.attention_block_n is not None
            else (
                16 if self.context_rows < 32 else
                128 if self.context_rows >= 128 else 32
            )
        )
        if self.context_rows % block_n != 0:
            raise ValueError(
                f"attention_block_n={block_n} must divide context_rows={self.context_rows}"
            )

        self.kernel = _bind_unique_so(tilelang.compile(
            paged_latent_attention_fp8_fused_kernel(
                config.num_heads,
                total_cache_rows,
                self.context_rows,
                config.q_lora_rank,
                config.kv_lora_rank,
                config.qk_nope_head_dim,
                config.qk_rope_head_dim,
                config.value_head_dim,
                block_size=config.quant_block_size,
                paged_block_size=config.paged_cache_block_size,
                block_n=block_n,
                score_scale=config.softmax_scale,
            ),
            out_idx=[5, 6, 13, 14, 15, 16],
            target="tpu",
        ), f"pla_fused_mc8_{suffix}",
            multicore_fused=(config.num_heads, 8))

        self._scale_cache_key = None
        self._wuq_scale_expanded = None
        self._wukv_scale_expanded = None

    @staticmethod
    def _check_kernel_return(name: str, result) -> None:
        if int(result) != 0:
            raise RuntimeError(f"{name} TPU kernel returned {int(result)}")

    def _prepare_scales(self, wuq, wukv, wuq_scale, wukv_scale):
        """Validate layouts and cache row-expanded block-scale metadata."""
        c = self.config
        q_out_features = c.num_heads * (c.qk_nope_head_dim + c.qk_rope_head_dim)
        wukv_out_features = c.num_heads * (c.qk_nope_head_dim + c.value_head_dim)
        expected = {
            "wuq": (q_out_features, c.q_lora_rank),
            "wukv": (wukv_out_features, c.kv_lora_rank),
            "wuq_scale": (
                q_out_features // c.quant_block_size,
                c.q_lora_rank // c.quant_block_size,
            ),
            "wukv_scale": (
                wukv_out_features // c.quant_block_size,
                c.kv_lora_rank // c.quant_block_size,
            ),
        }
        tensors = {
            "wuq": wuq,
            "wukv": wukv,
            "wuq_scale": wuq_scale,
            "wukv_scale": wukv_scale,
        }
        for name, tensor in tensors.items():
            if tuple(tensor.shape) != expected[name]:
                raise ValueError(
                    f"{name} shape must be {expected[name]}, got {tuple(tensor.shape)}"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        cache_key = (wuq_scale.data_ptr(), wukv_scale.data_ptr())
        if self._scale_cache_key != cache_key:
            self._wuq_scale_expanded = wuq_scale.repeat_interleave(
                c.quant_block_size, dim=0
            ).contiguous()
            self._wukv_scale_expanded = wukv_scale.repeat_interleave(
                c.quant_block_size, dim=0
            ).contiguous()
            self._scale_cache_key = cache_key
        return self._wuq_scale_expanded, self._wukv_scale_expanded

    def __call__(self, out, query, kv_latent, key_pe, wuq, wukv,
                 kv_cache, pe_cache, cos, sin, wuq_scale, wukv_scale,
                 block_tables, save_slots, kvu, mask, input_lengths,
                 cache_lengths, num_heads, *model_scalars):
        # The public model ABI includes generate_token after num_heads.  Keep
        # accepting the earlier direct-adapter ABI so existing Decode-only
        # diagnostics remain usable; torch.ops always uses the public form.
        if len(model_scalars) == 12:
            (generate_token, q_lora_rank, kv_lora_rank, qk_nope_head_dim,
             qk_rope_head_dim, value_head_dim, mask_size, quant_block_size,
             max_paged_block_num, paged_cache_block_size, softmax_scale,
             attention_mode) = model_scalars
        elif len(model_scalars) == 11:
            generate_token = 1
            (q_lora_rank, kv_lora_rank, qk_nope_head_dim,
             qk_rope_head_dim, value_head_dim, mask_size, quant_block_size,
             max_paged_block_num, paged_cache_block_size, softmax_scale,
             attention_mode) = model_scalars
        else:
            raise TypeError("decode expects 11 legacy or 12 model scalar arguments")
        del cache_lengths, mask_size
        c = self.config
        scalars = (num_heads, q_lora_rank, kv_lora_rank, qk_nope_head_dim,
                   qk_rope_head_dim, value_head_dim, quant_block_size,
                   max_paged_block_num, paged_cache_block_size)
        expected = (c.num_heads, c.q_lora_rank, c.kv_lora_rank,
                    c.qk_nope_head_dim, c.qk_rope_head_dim, c.value_head_dim,
                    c.quant_block_size, c.logical_blocks, c.paged_cache_block_size)
        if scalars != expected or generate_token != 1 or attention_mode != 3:
            raise ValueError("model scalar ABI does not match compiled decode kernel")
        if kvu is not None or mask is not None:
            raise ValueError("paged decode expects kvu=None and mask=None")
        if block_tables is None:
            raise ValueError("paged decode requires block_tables")
        if input_lengths.numel() != 1 or int(input_lengths.reshape(-1)[0]) != self.context_rows:
            raise ValueError("runtime context length differs from compiled kernel")
        if abs(float(softmax_scale) - c.softmax_scale) > 1e-8:
            raise ValueError("runtime softmax_scale differs from compiled kernel")

        bf16_inputs = {
            "query": query,
            "kv_latent": kv_latent,
            "key_pe": key_pe,
            "cos": cos,
            "sin": sin,
        }
        for name, tensor in bf16_inputs.items():
            if tensor.dtype != torch.bfloat16:
                raise TypeError(f"{name} must be torch.bfloat16")
        if wuq_scale.dtype != torch.bfloat16 or wukv_scale.dtype != torch.bfloat16:
            raise TypeError("FP8 block scales must be torch.bfloat16")
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None or wuq.dtype != fp8_dtype or wukv.dtype != fp8_dtype:
            raise TypeError("WUQ and WUKV must be torch.float8_e4m3fn")
        if out.dtype != torch.bfloat16:
            raise TypeError("out must be torch.bfloat16")
        if tuple(out.shape) != (1, num_heads, value_head_dim):
            raise ValueError(
                f"out shape must be {(1, num_heads, value_head_dim)}, "
                f"got {tuple(out.shape)}"
            )

        wuq_scale_expanded, wukv_scale_expanded = self._prepare_scales(
            wuq, wukv, wuq_scale, wukv_scale
        )
        kv_flat = kv_cache.reshape(self.total_cache_rows, kv_lora_rank)
        pe_flat = pe_cache.reshape(self.total_cache_rows, qk_rope_head_dim)
        kv_gather = torch.empty(self.context_rows, kv_lora_rank, dtype=torch.bfloat16)
        pe_gather = torch.empty(self.context_rows, qk_rope_head_dim, dtype=torch.bfloat16)
        pe_rotated = torch.empty(1, qk_rope_head_dim, dtype=torch.bfloat16)
        value = out.reshape(1, num_heads, 1, value_head_dim)
        ret = self.kernel(
            query.reshape(1, q_lora_rank).contiguous(),
            wuq,
            wuq_scale_expanded,
            wukv,
            wukv_scale_expanded,
            kv_flat,
            pe_flat,
            kv_latent.reshape(1, kv_lora_rank).contiguous(),
            key_pe.reshape(1, qk_rope_head_dim).contiguous(),
            cos.reshape(1, qk_rope_head_dim).contiguous(),
            sin.reshape(1, qk_rope_head_dim).contiguous(),
            block_tables.reshape(-1, 1).contiguous().view(torch.uint32),
            save_slots.reshape(1, 1).contiguous().view(torch.uint32),
            kv_gather,
            pe_gather,
            pe_rotated,
            value,
        )
        self._check_kernel_return("fused paged latent attention", ret)
        return out


_LIBRARIES = []
_OPERATOR = None
_SCHEMA = """paged_latent_attention_fp8(
 Tensor(a!) out, Tensor query, Tensor kv_latent, Tensor key_pe,
 Tensor wuq, Tensor wukv, Tensor(b!) kv_cache, Tensor(c!) pe_cache,
 Tensor cos, Tensor sin, Tensor wuq_scale, Tensor wukv_scale,
 Tensor? block_tables, Tensor save_slots, Tensor? kvu, Tensor? mask,
 Tensor input_lengths, Tensor cache_lengths, int num_heads, int generate_token,
 int q_lora_rank, int kv_lora_rank, int qk_nope_head_dim,
 int qk_rope_head_dim, int value_head_dim, int mask_size,
 int quant_block_size, int max_paged_block_num, int paged_cache_block_size,
 float softmax_scale, int attention_mode) -> Tensor(a!)""".replace("\n", " ")


def paged_latent_attention_fp8_prefill_cache_kernel(
    seq_len: int,
    total_rows: int,
    kv_lora_rank: int,
    rope_dim: int,
    block_m: int = 32,
):
    """Rotate and scatter one fixed prefill sequence into the paged cache."""
    dtype = "bfloat16"
    neg_one = T.Cast(dtype, -1.0)

    @T.prim_func
    def main_kernel_inner(
        kv_new: T.Tensor([seq_len, kv_lora_rank], dtype),
        pe_new: T.Tensor([seq_len, rope_dim], dtype),
        cos: T.Tensor([seq_len, rope_dim], dtype),
        sin: T.Tensor([seq_len, rope_dim], dtype),
        slots: T.Tensor([seq_len, 1], "uint32"),
        kv_cache: T.Tensor([total_rows, kv_lora_rank], dtype),
        pe_cache: T.Tensor([total_rows, rope_dim], dtype),
        pe_rotated: T.Tensor([seq_len, rope_dim], dtype),
    ):
        with T.Kernel(1, is_cpu=True):
            for qb in T.serial(seq_len // block_m):
                pe_s = T.alloc_shared([block_m, rope_dim], dtype)
                cos_s = T.alloc_shared([block_m, rope_dim], dtype)
                sin_s = T.alloc_shared([block_m, rope_dim], dtype)
                x_cos = T.alloc_shared([block_m, rope_dim], dtype)
                x_sin = T.alloc_shared([block_m, rope_dim], dtype)
                x_neg = T.alloc_shared([block_m, rope_dim], dtype)
                x_neg_sin = T.alloc_shared([block_m, rope_dim], dtype)
                rotated = T.alloc_shared([block_m, rope_dim], dtype)
                T.copy(pe_new[qb * block_m:(qb + 1) * block_m, :], pe_s)
                T.copy(cos[qb * block_m:(qb + 1) * block_m, :], cos_s)
                T.copy(sin[qb * block_m:(qb + 1) * block_m, :], sin_s)
                T.ppl_mul(x_cos, pe_s, cos_s)
                T.ppl_mul(x_sin, pe_s, sin_s)
                T.ppl_mul_C(x_neg, pe_s, neg_one)
                T.ppl_mul(x_neg_sin, x_neg, sin_s)
                T.ppl_rope_add(rotated, x_cos, x_neg_sin, x_cos, x_sin)
                T.copy(rotated,
                       pe_rotated[qb * block_m:(qb + 1) * block_m, :])
            T.ppl_scatter(kv_cache, kv_new, slots, seq_len)
            T.ppl_scatter(pe_cache, pe_rotated, slots, seq_len)

    return main_kernel_inner


def paged_latent_attention_fp8_prefill_head_kernel(
    seq_len: int,
    heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    nope_dim: int,
    rope_dim: int,
    value_dim: int,
    block_size: int = 128,
    score_scale: float | None = None,
):
    """Project FP8 weights and compute full causal prefill, one head per tile."""
    assert q_lora_rank % block_size == 0
    assert kv_lora_rank % block_size == 0
    q_head_dim = nope_dim + rope_dim
    kv_head_dim = nope_dim + value_dim
    q_in_blocks = q_lora_rank // block_size
    kv_in_blocks = kv_lora_rank // block_size
    scale = q_head_dim ** -0.5 if score_scale is None else float(score_scale)
    fp8_dtype = "e4m3_float8"
    dtype = "bfloat16"
    accum_dtype = "float"
    neg_one = T.Cast(dtype, -1.0)

    @T.prim_func
    def main_kernel_inner(
        query: T.Tensor([seq_len, q_lora_rank], dtype),
        kv_new: T.Tensor([seq_len, kv_lora_rank], dtype),
        pe_rotated: T.Tensor([seq_len, rope_dim], dtype),
        cos: T.Tensor([seq_len, rope_dim], dtype),
        sin: T.Tensor([seq_len, rope_dim], dtype),
        wuq: T.Tensor([heads * q_head_dim, q_lora_rank], fp8_dtype),
        wuq_scale: T.Tensor([heads * q_head_dim, q_in_blocks], dtype),
        wukv: T.Tensor([heads * kv_head_dim, kv_lora_rank], fp8_dtype),
        wukv_scale: T.Tensor([heads * kv_head_dim, kv_in_blocks], dtype),
        mask: T.Tensor([seq_len, seq_len], dtype),
        q_projected: T.Tensor([seq_len, heads, q_head_dim], dtype),
        kvu_projected: T.Tensor([seq_len, heads, kv_head_dim], dtype),
        output: T.Tensor([seq_len, heads, value_dim], dtype),
    ):
        with T.Kernel(1, heads, is_cpu=True) as (_, h):
            q_acc = T.alloc_shared([seq_len, q_head_dim], accum_dtype)
            T.ppl_clear(q_acc)
            for ib in T.serial(q_in_blocks):
                q_block = T.alloc_shared([seq_len, block_size], dtype)
                w_fp8 = T.alloc_shared([q_head_dim, block_size], fp8_dtype)
                w_bf16 = T.alloc_shared([q_head_dim, block_size], dtype)
                scales = T.alloc_shared([q_head_dim, 1], dtype)
                T.copy(query[:, ib * block_size:(ib + 1) * block_size], q_block)
                T.copy(
                    wuq[
                        h * q_head_dim:(h + 1) * q_head_dim,
                        ib * block_size:(ib + 1) * block_size,
                    ],
                    w_fp8,
                )
                T.copy(w_fp8, w_bf16)
                T.copy(
                    wuq_scale[
                        h * q_head_dim:(h + 1) * q_head_dim, ib:ib + 1
                    ],
                    scales,
                )
                T.ppl_mul(w_bf16, w_bf16, scales)
                T.ppl_gemm(q_block, w_bf16, q_acc, transpose_B=True)

            kvu_acc = T.alloc_shared([seq_len, kv_head_dim], accum_dtype)
            T.ppl_clear(kvu_acc)
            for ib in T.serial(kv_in_blocks):
                kv_block = T.alloc_shared([seq_len, block_size], dtype)
                w_fp8 = T.alloc_shared([kv_head_dim, block_size], fp8_dtype)
                w_bf16 = T.alloc_shared([kv_head_dim, block_size], dtype)
                scales = T.alloc_shared([kv_head_dim, 1], dtype)
                T.copy(kv_new[:, ib * block_size:(ib + 1) * block_size], kv_block)
                T.copy(
                    wukv[
                        h * kv_head_dim:(h + 1) * kv_head_dim,
                        ib * block_size:(ib + 1) * block_size,
                    ],
                    w_fp8,
                )
                T.copy(w_fp8, w_bf16)
                T.copy(
                    wukv_scale[
                        h * kv_head_dim:(h + 1) * kv_head_dim, ib:ib + 1
                    ],
                    scales,
                )
                T.ppl_mul(w_bf16, w_bf16, scales)
                T.ppl_gemm(kv_block, w_bf16, kvu_acc, transpose_B=True)

            q_nope = T.alloc_shared([seq_len, nope_dim], dtype)
            q_rope = T.alloc_shared([seq_len, rope_dim], dtype)
            T.copy(q_acc[:, 0:nope_dim], q_nope)
            T.copy(q_acc[:, nope_dim:q_head_dim], q_rope)
            T.copy(kvu_acc[:, 0:nope_dim], k_nope)
            T.copy(kvu_acc[:, nope_dim:kv_head_dim], value)
            T.copy(pe_rotated, pe_s)
            T.copy(q_acc, q_projected[:, h:h + 1, :])
            T.copy(kvu_acc, kvu_projected[:, h:h + 1, :])

            cos_s = T.alloc_shared([seq_len, rope_dim], dtype)
            sin_s = T.alloc_shared([seq_len, rope_dim], dtype)
            q_x_cos = T.alloc_shared([seq_len, rope_dim], dtype)
            q_x_sin = T.alloc_shared([seq_len, rope_dim], dtype)
            q_neg = T.alloc_shared([seq_len, rope_dim], dtype)
            q_neg_sin = T.alloc_shared([seq_len, rope_dim], dtype)
            q_pe = T.alloc_shared([seq_len, rope_dim], dtype)
            T.copy(cos, cos_s)
            T.copy(sin, sin_s)
            T.ppl_mul(q_x_cos, q_rope, cos_s)
            T.ppl_mul(q_x_sin, q_rope, sin_s)
            T.ppl_mul_C(q_neg, q_rope, neg_one)
            T.ppl_mul(q_neg_sin, q_neg, sin_s)
            T.ppl_rope_add(q_pe, q_x_cos, q_neg_sin, q_x_cos, q_x_sin)

            score = T.alloc_shared([seq_len, seq_len], accum_dtype)
            score_pe = T.alloc_shared([seq_len, seq_len], accum_dtype)
            mask_f32 = T.alloc_shared([seq_len, seq_len], accum_dtype)
            max_v = T.alloc_shared([seq_len, 1], accum_dtype)
            score_sum = T.alloc_shared([seq_len, 1], accum_dtype)
            work0 = T.alloc_shared([seq_len, seq_len], accum_dtype)
            work1 = T.alloc_shared([seq_len, seq_len], accum_dtype)
            coeff = T.alloc_shared([64, 32], accum_dtype)
            table = T.alloc_shared([64, 192], accum_dtype)
            prob = T.alloc_shared([seq_len, seq_len], dtype)
            out_acc = T.alloc_shared([seq_len, value_dim], accum_dtype)
            out_bf16 = T.alloc_shared([seq_len, value_dim], dtype)

            T.ppl_clear(score)
            T.ppl_gemm(q_nope, k_nope, score, transpose_B=True)
            T.ppl_clear(score_pe)
            T.ppl_gemm(q_pe, pe_s, score_pe, transpose_B=True)
            T.ppl_add(score, score, score_pe)
            T.ppl_mul_C(score, score, scale)
            T.copy(mask, mask_f32)
            T.ppl_add(score, score, mask_f32)
            T.ppl_fill(max_v, -T.infinity(accum_dtype))
            T.ppl_reduce_max(score, max_v, dim=1, clear=False)
            T.ppl_subtract(score, score, max_v)
            # The TPU FP32 exponential approximation is not defined for the
            # usual additive causal-mask sentinel (about -10000).  Values
            # below -80 are already numerically zero for softmax, so clamp to
            # the stable exp domain without changing observable BF16 output.
            T.ppl_fill(work0, T.float32(-80.0))
            T.ppl_max(score, score, work0)
            T.ppl_exp2(score, work0, work1, coeff, table)
            T.ppl_reduce_sum(score, score_sum, dim=1)
            T.copy(score, prob)
            T.ppl_clear(out_acc)
            T.ppl_gemm(prob, value, out_acc)
            T.ppl_div(out_acc, out_acc, score_sum)
            T.copy(out_acc, out_bf16)
            T.copy(out_bf16, output[:, h:h + 1, :])

    return main_kernel_inner


def paged_latent_attention_fp8_prefill_projection_kernel(
    seq_len: int,
    heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    q_head_dim: int,
    kv_head_dim: int,
    block_size: int = 128,
    block_m: int = 32,
    heads_per_group: int = 1,
):
    """Token-tiled FP8 projection with one contiguous head group per core."""
    assert seq_len % block_m == 0
    assert heads % heads_per_group == 0
    head_groups = heads // heads_per_group
    q_group_dim = heads_per_group * q_head_dim
    kv_group_dim = heads_per_group * kv_head_dim
    q_in_blocks = q_lora_rank // block_size
    kv_in_blocks = kv_lora_rank // block_size
    fp8_dtype = "e4m3_float8"
    dtype = "bfloat16"
    accum_dtype = "float"

    @T.prim_func
    def main_kernel_inner(
        query: T.Tensor([seq_len, q_lora_rank], dtype),
        kv_new: T.Tensor([seq_len, kv_lora_rank], dtype),
        wuq: T.Tensor([heads * q_head_dim, q_lora_rank], fp8_dtype),
        wuq_scale: T.Tensor([heads * q_head_dim, q_in_blocks], dtype),
        wukv: T.Tensor([heads * kv_head_dim, kv_lora_rank], fp8_dtype),
        wukv_scale: T.Tensor([heads * kv_head_dim, kv_in_blocks], dtype),
        q_projected: T.Tensor([heads * seq_len, q_head_dim], dtype),
        kvu_projected: T.Tensor([heads * seq_len, kv_head_dim], dtype),
    ):
        with T.Kernel(seq_len // block_m, head_groups,
                      is_cpu=True) as (qb, hg):
            q_acc = T.alloc_shared([block_m, q_group_dim], accum_dtype)
            T.ppl_clear(q_acc)
            for ib in T.serial(q_in_blocks):
                x = T.alloc_shared([block_m, block_size], dtype)
                w_fp8 = T.alloc_shared([q_group_dim, block_size], fp8_dtype)
                w = T.alloc_shared([q_group_dim, block_size], dtype)
                s = T.alloc_shared([q_group_dim, 1], dtype)
                T.copy(query[qb * block_m:(qb + 1) * block_m,
                             ib * block_size:(ib + 1) * block_size], x)
                T.copy(wuq[hg * q_group_dim:(hg + 1) * q_group_dim,
                           ib * block_size:(ib + 1) * block_size], w_fp8)
                T.copy(w_fp8, w)
                T.copy(wuq_scale[hg * q_group_dim:(hg + 1) * q_group_dim,
                                 ib:ib + 1], s)
                T.ppl_mul(w, w, s)
                T.ppl_gemm(x, w, q_acc, transpose_B=True, result_add=True)
            q_out = T.alloc_shared([block_m, q_group_dim], dtype)
            T.copy(q_acc, q_out)
            for hi in T.serial(heads_per_group):
                T.copy(q_out[:, hi * q_head_dim:(hi + 1) * q_head_dim],
                       q_projected[
                           (hg * heads_per_group + hi) * seq_len + qb * block_m:
                           (hg * heads_per_group + hi) * seq_len +
                           (qb + 1) * block_m, :])

            kv_acc = T.alloc_shared([block_m, kv_group_dim], accum_dtype)
            T.ppl_clear(kv_acc)
            for ib in T.serial(kv_in_blocks):
                x = T.alloc_shared([block_m, block_size], dtype)
                w_fp8 = T.alloc_shared([kv_group_dim, block_size], fp8_dtype)
                w = T.alloc_shared([kv_group_dim, block_size], dtype)
                s = T.alloc_shared([kv_group_dim, 1], dtype)
                T.copy(kv_new[qb * block_m:(qb + 1) * block_m,
                              ib * block_size:(ib + 1) * block_size], x)
                T.copy(wukv[hg * kv_group_dim:(hg + 1) * kv_group_dim,
                            ib * block_size:(ib + 1) * block_size], w_fp8)
                T.copy(w_fp8, w)
                T.copy(wukv_scale[hg * kv_group_dim:(hg + 1) * kv_group_dim,
                                  ib:ib + 1], s)
                T.ppl_mul(w, w, s)
                T.ppl_gemm(x, w, kv_acc, transpose_B=True, result_add=True)
            kv_out = T.alloc_shared([block_m, kv_group_dim], dtype)
            T.copy(kv_acc, kv_out)
            for hi in T.serial(heads_per_group):
                T.copy(kv_out[:, hi * kv_head_dim:(hi + 1) * kv_head_dim],
                       kvu_projected[
                           (hg * heads_per_group + hi) * seq_len + qb * block_m:
                           (hg * heads_per_group + hi) * seq_len +
                           (qb + 1) * block_m, :])

    return main_kernel_inner


def paged_latent_attention_fp8_prefill_qk_kernel(
    seq_len: int, heads: int, block_m: int = 32,
):
    """Compute QK scores in a projection-shaped, isolated kernel."""
    assert seq_len % block_m == 0
    padded_seq_len = 384
    dtype = "bfloat16"

    @T.prim_func
    def main_kernel_inner(
        q_packed: T.Tensor([heads * seq_len, 512], dtype),
        k_packed: T.Tensor([heads * padded_seq_len, 512], dtype),
        scores: T.Tensor([heads * seq_len, padded_seq_len], dtype),
    ):
        with T.Kernel(seq_len // block_m, heads, is_cpu=True) as (qb, h):
            score = T.alloc_shared([block_m, padded_seq_len], "float")
            T.ppl_clear(score)
            for ib in T.serial(4):
                q_chunk = T.alloc_shared([block_m, 128], dtype)
                k_chunk = T.alloc_shared([padded_seq_len, 128], dtype)
                T.copy(q_packed[
                    h * seq_len + qb * block_m:
                    h * seq_len + (qb + 1) * block_m,
                    ib * 128:(ib + 1) * 128], q_chunk)
                T.copy(k_packed[
                    h * padded_seq_len:(h + 1) * padded_seq_len,
                    ib * 128:(ib + 1) * 128], k_chunk)
                T.ppl_gemm(q_chunk, k_chunk, score, transpose_B=True,
                           result_add=True)
            score_bf16 = T.alloc_shared([block_m, padded_seq_len], dtype)
            T.copy(score, score_bf16)
            T.copy(score_bf16, scores[
                h * seq_len + qb * block_m:
                h * seq_len + (qb + 1) * block_m, :])

    return main_kernel_inner


def paged_latent_attention_fp8_prefill_attention_kernel(
    seq_len: int, heads: int, value_dim: int, softmax_scale: float,
    block_m: int = 32,
):
    """Run row-wise stable causal softmax and PV entirely on TPU.

    BM1690's current elementwise/reduction lowering is reliable for the M=1
    layout used by Decode, but not for an M=32 score tile.  Keep QK blocked at
    M=32 and serialize only the softmax/PV rows inside each independently
    scheduled (query block, head) work item.
    """
    assert seq_len % block_m == 0
    padded_seq_len = 384
    block_n = 128
    softmax_rows = 16
    dtype = "bfloat16"

    @T.prim_func
    def main_kernel_inner(
        scores: T.Tensor([heads * seq_len, padded_seq_len], dtype),
        additive_mask: T.Tensor([seq_len, padded_seq_len], dtype),
        v_transposed: T.Tensor(
            [heads * padded_seq_len, padded_seq_len], dtype),
        output: T.Tensor([heads * seq_len, value_dim], dtype),
    ):
        with T.Kernel(seq_len // block_m, heads, is_cpu=True) as (qb, h):
            for rg in T.serial(block_m // softmax_rows):
                row = qb * block_m + rg * softmax_rows
                score = T.alloc_shared(
                    [softmax_rows, padded_seq_len], "float")
                mask_bf16 = T.alloc_shared(
                    [softmax_rows, padded_seq_len], dtype)
                mask_acc = T.alloc_shared(
                    [softmax_rows, padded_seq_len], "float")
                prob = T.alloc_shared(
                    [softmax_rows, padded_seq_len], dtype)
                max_v = T.alloc_shared([softmax_rows, 1], "float")
                logsum = T.alloc_shared([softmax_rows, 1], "float")
                work0 = T.alloc_shared(
                    [softmax_rows, padded_seq_len], "float")
                work1 = T.alloc_shared(
                    [softmax_rows, padded_seq_len], "float")
                coeff = T.alloc_shared([64, 32], "float")
                table = T.alloc_shared([64, 192], "float")
                out_acc = T.alloc_shared(
                    [softmax_rows, padded_seq_len], "float")
                out_bf16 = T.alloc_shared(
                    [softmax_rows, padded_seq_len], dtype)

                T.copy(scores[
                    h * seq_len + row:
                    h * seq_len + row + softmax_rows, :], prob)
                T.copy(prob, score)
                T.ppl_mul_C(score, score, softmax_scale)
                T.copy(additive_mask[
                    row:row + softmax_rows, :], mask_bf16)
                T.copy(mask_bf16, mask_acc)
                T.ppl_add(score, score, mask_acc)
                T.ppl_fill(max_v, T.float32(-15000.0))
                T.ppl_reduce_max(score, max_v, dim=1, clear=False)
                T.ppl_subtract(score, score, max_v)
                T.ppl_exp2(score, work0, work1, coeff, table)
                T.ppl_reduce_sum(score, logsum, dim=1)
                T.copy(score, prob)

                T.ppl_clear(out_acc)
                for kb in T.serial(padded_seq_len // block_n):
                    prob_tile = T.alloc_shared([softmax_rows, block_n], dtype)
                    value_t = T.alloc_shared([padded_seq_len, block_n], dtype)
                    T.copy(prob[:, kb * block_n:(kb + 1) * block_n],
                           prob_tile)
                    T.copy(v_transposed[
                        h * padded_seq_len:(h + 1) * padded_seq_len,
                        kb * block_n:(kb + 1) * block_n], value_t)
                    T.ppl_gemm(prob_tile, value_t, out_acc,
                               transpose_B=True, result_add=True)
                T.ppl_div(out_acc, out_acc, logsum)
                T.copy(out_acc, out_bf16)
                T.copy(out_bf16[:, :value_dim], output[
                    h * seq_len + row:
                    h * seq_len + row + softmax_rows, :])

    return main_kernel_inner


@dataclass(frozen=True)
class PagedLatentAttentionFp8PrefillConfig:
    seq_len: int = 192
    num_heads: int = 16
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    value_head_dim: int = 128
    quant_block_size: int = 128
    paged_cache_block_size: int = 16
    softmax_scale: float = 0.07216878364870322
    core_num: int = 8
    attention_block_m: int = 32

    @property
    def logical_blocks(self) -> int:
        return self.seq_len // self.paged_cache_block_size


class PagedLatentAttentionFp8Prefill:
    """Fixed batch-1, zero-cache paged-prefill adapter (attention mode 2)."""

    def __init__(self, config: PagedLatentAttentionFp8PrefillConfig,
                 total_cache_rows: int):
        self.config = config
        self.total_cache_rows = total_cache_rows
        c = config
        if c.seq_len % c.paged_cache_block_size:
            raise ValueError("seq_len must be page aligned in the first prototype")
        if c.seq_len % c.attention_block_m:
            raise ValueError("seq_len must be divisible by attention_block_m")
        # Keep the device module basename below the runtime's 64-byte limit.
        suffix = (f"s{c.seq_len}_h{c.num_heads}_c{c.core_num}_"
                  f"m{c.attention_block_m}")
        self.cache_kernel = _bind_unique_so(
            tilelang.compile(
                paged_latent_attention_fp8_prefill_cache_kernel(
                    c.seq_len, total_cache_rows, c.kv_lora_rank,
                    c.qk_rope_head_dim, 32,
                ),
                out_idx=[5, 6, 7],
                target="tpu",
            ),
            f"pla_prefill_cache_{suffix}",
        )
        q_head_dim = c.qk_nope_head_dim + c.qk_rope_head_dim
        kv_head_dim = c.qk_nope_head_dim + c.value_head_dim
        heads_per_core = (c.num_heads + c.core_num - 1) // c.core_num
        if c.num_heads % c.core_num:
            raise ValueError("Prefill num_heads must be divisible by core_num")
        self.projection_kernel = _bind_unique_so(
            tilelang.compile(
                paged_latent_attention_fp8_prefill_projection_kernel(
                    c.seq_len, c.num_heads, c.q_lora_rank, c.kv_lora_rank,
                    q_head_dim, kv_head_dim, c.quant_block_size, 32,
                    heads_per_core,
                ),
                out_idx=[6, 7],
                target="tpu",
            ),
            f"pla_prefill_projection_{suffix}",
            multicore_head_loop=("hg", c.core_num, c.core_num),
        )
        self.qk_kernel = _bind_unique_so(
            tilelang.compile(
                paged_latent_attention_fp8_prefill_qk_kernel(
                    c.seq_len, c.num_heads, c.attention_block_m,
                ),
                out_idx=[2],
                target="tpu",
            ),
            f"pla_prefill_qk_{suffix}",
            multicore_head_loop=("h", c.num_heads, c.core_num),
        )
        self.attention_kernel = _bind_unique_so(
            tilelang.compile(
                paged_latent_attention_fp8_prefill_attention_kernel(
                    c.seq_len, c.num_heads, c.value_head_dim,
                    c.softmax_scale,
                    c.attention_block_m,
                ),
                out_idx=[3],
                target="tpu",
            ),
            f"pla_prefill_softmax_pv_{suffix}",
            multicore_head_loop=("h", c.num_heads, c.core_num),
        )
        self._scale_cache_key = None
        self._wuq_scale_expanded = None
        self._wukv_scale_expanded = None
        self._slots_cache_key = None
        self._slots = None

    @staticmethod
    def _check_kernel_return(name: str, result) -> None:
        if int(result) != 0:
            raise RuntimeError(f"{name} TPU kernel returned {int(result)}")

    def _prepare_scales(self, wuq, wukv, wuq_scale, wukv_scale):
        c = self.config
        q_rows = c.num_heads * (c.qk_nope_head_dim + c.qk_rope_head_dim)
        kv_rows = c.num_heads * (c.qk_nope_head_dim + c.value_head_dim)
        expected = {
            "wuq": (q_rows, c.q_lora_rank),
            "wukv": (kv_rows, c.kv_lora_rank),
            "wuq_scale": (q_rows // c.quant_block_size,
                           c.q_lora_rank // c.quant_block_size),
            "wukv_scale": (kv_rows // c.quant_block_size,
                            c.kv_lora_rank // c.quant_block_size),
        }
        for name, tensor in (("wuq", wuq), ("wukv", wukv),
                             ("wuq_scale", wuq_scale),
                             ("wukv_scale", wukv_scale)):
            if tuple(tensor.shape) != expected[name] or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous with shape {expected[name]}")
        key = (wuq_scale.data_ptr(), wukv_scale.data_ptr())
        if key != self._scale_cache_key:
            self._wuq_scale_expanded = wuq_scale.repeat_interleave(
                c.quant_block_size, dim=0).contiguous()
            self._wukv_scale_expanded = wukv_scale.repeat_interleave(
                c.quant_block_size, dim=0).contiguous()
            self._scale_cache_key = key
        return self._wuq_scale_expanded, self._wukv_scale_expanded

    def _prepare_slots(self, block_tables):
        c = self.config
        key = (block_tables.data_ptr(), tuple(block_tables.shape))
        if key != self._slots_cache_key:
            blocks = block_tables.reshape(-1)[:c.logical_blocks].cpu().to(torch.int64)
            offsets = torch.arange(c.paged_cache_block_size, dtype=torch.int64)
            slots = (blocks[:, None] * c.paged_cache_block_size + offsets).reshape(-1)
            if int(slots.max()) >= self.total_cache_rows:
                raise ValueError("block table points outside the compiled cache")
            self._slots = slots.to(device=block_tables.device,
                                   dtype=torch.int32).contiguous().view(torch.uint32)
            self._slots_cache_key = key
        return self._slots.reshape(c.seq_len, 1)

    def __call__(self, out, query, kv_latent, key_pe, wuq, wukv,
                 kv_cache, pe_cache, cos, sin, wuq_scale, wukv_scale,
                 block_tables, save_slots, kvu, mask, input_lengths,
                 cache_lengths, num_heads, *model_scalars):
        if len(model_scalars) == 12:
            (generate_token, q_lora_rank, kv_lora_rank, qk_nope_head_dim,
             qk_rope_head_dim, value_head_dim, mask_size, quant_block_size,
             max_paged_block_num, paged_cache_block_size, softmax_scale,
             attention_mode) = model_scalars
        elif len(model_scalars) == 11:
            generate_token = self.config.seq_len
            (q_lora_rank, kv_lora_rank, qk_nope_head_dim,
             qk_rope_head_dim, value_head_dim, mask_size, quant_block_size,
             max_paged_block_num, paged_cache_block_size, softmax_scale,
             attention_mode) = model_scalars
        else:
            raise TypeError("prefill expects 11 legacy or 12 model scalar arguments")
        del save_slots, kvu
        c = self.config
        scalars = (num_heads, q_lora_rank, kv_lora_rank, qk_nope_head_dim,
                   qk_rope_head_dim, value_head_dim, quant_block_size,
                   max_paged_block_num, paged_cache_block_size)
        expected = (c.num_heads, c.q_lora_rank, c.kv_lora_rank,
                    c.qk_nope_head_dim, c.qk_rope_head_dim, c.value_head_dim,
                    c.quant_block_size, c.logical_blocks,
                    c.paged_cache_block_size)
        if (scalars != expected or generate_token != c.seq_len or
                attention_mode != 2):
            raise ValueError("model scalar ABI does not match compiled prefill kernel")
        if mask is None:
            raise ValueError("fixed prefill requires an additive causal mask")
        if block_tables is None:
            raise ValueError("paged prefill requires block_tables")
        if mask_size != c.seq_len:
            raise ValueError("mask_size must equal the fixed prefill sequence length")
        if int(input_lengths.reshape(-1)[0]) != c.seq_len:
            raise ValueError("runtime input length differs from compiled prefill kernel")
        if int(cache_lengths.reshape(-1)[0]) != 0:
            raise ValueError("the first prefill prototype supports cache length 0 only")
        if abs(float(softmax_scale) - c.softmax_scale) > 1e-8:
            raise ValueError("runtime softmax_scale differs from compiled kernel")

        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None or wuq.dtype != fp8_dtype or wukv.dtype != fp8_dtype:
            raise TypeError("WUQ and WUKV must be torch.float8_e4m3fn")
        for name, tensor in (("out", out), ("query", query),
                             ("kv_latent", kv_latent), ("key_pe", key_pe),
                             ("kv_cache", kv_cache), ("pe_cache", pe_cache),
                             ("cos", cos), ("sin", sin), ("mask", mask),
                             ("wuq_scale", wuq_scale),
                             ("wukv_scale", wukv_scale)):
            if tensor.dtype != torch.bfloat16:
                raise TypeError(f"{name} must be torch.bfloat16")

        wuq_scale_e, wukv_scale_e = self._prepare_scales(
            wuq, wukv, wuq_scale, wukv_scale)
        slots = self._prepare_slots(block_tables)
        kv_flat = kv_cache.reshape(self.total_cache_rows, c.kv_lora_rank)
        pe_flat = pe_cache.reshape(self.total_cache_rows, c.qk_rope_head_dim)
        query_2d = query.reshape(c.seq_len, c.q_lora_rank).contiguous()
        kv_2d = kv_latent.reshape(c.seq_len, c.kv_lora_rank).contiguous()
        pe_2d = key_pe.reshape(c.seq_len, c.qk_rope_head_dim).contiguous()
        cos_2d = cos.reshape(c.seq_len, c.qk_rope_head_dim).contiguous()
        sin_2d = sin.reshape(c.seq_len, c.qk_rope_head_dim).contiguous()
        pe_rotated = torch.empty_like(pe_2d)
        ret = self.cache_kernel(kv_2d, pe_2d, cos_2d, sin_2d, slots,
                                kv_flat, pe_flat, pe_rotated)
        self._check_kernel_return("prefill cache update", ret)
        self.last_pe_rotated = pe_rotated
        output = out.reshape(c.seq_len, c.num_heads, c.value_head_dim)
        q_projected = torch.empty(
            c.num_heads * c.seq_len,
            c.qk_nope_head_dim + c.qk_rope_head_dim,
            dtype=torch.bfloat16,
        )
        kvu_projected = torch.empty(
            c.num_heads * c.seq_len,
            c.qk_nope_head_dim + c.value_head_dim,
            dtype=torch.bfloat16,
        )
        ret = self.projection_kernel(
            query_2d, kv_2d, wuq, wuq_scale_e, wukv, wukv_scale_e,
            q_projected, kvu_projected,
        )
        self.last_q_projected = q_projected.reshape(
            c.num_heads, c.seq_len, -1)
        self.last_kvu_projected = kvu_projected.reshape(
            c.num_heads, c.seq_len, -1)
        self._check_kernel_return("prefill projection", ret)
        q_view = q_projected.reshape(c.num_heads, c.seq_len, -1)
        q_rope_raw = q_view[:, :, c.qk_nope_head_dim:]
        cos_heads = cos_2d.unsqueeze(0)
        sin_heads = sin_2d.unsqueeze(0)
        q_even = q_rope_raw[:, :, 0::2]
        q_odd = q_rope_raw[:, :, 1::2]
        q_pe_torch = torch.stack((
            q_even * cos_heads[:, :, 0::2] - q_odd * sin_heads[:, :, 1::2],
            q_odd * cos_heads[:, :, 1::2] + q_even * sin_heads[:, :, 0::2],
        ), dim=-1).reshape(c.num_heads, c.seq_len,
                          c.qk_rope_head_dim).contiguous()
        q_packed = torch.cat((
            q_view[:, :, :c.qk_nope_head_dim], q_pe_torch,
            torch.zeros(c.num_heads, c.seq_len,
                        512 - c.qk_nope_head_dim - c.qk_rope_head_dim,
                        dtype=torch.bfloat16, device=q_projected.device),
        ), dim=-1).contiguous().reshape(c.num_heads * c.seq_len, 512)
        attention_padded_seq_len = 384
        kvu_padded = torch.cat((
            kvu_projected.reshape(c.num_heads, c.seq_len, -1),
            torch.zeros(c.num_heads, attention_padded_seq_len - c.seq_len,
                        c.qk_nope_head_dim + c.value_head_dim,
                        dtype=torch.bfloat16, device=kvu_projected.device),
        ), dim=1).contiguous()
        pe_padded = torch.cat((
            pe_rotated,
            torch.zeros(attention_padded_seq_len - c.seq_len,
                        c.qk_rope_head_dim,
                        dtype=torch.bfloat16, device=pe_rotated.device),
        ), dim=0).contiguous()
        k_packed = torch.cat((
            kvu_padded[:, :, :c.qk_nope_head_dim],
            pe_padded.unsqueeze(0).expand(c.num_heads, -1, -1),
            torch.zeros(c.num_heads, attention_padded_seq_len,
                        512 - c.qk_nope_head_dim - c.qk_rope_head_dim,
                        dtype=torch.bfloat16, device=kvu_projected.device),
        ), dim=-1).contiguous().reshape(
            c.num_heads * attention_padded_seq_len, 512)
        v_packed = kvu_padded[:, :, c.qk_nope_head_dim:].contiguous().reshape(
            c.num_heads * attention_padded_seq_len, c.value_head_dim)
        additive_mask = torch.cat((
            torch.where(
                mask.reshape(c.seq_len, c.seq_len).float() > -1.0,
                torch.zeros((), dtype=torch.bfloat16, device=mask.device),
                torch.full((), -15000.0, dtype=torch.bfloat16,
                           device=mask.device),
            ),
            torch.full((c.seq_len,
                        attention_padded_seq_len - c.seq_len),
                       -15000.0, dtype=torch.bfloat16, device=mask.device),
        ), dim=1).contiguous()
        v_output_padded = torch.cat((
            v_packed.reshape(c.num_heads, attention_padded_seq_len,
                             c.value_head_dim),
            torch.zeros(c.num_heads, attention_padded_seq_len,
                        attention_padded_seq_len - c.value_head_dim,
                        dtype=torch.bfloat16, device=v_packed.device),
        ), dim=-1)
        v_transposed = v_output_padded.transpose(1, 2).contiguous().reshape(
            c.num_heads * attention_padded_seq_len,
            attention_padded_seq_len)
        scores = torch.empty(c.num_heads * c.seq_len,
                             attention_padded_seq_len,
                             dtype=torch.bfloat16)
        ret = self.qk_kernel(q_packed, k_packed, scores)
        self._check_kernel_return("prefill QK", ret)
        self.last_qk_scores = scores
        attention_output = torch.empty(c.num_heads * c.seq_len,
                                       c.value_head_dim,
                                       dtype=torch.bfloat16)
        ret = self.attention_kernel(
            scores, additive_mask, v_transposed, attention_output)
        output.copy_(attention_output.reshape(
            c.num_heads, c.seq_len, c.value_head_dim).permute(1, 0, 2))
        self._check_kernel_return("prefill attention", ret)
        return out


class PagedLatentAttentionFp8Dispatcher:
    """One model-facing callable with separate Prefill and Decode backends."""

    def __init__(self, decode=None, prefill=None):
        self._operators = {}
        if decode is not None:
            self.register(decode)
        if prefill is not None:
            self.register(prefill)

    def register(self, operator) -> None:
        if isinstance(operator, PagedLatentAttentionFp8Prefill):
            mode = 2
        elif isinstance(operator, PagedLatentAttentionFp8):
            mode = 3
        else:
            raise TypeError(f"unsupported paged latent attention adapter: {type(operator)!r}")
        self._operators[mode] = operator

    @property
    def modes(self):
        return tuple(sorted(self._operators))

    def __call__(self, *args):
        if not args:
            raise TypeError("paged_latent_attention_fp8 expects model arguments")
        attention_mode = int(args[-1])
        operator = self._operators.get(attention_mode)
        if operator is None:
            available = ", ".join(str(mode) for mode in self.modes) or "none"
            raise ValueError(
                f"attention_mode={attention_mode} is not installed; "
                f"available modes: {available}"
            )
        return operator(*args)


def install_torch_custom_op(operator=None, *, decode=None, prefill=None) -> None:
    """Install/update the single model-facing ``torch.ops.my_ops`` symbol.

    ``operator`` preserves the original one-adapter call style.  Passing
    ``decode=`` and/or ``prefill=`` makes the two-mode structure explicit.
    Repeated calls add or replace a mode without redefining the Torch schema.
    """
    global _OPERATOR
    if _OPERATOR is None:
        _OPERATOR = PagedLatentAttentionFp8Dispatcher()
    if operator is not None:
        if isinstance(operator, PagedLatentAttentionFp8Dispatcher):
            for mode_operator in operator._operators.values():
                _OPERATOR.register(mode_operator)
        else:
            _OPERATOR.register(operator)
    if decode is not None:
        _OPERATOR.register(decode)
    if prefill is not None:
        _OPERATOR.register(prefill)
    if not _OPERATOR.modes:
        raise ValueError("at least one Prefill or Decode adapter must be provided")
    if _LIBRARIES:
        return
    lib = torch.library.Library("my_ops", "FRAGMENT")
    lib.define(_SCHEMA)
    lib.impl("paged_latent_attention_fp8", lambda *args: _OPERATOR(*args),
             "CompositeExplicitAutograd")
    _LIBRARIES.append(lib)


__all__ = [
    "PagedLatentAttentionFp8",
    "PagedLatentAttentionFp8Config",
    "PagedLatentAttentionFp8Prefill",
    "PagedLatentAttentionFp8PrefillConfig",
    "PagedLatentAttentionFp8Dispatcher",
    "dequant_block_weight",
    "fp8_block_dequant_kernel",
    "install_torch_custom_op",
    "latent_absorb_kernel",
    "paged_latent_attention_fp8_fused_kernel",
    "paged_latent_attention_fp8_prefill_cache_kernel",
    "paged_latent_attention_fp8_prefill_projection_kernel",
    "paged_latent_attention_fp8_prefill_qk_kernel",
    "paged_latent_attention_fp8_prefill_attention_kernel",
    "paged_latent_attention_fp8_prefill_head_kernel",
    "paged_latent_attention_fp8_kernel",
    "wuq_projection_kernel",
    "wuv_projection_kernel",
]
