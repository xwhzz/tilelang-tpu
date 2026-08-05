"""Final single-file delivery for ``paged_latent_attention_fp8``.

This file contains the TileLang TPU kernels, the Soph DeepSeek V3 model ABI
adapter, and the ``torch.ops.my_ops.paged_latent_attention_fp8`` registration.
The model path keeps FP8 weight dequantization, WUQ projection, latent
absorption, paged attention, and WUV projection on TPU. The supported scope is
batch-1 paged decode with BF16 KV/PE cache and FP8 Q-B/KV-B weights.
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


def _bind_unique_so(kernel, name: str, multicore_head_loop: tuple[str, int, int] | None = None, multicore_attention: tuple[int, int] | None = None, multicore_fused: tuple[int, int] | None = None):
    """Bind matching host and device libraries for one compiled TPU kernel."""
    kernel_dir = os.environ["TPU_KERNEL_PATH"]
    bound_dir = os.path.join(kernel_dir, "pla_bound")
    os.makedirs(bound_dir, exist_ok=True)
    rebuilt_artifacts = None
    if multicore_head_loop is not None:
        _patch_generated_head_multicore(*multicore_head_loop)
        rebuilt_artifacts = _rebuild_current_tpu_artifacts()
    if multicore_attention is not None:
        _patch_generated_attention_multicore(*multicore_attention)
        rebuilt_artifacts = _rebuild_current_tpu_artifacts()
    if multicore_fused is not None:
        _patch_generated_fused_multicore(*multicore_fused)
        rebuilt_artifacts = _rebuild_current_tpu_artifacts()
    if rebuilt_artifacts is None:
        host_source = os.path.join(kernel_dir, "main.so")
        device_source = os.path.join(kernel_dir, "libkernel.so")
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
            q_nope_acc = T.alloc_shared([1, nope_dim], accum_dtype)
            q_rope_acc = T.alloc_shared([1, rope_dim], accum_dtype)
            q_abs = T.alloc_shared([1, kv_lora_rank], dtype)
            T.ppl_clear(q_nope_acc)
            T.ppl_clear(q_rope_acc)

            for ib in T.serial(q_in_blocks):
                query_shared = T.alloc_shared([1, block_size], dtype)
                weight_nope_fp8 = T.alloc_shared([nope_dim, block_size], fp8_dtype)
                weight_nope_bf16 = T.alloc_shared([nope_dim, block_size], dtype)
                scale_nope = T.alloc_shared([nope_dim, 1], dtype)
                weight_nope = T.alloc_shared([nope_dim, block_size], dtype)
                weight_rope_fp8 = T.alloc_shared([rope_dim, block_size], fp8_dtype)
                weight_rope_bf16 = T.alloc_shared([rope_dim, block_size], dtype)
                scale_rope = T.alloc_shared([rope_dim, 1], dtype)
                weight_rope = T.alloc_shared([rope_dim, block_size], dtype)
                T.copy(query[:, ib * block_size:(ib + 1) * block_size],
                       query_shared)
                T.copy(
                    wuq[
                        by * q_head_dim:by * q_head_dim + nope_dim,
                        ib * block_size:(ib + 1) * block_size,
                    ],
                    weight_nope_fp8,
                )
                T.copy(weight_nope_fp8, weight_nope_bf16)
                T.copy(
                    wuq_scale_expanded[
                        by * q_head_dim:by * q_head_dim + nope_dim,
                        ib:ib + 1,
                    ],
                    scale_nope,
                )
                T.ppl_mul(weight_nope, weight_nope_bf16, scale_nope)
                # ppl_gemm accumulates into C.  Keep the projection result in
                # its final FP32 accumulator instead of clearing a temporary
                # tensor and adding that tensor after every input block.
                T.ppl_gemm(query_shared, weight_nope, q_nope_acc,
                           transpose_B=True)

                T.copy(
                    wuq[
                        by * q_head_dim + nope_dim:(by + 1) * q_head_dim,
                        ib * block_size:(ib + 1) * block_size,
                    ],
                    weight_rope_fp8,
                )
                T.copy(weight_rope_fp8, weight_rope_bf16)
                T.copy(
                    wuq_scale_expanded[
                        by * q_head_dim + nope_dim:(by + 1) * q_head_dim,
                        ib:ib + 1,
                    ],
                    scale_rope,
                )
                T.ppl_mul(weight_rope, weight_rope_bf16, scale_rope)
                T.ppl_gemm(query_shared, weight_rope, q_rope_acc,
                           transpose_B=True)

            T.copy(q_nope_acc, q_nope)
            T.copy(q_rope_acc, q_rope)

            for ib in T.serial(kv_in_blocks):
                weight_fp8 = T.alloc_shared([block_size, block_size], fp8_dtype)
                weight_bf16 = T.alloc_shared([block_size, block_size], dtype)
                scale_rows = T.alloc_shared([block_size, 1], dtype)
                weight_dequant = T.alloc_shared([block_size, block_size], dtype)
                acc = T.alloc_shared([1, block_size], accum_dtype)
                out_shared = T.alloc_shared([1, block_size], dtype)
                T.copy(
                    wukv[
                        by * wukv_head_dim:by * wukv_head_dim + block_size,
                        ib * block_size:(ib + 1) * block_size,
                    ],
                    weight_fp8,
                )
                T.copy(weight_fp8, weight_bf16)
                T.copy(
                    wukv_scale_expanded[
                        by * wukv_head_dim:by * wukv_head_dim + block_size,
                        ib:ib + 1,
                    ],
                    scale_rows,
                )
                T.ppl_mul(weight_dequant, weight_bf16, scale_rows)
                T.ppl_clear(acc)
                T.ppl_gemm(q_nope, weight_dequant, acc)
                T.copy(acc, out_shared)
                T.copy(out_shared, q_abs[:, ib * block_size:(ib + 1) * block_size])

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
            max_v = T.alloc_shared([1, 1], accum_dtype)
            max_prev = T.alloc_shared([1, 1], accum_dtype)
            max_curr = T.alloc_shared([1, 1], accum_dtype)
            max_pair = T.alloc_shared([1, 2], accum_dtype)
            rescale = T.alloc_shared([1, 1], accum_dtype)
            score_sum = T.alloc_shared([1, 1], accum_dtype)
            logsum = T.alloc_shared([1, 1], accum_dtype)
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

                # Preserve the global running maximum across KV tiles.  This
                # keeps both the previous accumulator rescale and the current
                # tile probabilities in the same softmax normalization frame.
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
                weight_fp8 = T.alloc_shared([block_size, block_size], fp8_dtype)
                weight_bf16 = T.alloc_shared([block_size, block_size], dtype)
                scale_rows = T.alloc_shared([block_size, 1], dtype)
                weight_dequant = T.alloc_shared([block_size, block_size], dtype)
                T.copy(acc_o[:, ib * block_size:(ib + 1) * block_size],
                       latent_shared)
                T.copy(
                    wukv[
                        by * wukv_head_dim + nope_dim:(by + 1) * wukv_head_dim,
                        ib * block_size:(ib + 1) * block_size,
                    ],
                    weight_fp8,
                )
                T.copy(weight_fp8, weight_bf16)
                T.copy(
                    wukv_scale_expanded[
                        by * wukv_head_dim + nope_dim:(by + 1) * wukv_head_dim,
                        ib:ib + 1,
                    ],
                    scale_rows,
                )
                T.ppl_mul(weight_dequant, weight_bf16, scale_rows)
                # Accumulate each input-rank block directly into the final
                # output accumulator, mirroring the attention GEMM path.
                T.ppl_gemm(latent_shared, weight_dequant, value_acc,
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
                64 if self.context_rows >= 128 else 32
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
                 cache_lengths, num_heads, q_lora_rank, kv_lora_rank,
                 qk_nope_head_dim, qk_rope_head_dim, value_head_dim,
                 mask_size, quant_block_size, max_paged_block_num,
                 paged_cache_block_size, softmax_scale, attention_mode):
        del cache_lengths, mask_size
        c = self.config
        scalars = (num_heads, q_lora_rank, kv_lora_rank, qk_nope_head_dim,
                   qk_rope_head_dim, value_head_dim, quant_block_size,
                   max_paged_block_num, paged_cache_block_size)
        expected = (c.num_heads, c.q_lora_rank, c.kv_lora_rank,
                    c.qk_nope_head_dim, c.qk_rope_head_dim, c.value_head_dim,
                    c.quant_block_size, c.logical_blocks, c.paged_cache_block_size)
        if scalars != expected or attention_mode != 3:
            raise ValueError("model scalar ABI does not match compiled decode kernel")
        if kvu is not None or mask is not None:
            raise ValueError("paged decode expects kvu=None and mask=None")
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
 Tensor wuq, Tensor wukv, Tensor kv_cache, Tensor pe_cache,
 Tensor cos, Tensor sin, Tensor wuq_scale, Tensor wukv_scale,
 Tensor block_tables, Tensor save_slots, Tensor? kvu, Tensor? mask,
 Tensor input_lengths, Tensor cache_lengths, int num_heads,
 int q_lora_rank, int kv_lora_rank, int qk_nope_head_dim,
 int qk_rope_head_dim, int value_head_dim, int mask_size,
 int quant_block_size, int max_paged_block_num, int paged_cache_block_size,
 float softmax_scale, int attention_mode) -> Tensor(a!)""".replace("\n", " ")


def install_torch_custom_op(operator: PagedLatentAttentionFp8) -> None:
    """Install the exact model-facing ``torch.ops.my_ops`` symbol."""
    global _OPERATOR
    _OPERATOR = operator
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
    "dequant_block_weight",
    "fp8_block_dequant_kernel",
    "install_torch_custom_op",
    "latent_absorb_kernel",
    "paged_latent_attention_fp8_fused_kernel",
    "paged_latent_attention_fp8_kernel",
    "wuq_projection_kernel",
    "wuv_projection_kernel",
]
