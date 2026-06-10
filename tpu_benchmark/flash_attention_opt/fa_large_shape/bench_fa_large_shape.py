"""FA large-shape FlashAttention benchmark.

This runner targets the non-causal large-shape FA cases:

B8H8S512: B=8, H=8, S=512, D=128
B8H32S512: B=8, H=32, S=512, D=128
S1024: B=64, H=64, S=1024, D=128
S2048: B=64, H=64, S=2048, D=128

The TileLang-TPU kernel uses head blocking (`BLOCK_H`) so the H=64 case does
not allocate all heads in local memory at once.  Timing still uses the TPU host
template; cmodel runs default to 0 warmup and 1 measured run via environment
variables because emulator correctness checks are otherwise prohibitively slow.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
OPT_DIR = THIS_DIR.parent
BENCHMARK_ROOT = OPT_DIR.parent
FLASH_DIR = BENCHMARK_ROOT / "flash_attention"
REPO_ROOT = BENCHMARK_ROOT.parent
RESULTS_DIR = BENCHMARK_ROOT / "results" / "flash_attention_opt" / "fa_large_shape"
PPL_PL_DIR = FLASH_DIR / "pl"
TL_KERNEL_PATH = REPO_ROOT / "src" / "tl_templates" / "tpu" / "libkernel.so"
sys.path.insert(0, str(BENCHMARK_ROOT))

import tilelang
import tilelang.language as T
DTYPES = {
    "fp16": {
        "tl": "float16",
        "torch": torch.float16,
        "ppl": "fp16",
        "atol": 1e-2,
        "rtol": 1e-2,
    },
    "bf16": {
        "tl": "bfloat16",
        "torch": torch.bfloat16,
        "ppl": "bf16",
        "atol": 2e-2,
        "rtol": 2e-2,
    },
}

CONFIGS = {
    "s1024": {
        "batch": 64,
        "heads": 64,
        "seq_len": 1024,
        "dim": 128,
        "block_m": 256,
        "block_n": 256,
        "block_h": 8,
        "num_stages": 2,
        "ppl_block_m": 64,
        "ppl_block_k": 64,
    },
    "s2048": {
        "batch": 64,
        "heads": 64,
        "seq_len": 2048,
        "dim": 128,
        "block_m": 128,
        "block_n": 512,
        "block_h": 8,
        "num_stages": 2,
        "ppl_block_m": 64,
        "ppl_block_k": 64,
    },
    "b8_h8_s512_d128": {
        "batch": 8,
        "heads": 8,
        "seq_len": 512,
        "dim": 128,
        "block_m": 256,
        "block_n": 256,
        "block_h": 8,
        "num_stages": 2,
        "ppl_block_m": 32,
        "ppl_block_k": 32,
        "check": "full",
        "input_pattern": "random",
    },
    "b8_h32_s512_d128": {
        "batch": 8,
        "heads": 32,
        "seq_len": 512,
        "dim": 128,
        "block_m": 256,
        "block_n": 256,
        "block_h": 8,
        "num_stages": 2,
        "ppl_block_m": 32,
        "ppl_block_k": 32,
        "check": "full",
        "input_pattern": "random",
    },
}

PPL_TEMPLATE = r'''#include "ppl.h"
#include "ppl_wrapper_func.h"

using namespace ppl;

__KERNEL__ void main_kernel({ppl_t} *ptr_q, {ppl_t} *ptr_k, {ppl_t} *ptr_v,
                            {ppl_t} *ptr_out) {{
  int batch = {batch};
  int heads = {heads};
  int seq_len = {seq_len};
  int dim = {dim};
  float sqrt_d = 1.0f / sqrt((float)dim);
  int block_m = {block_m};
  int block_k = {block_k};
  int block_h = heads;

  int order[4] = {{2, 1, 0, 3}};
  dim4 qo_global_shape = {{batch, seq_len, heads, dim}};
  auto q_global = make_gtensor_permute<{ppl_t}>(qo_global_shape, GLOBAL, ptr_q, order);
  auto k_global = make_gtensor_permute<{ppl_t}>(qo_global_shape, GLOBAL, ptr_k, order);
  auto v_global = make_gtensor_permute<{ppl_t}>(qo_global_shape, GLOBAL, ptr_v, order);
  auto out_global = make_gtensor_permute<{ppl_t}>(qo_global_shape, GLOBAL, ptr_out, order);

  dim4 qk_shape = {{block_h, block_m, 1, block_k}};
  dim4 mi_shape = {{block_h, block_m, 1, 1}};
  dim4 acc_shape = {{block_h, block_m, 1, dim}};

  for (int _b = 0; _b < batch; _b++) {{
    for (int _h = 0; _h < heads; _h += block_h) {{
      int real_h = min(block_h, heads - _h);
      for (int _m = 0; _m < seq_len; _m += block_m) {{
        int real_m = min(block_m, seq_len - _m);
        dim4 qi_real_shape = {{real_h, real_m, 1, dim}};
        dim4 qi_offset = {{_h, _m, _b, 0}};
        tensor<{ppl_t}> qi_tensor;
        dma::load(qi_tensor, q_global.sub_view(qi_real_shape, qi_offset));
        tensor<{ppl_t}> qi_scaled;
        tiu::fmul(qi_scaled, qi_tensor, sqrt_d);

        dim4 mi_real = {{real_h, real_m, 1, 1}};
        dim4 acc_real = {{real_h, real_m, 1, dim}};
        auto mi_sub = make_tensor<{ppl_t}>(mi_shape, mi_real);
        auto li_sub = make_tensor<{ppl_t}>(mi_shape, mi_real);
        auto acc_sub = make_tensor<{ppl_t}>(acc_shape, acc_real);
        tiu::fill(mi_sub, -15000);
        tiu::zero(li_sub);
        tiu::zero(acc_sub);

        int block_k_iter = max(min(block_k, seq_len / 2), 1);
        for (int _k = 0; _k < seq_len; _k += block_k_iter) {{
          ppl::enable_pipeline();
          int real_k = min(block_k_iter, seq_len - _k);
          dim4 kvi_real = {{real_h, real_k, 1, dim}};
          dim4 kvi_offset = {{_h, _k, _b, 0}};
          dim4 qk_real = {{real_h, real_m, 1, real_k}};

          tensor<{ppl_t}> ki_tensor, vi_tensor;
          dma::load(ki_tensor, k_global.sub_view(kvi_real, kvi_offset));
          dma::load(vi_tensor, v_global.sub_view(kvi_real, kvi_offset));

          dim4 qk_batch = {{1, real_m, 1, real_k}};
          dim4 qi_batch = {{1, real_m, 1, dim}};
          dim4 ki_batch = {{1, real_k, 1, dim}};
          auto qk_sub = make_tensor<{ppl_t}>(qk_shape, qk_real);
          auto ki_sub = ki_tensor.view(kvi_real);
          for (int i = 0; i < real_h; i++) {{
            dim4 bo = {{i, 0, 0, 0}};
            tiu::fmm2(qk_sub.sub_view(qk_batch, bo),
                      qi_scaled.sub_view(qi_batch, bo),
                      ki_sub.sub_view(ki_batch, bo), false, true, false);
          }}

          tensor<{ppl_t}> max_out, mi_new;
          quick_pooling(max_out, qk_sub, &qk_shape, &qk_real, -15000, 0);
          tiu::fmax(mi_new, mi_sub, max_out);

          tensor<{ppl_t}> alpha, sub_out, li_tmp;
          tiu::fsub(sub_out, mi_sub, mi_new);
          tiu::move(mi_sub, mi_new);
          exp_no_overflow(alpha, sub_out, &mi_shape, &mi_real);
          tiu::fmul(acc_sub, acc_sub, alpha);
          tiu::fmul(li_tmp, li_sub, alpha);

          tensor<{ppl_t}> sub_out1, p_T, sum_val;
          tiu::fsub(sub_out1, qk_sub, mi_new);
          exp_no_overflow(p_T, sub_out1, &qk_shape, &qk_real);
          quick_pooling(sum_val, p_T, &qk_shape, &qk_real, 0, 1);
          tiu::fadd(li_sub, li_tmp, sum_val);

          auto pv = make_tensor<{ppl_t}>(acc_shape, acc_real);
          dim4 pv_batch = {{1, real_m, 1, dim}};
          dim4 p_batch = {{1, real_m, 1, real_k}};
          dim4 vi_batch = {{1, real_k, 1, dim}};
          auto vi_sub = vi_tensor.view(kvi_real);
          for (int i = 0; i < real_h; i++) {{
            dim4 bo = {{i, 0, 0, 0}};
            tiu::fmm2(pv.sub_view(pv_batch, bo),
                      p_T.sub_view(p_batch, bo),
                      vi_sub.sub_view(vi_batch, bo));
          }}
          tiu::fadd(acc_sub, acc_sub, pv);
        }}

        tensor<fp32> li_fp32, div_li;
        tensor<{ppl_t}> div_li_T, final_out;
        tiu::cast(li_fp32, li_sub);
        tiu::fdiv(div_li, 1.0f, li_fp32, 3);
        tiu::cast(div_li_T, div_li);
        tiu::fmul(final_out, acc_sub, div_li_T);
        dma::store(out_global.sub_view(acc_real, qi_offset), final_out);
      }}
    }}
  }}
}}
'''


def tl_flashattn_large(cfg, dtype_label):
    batch = cfg["batch"]
    heads = cfg["heads"]
    seq_len = cfg["seq_len"]
    dim = cfg["dim"]
    block_m = cfg["block_m"]
    block_n = cfg["block_n"]
    block_h = cfg["block_h"]
    num_stages = cfg.get("num_stages", 1)
    dtype = DTYPES[dtype_label]["tl"]
    accum_dtype = "float"
    scale = (1.0 / dim) ** 0.5
    shape = [batch, seq_len, heads, dim]

    @T.macro
    def LoadQ(Q, Q_shared, bx, bh, bz):
        for h in T.serial(block_h):
            T.copy(Q[bz, bx * block_m:(bx + 1) * block_m, bh * block_h + h, :],
                   Q_shared[h, :, :])

    @T.macro
    def LoadK(K, K_shared, k, bh, bz):
        for h in T.serial(block_h):
            T.copy(K[bz, k * block_n:(k + 1) * block_n, bh * block_h + h, :],
                   K_shared[h, :, :])

    @T.macro
    def LoadV(V, V_shared, k, bh, bz):
        for h in T.serial(block_h):
            T.copy(V[bz, k * block_n:(k + 1) * block_n, bh * block_h + h, :],
                   V_shared[h, :, :])

    @T.macro
    def StoreO(O_shared, Output, bx, bh, bz):
        for h in T.serial(block_h):
            T.copy(O_shared[h, :, :],
                   Output[bz, bx * block_m:(bx + 1) * block_m, bh * block_h + h, :])

    @T.macro
    def MMA0(K, K_shared, Q_shared, acc_s, k, bh, bz):
        LoadK(K, K_shared, k, bh, bz)
        T.fill(acc_s, T.float32(0))
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)

    @T.macro
    def MMA1(V, V_shared, acc_s_cast, acc_o, k, bh, bz):
        LoadV(V, V_shared, k, bh, bz)
        T.gemm(acc_s_cast, V_shared, acc_o)

    @T.macro
    def Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, block_max,
                scores_scale, scores_sum, logsum):
        T.copy(scores_max, scores_max_prev)
        T.reduce_max(acc_s, block_max, dim=2)
        T.max(scores_max, scores_max_prev, block_max)
        T.subtract(scores_scale, scores_max_prev, scores_max)
        T.exp(scores_scale)
        T.subtract(acc_s, acc_s, scores_max)
        T.exp(acc_s)
        T.reduce_sum(acc_s, scores_sum, dim=2)
        T.mul(logsum, logsum, scores_scale)
        T.add(logsum, logsum, scores_sum)
        T.copy(acc_s, acc_s_cast)

    @T.prim_func
    def main_kernel_inner(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_m), T.ceildiv(heads, block_h), batch,
                      is_cpu=True) as (bx, bh, bz):
            Q_shared = T.alloc_shared([block_h, block_m, dim], dtype)
            K_shared = T.alloc_shared([block_h, block_n, dim], dtype)
            V_shared = T.alloc_shared([block_h, block_n, dim], dtype)
            O_shared = T.alloc_shared([block_h, block_m, dim], dtype)
            acc_s = T.alloc_shared([block_h, block_m, block_n], accum_dtype)
            acc_s_cast = T.alloc_shared([block_h, block_m, block_n], dtype)
            acc_o = T.alloc_shared([block_h, block_m, dim], accum_dtype)
            scores_max = T.alloc_shared([block_h, block_m, 1], accum_dtype)
            scores_max_prev = T.alloc_shared([block_h, block_m, 1], accum_dtype)
            block_max = T.alloc_shared([block_h, block_m, 1], accum_dtype)
            scores_scale = T.alloc_shared([block_h, block_m, 1], accum_dtype)
            scores_sum = T.alloc_shared([block_h, block_m, 1], accum_dtype)
            logsum = T.alloc_shared([block_h, block_m, 1], accum_dtype)

            LoadQ(Q, Q_shared, bx, bh, bz)
            T.mul_C(Q_shared, Q_shared, T.float32(scale))
            T.fill(acc_o, T.float32(0))
            T.fill(logsum, T.float32(0))
            T.fill(scores_max, -T.infinity(accum_dtype))

            for k in T.Pipelined(T.ceildiv(seq_len, block_n), num_stages=num_stages):
                MMA0(K, K_shared, Q_shared, acc_s, k, bh, bz)
                Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, block_max, scores_scale,
                        scores_sum, logsum)
                T.mul(acc_o, acc_o, scores_scale)
                MMA1(V, V_shared, acc_s_cast, acc_o, k, bh, bz)
            T.div(acc_o, acc_o, logsum)
            T.copy(acc_o, O_shared)
            StoreO(O_shared, Output, bx, bh, bz)

    return main_kernel_inner


def dtype_info(dtype_label):
    return DTYPES[dtype_label]


def tensor_bytes(cfg, dtype_label):
    return cfg["batch"] * cfg["seq_len"] * cfg["heads"] * cfg["dim"] * 2


def format_bytes(nbytes):
    gib = nbytes / (1024**3)
    if gib >= 1:
        return f"{gib:.2f} GiB"
    return f"{nbytes / (1024**2):.2f} MiB"


def make_inputs(cfg, dtype_label, seed, pattern):
    torch.manual_seed(seed)
    info = dtype_info(dtype_label)
    shape = (cfg["batch"], cfg["seq_len"], cfg["heads"], cfg["dim"])
    if pattern == "random":
        q = torch.randn(shape, dtype=info["torch"])
        k = torch.randn(shape, dtype=info["torch"])
        v = torch.randn(shape, dtype=info["torch"])
        return q, k, v
    if pattern == "constant":
        q = torch.zeros(shape, dtype=info["torch"])
        k = torch.zeros(shape, dtype=info["torch"])
        v = torch.ones(shape, dtype=info["torch"])
        return q, k, v
    if pattern == "v-ramp":
        q = torch.zeros(shape, dtype=info["torch"])
        k = torch.zeros(shape, dtype=info["torch"])
        ramp = torch.linspace(-0.5, 0.5, cfg["seq_len"], dtype=torch.float32).to(info["torch"])
        v = ramp.view(1, cfg["seq_len"], 1, 1).expand(shape).contiguous()
        return q, k, v
    raise ValueError(f"Unsupported input pattern: {pattern}")
    return q, k, v


def sample_positions(cfg, max_samples):
    batch = cfg["batch"]
    heads = cfg["heads"]
    seq_len = cfg["seq_len"]
    raw = [
        (0, 0, 0),
        (0, 0, seq_len // 2),
        (0, 0, seq_len - 1),
        (batch // 2, heads // 2, seq_len // 3),
        (batch // 2, heads // 2, (2 * seq_len) // 3),
        (batch - 1, heads - 1, 0),
        (batch - 1, heads - 1, seq_len // 2),
        (batch - 1, heads - 1, seq_len - 1),
    ]
    seen = []
    for item in raw:
        if item not in seen:
            seen.append(item)
    return seen[:max_samples]


def reference_row(q, k, v, b, h, row, dim):
    scores = torch.matmul(k[b, :, h, :].float(), q[b, row, h, :].float()) / math.sqrt(dim)
    probs = F.softmax(scores, dim=0)
    return torch.matmul(probs, v[b, :, h, :].float())


def check_sampled(name, out, q, k, v, cfg, dtype_label, max_samples):
    info = dtype_info(dtype_label)
    samples = sample_positions(cfg, max_samples)
    max_diff = 0.0
    total = 0.0
    count = 0
    for b, h, row in samples:
        ref = reference_row(q, k, v, b, h, row, cfg["dim"])
        got = out[b, row, h, :].float()
        diff = (got - ref).abs()
        max_diff = max(max_diff, diff.max().item())
        total += diff.mean().item()
        count += 1
    avg_diff = total / max(count, 1)
    correct = max_diff <= info["atol"] + info["rtol"] * max(1.0, max_diff)
    result = {
        "name": name,
        "check": "sampled",
        "samples": samples,
        "correct": bool(correct),
        "max_diff": max_diff,
        "avg_diff": avg_diff,
    }
    print(f"  [{name}] sampled_correct={correct} max_diff={max_diff:.6f} avg_diff={avg_diff:.8f}")
    print("RESULT_JSON:" + json.dumps(result, sort_keys=True), file=sys.stderr, flush=True)
    return correct, result


def check_full(name, out, q, k, v, cfg, dtype_label):
    info = dtype_info(dtype_label)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / math.sqrt(cfg["dim"])
    ref = torch.einsum("bhqk,bkhd->bqhd", F.softmax(scores, dim=-1), v.float())
    diff = (out.float() - ref).abs()
    max_diff = diff.max().item()
    avg_diff = diff.mean().item()
    correct = torch.allclose(out.float(), ref, atol=info["atol"], rtol=info["rtol"])
    result = {
        "name": name,
        "check": "full",
        "correct": bool(correct),
        "max_diff": max_diff,
        "avg_diff": avg_diff,
    }
    print(f"  [{name}] full_correct={correct} max_diff={max_diff:.6f} avg_diff={avg_diff:.8f}")
    print("RESULT_JSON:" + json.dumps(result, sort_keys=True), file=sys.stderr, flush=True)
    return bool(correct), result


def check_output(name, out, q, k, v, cfg, dtype_label, check_mode, max_samples):
    if check_mode == "full":
        return check_full(name, out, q, k, v, cfg, dtype_label)
    return check_sampled(name, out, q, k, v, cfg, dtype_label, max_samples)


def render_ppl_pl(path, cfg, dtype_label):
    info = dtype_info(dtype_label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        PPL_TEMPLATE.format(
            ppl_t=info["ppl"],
            batch=cfg["batch"],
            heads=cfg["heads"],
            seq_len=cfg["seq_len"],
            dim=cfg["dim"],
            block_m=cfg["ppl_block_m"],
            block_k=cfg["ppl_block_k"],
        ),
        encoding="utf-8",
    )


def ppl_pl_path(config_name, cfg, dtype_label):
    shape_name = (
        f"b{cfg['batch']}_h{cfg['heads']}_s{cfg['seq_len']}_d{cfg['dim']}"
        if config_name in ("s1024", "s2048") else config_name
    )
    static_path = PPL_PL_DIR / f"flashattn_{dtype_label}_{shape_name}.pl"
    if static_path.exists():
        return static_path
    generated = RESULTS_DIR / "generated_pl" / f"flashattn_{dtype_label}_{shape_name}.pl"
    render_ppl_pl(generated, cfg, dtype_label)
    return generated


def compile_tilelang(cfg, dtype_label, mode):
    print("\n--- tilelang optimized ---", flush=True)
    print(
        "  tile={block_m}x{block_n} block_h={block_h} stages={num_stages} mode={mode}".format(
            block_m=cfg["block_m"],
            block_n=cfg["block_n"],
            block_h=cfg["block_h"],
            num_stages=cfg.get("num_stages", 1),
            mode=mode,
        ),
        flush=True,
    )
    kernel = tilelang.compile(tl_flashattn_large(cfg, dtype_label), out_idx=-1, target="tpu",
                              mode=mode)
    os.environ["PPL_KERNEL_PATH"] = str(TL_KERNEL_PATH)
    return kernel


def compile_ppl(config_name, cfg, dtype_label, mode):
    print("\n--- PPL ---", flush=True)
    print(
        "  block_m={block_m} block_k={block_k} block_h=heads mode={mode}".format(
            block_m=cfg["ppl_block_m"],
            block_k=cfg["ppl_block_k"],
            mode=mode,
        ),
        flush=True,
    )
    from ppl_utils import compile_ppl_kernel

    info = dtype_info(dtype_label)
    shape = (cfg["batch"], cfg["seq_len"], cfg["heads"], cfg["dim"])
    arg_specs = [(shape, info["torch"])] * 4
    return compile_ppl_kernel(str(ppl_pl_path(config_name, cfg, dtype_label)), arg_specs,
                              result_idx=[3], mode=mode)


def run_backend(name, kernel, kernel_path, q, k, v, cfg, dtype_label, args):
    print(f"[stage] {name}: allocate output", flush=True)
    out = torch.zeros_like(q)
    print(f"[stage] {name}: launch", flush=True)
    old_kernel_path = os.environ.get("PPL_KERNEL_PATH")
    if kernel_path:
        os.environ["PPL_KERNEL_PATH"] = kernel_path
    start = time.time()
    try:
        ret = kernel(q, k, v, out)
    finally:
        if old_kernel_path is None:
            os.environ.pop("PPL_KERNEL_PATH", None)
        else:
            os.environ["PPL_KERNEL_PATH"] = old_kernel_path
    print(f"[stage] {name}: returned in {time.time() - start:.3f}s", flush=True)
    correct, result = check_output(name, out, q, k, v, cfg, dtype_label, args.check,
                                   args.max_samples)
    result["returncode"] = int(ret) if isinstance(ret, int) else ret
    return out, correct, result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(CONFIGS), default="b8_h8_s512_d128")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--backend", choices=("tl", "ppl", "both"), default="both")
    parser.add_argument("--mode", choices=("cmodel", "pcie"), default="cmodel")
    parser.add_argument("--check", choices=("sample", "full"), default="sample")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-pattern", choices=("auto", "random", "constant", "v-ramp"),
                        default="auto")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--tl-block-m", type=int)
    parser.add_argument("--tl-block-n", type=int)
    parser.add_argument("--tl-block-h", type=int)
    parser.add_argument("--tl-num-stages", type=int)
    return parser.parse_args()


def run_config(args):
    cfg = CONFIGS[args.config].copy()
    if getattr(args, "tl_block_m", None) is not None:
        cfg["block_m"] = args.tl_block_m
    if getattr(args, "tl_block_n", None) is not None:
        cfg["block_n"] = args.tl_block_n
    if getattr(args, "tl_block_h", None) is not None:
        cfg["block_h"] = args.tl_block_h
    if getattr(args, "tl_num_stages", None) is not None:
        cfg["num_stages"] = args.tl_num_stages
    if cfg["heads"] % cfg["block_h"] != 0:
        raise ValueError("heads must be divisible by block_h for this FA large-shape runner")
    if cfg["seq_len"] % cfg["block_m"] != 0 or cfg["seq_len"] % cfg["block_n"] != 0:
        raise ValueError("seq_len must be divisible by block_m/block_n for this runner")

    tensor_nbytes = tensor_bytes(cfg, args.dtype)
    input_pattern = args.input_pattern
    if input_pattern == "auto":
        input_pattern = CONFIGS[args.config].get("input_pattern", "v-ramp")

    print("=" * 72)
    print(
        "FlashAttention {config} dtype={dtype} B={batch} H={heads} S={seq_len} D={dim} "
        "mode={mode} backend={backend}".format(
            config=args.config,
            dtype=args.dtype,
            mode=args.mode,
            backend=args.backend,
            **cfg,
        )
    )
    print(f"One Q/K/V/O tensor: {format_bytes(tensor_nbytes)}")
    print(f"Input pattern: {input_pattern}")
    print("=" * 72, flush=True)

    kernels = {}
    kernel_paths = {}
    if args.backend in ("tl", "both"):
        kernels["tilelang-opt"] = compile_tilelang(cfg, args.dtype, args.mode)
        kernel_paths["tilelang-opt"] = str(TL_KERNEL_PATH)
        print("[stage] tilelang compile done", flush=True)
    if args.backend in ("ppl", "both"):
        kernels["PPL"] = compile_ppl(args.config, cfg, args.dtype, args.mode)
        kernel_paths["PPL"] = os.environ.get("PPL_KERNEL_PATH")
        print("[stage] PPL compile done", flush=True)
    if args.compile_only:
        print("compile_only=True; skip input allocation and execution")
        return

    print("[stage] make inputs", flush=True)
    start = time.time()
    q, k, v = make_inputs(cfg, args.dtype, args.seed, input_pattern)
    print(f"[stage] inputs ready in {time.time() - start:.3f}s", flush=True)
    results = {}
    outputs = {}
    keep_outputs = args.backend == "both" and tensor_nbytes <= 512 * 1024 * 1024
    ok = True
    for name, kernel in kernels.items():
        out, correct, result = run_backend(name, kernel, kernel_paths.get(name), q, k, v, cfg,
                                           args.dtype, args)
        if keep_outputs:
            outputs[name] = out
        else:
            del out
        results[name] = result
        ok = ok and correct
        gc.collect()

    if "tilelang-opt" in outputs and "PPL" in outputs:
        samples = sample_positions(cfg, args.max_samples)
        max_diff = 0.0
        for b, h, row in samples:
            diff = (outputs["tilelang-opt"][b, row, h, :].float() -
                    outputs["PPL"][b, row, h, :].float()).abs()
            max_diff = max(max_diff, diff.max().item())
        print(f"  [TL-vs-PPL] sampled_max_diff={max_diff:.6f}")

    print("SUMMARY_JSON:" + json.dumps(results, sort_keys=True), file=sys.stderr, flush=True)
    if not ok:
        raise SystemExit(2)


def run_fixed(config_name, dtype_label):
    cfg = CONFIGS[config_name]
    args = argparse.Namespace(
        config=config_name,
        dtype=dtype_label,
        backend="both",
        mode="pcie",
        check=cfg.get("check", "sample"),
        max_samples=8,
        seed=0,
        input_pattern=cfg.get("input_pattern", "v-ramp"),
        compile_only=False,
    )
    run_config(args)


def main():
    run_config(parse_args())


if __name__ == "__main__":
    main()
