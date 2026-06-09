"""Benchmark fused W8A16 dequantized MLP forward on TileLang-TPU and PPL.

The TileLang-TPU path uses `kernel_v2.py`, where int8 weights are assumed to
have already been dequantized to fp16.  PPL baselines use the same fp16 input,
fp16 dequantized weights, and fp16 output layout.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from string import Template

import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_ROOT = os.path.dirname(THIS_DIR)
REPO_ROOT = os.path.dirname(BENCHMARK_ROOT)
RESULTS_DIR = os.path.join(BENCHMARK_ROOT, "results", "mlp_w8a16_dq_forward")

sys.path.insert(0, THIS_DIR)
sys.path.insert(0, BENCHMARK_ROOT)

from kernel_v2 import (
    mlp_w8a16_fused_kernel_ppl_layout_streaming,
    mlp_w8a16_fused_kernel_streaming_fullk,
    mlp_w8a16_fused_kernel_v2,
)


CONFIGS = {
    "tiny": {
        "M": 4,
        "hidden": 128,
        "intermediate": 128,
        "tl_block_M": 4,
        "tl_block_N": 128,
        "ppl_block_w": 128,
        "tl_strategy": "streaming_fullk",
    },
    "stage2": {
        "M": 16,
        "hidden": 256,
        "intermediate": 128,
        "tl_block_M": 16,
        "tl_block_N": 128,
        "ppl_block_w": 128,
        "tl_strategy": "streaming_fullk",
    },
    "stage3": {
        "M": 16,
        "hidden": 256,
        "intermediate": 256,
        "tl_block_M": 16,
        "tl_block_N": 256,
        "ppl_block_w": 128,
        "tl_strategy": "streaming_fullk",
    },
    "small_decode_scaled": {
        "M": 1,
        "hidden": 512,
        "intermediate": 2048,
        "tl_block_M": 1,
        "tl_block_N": 128,
        "ppl_block_w": 128,
        "tl_strategy": "streaming_fullk",
    },
    "medium_prefill_scaled": {
        "M": 16,
        "hidden": 1024,
        "intermediate": 4096,
        "tl_block_M": 16,
        "tl_block_N": 128,
        "ppl_block_w": 128,
        "tl_strategy": "streaming_fullk",
    },
    "moe_decode_dsv3": {
        "M": 1,
        "hidden": 7168,
        "intermediate": 2048,
        "tl_block_M": 1,
        "tl_block_N": 128,
        "ppl_block_w": 128,
        "tl_strategy": "streaming_fullk_pipe2",
    },
    "moe_prefill_dsv3": {
        "M": 16,
        "hidden": 7168,
        "intermediate": 2048,
        "tl_block_M": 16,
        "tl_block_N": 128,
        "ppl_block_w": 128,
        "tl_strategy": "ppl_layout_pipe1",
    },
    "dense_decode_dsv3": {
        "M": 1,
        "hidden": 7168,
        "intermediate": 18432,
        "tl_block_M": 1,
        "tl_block_N": 128,
        "ppl_block_w": 128,
        "tl_strategy": "streaming_fullk_pipe2",
    },
}

CONFIG_ORDER = [
    "tiny",
    "stage2",
    "stage3",
    "small_decode_scaled",
    "medium_prefill_scaled",
    "moe_decode_dsv3",
    "moe_prefill_dsv3",
    "dense_decode_dsv3",
]
BACKEND_CHOICES = [
    "tl",
    "tl_act",
    "tl_stream",
    "tl_pipe1",
    "tl_pipe2",
    "tl_ppl",
    "tl_ppl_pipe1",
    "tl_ppl_pipe2",
    "ppl_single",
]

ATOL = 2e-1
RTOL = 2e-1


PPL_SINGLE_TEMPLATE = Template(r'''
#include "ppl.h"
#include "ppl_wrapper_func.h"

using namespace ppl;

__KERNEL__ void main_kernel(fp16 *ptr_input, fp16 *ptr_weight0, fp16 *ptr_weight1,
                            fp16 *ptr_weight2, fp16 *ptr_output) {
  const int batch = $M;
  const int input_w = $HIDDEN;
  const int middle_w = $INTERMEDIATE;
  const int block_b = $BLOCK_B;
  const int block_iw = $HIDDEN;
  const int block_w = $BLOCK_W;
  ppl::set_core_num(1);

  int b_loop = div_up(batch, NPU_NUM);
  for (int b_idx = 0; b_idx < b_loop; b_idx++) {
    int b_offset = b_idx * NPU_NUM;
    int b_slice = min(NPU_NUM, batch - b_offset);
    dim4 g_input_shape = {1, batch, 1, input_w};
    dim4 g_offset = {0, b_offset, 0, 0};
    dim4 input_shape = {1, b_slice, 1, input_w};
    dim4 input_block_shape = {1, block_b, 1, block_iw};
    dim4 weight0_global_shape = {1, input_w, 1, middle_w};
    dim4 weight2_global_shape = {1, middle_w, 1, input_w};
    dim4 weight0_block_shape = {1, block_iw, 1, block_w};
    dim4 weight2_block_shape = {1, block_w, 1, block_iw};
    dim4 middle_buffer_shape = {1, block_b, 1, block_w};

    auto input_gtensor =
        gtensor<fp16>(g_input_shape, GLOBAL, ptr_input).sub_view(input_shape, g_offset);
    auto output_gtensor =
        gtensor<fp16>(g_input_shape, GLOBAL, ptr_output).sub_view(input_shape, g_offset);
    auto weight0_gtensor = gtensor<fp16>(weight0_global_shape, GLOBAL, ptr_weight0);
    auto weight1_gtensor = gtensor<fp16>(weight0_global_shape, GLOBAL, ptr_weight1);
    auto weight2_gtensor = gtensor<fp16>(weight2_global_shape, GLOBAL, ptr_weight2);

    auto input_local = make_tensor<fp16>(input_block_shape, input_shape);
    auto out_f32_local = make_tensor<fp32>(input_block_shape, input_shape);
    dma::load(input_local, input_gtensor);
    tiu::fill(out_f32_local, 0.0f);

    for (int w_idx = 0; w_idx < middle_w; w_idx += block_w) {
      ppl::enable_pipeline();
      int middle_slice = min(block_w, middle_w - w_idx);
      dim4 weight0_shape = {1, input_w, 1, middle_slice};
      dim4 weight2_shape = {1, middle_slice, 1, input_w};
      dim4 weight0_offset = {0, 0, 0, w_idx};
      dim4 weight2_offset = {0, w_idx, 0, 0};
      auto weight0_local = make_tensor<fp16>(weight0_block_shape, weight0_shape);
      auto weight1_local = make_tensor<fp16>(weight0_block_shape, weight0_shape);
      auto weight2_local = make_tensor<fp16>(weight2_block_shape, weight2_shape);

      dma::load(weight0_local, weight0_gtensor.sub_view(weight0_shape, weight0_offset));
      dma::load(weight1_local, weight1_gtensor.sub_view(weight0_shape, weight0_offset));
      dma::load(weight2_local, weight2_gtensor.sub_view(weight2_shape, weight2_offset));

      dim4 middle_real_shape = {1, b_slice, 1, middle_slice};
      auto gate_local = make_tensor<fp16>(middle_buffer_shape, middle_real_shape);
      auto work_f16 = make_tensor<fp16>(middle_buffer_shape, middle_real_shape);
      auto work_f32 = make_tensor<fp32>(middle_buffer_shape, middle_real_shape);

      tiu::fmm2(gate_local, input_local, weight1_local, false, false, false, false, false,
                DT_FP16);
      tiu::fmul(work_f16, gate_local, -1.0);
      tiu::cast(work_f32, work_f16);
      exp_no_overflow(work_f32, work_f32, &middle_buffer_shape, &middle_real_shape);
      tiu::fadd(work_f32, work_f32, 1.0);
      tiu::fdiv(work_f32, 1.0, work_f32, 4);
      tiu::cast(work_f16, work_f32);
      tiu::fmul(gate_local, gate_local, work_f16);
      tiu::fmm2(work_f16, input_local, weight0_local, false, false, false, false, false,
                DT_FP16);
      tiu::fmul(gate_local, gate_local, work_f16);
      tiu::fmm2(out_f32_local, gate_local, weight2_local, false, false, false, false, true,
                DT_FP32);
    }

    auto out_local = make_tensor<fp16>(input_block_shape, input_shape);
    tiu::cast(out_local, out_f32_local);
    dma::store(output_gtensor, out_local);
  }
}
''')



def make_inputs(cfg, seed=0):
    torch.manual_seed(seed)
    m, hidden, intermediate = cfg["M"], cfg["hidden"], cfg["intermediate"]
    x = (torch.randn(m, hidden, dtype=torch.float16) * 0.02).contiguous()
    gate_w = (torch.randn(intermediate, hidden, dtype=torch.float16) * 0.05).contiguous()
    up_w = (torch.randn(intermediate, hidden, dtype=torch.float16) * 0.05).contiguous()
    down_w = (torch.randn(hidden, intermediate, dtype=torch.float16) * 0.05).contiguous()
    return x, gate_w, up_w, down_w


def tl_reference(x, gate_w, up_w, down_w):
    gate = torch.matmul(x.float(), gate_w.T.float())
    up = torch.matmul(x.float(), up_w.T.float())
    act = (F.silu(gate) * up).clamp(-65500.0, 65500.0).half()
    return torch.matmul(act.float(), down_w.T.float()).half()


def ppl_reference(x, gate_w, up_w, down_w):
    gate = torch.matmul(x.float(), gate_w.T.float()).half()
    sigmoid = (1.0 / (1.0 + torch.exp(-gate.float()))).half()
    silu_gate = (gate * sigmoid).half()
    up = torch.matmul(x.float(), up_w.T.float()).half()
    act = (silu_gate * up).half()
    return torch.matmul(act.float(), down_w.T.float()).half()


def to_ppl_layout(x, gate_w, up_w, down_w):
    m, hidden = x.shape
    intermediate = gate_w.shape[0]
    x4 = x.reshape(1, m, 1, hidden).contiguous()
    weight0 = up_w.T.reshape(1, hidden, 1, intermediate).contiguous()
    weight1 = gate_w.T.reshape(1, hidden, 1, intermediate).contiguous()
    weight2 = down_w.T.reshape(1, intermediate, 1, hidden).contiguous()
    return x4, weight0, weight1, weight2


def check_output(name, out, ref, canonical_ref=None):
    out_f = out.float()
    ref_f = ref.float()
    finite = not (torch.isnan(out_f).any().item() or torch.isinf(out_f).any().item())
    correct = finite and torch.allclose(out_f, ref_f, atol=ATOL, rtol=RTOL)
    max_diff = (out_f - ref_f).abs().max().item()
    avg_diff = (out_f - ref_f).abs().mean().item()
    result = {
        "backend": name,
        "correct": bool(correct),
        "max_diff": max_diff,
        "avg_diff": avg_diff,
        "has_nan": bool(torch.isnan(out_f).any().item()),
        "has_inf": bool(torch.isinf(out_f).any().item()),
    }
    if canonical_ref is not None:
        canonical_diff = (out_f - canonical_ref.float()).abs()
        result["canonical_max_diff"] = canonical_diff.max().item()
        result["canonical_avg_diff"] = canonical_diff.mean().item()
    print(
        f"  [{name}] correct={correct} max_diff={max_diff:.6f} "
        f"avg_diff={avg_diff:.8f} nan={result['has_nan']} inf={result['has_inf']}",
        flush=True,
    )
    print("RESULT_JSON:" + json.dumps(result, sort_keys=True), file=sys.stderr, flush=True)
    return result


def _artifact_dir(out_dir, config_name, backend):
    return os.path.join(out_dir, "artifacts", config_name, backend)


def _save_tilelang_kernel_c(out_dir, config_name, backend):
    if out_dir is None:
        return
    src = os.path.join(REPO_ROOT, "src", "tl_templates", "tpu", "kernel.c")
    if not os.path.isfile(src):
        return
    dst_dir = _artifact_dir(out_dir, config_name, backend)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dst_dir, "tilelang_kernel.c"))


def run_tilelang(config_name, mode, strategy=None, out_dir=None, artifact_backend=None):
    import tilelang

    cfg = CONFIGS[config_name]
    m, hidden, intermediate = cfg["M"], cfg["hidden"], cfg["intermediate"]
    block_m = cfg["tl_block_M"]
    block_n = cfg["tl_block_N"]
    strategy = strategy or cfg.get("tl_strategy", "act_buffer")
    x, gate_w, up_w, down_w = make_inputs(cfg)
    ref = tl_reference(x, gate_w, up_w, down_w)
    act = torch.empty(m, intermediate, dtype=torch.float16)
    out = torch.empty(m, hidden, dtype=torch.float16)

    num_stages = 0
    use_ppl_layout = False
    if strategy == "act_buffer":
        kernel_func = mlp_w8a16_fused_kernel_v2
        label = "tilelang-act-buffer"
    elif strategy == "streaming_fullk":
        kernel_func = mlp_w8a16_fused_kernel_streaming_fullk
        label = "tilelang-streaming-fullk"
    elif strategy == "streaming_fullk_pipe1":
        kernel_func = mlp_w8a16_fused_kernel_streaming_fullk
        label = "tilelang-streaming-fullk-pipe1"
        num_stages = 1
    elif strategy == "streaming_fullk_pipe2":
        kernel_func = mlp_w8a16_fused_kernel_streaming_fullk
        label = "tilelang-streaming-fullk-pipe2"
        num_stages = 2
    elif strategy == "ppl_layout_streaming":
        kernel_func = mlp_w8a16_fused_kernel_ppl_layout_streaming
        label = "tilelang-ppl-layout-streaming"
        use_ppl_layout = True
    elif strategy == "ppl_layout_pipe1":
        kernel_func = mlp_w8a16_fused_kernel_ppl_layout_streaming
        label = "tilelang-ppl-layout-pipe1"
        num_stages = 1
        use_ppl_layout = True
    elif strategy == "ppl_layout_pipe2":
        kernel_func = mlp_w8a16_fused_kernel_ppl_layout_streaming
        label = "tilelang-ppl-layout-pipe2"
        num_stages = 2
        use_ppl_layout = True
    else:
        raise ValueError(f"Unknown TileLang strategy: {strategy}")

    if use_ppl_layout:
        gate_arg = gate_w.T.contiguous()
        up_arg = up_w.T.contiguous()
        down_arg = down_w.T.contiguous()
    else:
        gate_arg = gate_w
        up_arg = up_w
        down_arg = down_w

    print(
        f"--- TileLang {strategy} {config_name} mode={mode} "
        f"M={m} H={hidden} I={intermediate}"
    )
    kernel_kwargs = dict(
        block_M=block_m,
        block_N=block_n,
        block_K=block_n,
        block_D=block_n,
    )
    if kernel_func in (
        mlp_w8a16_fused_kernel_streaming_fullk,
        mlp_w8a16_fused_kernel_ppl_layout_streaming,
    ):
        kernel_kwargs["num_stages"] = num_stages

    kernel = tilelang.compile(
        kernel_func(m, hidden, intermediate, **kernel_kwargs),
        out_idx=-1,
        target="tpu",
        mode=mode,
    )
    _save_tilelang_kernel_c(out_dir, config_name, artifact_backend or label)
    kernel(x, gate_arg, up_arg, down_arg, act, out)
    return check_output(label, out, ref)


def render_ppl_pl(config_name, backend, out_dir):
    cfg = CONFIGS[config_name]
    pl_dir = os.path.join(out_dir, "generated_pl")
    os.makedirs(pl_dir, exist_ok=True)
    path = os.path.join(pl_dir, f"mlp_w8a16_dq_forward_fp16_{config_name}_{backend}.pl")
    text = PPL_SINGLE_TEMPLATE.substitute(
        M=cfg["M"],
        HIDDEN=cfg["hidden"],
        INTERMEDIATE=cfg["intermediate"],
        BLOCK_B=cfg["tl_block_M"],
        BLOCK_W=cfg["ppl_block_w"],
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def run_ppl(config_name, backend, mode, out_dir):
    from ppl_utils import compile_ppl_kernel

    cfg = CONFIGS[config_name]
    m, hidden, intermediate = cfg["M"], cfg["hidden"], cfg["intermediate"]
    x, gate_w, up_w, down_w = make_inputs(cfg)
    canonical_ref = tl_reference(x, gate_w, up_w, down_w)
    ref = ppl_reference(x, gate_w, up_w, down_w)
    x4, weight0, weight1, weight2 = to_ppl_layout(x, gate_w, up_w, down_w)
    out = torch.empty(1, m, 1, hidden, dtype=torch.float16)

    print(f"--- PPL {backend} {config_name} mode={mode} M={m} H={hidden} I={intermediate}")
    pl_path = render_ppl_pl(config_name, backend, out_dir)
    arg_specs = [
        ((1, m, 1, hidden), torch.float16),
        ((1, hidden, 1, intermediate), torch.float16),
        ((1, hidden, 1, intermediate), torch.float16),
        ((1, intermediate, 1, hidden), torch.float16),
        ((1, m, 1, hidden), torch.float16),
    ]
    artifact_dir = _artifact_dir(out_dir, config_name, backend)
    forward = compile_ppl_kernel(
        pl_path,
        arg_specs,
        result_idx=[4],
        mode=mode,
        artifact_dir=artifact_dir,
    )
    forward(x4, weight0, weight1, weight2, out)
    return check_output(backend, out.reshape(m, hidden), ref, canonical_ref=canonical_ref)


def parse_avg_us(text):
    matches = re.findall(r"Average execution time:\s+([\d.]+)\s+ms\s+\((\d+)\s+us\)", text)
    if not matches:
        return None
    return int(matches[-1][1])


def parse_result(text):
    matches = re.findall(r"RESULT_JSON:(\{.*?\})", text)
    if not matches:
        return {}
    return json.loads(matches[-1])


def run_child(config_name, backend, mode, out_dir):
    if backend == "tl":
        return run_tilelang(config_name, mode, out_dir=out_dir, artifact_backend=backend)
    if backend == "tl_act":
        return run_tilelang(
            config_name, mode, strategy="act_buffer", out_dir=out_dir, artifact_backend=backend)
    if backend == "tl_stream":
        return run_tilelang(
            config_name, mode, strategy="streaming_fullk", out_dir=out_dir, artifact_backend=backend)
    if backend == "tl_pipe1":
        return run_tilelang(
            config_name, mode, strategy="streaming_fullk_pipe1", out_dir=out_dir,
            artifact_backend=backend)
    if backend == "tl_pipe2":
        return run_tilelang(
            config_name, mode, strategy="streaming_fullk_pipe2", out_dir=out_dir,
            artifact_backend=backend)
    if backend == "tl_ppl":
        return run_tilelang(
            config_name, mode, strategy="ppl_layout_streaming", out_dir=out_dir,
            artifact_backend=backend)
    if backend == "tl_ppl_pipe1":
        return run_tilelang(
            config_name, mode, strategy="ppl_layout_pipe1", out_dir=out_dir,
            artifact_backend=backend)
    if backend == "tl_ppl_pipe2":
        return run_tilelang(
            config_name, mode, strategy="ppl_layout_pipe2", out_dir=out_dir,
            artifact_backend=backend)
    if backend == "ppl_single":
        return run_ppl(config_name, backend, mode, out_dir)
    raise ValueError(f"Unknown backend: {backend}")


def run_one_subprocess(config_name, backend, mode, out_dir, timeout):
    stem = f"{config_name}_{backend}_{mode}"
    stdout_path = os.path.join(out_dir, stem + ".stdout.txt")
    stderr_path = os.path.join(out_dir, stem + ".stderr.txt")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        sys.executable,
        __file__,
        "--run-one",
        "--config",
        config_name,
        "--backend",
        backend,
        "--mode",
        mode,
        "--out-dir",
        out_dir,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        status = "ok" if proc.returncode == 0 else "failed"
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        status = "timeout"
        returncode = None

    with open(stdout_path, "w", encoding="utf-8") as f:
        f.write(stdout)
    with open(stderr_path, "w", encoding="utf-8") as f:
        f.write(stderr)

    result = parse_result(stdout + "\n" + stderr)
    row = {
        "config": config_name,
        "backend": backend,
        "mode": mode,
        "status": status,
        "returncode": returncode,
        "avg_us": parse_avg_us(stdout),
        "stdout": os.path.relpath(stdout_path, REPO_ROOT),
        "stderr": os.path.relpath(stderr_path, REPO_ROOT),
    }
    row.update(result)
    row["backend_label"] = result.get("backend")
    row["backend"] = backend
    return row


def parse_csv_arg(value, all_values):
    if value == "all":
        return list(all_values)
    result = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in result if item not in all_values]
    if unknown:
        raise ValueError(f"Unknown values: {unknown}")
    return result


def write_summary(rows, out_dir):
    json_path = os.path.join(out_dir, "summary.json")
    md_path = os.path.join(out_dir, "summary.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    lines = [
        "# MLP W8A16 Dequantized Forward Summary",
        "",
        f"Run directory: `{os.path.relpath(out_dir, REPO_ROOT)}`",
        "",
        "`Speedup = PPL single-core avg us / TileLang avg us`.",
        "",
        "| Config | Backend | Correct | Avg us | Max diff | Canonical max diff | Status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {config} | {backend} | {correct} | {avg_us} | {max_diff} | {canonical} | {status} |".format(
                config=row["config"],
                backend=row["backend"],
                correct=row.get("correct", "-"),
                avg_us=row.get("avg_us") if row.get("avg_us") is not None else "-",
                max_diff=(
                    f"{row['max_diff']:.6f}" if isinstance(row.get("max_diff"), float) else "-"
                ),
                canonical=(
                    f"{row['canonical_max_diff']:.6f}"
                    if isinstance(row.get("canonical_max_diff"), float)
                    else "-"
                ),
                status=row["status"],
            )
        )

    lines += ["", "## TL vs PPL Single-Core", ""]
    lines += ["| Config | TL avg us | PPL single avg us | Speedup |", "|---|---:|---:|---:|"]
    by_key = {(row["config"], row["backend"]): row for row in rows}
    for config_name in CONFIG_ORDER:
        if not any(row["config"] == config_name for row in rows):
            continue
        tl = by_key.get((config_name, "tl"), {})
        ppl = by_key.get((config_name, "ppl_single"), {})
        speedup = "-"
        if (
            tl.get("avg_us")
            and ppl.get("avg_us")
            and tl.get("correct") is True
            and ppl.get("correct") is True
        ):
            speedup = f"{ppl['avg_us'] / tl['avg_us']:.2f}x"
        lines.append(
            f"| `{config_name}` | {tl.get('avg_us', '-')} | {ppl.get('avg_us', '-')} | {speedup} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return json_path, md_path


def run_suite(args):
    configs = parse_csv_arg(args.configs, CONFIG_ORDER)
    backends = parse_csv_arg(args.backends, BACKEND_CHOICES)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(RESULTS_DIR, f"{args.mode}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    total = len(configs) * len(backends)
    index = 0
    for config_name in configs:
        for backend in backends:
            index += 1
            print(f"[{index}/{total}] {config_name} {backend} mode={args.mode}", flush=True)
            row = run_one_subprocess(config_name, backend, args.mode, out_dir, args.timeout)
            rows.append(row)
            print(
                "  status={status} correct={correct} avg_us={avg_us}".format(
                    status=row["status"],
                    correct=row.get("correct"),
                    avg_us=row.get("avg_us"),
                ),
                flush=True,
            )
    json_path, md_path = write_summary(rows, out_dir)
    print(f"\nWrote {os.path.relpath(json_path, REPO_ROOT)}")
    print(f"Wrote {os.path.relpath(md_path, REPO_ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="all", help="all or comma-separated config names")
    parser.add_argument("--backends", default="tl,ppl_single", help="all or comma-separated backends")
    parser.add_argument("--backend", choices=BACKEND_CHOICES)
    parser.add_argument("--config", choices=CONFIG_ORDER)
    parser.add_argument("--mode", default="pcie", choices=["pcie", "cmodel"])
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--out-dir")
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--run-one", action="store_true")
    args = parser.parse_args()

    if args.list_configs:
        for name in CONFIG_ORDER:
            cfg = CONFIGS[name]
            print(
                f"{name}: M={cfg['M']} H={cfg['hidden']} I={cfg['intermediate']} "
                f"strategy={cfg['tl_strategy']}"
            )
        return

    if args.run_one:
        if not args.config or not args.backend:
            raise ValueError("--run-one requires --config and --backend")
        out_dir = args.out_dir or RESULTS_DIR
        os.makedirs(out_dir, exist_ok=True)
        run_child(args.config, args.backend, args.mode, out_dir)
        return

    run_suite(args)


if __name__ == "__main__":
    main()
