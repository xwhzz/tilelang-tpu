"""Correctness and latency harness for the fixed FP8 paged-prefill prototype."""

import os
import statistics
import sys
import time

import torch


sys.path.insert(0, "/mnt2/users/tilelanguser11/tilelang-tpu/tpu_demo")
import paged_latent_attention_fp8 as pla  # noqa: E402


DEVICE_ID = int(os.environ.get("PLA_DEVICE_ID", "7"))
WARMUP = int(os.environ.get("PLA_WARMUP", "2"))
RUNS = int(os.environ.get("PLA_RUNS", "5"))
NUM_HEADS = int(os.environ.get("PLA_NUM_HEADS", "32"))
CORE_NUM = int(os.environ.get("PLA_CORE_NUM", "8"))
ATTENTION_BLOCK_M = int(os.environ.get("PLA_ATTENTION_BLOCK_M", "32"))
TEST_DECODE = os.environ.get("PLA_TEST_DECODE", "1") != "0"


def patch_build_device() -> None:
    original = pla._rebuild_current_tpu_artifacts

    def rebuild():
        main_cpp = os.path.join(os.environ["TPU_KERNEL_PATH"], "main.cpp")
        with open(main_cpp, "r") as source:
            text = source.read()
        old = "tpuRtSetDevice(0);"
        new = f"tpuRtSetDevice({DEVICE_ID});"
        if old not in text and new not in text:
            raise RuntimeError("generated host source has no TPU device selector")
        with open(main_cpp, "w") as destination:
            destination.write(text.replace(old, new, 1))
        return original()

    pla._rebuild_current_tpu_artifacts = rebuild


def rotate_interleaved(x: torch.Tensor, cos: torch.Tensor,
                       sin: torch.Tensor) -> torch.Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos[..., 0::2] - odd * sin[..., 1::2]
    out[..., 1::2] = odd * cos[..., 1::2] + even * sin[..., 0::2]
    return out


def dequant(weight: torch.Tensor, scale: torch.Tensor,
            block: int) -> torch.Tensor:
    expanded = scale.repeat_interleave(block, 0).repeat_interleave(block, 1)
    return weight.to(torch.bfloat16) * expanded.to(torch.bfloat16)


def make_inputs(c, total_rows: int):
    torch.manual_seed(20260801)
    q_rows = c.num_heads * (c.qk_nope_head_dim + c.qk_rope_head_dim)
    kv_rows = c.num_heads * (c.qk_nope_head_dim + c.value_head_dim)
    fp8 = torch.float8_e4m3fn
    angles = torch.randn(c.seq_len, c.qk_rope_head_dim // 2) * 0.2
    cos = angles.cos().repeat_interleave(2, dim=-1).to(torch.bfloat16)
    sin = angles.sin().repeat_interleave(2, dim=-1).to(torch.bfloat16)
    mask = torch.zeros(c.seq_len, c.seq_len, dtype=torch.bfloat16)
    mask.masked_fill_(torch.triu(torch.ones(c.seq_len, c.seq_len,
                                           dtype=torch.bool), diagonal=1),
                      -10000.0)
    # A non-identity page order catches implementations that assume contiguous
    # cache placement.  Extra rows retain sentinels to detect overwrite.
    block_ids = torch.tensor([5, 1, 9, 0, 7, 3, 10, 2, 8, 4, 11, 6],
                             dtype=torch.int32)
    return {
        "query": (torch.randn(c.seq_len, c.q_lora_rank) * 0.02).to(torch.bfloat16),
        "kv": (torch.randn(c.seq_len, c.kv_lora_rank) * 0.02).to(torch.bfloat16),
        "pe": (torch.randn(c.seq_len, c.qk_rope_head_dim) * 0.02).to(torch.bfloat16),
        "wuq": (torch.randn(q_rows, c.q_lora_rank) * 0.05).to(fp8),
        "wukv": (torch.randn(kv_rows, c.kv_lora_rank) * 0.05).to(fp8),
        "wuq_scale": (torch.rand(q_rows // c.quant_block_size,
                                  c.q_lora_rank // c.quant_block_size)
                       * 0.2 + 0.9).to(torch.bfloat16),
        "wukv_scale": (torch.rand(kv_rows // c.quant_block_size,
                                   c.kv_lora_rank // c.quant_block_size)
                        * 0.2 + 0.9).to(torch.bfloat16),
        "kv_cache": torch.full((total_rows, c.kv_lora_rank), 0.125,
                                dtype=torch.bfloat16),
        "pe_cache": torch.full((total_rows, c.qk_rope_head_dim), -0.25,
                                dtype=torch.bfloat16),
        "cos": cos,
        "sin": sin,
        "mask": mask,
        "block_tables": block_ids.reshape(1, 1, 1, -1),
        "save_slots": torch.zeros(1, 1, 1, c.logical_blocks,
                                  dtype=torch.int32),
    }


def clone_inputs(inputs):
    return {name: value.clone() for name, value in inputs.items()}


def cpu_reference(c, inputs):
    wuq = dequant(inputs["wuq"], inputs["wuq_scale"], c.quant_block_size)
    wukv = dequant(inputs["wukv"], inputs["wukv_scale"], c.quant_block_size)
    q_full = (inputs["query"].float() @ wuq.float().t()).to(torch.bfloat16)
    kvu = (inputs["kv"].float() @ wukv.float().t()).to(torch.bfloat16)
    pe_rope = rotate_interleaved(inputs["pe"].float(), inputs["cos"].float(),
                                 inputs["sin"].float()).to(torch.bfloat16)
    q_dim = c.qk_nope_head_dim + c.qk_rope_head_dim
    kv_dim = c.qk_nope_head_dim + c.value_head_dim
    outputs = []
    for head in range(c.num_heads):
        q = q_full[:, head * q_dim:(head + 1) * q_dim]
        q_pe = rotate_interleaved(q[:, c.qk_nope_head_dim:].float(),
                                  inputs["cos"].float(),
                                  inputs["sin"].float()).to(torch.bfloat16)
        head_kvu = kvu[:, head * kv_dim:(head + 1) * kv_dim]
        score = (q[:, :c.qk_nope_head_dim].float()
                 @ head_kvu[:, :c.qk_nope_head_dim].float().t())
        score += q_pe.float() @ pe_rope.float().t()
        score = score * c.softmax_scale + inputs["mask"].float()
        prob = torch.softmax(score, dim=-1)
        outputs.append(prob @ head_kvu[:, c.qk_nope_head_dim:].float())
    output = torch.stack(outputs, dim=1).to(torch.bfloat16)
    kv_cache = inputs["kv_cache"].clone()
    pe_cache = inputs["pe_cache"].clone()
    slots = (inputs["block_tables"].reshape(-1).to(torch.int64)[:, None]
             * c.paged_cache_block_size
             + torch.arange(c.paged_cache_block_size)).reshape(-1)
    kv_cache[slots] = inputs["kv"]
    pe_cache[slots] = pe_rope
    return (output, kv_cache, pe_cache, slots,
            q_full.reshape(c.seq_len, c.num_heads, q_dim).permute(1, 0, 2).contiguous(),
            kvu.reshape(c.seq_len, c.num_heads, kv_dim).permute(1, 0, 2).contiguous())


def invoke(op, c, inputs):
    out = torch.empty(c.seq_len, c.num_heads, c.value_head_dim,
                      dtype=torch.bfloat16)
    input_lengths = torch.tensor([c.seq_len], dtype=torch.int32)
    cache_lengths = torch.tensor([0], dtype=torch.int32)
    op(out, inputs["query"], inputs["kv"], inputs["pe"], inputs["wuq"],
       inputs["wukv"], inputs["kv_cache"], inputs["pe_cache"], inputs["cos"],
       inputs["sin"], inputs["wuq_scale"], inputs["wukv_scale"],
       inputs["block_tables"], inputs["save_slots"], None, inputs["mask"],
       input_lengths, cache_lengths, c.num_heads, c.seq_len, c.q_lora_rank,
       c.kv_lora_rank, c.qk_nope_head_dim, c.qk_rope_head_dim,
       c.value_head_dim, c.seq_len, c.quant_block_size, c.logical_blocks,
       c.paged_cache_block_size, c.softmax_scale, 2)
    return out, inputs["kv_cache"], inputs["pe_cache"]


def invoke_decode(op, prefill_c, decode_c, inputs):
    out = torch.empty(1, decode_c.num_heads, decode_c.value_head_dim,
                      dtype=torch.bfloat16)
    block_tables = inputs["block_tables"].reshape(-1)[:decode_c.logical_blocks]
    save_slot = torch.tensor([13], dtype=torch.int32)
    op(
        out, inputs["query"][:1], inputs["kv"][:1], inputs["pe"][:1],
        inputs["wuq"], inputs["wukv"], inputs["kv_cache"],
        inputs["pe_cache"], inputs["cos"][:1], inputs["sin"][:1],
        inputs["wuq_scale"], inputs["wukv_scale"], block_tables, save_slot,
        None, None, torch.tensor([decode_c.logical_blocks *
                                  decode_c.paged_cache_block_size],
                                 dtype=torch.int32),
        torch.tensor([decode_c.logical_blocks *
                      decode_c.paged_cache_block_size - 1], dtype=torch.int32),
        decode_c.num_heads, 1, decode_c.q_lora_rank, decode_c.kv_lora_rank,
        decode_c.qk_nope_head_dim, decode_c.qk_rope_head_dim,
        decode_c.value_head_dim,
        decode_c.logical_blocks * decode_c.paged_cache_block_size,
        decode_c.quant_block_size, decode_c.logical_blocks,
        decode_c.paged_cache_block_size, decode_c.softmax_scale, 3,
    )
    if not bool(torch.isfinite(out).all()):
        raise AssertionError("Decode output through unified model op is not finite")
    if not torch.equal(inputs["kv_cache"].reshape(-1, prefill_c.kv_lora_rank)[13],
                       inputs["kv"][0]):
        raise AssertionError("Decode dispatch did not update KV cache")
    print("UNIFIED_DECODE_MODE3_FINITE True", flush=True)


def synchronize():
    if hasattr(torch, "tpu") and hasattr(torch.tpu, "synchronize"):
        torch.tpu.synchronize()


def report(name: str, actual: torch.Tensor, expected: torch.Tensor,
           atol: float, rtol: float):
    diff = (actual.float() - expected.float()).abs()
    ok = torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    print(f"{name} finite={bool(torch.isfinite(actual).all())} "
          f"allclose={bool(ok)} max_abs={float(diff.max()):.6f} "
          f"mean_abs={float(diff.mean()):.6f}", flush=True)
    return bool(ok)


def main():
    patch_build_device()
    c = pla.PagedLatentAttentionFp8PrefillConfig(
        num_heads=NUM_HEADS, core_num=CORE_NUM,
        attention_block_m=ATTENTION_BLOCK_M)
    total_rows = c.seq_len + 32
    base = make_inputs(c, total_rows)
    expected = cpu_reference(c, clone_inputs(base))
    op = pla.PagedLatentAttentionFp8Prefill(c, total_rows)
    decode_op = None
    if TEST_DECODE:
        decode_config = pla.PagedLatentAttentionFp8Config(
            num_heads=NUM_HEADS, logical_blocks=8, attention_block_n=128)
        decode_op = pla.PagedLatentAttentionFp8(decode_config, total_rows)
    pla.install_torch_custom_op(prefill=op, decode=decode_op)
    model_op = torch.ops.my_ops.paged_latent_attention_fp8.default
    print(f"MODEL_OP_MODES {pla._OPERATOR.modes}", flush=True)
    print(f"MODEL_OP_SCHEMA {model_op._schema}", flush=True)

    run = clone_inputs(base)
    actual = invoke(model_op, c, run)
    checks = []
    checks.append(report("Q_PROJECT_VS_CPU", op.last_q_projected, expected[4],
                         atol=0.03, rtol=0.03))
    checks.append(report("KVU_PROJECT_VS_CPU", op.last_kvu_projected, expected[5],
                         atol=0.03, rtol=0.03))
    print("QK_SCORE_STATS "
          f"finite={bool(torch.isfinite(op.last_qk_scores).all())} "
          f"min={float(torch.nan_to_num(op.last_qk_scores.float()).min()):.6f} "
          f"max={float(torch.nan_to_num(op.last_qk_scores.float()).max()):.6f}",
          flush=True)
    checks.append(report("OUTPUT_VS_CPU", actual[0], expected[0],
                         atol=0.09, rtol=0.04))
    checks.append(report("KV_CACHE_VS_CPU", actual[1], expected[1],
                         atol=0.0, rtol=0.0))
    checks.append(report("PE_CACHE_VS_CPU", actual[2], expected[2],
                         atol=0.01, rtol=0.01))
    print(f"Q_PROJECT_STATS actual_abs_mean={float(op.last_q_projected.float().abs().mean()):.6f} "
          f"expected_abs_mean={float(expected[4].float().abs().mean()):.6f}", flush=True)
    touched = torch.zeros(total_rows, dtype=torch.bool)
    touched[expected[3]] = True
    if not torch.equal(actual[1][~touched], base["kv_cache"][~touched]):
        raise AssertionError("KV cache rows outside block table were overwritten")
    if not torch.equal(actual[2][~touched], base["pe_cache"][~touched]):
        raise AssertionError("PE cache rows outside block table were overwritten")
    print("CACHE_UNTOUCHED_ROWS_MATCH True", flush=True)

    repeat = invoke(model_op, c, clone_inputs(base))
    if not torch.equal(actual[0], repeat[0]):
        raise AssertionError("repeated prefill output is not bitwise stable")
    print("REPEAT_OUTPUT_MATCH True", flush=True)

    if decode_op is not None:
        invoke_decode(model_op, c, decode_config, clone_inputs(base))

    if not all(checks):
        raise AssertionError("one or more correctness checks failed")

    for _ in range(WARMUP):
        invoke(model_op, c, clone_inputs(base))
    synchronize()
    samples = []
    for _ in range(RUNS):
        one = clone_inputs(base)
        start = time.perf_counter_ns()
        invoke(model_op, c, one)
        synchronize()
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    print("PREFILL_HOST_SYNC_MS "
          f"mean={statistics.mean(samples):.6f} "
          f"median={statistics.median(samples):.6f} "
          f"samples={','.join(f'{x:.6f}' for x in samples)}", flush=True)


if __name__ == "__main__":
    main()
