# MLP W8A16 Dequantized Forward 优化算子

本目录是 TileLang-TPU 版 `MLP W8A16 Dequantized Forward` 的集中实现目录。
默认 `tl` 后端已经按 config 固化当前最稳的策略；直接运行 benchmark runner 即可对比
TileLang-TPU 和 PPL 单核 baseline。

## 目录内容

| 文件 | 作用 |
|---|---|
| `kernel_v2.py` | TileLang-TPU kernel 实现，包含当前固化策略和诊断用候选策略。 |
| `bench_mlp_w8a16_dq_forward.py` | 固定 config、输入生成、correctness 检查、PPL 单核 baseline 和计时入口。 |
| `README.md` | 当前算子语义、策略选择、性能结果和维护规则。 |

## 算子语义

`dq` 表示 int8 权重已经在调用前反量化为 fp16；本算子接收 fp16 输入、fp16
反量化权重，并输出 fp16。计算语义是 Transformer feed-forward network（前馈网络）
中的 SwiGLU MLP forward：

```text
Output = (silu(X @ W_gate.T) * (X @ W_up.T)) @ W_down.T
```

逻辑 shape：

| Tensor | Shape |
|---|---|
| `X` | `[M, hidden]` |
| `W_gate`, `W_up` | `[intermediate, hidden]` |
| `W_down` | `[hidden, intermediate]` |
| `Output` | `[M, hidden]` |

## 固化策略

当前默认 `tl` 后端根据 config 自动选择策略。

| Config | Shape | 固定 TL 策略 | tile_N | 说明 |
|---|---|---|---:|---|
| `tiny` | `M=4,H=128,I=128` | `streaming_fullk` | 128 | 小 shape，pipeline 固定开销不划算。 |
| `stage2` | `M=16,H=256,I=128` | `streaming_fullk` | 128 | 小 intermediate，保持最简单稳定路径。 |
| `stage3` | `M=16,H=256,I=256` | `streaming_fullk` | 256 | 小 shape 中 `tile_N=256` 可稳定运行。 |
| `small_decode_scaled` | `M=1,H=512,I=2048` | `streaming_fullk` | 128 | decode 缩放场景，与 PPL 接近。 |
| `medium_prefill_scaled` | `M=16,H=1024,I=4096` | `streaming_fullk` | 128 | prefill 缩放场景，与 PPL 接近。 |
| `moe_decode_dsv3` | `M=1,H=7168,I=2048` | `streaming_fullk_pipe2` | 128 | pipeline 对真实 MoE decode 有明显收益。 |
| `moe_prefill_dsv3` | `M=16,H=7168,I=2048` | `ppl_layout_pipe1` | 128 | 原布局 pipeline 偶发 NaN，固化稳定路径。 |
| `dense_decode_dsv3` | `M=1,H=7168,I=18432` | `streaming_fullk_pipe2` | 128 | 长 intermediate loop 中 pipeline 收益最大。 |

策略含义：

| 策略 | 做法 | 使用场景 |
|---|---|---|
| `streaming_fullk` | 每个 intermediate tile 计算 gate/up/SiLU 后立即做 down 累加，不写完整 global `Act`。 | 小中 shape 和缩放 shape。 |
| `streaming_fullk_pipe2` | 在 `streaming_fullk` 上使用 `T.Pipelined(..., num_stages=2)`。 | H=7168 的 decode/dense 大 shape。 |
| `ppl_layout_pipe1` | 使用 PPL baseline 风格的权重布局，让 down projection 走 non-transpose accumulate GEMM。 | `moe_prefill_dsv3` 的稳定固化路径。 |

## 固定结果

计时口径为 5 次 warmup 后重复 10 次取平均。`Speedup = PPL(us) / TL(us)`，
大于 1 表示 TileLang-TPU 更快。

| Config | TL 策略 | TL (us) | PPL 单核 (us) | Speedup |
|---|---|---:|---:|---:|
| `tiny` | `streaming_fullk` | 90 | 103 | 1.14x |
| `stage2` | `streaming_fullk` | 110 | 106 | 0.96x |
| `stage3` | `streaming_fullk` | 109 | 120 | 1.10x |
| `small_decode_scaled` | `streaming_fullk` | 314 | 306 | 0.97x |
| `medium_prefill_scaled` | `streaming_fullk` | 545 | 557 | 1.02x |
| `moe_decode_dsv3` | `streaming_fullk_pipe2` | 1,001 | 999 | 1.00x |
| `moe_prefill_dsv3` | `ppl_layout_pipe1` | 1,116 | 1,011 | 0.91x |
| `dense_decode_dsv3` | `streaming_fullk_pipe2` | 8,215 | 8,109 | 0.99x |

8 组配置 correctness 全部通过，平均 `PPL(us) / TL(us) = 1.01x`。H=7168 真实
shape 相对上一版 `streaming_fullk` 的 TL 内部提升为：

| Config | 上一版 TL (us) | 当前 TL (us) | TL 内部提升 |
|---|---:|---:|---:|
| `moe_decode_dsv3` | 1,383 | 1,001 | 1.38x |
| `moe_prefill_dsv3` | 1,393 | 1,116 | 1.25x |
| `dense_decode_dsv3` | 11,639 | 8,215 | 1.42x |

## 优化结论

- 旧 `act_buffer` 路径会把完整 `Act=Mxintermediate` 写到 global，再读回来做 down projection；大 shape 上冗余明显。
- `streaming_fullk` 去掉 global `Act`，把 gate/up/SiLU 和 down projection 放进同一个 intermediate tile 流水。
- PPL baseline 的 down projection 可以直接累加到 fp32 输出；当前原布局 TileLang 仍会生成 `down_temp + add`，这是剩余差距之一。
- H=7168 decode/dense shape 中，`T.Pipelined` 能显著降低串行 DMA/compute 开销，性能已接近 PPL 单核。
- `moe_prefill_dsv3` 的最快原布局 pipeline 会偶发 NaN，因此固化为较慢但稳定的 `ppl_layout_pipe1`。
- H=7168 下 `tile_N=256` 超出 local memory 可分配边界，`tile_N=64` loop 次数更多且更慢，因此固定 `tile_N=128`。

## Shape 选择依据

| Config 类型 | 目的 |
|---|---|
| `tiny/stage2/stage3` | 验证调度、tile 和固定开销边界。 |
| `small_decode_scaled/medium_prefill_scaled` | 覆盖缩放版 decode/prefill 趋势。 |
| `moe_*_dsv3/dense_decode_dsv3` | 覆盖 DeepSeek-V3/R1 相关真实 hidden/intermediate 维度。 |

`M` 表示本次 MLP 调用处理的 token 行数。dense MLP 中通常对应
`batch * seq_chunk`；MoE expert 中对应路由到单个 expert 的 token 数，因此需要同时覆盖
decode 小 `M`、prefill 中等 `M` 和真实模型维度。

## 运行方式

列出固定配置：

```bash
.venv/bin/python tpu_benchmark/mlp_w8a16_dq_forward/bench_mlp_w8a16_dq_forward.py --list-configs
```

PCIe 固定 suite：

```bash
.venv/bin/python tpu_benchmark/mlp_w8a16_dq_forward/bench_mlp_w8a16_dq_forward.py \
  --configs all --backends tl,ppl_single --mode pcie --timeout 3600
```

调试候选策略：

```bash
.venv/bin/python tpu_benchmark/mlp_w8a16_dq_forward/bench_mlp_w8a16_dq_forward.py \
  --configs moe_decode_dsv3,dense_decode_dsv3 \
  --backends tl_stream,tl_pipe1,tl_pipe2,ppl_single --mode pcie --timeout 2400
```

## 维护规则

- 用户默认只需要使用 `tl` 和 `ppl_single`；其它 backend 只用于回归诊断。
- 新增 shape 必须先跑 correctness，再记录性能。
- H=7168 真实 shape 不要直接放大 `tile_N`；先确认 local memory 是否能分配。
- pipeline 候选如果出现 NaN，即使单轮速度更快，也不能固化。
- 本目录只保留源码、benchmark runner 和 README；运行日志和生成产物不写入本目录。
