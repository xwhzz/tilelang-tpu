# FlashAttention 优化算子

本目录存放已经固化的 TileLang-TPU FlashAttention 优化算子。每个
`bench_flashattn_*.py` 都是一个具体 dtype/shape 的 fixed best 实现，脚本内部直接
运行优化后的 TileLang-TPU 算子并对照 PPL baseline。

当前 fixed suite 覆盖 8 个 unique shape，每个 shape 提供 fp16 和 bf16 两个脚本。
计时口径统一为 BM1690 PCIe 上 5 次 warmup 后重复 10 次取平均。

`Speedup = PPL(us) / TL(us)`，大于 1 表示 TileLang-TPU 更快。

## 配置选择

| Shape | 覆盖目的 | 固化 TL tile | 固定脚本 |
|---|---|---|---|
| `B1H8S64D32` | 短上下文、小 head dim | `64x64` | `bench_flashattn_{dtype}_b1_h8_s64_d32.py` |
| `B1H8S128D64` | 中短上下文、中等 head dim | `128x128` | `bench_flashattn_{dtype}_b1_h8_s128_d64.py` |
| `B1H8S256D128` | 中等上下文、标准 `D=128` | `128x128` | `bench_flashattn_{dtype}_b1_h8_s256_d128.py` |
| `B1H8S512D128` | 较长上下文、单 batch/head baseline | `128x128` | `bench_flashattn_{dtype}_b1_h8_s512_d128.py` |
| `B8H8S512D128` | batch 扩展对照 | `256x256,H8,s2` | `bench_flashattn_{dtype}_b8_h8_s512_d128.py` |
| `B8H32S512D128` | batch 和 head 同时扩展对照 | `256x256,H8,s2` | `bench_flashattn_{dtype}_b8_h32_s512_d128.py` |
| `B64H64S1024D128` | 大 batch/head、长上下文 | `256x256,H8,s2` | `bench_flashattn_{dtype}_b64_h64_s1024_d128.py` |
| `B64H64S2048D128` | 大 batch/head、更长上下文 | `128x512,H8,s2` | `bench_flashattn_{dtype}_b64_h64_s2048_d128.py` |

`{dtype}` 取 `fp16` 或 `bf16`。其中 `H8` 表示每个 kernel block 处理 8 个 head，
`s2` 表示 `num_stages=2`。

## 实测结果

| Dtype | Shape | TL | PPL | Speedup |
|---|---|---:|---:|---:|
| bf16 | `B1H8S64D32` | 123 us | 2,123 us | 17.26x |
| bf16 | `B1H8S128D64` | 141 us | 2,183 us | 15.48x |
| bf16 | `B1H8S256D128` | 245 us | 1,958 us | 7.99x |
| bf16 | `B1H8S512D128` | 671 us | 7,535 us | 11.23x |
| bf16 | `B8H8S512D128` | 2,054 us | 59,654 us | 29.04x |
| bf16 | `B8H32S512D128` | 7,821 us | 101,678 us | 13.00x |
| bf16 | `B64H64S1024D128` | 431,822 us | 1,237,966 us | 2.87x |
| bf16 | `B64H64S2048D128` | 1,548,654 us | 4,834,021 us | 3.12x |
| fp16 | `B1H8S64D32` | 138 us | 2,098 us | 15.20x |
| fp16 | `B1H8S128D64` | 138 us | 2,209 us | 16.01x |
| fp16 | `B1H8S256D128` | 243 us | 2,049 us | 8.43x |
| fp16 | `B1H8S512D128` | 671 us | 7,674 us | 11.44x |
| fp16 | `B8H8S512D128` | 2,043 us | 61,190 us | 29.95x |
| fp16 | `B8H32S512D128` | 7,823 us | 100,203 us | 12.81x |
| fp16 | `B64H64S1024D128` | 431,822 us | 1,285,176 us | 2.98x |
| fp16 | `B64H64S2048D128` | 1,548,585 us | 4,986,000 us | 3.22x |

| 范围 | Shape | 平均 Speedup |
|---|---|---:|
| sequence sweep | `B1H8S64-512D32-128` | 12.88x |
| batch/head variation | `B8H8S512D128`, `B8H32S512D128` | 21.20x |
| large real | `B64H64S1024D128`, `B64H64S2048D128` | 3.05x |
| all | 8 个 unique shape，fp16/bf16 共 16 项 | 12.50x |

## 固化策略

常规 `B1H8` shape 使用 head-batched（按 head 打包）实现：

- kernel grid 为 `(seq_block, batch)`，head 维度放入本地 rank-3 tensor。
- 本地张量布局为 `[heads, block, dim]` 和 `[heads, block_m, block_n]`。
- 直接调用普通 `T.gemm` 处理 rank-3 local tile；codegen 在 C 层逐 head 发二维
  GEMM，reduce、exp、状态缩放和累加不再在外层按 head 重复。

`H > 8` 或大 batch/head shape 使用 head blocking（按 head 分块）实现：

- 固定 `BLOCK_H=8`，避免一次把所有 head 放入本地内存。
- `B8H8S512D128` 和 `B8H32S512D128` 使用 `256x256,H8,s2`。
- `B64H64S1024D128` 使用 `256x256,H8,s2`。
- `B64H64S2048D128` 使用 `128x512,H8,s2`。

## 正确性策略

当前 fixed suite 测试的是 `q_len = kv_len = seq_len` 的 non-causal square
self-attention，适合 prefill 型场景。

| Shape 范围 | 正确性方式 |
|---|---|
| `B1H8S64-512D32-128` | full reference |
| `B8H8S512D128`, `B8H32S512D128` | full reference |
| `B64H64S1024D128`, `B64H64S2048D128` | sampled reference |

大 shape 的完整 score 矩阵过大：

| Shape | full score 大小 |
|---|---:|
| `B64H64S1024D128` | 约 16 GiB fp32 |
| `B64H64S2048D128` | 约 64 GiB fp32 |

decode attention 常见的 `q_len=1, kv_len=S` 不是当前 fixed suite 的语义，需要单独
实现和测试，不能由本表直接外推。

## 运行方式

运行单个固定脚本：

```bash
python tpu_benchmark/flash_attention_opt/bench_flashattn_fp16_b1_h8_s128_d64.py
```

列出全部固化配置：

```bash
python tpu_benchmark/flash_attention_opt/run_head_batched_suite.py --list-configs
```

运行完整 fixed suite：

```bash
python tpu_benchmark/flash_attention_opt/run_head_batched_suite.py --configs all --timeout 1800
```

## 维护规则

- 本目录只保存固化后的 best 脚本和必要说明。
- 临时 stdout/stderr、JSON、Markdown 汇总不应保留在本源码目录。
- 若同一进程内先编译多个 TPU/PPL kernel，运行每个 backend 前必须恢复对应的
  `PPL_KERNEL_PATH`，避免 TL 实际加载到 PPL module。
