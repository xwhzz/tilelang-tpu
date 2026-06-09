# BM1690 PPL 性能追踪表

本文件是后续性能提升后的固定更新入口。新的完整串行测试或优化算子测试完成后，
直接更新本文对应表格和分析，不再另建同类 summary。

## 当前结论

- 初始性能对比表表示优化接入前的 baseline；其中 Matmul 行已按 best-of-5 复测刷新。
- Matmul: 15 个配置 best speedup 均大于 1x，平均 `PPL(us) / TL(us) = 1.14x`。
- FlashAttention 优化算子: 8 个 unique shape、fp16/bf16 共 16 项 correctness 全部通过，平均 `PPL(us) / TL(us) = 12.50x`。
- RMSNorm 优化算子: 15 个 dtype/shape correctness 全部通过，平均 `PPL(us) / TL(us) = 1.23x`。
- MLP W8A16 Dequantized Forward 优化算子: 8 个配置 correctness 全部通过，平均 `PPL(us) / TL(us) = 1.01x`；H=7168 真实 shape 相对上一版 TL 内部提升 `1.25x-1.42x`。
- 运行日志、stdout/stderr 和生成产物由各 benchmark runner 自动写入忽略目录，不写入源码目录。

## PPL 分块参考

本节只记录 PPL baseline 的实现分块，作为读表参考，不再要求 TL 与 PPL 分块一致。

| Op | 范围 | PPL 分块 |
|---|---|---|
| Matmul | 15 个 dtype/shape | `32x32x32` |
| RMSNorm direct | `64x64`, `128x128`, `512x512` | `block_m=64` |
| RMSNorm direct | `256x256` | `block_m=128` |
| RMSNorm direct | `1024x1024` | `block_m=32` |
| RMSNorm-SplitK 对照 | 全部 shape | 使用 direct RMSNorm PPL baseline |
| FlashAttention | `B1H2S4D8` | `block_m=2, block_k=2` |
| FlashAttention | `B1H8S64D32` | `block_m=8, block_k=8` |
| FlashAttention | `B1H8S128D64` | `block_m=16, block_k=16` |
| FlashAttention | `B1H8S256D128`, `B1H8S512D128` | `block_m=32, block_k=32` |
| FlashAttention FA 大 shape | `B64H64S1024D128`, `B64H64S2048D128` | `block_m=64, block_k=64` |
| MLP W8A16 dq forward | 全部固定 config | `block_w=128, block_iw=hidden, block_b=tl_block_M` |

## 初始性能对比

`Speedup = PPL(us) / TL(us)`，大于 1 表示 TL 更快。Avg Speedup 只统计 TL/PPL correctness 都通过且 PPL 未跳过的有效行。

| Op | Dtype | Shape | TL (us) | PPL (us) | Speedup |
|---|---|---:|---:|---:|---:|
| Matmul | fp32 | 64×64×64 | 108 | 136 | 1.26x |
| Matmul | fp32 | 128×128×128 | 300 | 321 | 1.07x |
| Matmul | fp32 | 256×256×256 | 1551 | 1768 | 1.14x |
| Matmul | fp32 | 512×512×512 | 11135 | 12840 | 1.15x |
| Matmul | fp32 | 1024×1024×1024 | 86846 | 98204 | 1.13x |
| Matmul | fp32 | Avg Speedup |  |  | 1.15x |
| Matmul | bf16 | 64×64×64 | 102 | 132 | 1.29x |
| Matmul | bf16 | 128×128×128 | 242 | 265 | 1.10x |
| Matmul | bf16 | 256×256×256 | 1121 | 1260 | 1.12x |
| Matmul | bf16 | 512×512×512 | 7257 | 8422 | 1.16x |
| Matmul | bf16 | 1024×1024×1024 | 55043 | 64281 | 1.17x |
| Matmul | bf16 | Avg Speedup |  |  | 1.17x |
| Matmul | fp16 | 64×64×64 | 127 | 132 | 1.04x |
| Matmul | fp16 | 128×128×128 | 240 | 264 | 1.10x |
| Matmul | fp16 | 256×256×256 | 1070 | 1196 | 1.12x |
| Matmul | fp16 | 512×512×512 | 7283 | 8370 | 1.15x |
| Matmul | fp16 | 1024×1024×1024 | 55982 | 64836 | 1.16x |
| Matmul | fp16 | Avg Speedup |  |  | 1.11x |
| Matmul | ALL | Op Avg Speedup |  |  | 1.14x |
| RMSNorm | fp32 | 64×64 | 74 | 70 | 0.95x |
| RMSNorm | fp32 | 128×128 | 92 | 85 | 0.92x |
| RMSNorm | fp32 | 256×256 | 84 | 95 | 1.13x |
| RMSNorm | fp32 | 512×512 | 117 | 104 | 0.89x |
| RMSNorm | fp32 | 1024×1024 | 241 | 229 | 0.95x |
| RMSNorm | fp32 | Avg Speedup |  |  | 0.97x |
| RMSNorm | bf16 | 64×64 | 83 | 84 | 1.01x |
| RMSNorm | bf16 | 128×128 | 94 | 83 | 0.88x |
| RMSNorm | bf16 | 256×256 | 85 | 81 | 0.95x |
| RMSNorm | bf16 | 512×512 | 143 | 114 | 0.80x |
| RMSNorm | bf16 | 1024×1024 | 299 | 246 | 0.82x |
| RMSNorm | bf16 | Avg Speedup |  |  | 0.89x |
| RMSNorm | fp16 | 64×64 | 80 | 74 | 0.93x |
| RMSNorm | fp16 | 128×128 | 92 | 89 | 0.97x |
| RMSNorm | fp16 | 256×256 | 89 | 90 | 1.01x |
| RMSNorm | fp16 | 512×512 | 132 | 108 | 0.82x |
| RMSNorm | fp16 | 1024×1024 | 312 | 236 | 0.76x |
| RMSNorm | fp16 | Avg Speedup |  |  | 0.90x |
| RMSNorm | ALL | Op Avg Speedup |  |  | 0.92x |
| RMSNorm-SplitK | fp32 | 64×64 | 92 | 75 | 0.82x |
| RMSNorm-SplitK | fp32 | 128×128 | 109 | 79 | 0.72x |
| RMSNorm-SplitK | fp32 | 256×256 | 144 | 82 | 0.57x |
| RMSNorm-SplitK | fp32 | 512×512 | 594 | 118 | 0.20x |
| RMSNorm-SplitK | fp32 | 1024×1024 | 4061 | 199 | 0.05x |
| RMSNorm-SplitK | fp32 | Avg Speedup |  |  | 0.47x |
| RMSNorm-SplitK | bf16 | 64×64 | 89 | 84 | 0.94x |
| RMSNorm-SplitK | bf16 | 128×128 | 130 | 94 | 0.72x |
| RMSNorm-SplitK | bf16 | 256×256 | 180 | 85 | 0.47x |
| RMSNorm-SplitK | bf16 | 512×512 | 833 | 112 | 0.13x |
| RMSNorm-SplitK | bf16 | 1024×1024 | 5667 | 238 | 0.04x |
| RMSNorm-SplitK | bf16 | Avg Speedup |  |  | 0.46x |
| RMSNorm-SplitK | fp16 | 64×64 | 103 | 78 | 0.76x |
| RMSNorm-SplitK | fp16 | 128×128 | 126 | 77 | 0.61x |
| RMSNorm-SplitK | fp16 | 256×256 | 179 | 86 | 0.48x |
| RMSNorm-SplitK | fp16 | 512×512 | 842 | 116 | 0.14x |
| RMSNorm-SplitK | fp16 | 1024×1024 | 5920 | 242 | 0.04x |
| RMSNorm-SplitK | fp16 | Avg Speedup |  |  | 0.41x |
| RMSNorm-SplitK | ALL | Op Avg Speedup |  |  | 0.45x |
| FlashAttention | fp32 | B1H2S4D8 | N/A | N/A | N/A |
| FlashAttention | fp32 | B1H8S64D32 | N/A | N/A | N/A |
| FlashAttention | fp32 | B1H8S128D64 | N/A | N/A | N/A |
| FlashAttention | fp32 | B1H8S256D128 | N/A | N/A | N/A |
| FlashAttention | fp32 | B1H8S512D128 | N/A | N/A | N/A |
| FlashAttention | fp32 | Avg Speedup |  |  | N/A |
| FlashAttention | bf16 | B1H2S4D8 | 302 | 213 | 0.71x |
| FlashAttention | bf16 | B1H8S64D32 | 12083 | 2192 | 0.18x |
| FlashAttention | bf16 | B1H8S128D64 | 12087 | 2141 | 0.18x |
| FlashAttention | bf16 | B1H8S256D128 | 11204 | 2028 | 0.18x |
| FlashAttention | bf16 | B1H8S512D128 | 43403 | 7864 | 0.18x |
| FlashAttention | bf16 | Avg Speedup |  |  | 0.29x |
| FlashAttention | fp16 | B1H2S4D8 | 304 | 210 | 0.69x |
| FlashAttention | fp16 | B1H8S64D32 | 12014 | 2179 | 0.18x |
| FlashAttention | fp16 | B1H8S128D64 | 12018 | 2140 | 0.18x |
| FlashAttention | fp16 | B1H8S256D128 | 11113 | 2136 | 0.19x |
| FlashAttention | fp16 | B1H8S512D128 | 42692 | 7725 | 0.18x |
| FlashAttention | fp16 | Avg Speedup |  |  | 0.28x |
| FlashAttention | ALL | Op Avg Speedup |  |  | 0.28x |

## 初始结果分析

- Matmul: `2026-06-09` best-of-5 后 Op Avg Speedup 为 1.14x，15 个配置 best 均大于 1x。
- RMSNorm direct: 初始 Op Avg Speedup 为 0.92x，PPL direct baseline 整体更快。
- RMSNorm-SplitK: 初始 Op Avg Speedup 为 0.45x，大 shape 受多 pass split-k 开销影响明显。
- FlashAttention: 初始 fp16/bf16 Op Avg Speedup 为 0.28x，主要输在 H8 大 shape 的逐 head 外层调度。

## 优化算子性能对比

本节只记录已经放入优化目录的算子结果。计时口径均为 TPU 模板默认的
5 次 warmup 后重复 10 次取平均。

### FlashAttention Fixed Best

`Speedup = PPL(us) / TL(us)`。当前 fixed suite 覆盖 non-causal square
self-attention，`q_len = kv_len = seq_len`。

| Dtype | Shape | 固化 TL tile | 优化 TL (us) | PPL (us) | Speedup |
|---|---|---:|---:|---:|---:|
| bf16 | B1H8S64D32 | `64x64` | 123 | 2,123 | 17.26x |
| bf16 | B1H8S128D64 | `128x128` | 141 | 2,183 | 15.48x |
| bf16 | B1H8S256D128 | `128x128` | 245 | 1,958 | 7.99x |
| bf16 | B1H8S512D128 | `128x128` | 671 | 7,535 | 11.23x |
| bf16 | B8H8S512D128 | `256x256,H8,s2` | 2,054 | 59,654 | 29.04x |
| bf16 | B8H32S512D128 | `256x256,H8,s2` | 7,821 | 101,678 | 13.00x |
| bf16 | B64H64S1024D128 | `256x256,H8,s2` | 431,822 | 1,237,966 | 2.87x |
| bf16 | B64H64S2048D128 | `128x512,H8,s2` | 1,548,654 | 4,834,021 | 3.12x |
| fp16 | B1H8S64D32 | `64x64` | 138 | 2,098 | 15.20x |
| fp16 | B1H8S128D64 | `128x128` | 138 | 2,209 | 16.01x |
| fp16 | B1H8S256D128 | `128x128` | 243 | 2,049 | 8.43x |
| fp16 | B1H8S512D128 | `128x128` | 671 | 7,674 | 11.44x |
| fp16 | B8H8S512D128 | `256x256,H8,s2` | 2,043 | 61,190 | 29.95x |
| fp16 | B8H32S512D128 | `256x256,H8,s2` | 7,823 | 100,203 | 12.81x |
| fp16 | B64H64S1024D128 | `256x256,H8,s2` | 431,822 | 1,285,176 | 2.98x |
| fp16 | B64H64S2048D128 | `128x512,H8,s2` | 1,548,585 | 4,986,000 | 3.22x |

平均 speedup：sequence sweep `12.88x`，batch/head variation `21.20x`，
large real `3.05x`，全表 `12.50x`。decode attention 的
`q_len=1, kv_len=S` 形态需要单独实现和测试，不能由本表直接外推。

### RMSNorm Fixed Best

RMSNorm fixed suite 覆盖 fp32、bf16、fp16 三种 dtype，以及 `64x64` 到
`1024x1024` 五个 shape。

| Dtype | Shape | 固化 TL variant | 优化 TL (us) | PPL (us) | Speedup | 初始 TL/优化 TL |
|---|---|---|---:|---:|---:|---:|
| bf16 | 64x64 | `keep_square_b32` | 117 | 106 | 0.91x | 0.71x |
| bf16 | 128x128 | `keep_square_b128` | 108 | 111 | 1.03x | 0.87x |
| bf16 | 256x256 | `keep_square_b128` | 114 | 114 | 1.00x | 0.75x |
| bf16 | 512x512 | `keep_square_b256` | 115 | 141 | 1.23x | 1.24x |
| bf16 | 1024x1024 | `keep_square_b512` | 131 | 265 | 2.02x | 2.28x |
| fp16 | 64x64 | `reload_b64` | 112 | 112 | 1.00x | 0.71x |
| fp16 | 128x128 | `reload_b128` | 89 | 111 | 1.25x | 1.03x |
| fp16 | 256x256 | `keep_square_b256` | 106 | 96 | 0.91x | 0.84x |
| fp16 | 512x512 | `keep_square_b512` | 113 | 140 | 1.24x | 1.17x |
| fp16 | 1024x1024 | `keep_square_b256` | 150 | 265 | 1.77x | 2.08x |
| fp32 | 64x64 | `reload_b32` | 116 | 108 | 0.93x | 0.64x |
| fp32 | 128x128 | `keep_square_b128` | 82 | 95 | 1.16x | 1.12x |
| fp32 | 256x256 | `reload_b256` | 82 | 112 | 1.37x | 1.02x |
| fp32 | 512x512 | `keep_square_b512` | 94 | 112 | 1.19x | 1.24x |
| fp32 | 1024x1024 | `keep_square_b128` | 162 | 230 | 1.42x | 1.49x |

fixed suite 记录轮平均 `PPL(us) / TL(us) = 1.23x`，相对初始 TL 平均提升 `1.15x`。
RMSNorm 优化主要改善 `512x512` 和 `1024x1024`，小 shape 已接近固定调度开销下限，
单次 10-run 平均值会有明显波动。

### MLP W8A16 Dequantized Forward Fixed Best

MLP W8A16 DQ fixed suite 覆盖调度验证、缩放 decode/prefill 和 DeepSeek 相关真实
shape。

| Config | Shape | 固化 TL 策略 | 优化 TL (us) | PPL 单核 (us) | Speedup |
|---|---|---|---:|---:|---:|
| `tiny` | M4H128I128 | `streaming_fullk` | 90 | 103 | 1.14x |
| `stage2` | M16H256I128 | `streaming_fullk` | 110 | 106 | 0.96x |
| `stage3` | M16H256I256 | `streaming_fullk` | 109 | 120 | 1.10x |
| `small_decode_scaled` | M1H512I2048 | `streaming_fullk` | 314 | 306 | 0.97x |
| `medium_prefill_scaled` | M16H1024I4096 | `streaming_fullk` | 545 | 557 | 1.02x |
| `moe_decode_dsv3` | M1H7168I2048 | `streaming_fullk_pipe2` | 1,001 | 999 | 1.00x |
| `moe_prefill_dsv3` | M16H7168I2048 | `ppl_layout_pipe1` | 1,116 | 1,011 | 0.91x |
| `dense_decode_dsv3` | M1H7168I18432 | `streaming_fullk_pipe2` | 8,215 | 8,109 | 0.99x |

fixed suite 平均 `PPL(us) / TL(us) = 1.01x`。本轮关键收益来自 H=7168 真实
shape 的 per-config pipeline：`moe_decode_dsv3`、`moe_prefill_dsv3` 和
`dense_decode_dsv3` 相对上一版 TL 分别提升 `1.38x`、`1.25x` 和 `1.42x`。
`moe_prefill_dsv3` 的最快原布局 pipeline 曾出现 NaN，因此固化为更稳定的
PPL-layout pipeline。

## 后续更新规则

1. 完整串行测试用于刷新 `初始性能对比` 或形成新的主线 baseline。
2. 优化目录的局部测试写入 `优化算子性能对比`。
3. 分块只记录 PPL baseline 参考值；TL 优化可以按性能选择不同策略，不要求与 PPL 对齐。
4. 中间 stdout/stderr、summary JSON 和历史实验目录不写入源码目录。
