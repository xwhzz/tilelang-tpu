# RMSNorm 优化算子

本目录存放已经固化的 TileLang-TPU RMSNorm direct path 优化算子。每个
`bench_rmsnorm_<dtype>_<shape>.py` 都是一个具体 dtype/shape 的 best 实现，脚本内部
直接保存当前最优策略和 `BLK_M`，不保留运行时候选搜索逻辑。

## 优化效果

本目录覆盖 fp32、bf16、fp16 三种 dtype，以及 `64x64`、`128x128`、`256x256`、
`512x512`、`1024x1024` 五个 shape。fixed suite 的 TL/PPL correctness 全部通过。

`Speedup = PPL(us) / TL(us)`，大于 1 表示 TileLang-TPU 更快。当前 fixed suite 平均
`PPL(us) / TL(us) = 1.23x`，相对初始 TileLang-TPU 平均提升 `1.15x`。

| Config | 固化 TL variant | 优化重点 |
|---|---|---|
| `bf16_64x64` | `keep_square_b32` | 小 shape，控制固定调度开销 |
| `bf16_128x128` | `keep_square_b128` | 保留输入和平方 buffer，减少输出阶段读回 |
| `bf16_256x256` | `keep_square_b128` | 中等 shape，平衡 tile 数和 local memory |
| `bf16_512x512` | `keep_square_b256` | 中大 shape，减少外层 tile 数 |
| `bf16_1024x1024` | `keep_square_b512` | 大 shape，主要收益来自减少 global load |
| `fp16_64x64` | `reload_b64` | 小 shape，减少 local buffer |
| `fp16_128x128` | `reload_b128` | 小中 shape，重读输入换 local memory |
| `fp16_256x256` | `keep_square_b256` | 中等 shape，减少输出阶段读回 |
| `fp16_512x512` | `keep_square_b512` | 中大 shape，减少外层 tile 数 |
| `fp16_1024x1024` | `keep_square_b256` | 大 shape，平衡 local memory 和 tile 数 |
| `fp32_64x64` | `reload_b32` | 小 shape，固定开销主导 |
| `fp32_128x128` | `keep_square_b128` | 中小 shape，减少输出阶段读回 |
| `fp32_256x256` | `reload_b256` | 中等 shape，减少 local buffer |
| `fp32_512x512` | `keep_square_b512` | 中大 shape，减少外层 tile 数 |
| `fp32_1024x1024` | `keep_square_b128` | 大 shape，按实测选择更小 tile 降低压力 |

固化脚本的逐配置复测结果如下：

| Config | TL | PPL | Speedup |
|---|---:|---:|---:|
| `bf16_64x64` | 117 us | 106 us | 0.91x |
| `bf16_128x128` | 108 us | 111 us | 1.03x |
| `bf16_256x256` | 114 us | 114 us | 1.00x |
| `bf16_512x512` | 115 us | 141 us | 1.23x |
| `bf16_1024x1024` | 131 us | 265 us | 2.02x |
| `fp16_64x64` | 112 us | 112 us | 1.00x |
| `fp16_128x128` | 89 us | 111 us | 1.25x |
| `fp16_256x256` | 106 us | 96 us | 0.91x |
| `fp16_512x512` | 113 us | 140 us | 1.24x |
| `fp16_1024x1024` | 150 us | 265 us | 1.77x |
| `fp32_64x64` | 116 us | 108 us | 0.93x |
| `fp32_128x128` | 82 us | 95 us | 1.16x |
| `fp32_256x256` | 82 us | 112 us | 1.37x |
| `fp32_512x512` | 94 us | 112 us | 1.19x |
| `fp32_1024x1024` | 162 us | 230 us | 1.42x |

## 优化思路

RMSNorm 的计算可以拆成三步：

1. 对每行求 `sum(x^2)`。
2. 计算 `rsqrt(sum / N + eps)`。
3. 输出 `x * scale * weight`。

初始 TileLang-TPU direct path 的主要瓶颈在低精度大 shape：输出阶段为了恢复原始
输入，会再次从 global memory 读取 `x`。但如果始终把原始输入和平方结果都留在本地，
又会增加 local memory 压力，降低可用 tile 大小。

因此本目录固化两类策略：

| 策略 | 做法 | 优点 | 代价 |
|---|---|---|---|
| `keep_square` | 本地同时保留原始输入和平方 buffer | 输出阶段不再重读 global input | local memory 压力更高 |
| `reload` | 原地平方，输出阶段重新读输入 | local buffer 更少，适合小 shape 或 fp32 部分配置 | 需要额外 global load |

最终不是固定一种策略，而是按 dtype/shape 选择：

- 低精度中大 shape 更容易受重复 global load 影响，通常偏向 `keep_square`。
- 小 shape 的固定调度开销占比高，`reload` 有时更稳。
- `BLK_M` 必须跟策略一起搜索；它同时影响外层 tile 数、descriptor 重建次数和
  local memory 分配压力。

## 正确性与波动

RMSNorm 小 shape 的 TL/PPL 耗时经常在 100 us 左右，10 us 以内波动就可能造成排名反转。
因此选择 best 时需要同时看：

- TL/PPL correctness 必须通过。
- 大 shape 的收益优先级高于小 shape 的微小波动。
- 如果某个小 shape 的 speedup 接近 1x，通常不值得为了几微秒引入额外复杂度。

## 运行方式

运行单个固定配置：

```bash
python tpu_benchmark/rmsnorm_opt/bench_rmsnorm_fp16_1024x1024.py
```

列出全部固化配置：

```bash
python tpu_benchmark/rmsnorm_opt/run_rmsnorm_opt_suite.py --list-configs
```

运行全部固定配置：

```bash
python tpu_benchmark/rmsnorm_opt/run_rmsnorm_opt_suite.py --configs all --timeout 300
```

## 维护规则

- 本目录只保存固化后的 best 脚本和必要说明。
- 重新搜索时可以临时增加候选入口，但 best 确认后应回写到对应固定脚本。
- 不再恢复 `kernels.py` 或把搜索参数保留成运行时分支。
- 临时 stdout/stderr、JSON、Markdown 汇总不应保留在本源码目录。
