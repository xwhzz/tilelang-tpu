# FP8 Paged Prefill 多核优化记录

日期：2026-08-01

## 结论

Prefill 的主要计算阶段已按原版 PPL 的连续 head 分片方式绑定到 8 个 TPU
core。projection 从“每核逐个 head 重复加载输入”改成“每核一次处理连续的
`heads_per_core` 个 head”，16-head 配置下每核处理 2 heads，32-head 配置下每核
处理 4 heads。模型 ABI、attention mode 2/3 分派和 Decode kernel 均未改变。

固定 16-head、sequence 192 配置下，projection 从 `0.829 ms` 降到
`0.473 ms`，减少 `42.9%`；三个 device kernel 合计从 `20.470 ms` 降到
`20.185 ms`，端到端 device kernel 时间减少 `1.39%`。剩余瓶颈是未改变算法的
两遍 attention（`19.602 ms`，约占新合计的 `97.1%`）。

## 实现

- 原版参考：`torch-tpu-mla/mla_prefill.pl` 中
  `heads_per_core = div_up(n_heads, core_num)` 和连续的 WUQ/WUKV per-core view。
- TileLang projection 现在以 `head_group` 为 kernel 任务；8 核各自拥有一个连续
  head group。
- 同一个 query/KV input block 在每核每次循环只加载一次，再通过更宽的 GEMM
  同时生成该核所有 head 的投影。
- 输出仍拆回原有 head-major buffer，所以上层张量布局与模型接口无变化。
- attention 继续以连续 head 范围分给 8 核；本阶段没有混入 query tiling、online
  softmax 或 kernel fusion。

## 性能

设备：TPU device 7；状态 Active、利用率 0%、显存 0 MB（测试前）。配置：batch
1，paged prefill mode 2，sequence 192，cache length 0，16 heads，page size 16，
Q rank 1536，KV rank 512，nope/rope/value 128/64/128，FP8 E4M3 权重，BF16
输入输出，8 cores。

每组数据来自生成 host benchmark 内置的 5 次预热和 10 次 device execution；新
版本列取三次完整调用所报告均值的平均值（共 30 次 device execution）。

| 阶段 | 旧单核 | 旧 8 核 | 新 8 核 head-group | 新版相对旧 8 核 |
|---|---:|---:|---:|---:|
| cache / RoPE / scatter | 0.123 ms | 0.126 ms | 0.110 ms | 1.15x |
| FP8 projection | 5.665 ms | 0.829 ms | 0.473 ms | 1.75x |
| 两遍 attention | 155.753 ms | 19.515 ms | 19.602 ms | 1.00x（未改算法） |
| 三 kernel 合计 | 161.541 ms | 20.470 ms | 20.185 ms | 1.014x |

单核到新版 8 核的三 kernel 合计加速为 `8.00x`。旧 8 核数据来自同日固定输入
基线；原版 PPL/C Prefill 为 `0.260529 ms`，因此新版与原版仍相差约 `77.48x`。
这说明下一阶段应优化 attention 的 query tiling/softmax，而不是继续微调
projection 或 cache。

## 正确性和接口回归

16-head：

- Q projection：max abs `0.000488`，通过；
- KVU projection：max abs `0.000244`，通过；
- output：max abs `0.081806`、mean abs `0.002313`，通过；
- KV cache：bitwise match；PE cache max abs `0.000488`；
- 未触及 cache rows 保持不变；重复执行 output bitwise stable。

32-head 统一接口：

- 注册的同一个 `torch.ops.my_ops.paged_latent_attention_fp8` 同时包含 mode
  `(2, 3)`；
- Prefill output max abs `0.087048`、mean abs `0.002314`，cache 检查通过；
- Decode mode 3 输出 finite，原 Decode backend 未修改。

注意：benchmark 的 `PREFILL_HOST_SYNC_MS` 包含 Python 端临时张量拼装和 clone，
不作为 device kernel 性能指标。
