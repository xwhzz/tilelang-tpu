# Paged Latent Attention FP8：Decode 进度与 Prefill 开发交接

**更新时间：2026-08-01**
**用途：** 新对话开始实现和测试 FP8 Prefill 时，先完整读取本文件。

## 1. 新对话的任务目标

在保留当前 FP8 Paged Decode 成果的基础上，为 `paged_latent_attention_fp8` 增加独立的 Prefill kernel。第一阶段先完成固定配置的正确实现和性能基线，再逐步扩展 shape 和优化性能。

建议第一周目标：

1. 参考原版 PPL Prefill，梳理输入、mask、paged cache 写入和输出语义；
2. 完成 batch 1、FP8 权重、固定 context length 的 TileLang Prefill 原型；
3. 建立 CPU reference、PPL/C 和 TileLang 三方对比；
4. 验证输出、KV/PE cache 和重复调用一致性；
5. 在同一空闲 TPU device 上记录正式延迟。

不要直接承诺一周内完成所有动态长度、Normal/Paged 两种模式及充分性能优化。一周内完成固定配置原型与正确性验证是合理目标。

## 2. 仓库、用户与分支

- SSH 主机别名：`tpu`
- 服务器用户：`tilelanguser11`
- 服务器仓库：`/mnt2/users/tilelanguser11/tilelang-tpu`
- 工作分支：`feature/paged-latent-attention-fp8-sophgo-clean`
- 用户远程：`wyleeyuan`，地址为 `github-wyleeyuan:wyleeyuan/tilelang-tpu.git`
- GitHub 分支：`wyleeyuan/feature/paged-latent-attention-fp8-sophgo-clean`
- 不要修改或推送 `main`，不要操作其他用户、其他用户目录或其他人的远程分支。

截至 2026-08-01：

- 服务器 HEAD：`f832e36 Reuse transient WUKV weights in paged decode`
- `wyleeyuan` 远程分支：`96dd7a6 Optimize paged latent attention single-tile decode`
- 因此 `f832e36` 是已提交但尚未推送的最新 Decode 版本。
- 除非用户明确要求，不要自动 push。

服务器工作区还有已有的编译器修改、生成文件和实验文件。它们可能属于此前实验，必须保留；不要执行 `git reset --hard`、`git clean` 或覆盖无关修改。开始 Prefill 前先运行 `git status --short --branch`，新工作优先使用独立的新文件，避免污染稳定 Decode 入口。

## 3. 当前 Decode 的正式入口和版本

最新稳定入口：

```text
/mnt2/users/tilelanguser11/tilelang-tpu/tpu_demo/paged_latent_attention_fp8.py
```

最新提交 `f832e36` 只修改了上述文件，共 32 行新增、26 行删除。该版本在 `96dd7a6` 单-tile版本上增加了 kernel 内的临时 WUKV K/V 权重复用。

服务器还保留若干实验文件：

```text
tpu_demo/paged_latent_attention_fp8_single_tile.py
tpu_demo/paged_latent_attention_fp8_ppl_wukv_reuse.py
tpu_demo/paged_latent_attention_fp8_ppl_pipeline.py
tpu_demo/paged_latent_attention_fp8_ppl_inplace.py
tpu_demo/paged_latent_attention_fp8_cached_weights.py
```

其中部分文件是未跟踪实验文件，不能当作正式入口，也不要在未检查差异前全部提交。

## 4. Decode 已完成的主要优化

当前 Decode 是 batch 1、query length 1 的低延迟路径，已经完成：

1. 利用 GEMM 原地累加语义，减少临时 clear/add；
2. 将 128 维 `q_nope` 和 64 维 `q_rope` 合并为一次 192 维 WUQ 投影；
3. 对 128-token context 使用单个 `block_n=128` tile，删除两轮 64-token attention 的中间状态合并；
4. 删除单-tile场景不需要的 online-softmax 历史 max、rescale 等路径；
5. Query 每个 TPU core 加载一次，供该 core 负责的 4 个 head 复用；
6. Key RoPE 每个 TPU core 计算一次，供 4 个 head 复用；
7. cache 更新和 paged gather 避免在多个 core 上重复执行；
8. 合并 K/V 权重加载与 FP8 转换，K 用于 attention 前计算，V 临时留在 kernel 片上存储中，attention 后直接复用。

第 8 项不会增加长期显存或常驻 BF16 权重，只延长一次 kernel 调用中 V 权重的片上生命周期。

注意：“让一个 TPU core 把 4 个 head 的全部公共权重只加载、转换一次”仍是下一步优化方向，当前版本只完整实现了每核 Query/Key RoPE 复用以及一次 Decode 内的 K/V 临时复用，不要在汇报中说成已经全部完成。

## 5. Decode 性能和正确性基线

统一正式对比配置：

- batch：1
- attention heads：32
- context length：128 tokens
- q_lora_rank：1536
- kv_lora_rank：512
- qk_nope_head_dim：128
- qk_rope_head_dim：64
- v_head_dim：128
- page size：16
- quant block：128
- TPU core：8
- device：7
- 计时边界：kernel launch + stream synchronize

性能演进：

| 版本 | 延迟 | 说明 |
| --- | ---: | --- |
| 联合 WUQ 基线 | `0.7723 ms` | device 7 正式复测 |
| 单-tile Decode | 平均 `0.5285 ms`，中位 `0.5270 ms` | 相对基线提升 `31.6%` |
| 最新 K/V 临时复用版 | 稳定段 `0.506/0.508 ms`，典型约 `0.507 ms` | 相对上一版约提升 `4%` |
| corrected PPL/C Decode | 平均 `0.257661 ms`，中位 `0.252727 ms` | 相同 device 7 和计时边界 |

当前 TileLang Decode 典型耗时约为 corrected PPL/C 的 `1.97x`，绝对差距约 `0.249 ms`。

最新 K/V 临时复用版正确性已经通过：

- CPU reference 最大误差：`0.000549`
- 输出 finite/allclose：通过
- KV cache、PE cache：一致
- 重复调用：一致

单-tile版本此前 CPU reference 最大误差为 `0.000023`。不同版本误差数字不要混用。

## 6. Decode 测试入口

服务器已有：

```text
/mnt2/users/tilelanguser11/benchmark_fused_128.py
/mnt2/users/tilelanguser11/run_paged_latent_attention_experiment.sh
/mnt2/users/tilelanguser11/pla_benchmark_base.py
```

测试时通过 `PLA_MODULE` 选择版本，通过 `PLA_DEVICE_ID` 选择设备。正式统计使用 5 次预热和 10 次运行；需要 CPU reference 时设置 `PLA_SKIP_REFERENCE=0`。

示例：

```bash
PLA_MODULE=paged_latent_attention_fp8 \
PLA_DEVICE_ID=7 \
PLA_WARMUP=5 \
PLA_RUNS=10 \
/mnt2/users/tilelanguser11/run_paged_latent_attention_experiment.sh
```

日志中的 `Average execution time` 才是 kernel 性能；`HOST_TIMING_MS` 包含较大的框架调用开销，不能用于性能结论。每次测试前先检查设备状态和占用，前后版本必须使用同一设备、相同输入、相同预热/统计次数和相同计时边界。

本地保存的正式 Decode 日志位于：

```text
paged_latent_attention_fp8/results/ppl_vs_tilelang_20260728/
```

## 7. 原版 PPL/C 的来源

Decode PPL 源码：

```text
paged_latent_attention_fp8/torch-tpu-mla/mla_decode.pl
```

FP8 Decode kernel 入口：`mla_decode_bf16_fp8e4m3`，约在第 1396 行。

Prefill PPL 源码：

```text
paged_latent_attention_fp8/torch-tpu-mla/mla_prefill.pl
```

FP8 Prefill kernel 入口：`mla_prefill_bf16_fp8e4m3`，约在第 1086 行。

C++ 分发入口：

```text
paged_latent_attention_fp8/torch-tpu-mla/Mla.cpp
```

该文件本来就根据 `attention_mode` 分别调用 Decode 和 Prefill kernel，说明原版也是两个独立 kernel，不是一个 kernel 同时处理两个阶段。代码中 paged prefill 使用 `attention_mode == 2`，paged decode 使用 `attention_mode == 3`。

corrected PPL/C Decode 来自项目原有的 `mla_decode.pl`。原始 batch 1 路径有输出写回条件问题，只修复该条件后重新编译；attention、FP8 反量化、权重加载和多核计算仍保持原版流程。因此应称为“corrected PPL/C”，不要描述为完全未修改的官方二进制。

## 8. Prefill 与 Decode 的关系

二者数学主流程相近：Query 投影、RoPE、attention、输出投影。但工程实现明显不同：

- Decode 的 query length 通常为 1，主要读取较长 KV cache，目标是降低单 token 延迟；
- Prefill 一次处理多个 query token，需要因果 mask，并批量生成或写入 KV/PE cache，目标更偏向吞吐；
- Prefill 的二维 tiling、多核分工、mask、cache 写入和长度处理不能直接照搬 Decode；
- Decode 中针对单个 query 和 128-token单 tile 的快速路径，不能未经验证直接用于 Prefill。

因此应复用已有的 FP8 投影、反量化、RoPE、paged 地址计算和测试框架思路，但新建独立 Prefill kernel，而不是在 Decode 文件里硬加分支。

## 9. 建议的 Prefill 首个固定配置

原版 PPL 自带 FP8 Prefill 测试使用：batch 1、context length 192、cache length 0、page size 16、q rank 1536、kv rank 512、nope 128、RoPE 64、value 128、quant block 128、paged prefill、带 mask；源码测试中 head 数是 16。

开始实现时应先核对团队实际目标 shape。若没有其他要求，可先复现原版 PPL 自带的固定配置，以便直接建立 PPL/C 基线；随后再补与当前项目一致的 32-head 配置。不要默认把 Decode 的“128-token context”误当成 Prefill 的“128个 query token”。

首版建议新建：

```text
tpu_demo/paged_latent_attention_fp8_prefill.py
```

同时为 Prefill 新建独立的输入/reference 和 benchmark，避免修改已经稳定的 Decode benchmark。

## 10. Prefill 开发顺序与门禁

建议按以下顺序推进：

1. 读 `mla_prefill.pl` 的 kernel 参数、shape、mask、block table、save slots 和 cache 更新；
2. 建立 CPU reference，并先把 PPL/C Prefill 编译和计时跑通；
3. 实现固定 shape TileLang Prefill，不先做过多优化；
4. 检查输出 finite、CPU allclose、KV cache、PE cache和重复调用；
5. 正确性通过后再测性能，记录原始日志；
6. 再针对 tiling、core 分工、权重复用和搬运/计算重叠做优化；
7. 每次只改变一个主要变量，保留前后对照版本和回滚点。

Prefill 正确性至少检查：

- causal mask 是否正确；
- 每个 query token 的输出是否匹配 CPU reference；
- paged block table 和 save slots 是否正确映射；
- 新产生的 KV/PE 是否写入正确 cache 位置；
- cache 中原有内容是否未被误覆盖；
- 不同有效长度、边界 page 和重复调用是否稳定。

性能报告至少记录：设备号、设备状态、完整 shape、attention mode、是否带 mask、预热次数、统计次数、均值、中位数、原始样本、计时边界，以及 PPL/C 和 TileLang 是否使用相同输入。

## 11. 相关本地资料

- Decode/PPL 性能报告：`notes/paged_latent_attention_fp8_ppl_performance_comparison_2026-07-29.md`
- 单-tile优化报告：`notes/paged_latent_attention_fp8_single_tile_optimization_2026-07-28.md`
- 第三阶段报告：`notes/paged_latent_attention_fp8_third_stage_optimization_report_2026-07-29.md`
- 历史性能交接：`notes/paged_latent_attention_fp8_performance_handoff_2026-07-25.md`
- PPL/C完整对比：`notes/paged_latent_attention_fp8_vs_ppl_c_2026-07-28.md`
- 本地 PPL/C benchmark：`paged_latent_attention_fp8/ppl_vs_tilelang_benchmark/`

## 12. 新对话可直接使用的开场指令

```text
请先完整读取 notes/paged_latent_attention_fp8_prefill_handoff_2026-08-01.md，核对 TPU 服务器分支、HEAD、工作区和设备状态。不要修改 main，不要操作其他用户，不要清理服务器已有脏工作区，也不要未经我确认 push。然后参考 torch-tpu-mla/mla_prefill.pl，为当前 paged_latent_attention_fp8 新建独立的 FP8 Prefill 原型和 CPU/PPL/TileLang 性能测试。先建立固定配置基线并验证输出与 KV/PE cache 正确，再优化性能；保留现有 Decode 稳定入口不变。
```
