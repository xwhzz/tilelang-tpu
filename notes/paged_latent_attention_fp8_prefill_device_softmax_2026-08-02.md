# FP8 Paged Latent Attention Prefill：设备端 Softmax 优化记录

日期：2026-08-02

## 固定目标与范围

- batch=1，query sequence=192，8 TPU cores，attention mode=2。
- 16 heads 为性能目标；32 heads 做模型接口和正确性回归。
- 将 QK 之后的 scale、causal mask、row max、exp、row sum、归一化和 PV 从 host 下沉到 TPU。
- 保持统一 `torch.ops.my_ops.paged_latent_attention_fp8` ABI，Decode 使用 mode=3。
- 16-head 四个 Prefill device kernels 合计不超过原版 PPL/C 0.260529 ms 的 10 倍，即不超过 2.60529 ms。

## 实现

- QK 继续使用 M=32、N=384、K=128x4 的多核分块 GEMM。
- Softmax/PV 核使用 16-row 分组。每个 `(query block, head)` 内处理两组 16 行：
  1. 从全局 QK score 读取 BF16，转换为 FP32；
  2. TPU 上乘 `softmax_scale` 并加 additive causal mask；
  3. TPU 上执行稳定 softmax：row max、subtract、exp、row sum；
  4. 用三个 K=128 PV GEMM 累积 FP32 output，再在 TPU 上除 row sum；
  5. BF16 写回模型输出。
- `PLA_TPU_SINGLE_RUN=1` 会从生成的 host launcher 删除内置 5 次 warmup + 10 次 measure 段，用于测量没有重复污染的 host-sync 延迟。
- Prefill 与 Decode 保持两个后端 kernel 集合，通过同一个模型 ABI 按 attention mode 分发。

## 16-head 正确性与稳定性

固定随机输入、TPU device 7：

- Q projection：finite/allclose，max abs error 0.000488。
- KVU projection：finite/allclose，max abs error 0.000244。
- QK score：finite，范围 [-0.063965, 0.059570]。
- 最终 output：finite/allclose，max abs error 0.000244，mean abs error 0.000003。
- KV cache：精确相等；PE cache max abs error 0.000488。
- 未触及 cache rows 保持不变。
- 重复调用输出 bitwise equal。
- 统一 ABI Decode mode=3 输出 finite，cache 更新回归通过。

## 性能

### 稳定态 device kernel 平均值

| 阶段 | 16 heads |
| --- | ---: |
| Cache/RoPE update | 0.124 ms |
| FP8 Q/KVU projection | 0.656 ms |
| QK | 0.376 ms |
| Stable causal softmax + PV | 1.189 ms |
| 合计 | **2.345 ms** |

- 原版 PPL/C：0.260529 ms。
- 当前精确设备 softmax Prefill：约 **9.00x** 原版，满足不超过 10x 的目标。
- 相比此前 20.185 ms TileLang Prefill 基线，device-kernel 合计约 **8.61x 加速**。
- 单次冷启动四核观测约 0.216 + 0.773 + 0.483 + 1.344 = 2.816 ms；稳定态表用于与原版 kernel benchmark 对比。

### 无内置重复污染的 host-sync

命令设置 `PLA_TPU_SINGLE_RUN=1 PLA_WARMUP=0 PLA_RUNS=1 PLA_TEST_DECODE=0`，测得一次完整模型 adapter 调用为 **135787.480866 ms**。

这个数字揭示了下一层瓶颈：当前验证原型由四个独立 JIT host/device `.so` 组成，每段 launcher 都执行 runtime 初始化、device malloc、输入 H2D（包括大权重）和输出 D2H。它不是 TPU 算术耗时；四个 device kernels 的稳定态总计算时间是 2.345 ms。要得到可部署的模型端到端延迟，下一步必须把 persistent device buffers、权重驻留和四阶段异步/融合调度接入模型 runtime，不能继续使用测试 launcher 作为生产调用层。

## 32-head 回归

- 最终 output finite/allclose，max abs error 0.000244，mean abs error 0.000003。
- Q/KVU projection、QK、KV/PE cache 与 16-head 使用相同判据并全部通过。
- 该回归验证了统一模型参数和 mode=2 ABI 在 32-head 配置下仍然对齐。

## 结论

QK 后的 score-dependent softmax 权重不再由 PyTorch/host 物化；scale、causal mask、稳定 softmax、normalization 和 PV 已完整位于 TPU。16-row 分组在当前 BM1690 lowering 上同时保持精确输出和性能，使 16-head device-kernel 总时间达到 2.345 ms（原版约 9.00x）。真实 host-sync 测量也已完成，并把后续优化边界明确定位到模型 runtime/launcher，而不是 attention 数学核。
