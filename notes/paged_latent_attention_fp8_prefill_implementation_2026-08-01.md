# FP8 Paged Prefill 固定配置原型实现记录

日期：2026-08-01

## 范围

- 统一模型入口：`tpu_demo/paged_latent_attention_fp8.py`
- 兼容导入：`tpu_demo/paged_latent_attention_fp8_prefill.py`（仅 shim）
- 独立测试：`tpu_demo/benchmark_prefill_192.py`
- batch 1、query/cache-new length 192、cache length 0
- 16 heads、q rank 1536、kv rank 512、nope 128、RoPE 64、value 128
- FP8 E4M3 WUQ/WUKV、quant block 128、page size 16、attention mode 2
- additive causal mask、乱序 paged block table

内部 Prefill / Decode kernel 保持独立，模型侧通过同一个
`torch.ops.my_ops.paged_latent_attention_fp8` 按 `attention_mode` 分发：
Paged Prefill 为 2，Paged Decode 为 3。

## 模型接口对齐

1. Torch schema 参数顺序与 `torch-tpu-mla/Mla.cpp` 对齐，在 `num_heads`
   后补齐 `generate_token`。
2. `block_tables` 与原接口一致为 optional tensor；Paged 模式运行时要求提供。
3. schema 标注 `out`、`kv_cache`、`pe_cache` 的原地写语义。
4. `PagedLatentAttentionFp8Dispatcher` 可同时安装 Prefill 和 Decode adapter；
   `install_torch_custom_op(prefill=..., decode=...)` 只注册一个 Torch symbol。
5. Python adapter 继续兼容不含 `generate_token` 的旧直调形式；模型 Torch op
   使用完整新 ABI。

## 实现

1. 32-token tile 的 FP8 WUQ/WUKV 反量化与投影；transpose-B GEMM 显式累加 quant blocks。
2. 32-token tile 的 PE RoPE，随后使用 block table 展开的 token slots 对 KV/PE 各执行一次 paged scatter。
3. Query RoPE 与 Q/K/V packing 暂由 adapter 张量操作完成；Q/K 零填充到 512，KV sequence 从 192 padding 到 256。
4. TileLang attention 使用 M=1、N=128、K=512 的后端稳定 GEMM shape。
5. softmax 使用两遍 KV 扫描：第一遍求全局 row max，第二遍重新计算并累加 numerator/denominator；exp 输入下限为 -20。
6. 8-core 按 head 分工。

## 正确性结果

device 7，正式测试命令使用 5 次预热、10 次运行：

```bash
PLA_DEVICE_ID=7 PLA_WARMUP=5 PLA_RUNS=10 \
python3 tpu_demo/benchmark_prefill_192.py
```

- WUQ projection：finite/allclose，max abs `0.000488`
- WUKV projection：finite/allclose，max abs `0.000244`
- 最终输出：finite/allclose，max abs `0.081806`，mean abs `0.002313`
- KV cache：精确一致，max abs `0`
- PE cache：allclose，max abs `0.000488`，mean abs `0.000016`
- page-table 外 cache rows 未覆盖：通过
- 重复调用 bitwise output match：通过

测试使用非恒等 RoPE、乱序 12-page block table 和 cache 哨兵值。

### 32-head 模型接口复测

device 7，通过统一 Torch op 调用；`MODEL_OP_MODES` 为 `(2, 3)`：

- WUQ projection：finite/allclose，max abs `0.000488`
- WUKV projection：finite/allclose，max abs `0.000244`
- Prefill 输出：finite/allclose，max abs `0.087048`，mean abs `0.002314`
- KV cache：精确一致；PE cache max abs `0.000488`
- page-table 外 cache rows 未覆盖：通过
- Prefill 重复调用 bitwise output match：通过
- 同一 Torch op 的 Decode mode 3：输出 finite、KV cache 更新通过

32-head 测试使用 `atol=0.09, rtol=0.04`；门限调整只覆盖两遍 BF16
softmax 已观测的尾部极值，平均误差与 16-head 测试基本相同。

## 性能基线

运行时日志的典型 kernel 内部平均值：

- cache/RoPE/scatter：约 `0.12 ms`
- FP8 projection：约 `0.85 ms`
- 两遍 attention：约 `19.5 ms`

这是正确性优先原型，不是优化结果。attention 的 M=1 两遍扫描、512 维 padding、256-token padding、adapter 侧 packing/Query RoPE 都是明确的后续优化点。

## 后续工作

1. 将 Query RoPE 和 packing 移回 TileLang，并缓存不变的 padding metadata。
2. 修复或绕过当前后端在 Prefill M>1 QK/exp 路径上的限制，提升 query tile。
3. 使用稳定的 online softmax 单遍实现替代两遍扫描。
4. 建立 corrected PPL/C Prefill 的同输入正式基线，再做三方性能对比。
5. 在固定 16-head 原型稳定后补 32-head 配置和非零 cache length。
