# `paged_latent_attention_fp8` 长上下文 NaN 修复报告

日期：2026-07-11

## 问题

融合后的 8 核 `paged_latent_attention_fp8` 在短上下文正常，但当
`context_rows=64/128`（分别对应 `logical_blocks=4/8`）时输出出现
`NaN`。KV cache 与 PE cache 同时仍与对照路径一致，因此问题收敛到
tiled attention 的 softmax / value accumulation，而不是 RoPE、cache
scatter 或 paged gather。

## 根因

attention 循环原先每个 KV tile 都把 `max_v` 重置并写成当前 tile 的
row max。进入下一个 tile 后，rescale 使用的是：

```text
exp(previous_tile_max - current_tile_max)
```

而不是 online softmax 所需的：

```text
m_new = max(m_previous, m_current)
exp(m_previous - m_new)
```

当后一个 tile 的最大值更小，错误的 rescale 会大于 1 并可能溢出，最终
使 `logsum` 或 value accumulator 产生非法数值。

## 修复

在 fused kernel 和 four-kernel attention kernel 中都改为：

1. 计算当前 tile 的 `max_curr`；
2. 将 `max_prev` 和 `max_curr` 写入一个 `[heads, 2]` 的小 tile；
3. 对这个小 tile 调用 `ppl_reduce_max` 得到新的全局 running `max_v`；
4. 用这个 running max 统一计算历史 accumulator 的 rescale 和当前 tile
   的 softmax 概率。

这里没有使用 `ppl_max`：虽然 Python 接口存在，但当前已构建的 TPU
codegen 没有在 `kernel.c` 中发射该 elementwise 指令。`ppl_reduce_max`
已在当前运行时稳定生成和执行。

同时调整 binder，以适配当前 `LibraryGenerator` 对 host/device `.so`
使用唯一文件名的行为；绑定后的 `.so` 放进 `pla_bound/` 子目录，避免
后续 kernel 编译清理模板目录顶层 `.so` 时使此前绑定的 kernel 失效。

为让现有 FP8 路径重新编译，补齐了 live TileLang 树中缺失的：

- `ppl_scatter` / `ppl_gather_block` language exports；
- TPU Cython wrapper 的 `e4m3_float8 -> DT_FP8E4M3` type map。

## 验证

服务器上的完整回归脚本：

```text
/mnt2/users/tilelanguser11/pla_longctx_full_regression_20260711.py
```

每个 shape 都执行 fused 8-core、four-kernel 8-core、独立 PyTorch
reference，并检查 KV/PE cache 与重复调用稳定性。

| context rows | fused vs CPU max diff | finite | allclose | KV/PE cache | second call |
| ---: | ---: | --- | --- | --- | --- |
| 16 | 0.000131 | True | True | match | stable |
| 32 | 0.000134 | True | True | match | stable |
| 64 | 0.000113 | True | True | match | stable |
| 128 | 0.000116 | True | True | match | stable |

fused 与 four-kernel 8-core 的最大差异依次为 `0.000131`、`0.000134`、
`0.000121`、`0.000144`，四个 shape 均 finite、allclose，且 KV/PE cache
完全匹配。

## 保持的接口与并行语义

- `attention_mode=3` ABI guard 未变；
- fused 路径仍按 32 heads 进行 8 核分配；
- B8 生成的 `kernel.cpp` 验证为 `core_num = 8`、`block_num = core_num`；
- 本次测试生成的临时 kernel/source/shared-object 已从 repo 清理，未纳入提交。
