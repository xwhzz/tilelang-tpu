# Latent Attention FP8 Device Package

This directory contains the verified non-paged bf16+fp8 TileLang frontend,
generated device C, and a single TPU firmware library.

The library registers two independent device entries:

- `latent_attention_fp8_decode_kernel`: static `B=1,H=16,HEAD_BLOCK=2`, eight
  workitems, runtime cache slot.
- `latent_attention_fp8_prefill_kernel`: static `row_tile=8`, one head per
  launch. The integrating C++ operator owns the head and row-tile loops.

Submitted files:

- `latent_attention_bf16_fp8_tilelang.py`: TileLang frontend for decode and prefill.
- `latent_attention_fp8_*_kernel.c`: generated TPU device sources.
- `libkernel.so`: device firmware containing both registered entries.
- `dispatcher.cpp` and `dispatcher.h`: reference host-side entry selection and launch helper.

The integrating C++ operator loads `libkernel.so`, constructs the mode-specific
argument structure, and calls `launch_latent_attention_fp8_kernel`. The dispatcher
selects the decode or prefill entry from `attention_mode`; it does not register a
Torch operator. Tensor validation, device-address extraction, and prefill
head/row-tile loops remain the responsibility of the integrating C++ operator.

The device `libkernel.so` is different from the host extension that registers
`torch.ops.my_ops.latent_attention_fp8`.
