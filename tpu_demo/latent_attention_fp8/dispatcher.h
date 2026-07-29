#pragma once

#include <cstddef>

#include <tpuv7_rt.h>


enum LatentAttentionFp8Mode {
  LATENT_ATTENTION_FP8_PREFILL = 0,
  LATENT_ATTENTION_FP8_DECODE = 1,
};

// Launches one static TileLang kernel from the shared latent-attention firmware.
// The caller fills the mode-specific API struct and owns the head/row-tile loop.
int launch_latent_attention_fp8_kernel(
    tpuRtKernelModule_t module,
    tpuRtStream_t stream,
    int attention_mode,
    const void *api,
    std::size_t api_size,
    int decode_core_num = 8);
