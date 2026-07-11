#include "dispatcher.h"

#include <cstring>
#include <vector>


namespace {

constexpr const char *kDecodeKernel = "latent_attention_fp8_decode_kernel";
constexpr const char *kPrefillKernel = "latent_attention_fp8_prefill_kernel";

}  // namespace


int launch_latent_attention_fp8_kernel(
    tpuRtKernelModule_t module,
    tpuRtStream_t stream,
    int attention_mode,
    const void *api,
    std::size_t api_size,
    int decode_core_num) {
  if (module == nullptr || api == nullptr || api_size == 0) {
    return -1;
  }

  const char *kernel_name = nullptr;
  int block_num = 1;
  if (attention_mode == LATENT_ATTENTION_FP8_DECODE) {
    if (decode_core_num <= 0) {
      return -2;
    }
    kernel_name = kDecodeKernel;
    block_num = decode_core_num;
  } else if (attention_mode == LATENT_ATTENTION_FP8_PREFILL) {
    kernel_name = kPrefillKernel;
  } else {
    return -3;
  }

  std::vector<unsigned char> packed_args(api_size * block_num);
  for (int core = 0; core < block_num; ++core) {
    std::memcpy(packed_args.data() + core * api_size, api, api_size);
  }

  int ret = tpuRtKernelLaunch(
      module,
      kernel_name,
      packed_args.data(),
      packed_args.size(),
      1,
      block_num,
      stream);
  if (ret != 0) {
    return ret;
  }
  return tpuRtStreamSynchronize(stream);
}
