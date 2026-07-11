#include "kernel.h"
#include <ppl_mem.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tpuv7_rt.h>
#include <vector>

extern tpuRtStream_t stream;
extern tpuRtKernelModule_t tpu_module;

int {function_name}_check_mem({func_params}) {{
  return 0;
}}

int {function_name}_check_mem_s(tpu_kernel_api_{function_name}_t *api) {{
  return 0;
}}

tpu_kernel_api_{function_name}_t fill_{function_name}_struct({func_params}) {{
  tpu_kernel_api_{function_name}_t api;
  {struct_assignments}
  return api;
}}

int {function_name}({func_params}) {{
  int core_num = 1;
  const char *core_parallel_env = std::getenv("TL_TPU_ENABLE_CORE_PARALLEL");
  if (core_parallel_env != nullptr && std::strcmp(core_parallel_env, "1") == 0) {{
    core_num = 8;
    const char *core_num_env = std::getenv("TL_TPU_CORE_NUM");
    if (core_num_env != nullptr) {{
      char *end = nullptr;
      long parsed = std::strtol(core_num_env, &end, 10);
      if (end != core_num_env && *end == '\0' && parsed > 0 && parsed <= 64) {{
        core_num = static_cast<int>(parsed);
      }}
    }}
  }}

  tpu_kernel_api_{function_name}_t api;
  {struct_assignments}
  std::vector<tpu_kernel_api_{function_name}_t> apis(core_num, api);

  int ret = tpuRtKernelLaunch(
      tpu_module,
      "{function_name}",
      apis.data(),
      static_cast<uint32_t>(apis.size() * sizeof(apis[0])),
      1,
      core_num,
      stream);
  if (ret != static_cast<int>(tpuRtSuccess)) {{
    std::printf("tpu kernel launch failed with status %d\n", ret);
    return ret;
  }}
  ret = tpuRtStreamSynchronize(stream);
  if (ret != static_cast<int>(tpuRtSuccess)) {{
    std::printf("tpu stream synchronize failed with status %d\n", ret);
    return ret;
  }}
  return 0;
}}
