#include "kernel.h"
#include <ppl_mem.h>
#include <cstdio>
#include <mutex>
#include <memory>
#include <sstream>
#include <vector>
#include <unistd.h>
#include <numeric>
#include <string>
#include <tpuv7_rt.h>
extern tpuRtStream_t stream;
extern tpuRtKernelModule_t tpu_module;

#define MIN(x, y) (((x)) < ((y)) ? (x) : (y))
#define MAX(x, y) (((x)) > ((y)) ? (x) : (y))

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
  int core_num = {core_num};
  tpu_kernel_api_{function_name}_t api;
  int ret = 0;
  {struct_assignments}
  int group_num = 1;
  int block_num = core_num;
  tpu_kernel_api_{function_name}_t apis[core_num];
  for (int i = 0; i < core_num; ++i) {{
    apis[i] = api;
    apis[i].bx_id = i;
  }}
  ret = tpuRtKernelLaunch(tpu_module, "{function_name}", &apis, sizeof(apis), group_num, block_num, stream);
  if (ret != 0) {{
      printf("tpu kernel launch failed!");
      return ret;
  }}
  ret = tpuRtStreamSynchronize(stream);
  if (ret != 0) {{
      printf("tpu stream synchronize failed!");
      return ret;
  }}
  return 0;
}}
