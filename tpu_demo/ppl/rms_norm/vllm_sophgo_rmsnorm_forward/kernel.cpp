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

int main_kernel_check_mem(unsigned long long v1, unsigned long long v2, unsigned long long v3) {
  return 0;
}

int main_kernel_check_mem_s(tpu_kernel_api_main_kernel_t *api) {
  return 0;
}

tpu_kernel_api_main_kernel_t fill_main_kernel_struct(unsigned long long v1, unsigned long long v2, unsigned long long v3) {
  tpu_kernel_api_main_kernel_t api;
  api.v1 = v1;
  api.v2 = v2;
  api.v3 = v3;
  return api;
}

int main_kernel(unsigned long long v1, unsigned long long v2, unsigned long long v3) {
  int core_num = 8;
  tpu_kernel_api_main_kernel_t api;
  int ret = 0;
  api.v1 = v1;
  api.v2 = v2;
  api.v3 = v3;
  int group_num = 1;
  int block_num = 8;
  tpu_kernel_api_main_kernel_t apis[core_num];
  for (int i = 0; i < core_num; ++i) {
    apis[i] = api;
  }
  ret = tpuRtKernelLaunch(tpu_module, "main_kernel", &apis, sizeof(apis), group_num, block_num, stream);
  if (ret != 0) {
      printf("tpu kernel launch failed!");
      return ret;
  }
  return 0;
}