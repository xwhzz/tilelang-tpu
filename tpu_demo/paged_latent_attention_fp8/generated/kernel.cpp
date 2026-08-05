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

int main_kernel_check_mem(unsigned long long ptr_v1, unsigned long long ptr_v2, unsigned long long ptr_v3, unsigned long long ptr_v4, unsigned long long ptr_v5, unsigned long long ptr_v6, unsigned long long ptr_v7, unsigned long long ptr_v8, unsigned long long ptr_v9, unsigned long long ptr_v10, unsigned long long ptr_v11, unsigned long long ptr_v12, unsigned long long ptr_v13, unsigned long long ptr_v14, unsigned long long ptr_v15, unsigned long long ptr_v16, unsigned long long ptr_v17) {
  return 0;
}

int main_kernel_check_mem_s(tpu_kernel_api_main_kernel_t *api) {
  return 0;
}

tpu_kernel_api_main_kernel_t fill_main_kernel_struct(unsigned long long ptr_v1, unsigned long long ptr_v2, unsigned long long ptr_v3, unsigned long long ptr_v4, unsigned long long ptr_v5, unsigned long long ptr_v6, unsigned long long ptr_v7, unsigned long long ptr_v8, unsigned long long ptr_v9, unsigned long long ptr_v10, unsigned long long ptr_v11, unsigned long long ptr_v12, unsigned long long ptr_v13, unsigned long long ptr_v14, unsigned long long ptr_v15, unsigned long long ptr_v16, unsigned long long ptr_v17) {
  tpu_kernel_api_main_kernel_t api;
  api.ptr_v1 = ptr_v1;
  api.ptr_v2 = ptr_v2;
  api.ptr_v3 = ptr_v3;
  api.ptr_v4 = ptr_v4;
  api.ptr_v5 = ptr_v5;
  api.ptr_v6 = ptr_v6;
  api.ptr_v7 = ptr_v7;
  api.ptr_v8 = ptr_v8;
  api.ptr_v9 = ptr_v9;
  api.ptr_v10 = ptr_v10;
  api.ptr_v11 = ptr_v11;
  api.ptr_v12 = ptr_v12;
  api.ptr_v13 = ptr_v13;
  api.ptr_v14 = ptr_v14;
  api.ptr_v15 = ptr_v15;
  api.ptr_v16 = ptr_v16;
  api.ptr_v17 = ptr_v17;
  return api;
}

int main_kernel(unsigned long long ptr_v1, unsigned long long ptr_v2, unsigned long long ptr_v3, unsigned long long ptr_v4, unsigned long long ptr_v5, unsigned long long ptr_v6, unsigned long long ptr_v7, unsigned long long ptr_v8, unsigned long long ptr_v9, unsigned long long ptr_v10, unsigned long long ptr_v11, unsigned long long ptr_v12, unsigned long long ptr_v13, unsigned long long ptr_v14, unsigned long long ptr_v15, unsigned long long ptr_v16, unsigned long long ptr_v17) {
  int core_num = 8;
  tpu_kernel_api_main_kernel_t api;
  int ret = 0;
  api.ptr_v1 = ptr_v1;
  api.ptr_v2 = ptr_v2;
  api.ptr_v3 = ptr_v3;
  api.ptr_v4 = ptr_v4;
  api.ptr_v5 = ptr_v5;
  api.ptr_v6 = ptr_v6;
  api.ptr_v7 = ptr_v7;
  api.ptr_v8 = ptr_v8;
  api.ptr_v9 = ptr_v9;
  api.ptr_v10 = ptr_v10;
  api.ptr_v11 = ptr_v11;
  api.ptr_v12 = ptr_v12;
  api.ptr_v13 = ptr_v13;
  api.ptr_v14 = ptr_v14;
  api.ptr_v15 = ptr_v15;
  api.ptr_v16 = ptr_v16;
  api.ptr_v17 = ptr_v17;
  int group_num = 1;
  int block_num = core_num;
  tpu_kernel_api_main_kernel_t apis[core_num];
  for (int i = 0; i < core_num; ++i) {
    apis[i] = api;
  }
  ret = tpuRtKernelLaunch(tpu_module, "main_kernel", &apis, sizeof(apis), group_num, block_num, stream);
  if (ret != 0) {
      printf("tpu kernel launch failed!");
      return ret;
  }
  ret = tpuRtStreamSynchronize(stream);
  if (ret != 0) {
      printf("tpu stream synchronize failed!");
      return ret;
  }
  return 0;
}
