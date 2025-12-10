#include <tpuv7_rt.h>
#include "host_test_utils.h"
#include "kernel.h"
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>

tpuRtStream_t stream;
tpuRtKernelModule_t tpu_module;

int init(){{
  tpuRtStatus_t ret;
  ret = tpuRtInit();
  if (ret != tpuRtSuccess) {{
    return -1;
  }}
  tpuRtSetDevice(0);
  tpuRtStreamCreate(&stream);
  auto kernel_dir = getenv("PPL_KERNEL_PATH");
  if (!kernel_dir) {{
    return -2;
  }}
  tpu_module = tpuRtKernelLoadModuleFile(kernel_dir, stream);
  if (NULL == tpu_module) {{
    printf("tpuRtKernelLoadModuleFile failed\n");
    return -2;
  }}
  return 0;
}}

void post(){{
  tpuRtKernelUnloadModule(tpu_module, stream);
  tpuRtStreamDestroy(stream);
}}

int main(int argc, char** argv) {{
{arg_declarations}

  int res = init();
  if(res != 0){{
    return res;
  }}

  // 设备指针声明
{device_declarations}

  // 分配设备内存
{malloc_statements}

  // 拷贝数据到设备
{memcpy_s2d_statements}

  // 调用内核函数

  auto start = std::chrono::high_resolution_clock::now();  // 开始计时

{kernel_call}

  auto end = std::chrono::high_resolution_clock::now();    // 结束计时

  if (rst) {{
    printf("kernel_launch failed\n");
    return 1;
  }}
  printf("kernel_launch success\n");

  // 计算执行时间
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  double elapsed_time_ms = duration.count() / 1000.0;
  double elapsed_time_us = duration.count();

  printf("Kernel execution time: %.3f ms (%.0f us)\n", elapsed_time_ms, elapsed_time_us);
  
  // 拷贝输出数据回主机
{memcpy_d2s_statements}

  // 释放设备内存
{free_statements}

  post();
  return 0;
}}