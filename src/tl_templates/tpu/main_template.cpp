#include <tpuv7_rt.h>
#include "host_test_utils.h"
#include "kernel.h"
#include <cstdlib>
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
  tpuRtSetDevice(11); // Set TPU ID
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

extern "C" int tilelang_tpu_run(void** args) {{
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

  printf("Single kernel execution time: %.3f ms (%.0f us)\n", elapsed_time_ms, elapsed_time_us);

  // 性能测试
  const int warmup_runs = 5;
  const int measure_runs = 10;

  printf("\n=== Performance Benchmark (after %d warmup runs) ===\n", warmup_runs);

  // 预热运行
  for (int i = 0; i < warmup_runs; i++) {{
    {pure_kernel_call}
  }}

  printf("Runs: %d\n", measure_runs);

  // 测量运行
  double total_time_us = 0.0;
  double min_time_us = std::numeric_limits<double>::max();
  double max_time_us = 0.0;

  for (int i = 0; i < measure_runs; i++) {{
    auto run_start = std::chrono::high_resolution_clock::now();
    {pure_kernel_call}
    auto run_end = std::chrono::high_resolution_clock::now();
    
    auto run_duration = std::chrono::duration_cast<std::chrono::microseconds>(run_end - run_start);
    double run_time_us = run_duration.count();
    
    total_time_us += run_time_us;
    min_time_us = std::min(min_time_us, run_time_us);
    max_time_us = std::max(max_time_us, run_time_us);
  }}

  double avg_time_us = total_time_us / measure_runs;
  double avg_time_ms = avg_time_us / 1000.0;


  printf("Average execution time: %.3f ms (%.0f us)\n", avg_time_ms, avg_time_us);
  printf("Minimum execution time: %.3f ms (%.0f us)\n", min_time_us / 1000.0, min_time_us);
  printf("Maximum execution time: %.3f ms (%.0f us)\n", max_time_us / 1000.0, max_time_us);
  
  // 拷贝输出数据回主机
{memcpy_d2s_statements}
  tpuRtStreamSynchronize(stream);

  // 释放设备内存
{free_statements}

  post();
  return 0;
}}
