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

int init(){
  tpuRtStatus_t ret;
  ret = tpuRtInit();
  if (ret != tpuRtSuccess) {
    return -1;
  }
  tpuRtSetDevice(0);
  tpuRtStreamCreate(&stream);
  auto kernel_dir = getenv("PPL_KERNEL_PATH");
  if (!kernel_dir) {
    return -2;
  }
  tpu_module = tpuRtKernelLoadModuleFile(kernel_dir, stream);
  if (NULL == tpu_module) {
    printf("tpuRtKernelLoadModuleFile failed\n");
    return -2;
  }
  return 0;
}

void post(){
  tpuRtKernelUnloadModule(tpu_module, stream);
  tpuRtStreamDestroy(stream);
}

int main(int argc, char** argv) {
  char* G_in = argv[1];
  size_t G_in_size = 64 * 64 * sizeof(DT_FP32);
  char* G_right = argv[2];
  size_t G_right_size = 64 * 64 * sizeof(DT_FP32);
  char* G_out = argv[3];
  size_t G_out_size = 64 * 64 * sizeof(DT_FP32);

  int res = init();
  if(res != 0){
    return res;
  }

  // 设备指针声明
  void *dev_G_in;
  void *dev_G_right;
  void *dev_G_out;

  // 分配设备内存
  tpuRtMalloc((void **)(&dev_G_in), G_in_size, 0);
  tpuRtMalloc((void **)(&dev_G_right), G_right_size, 0);
  tpuRtMalloc((void **)(&dev_G_out), G_out_size, 0);

  // 拷贝数据到设备
  tpuRtMemcpyS2D(dev_G_in, G_in, G_in_size);
  tpuRtMemcpyS2D(dev_G_right, G_right, G_right_size);
  tpuRtMemcpyS2D(dev_G_out, G_out, G_out_size);

  // 调用内核函数

  auto start = std::chrono::high_resolution_clock::now();  // 开始计时

  int rst = main_kernel((unsigned long long)dev_G_in, (unsigned long long)dev_G_right, (unsigned long long)dev_G_out);

  auto end = std::chrono::high_resolution_clock::now();    // 结束计时

  if (rst) {
    printf("kernel_launch failed\n");
    return 1;
  }
  printf("kernel_launch success\n");

  // 计算执行时间
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  double elapsed_time_ms = duration.count() / 1000.0;
  double elapsed_time_us = duration.count();

  printf("Kernel execution time: %.3f ms (%.0f us)\n", elapsed_time_ms, elapsed_time_us);

  // 性能测试
  const int warmup_runs = 5;
  const int measure_runs = 10;

  printf("\n=== Performance Benchmark (after %d warmup runs) ===\n", warmup_runs);

  // 预热运行
  for (int i = 0; i < warmup_runs; i++) {
      rst = main_kernel((unsigned long long)dev_G_in, (unsigned long long)dev_G_right, (unsigned long long)dev_G_out);
  }

  printf("Runs: %d\n", measure_runs);

  // 测量运行
  double total_time_us = 0.0;
  double min_time_us = std::numeric_limits<double>::max();
  double max_time_us = 0.0;

  for (int i = 0; i < measure_runs; i++) {
    auto run_start = std::chrono::high_resolution_clock::now();
      rst = main_kernel((unsigned long long)dev_G_in, (unsigned long long)dev_G_right, (unsigned long long)dev_G_out);
    auto run_end = std::chrono::high_resolution_clock::now();
    
    auto run_duration = std::chrono::duration_cast<std::chrono::microseconds>(run_end - run_start);
    double run_time_us = run_duration.count();
    
    total_time_us += run_time_us;
    min_time_us = std::min(min_time_us, run_time_us);
    max_time_us = std::max(max_time_us, run_time_us);
  }

  double avg_time_us = total_time_us / measure_runs;
  double avg_time_ms = avg_time_us / 1000.0;


  printf("Average execution time: %.3f ms (%.0f us)\n", avg_time_ms, avg_time_us);
  printf("Minimum execution time: %.3f ms (%.0f us)\n", min_time_us / 1000.0, min_time_us);
  printf("Maximum execution time: %.3f ms (%.0f us)\n", max_time_us / 1000.0, max_time_us);
  
  // 拷贝输出数据回主机
  tpuRtMemcpyD2S(G_out, dev_G_out, G_out_size);

  // 释放设备内存
  tpuRtFree(&dev_G_in, 0);
  tpuRtFree(&dev_G_right, 0);
  tpuRtFree(&dev_G_out, 0);

  post();
  return 0;
}