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
  tpuRtSetDevice(14); // Set TPU ID
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
  char* X = argv[1];
  size_t X_size = 128 * 128 * 2;
  char* Weight = argv[2];
  size_t Weight_size = 1 * 128 * 2;
  char* Output = argv[3];
  size_t Output_size = 128 * 128 * 2;

  int res = init();
  if(res != 0){
    return res;
  }

  // 设备指针声明
  void *dev_X;
  void *dev_Weight;
  void *dev_Output;

  // 分配设备内存
  tpuRtMalloc((void **)(&dev_X), X_size, 0);
  tpuRtMalloc((void **)(&dev_Weight), Weight_size, 0);
  tpuRtMalloc((void **)(&dev_Output), Output_size, 0);

  // 拷贝数据到设备
  tpuRtMemcpyS2D(dev_X, X, X_size);
  tpuRtMemcpyS2D(dev_Weight, Weight, Weight_size);
  tpuRtMemcpyS2D(dev_Output, Output, Output_size);

  // 调用内核函数

  auto start = std::chrono::high_resolution_clock::now();  // 开始计时

  int rst = main_kernel((unsigned long long)dev_X, (unsigned long long)dev_Weight, (unsigned long long)dev_Output);
  tpuRtStreamSynchronize(stream);

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

  printf("Single kernel execution time: %.3f ms (%.0f us)\n", elapsed_time_ms, elapsed_time_us);

  // 性能测试
  const int warmup_runs = 5;
  const int measure_runs = 10;

  printf("\n=== Performance Benchmark (after %d warmup runs) ===\n", warmup_runs);

  // 预热运行
  for (int i = 0; i < warmup_runs; i++) {
      rst = main_kernel((unsigned long long)dev_X, (unsigned long long)dev_Weight, (unsigned long long)dev_Output);
    tpuRtStreamSynchronize(stream);
  }

  printf("Runs: %d\n", measure_runs);

  // 测量运行
  double total_time_us = 0.0;
  double min_time_us = std::numeric_limits<double>::max();
  double max_time_us = 0.0;

  for (int i = 0; i < measure_runs; i++) {
    auto run_start = std::chrono::high_resolution_clock::now();
      rst = main_kernel((unsigned long long)dev_X, (unsigned long long)dev_Weight, (unsigned long long)dev_Output);
    tpuRtStreamSynchronize(stream);
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
  tpuRtMemcpyD2S(Output, dev_Output, Output_size);

  // 释放设备内存
  tpuRtFree(&dev_X, 0);
  tpuRtFree(&dev_Weight, 0);
  tpuRtFree(&dev_Output, 0);

  post();
  return 0;
}
