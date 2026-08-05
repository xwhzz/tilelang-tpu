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

extern "C" int tilelang_tpu_run(void** args) {
  char* query = static_cast<char*>(args[0]);
  size_t query_size = 1 * 1536 * 2;
  char* wuq = static_cast<char*>(args[1]);
  size_t wuq_size = 6144 * 1536 * 1;
  char* wuq_scale_expanded = static_cast<char*>(args[2]);
  size_t wuq_scale_expanded_size = 6144 * 12 * 2;
  char* wukv = static_cast<char*>(args[3]);
  size_t wukv_size = 8192 * 512 * 1;
  char* wukv_scale_expanded = static_cast<char*>(args[4]);
  size_t wukv_scale_expanded_size = 8192 * 4 * 2;
  char* kv_cache = static_cast<char*>(args[5]);
  size_t kv_cache_size = 32 * 512 * 2;
  char* pe_cache = static_cast<char*>(args[6]);
  size_t pe_cache_size = 32 * 64 * 2;
  char* kv_new = static_cast<char*>(args[7]);
  size_t kv_new_size = 1 * 512 * 2;
  char* pe_new = static_cast<char*>(args[8]);
  size_t pe_new_size = 1 * 64 * 2;
  char* cos = static_cast<char*>(args[9]);
  size_t cos_size = 1 * 64 * 2;
  char* sin = static_cast<char*>(args[10]);
  size_t sin_size = 1 * 64 * 2;
  char* block_table = static_cast<char*>(args[11]);
  size_t block_table_size = 2 * 1 * 4;
  char* save_slots = static_cast<char*>(args[12]);
  size_t save_slots_size = 1 * 1 * 4;
  char* kv_gather = static_cast<char*>(args[13]);
  size_t kv_gather_size = 32 * 512 * 2;
  char* pe_gather = static_cast<char*>(args[14]);
  size_t pe_gather_size = 32 * 64 * 2;
  char* pe_new_rope = static_cast<char*>(args[15]);
  size_t pe_new_rope_size = 1 * 64 * 2;
  char* output = static_cast<char*>(args[16]);
  size_t output_size = 1 * 32 * 1 * 128 * 2;

  int res = init();
  if(res != 0){
    return res;
  }

  // 设备指针声明
  void *dev_query;
  void *dev_wuq;
  void *dev_wuq_scale_expanded;
  void *dev_wukv;
  void *dev_wukv_scale_expanded;
  void *dev_kv_cache;
  void *dev_pe_cache;
  void *dev_kv_new;
  void *dev_pe_new;
  void *dev_cos;
  void *dev_sin;
  void *dev_block_table;
  void *dev_save_slots;
  void *dev_kv_gather;
  void *dev_pe_gather;
  void *dev_pe_new_rope;
  void *dev_output;

  // 分配设备内存
  tpuRtMalloc((void **)(&dev_query), query_size, 0);
  tpuRtMalloc((void **)(&dev_wuq), wuq_size, 0);
  tpuRtMalloc((void **)(&dev_wuq_scale_expanded), wuq_scale_expanded_size, 0);
  tpuRtMalloc((void **)(&dev_wukv), wukv_size, 0);
  tpuRtMalloc((void **)(&dev_wukv_scale_expanded), wukv_scale_expanded_size, 0);
  tpuRtMalloc((void **)(&dev_kv_cache), kv_cache_size, 0);
  tpuRtMalloc((void **)(&dev_pe_cache), pe_cache_size, 0);
  tpuRtMalloc((void **)(&dev_kv_new), kv_new_size, 0);
  tpuRtMalloc((void **)(&dev_pe_new), pe_new_size, 0);
  tpuRtMalloc((void **)(&dev_cos), cos_size, 0);
  tpuRtMalloc((void **)(&dev_sin), sin_size, 0);
  tpuRtMalloc((void **)(&dev_block_table), block_table_size, 0);
  tpuRtMalloc((void **)(&dev_save_slots), save_slots_size, 0);
  tpuRtMalloc((void **)(&dev_kv_gather), kv_gather_size, 0);
  tpuRtMalloc((void **)(&dev_pe_gather), pe_gather_size, 0);
  tpuRtMalloc((void **)(&dev_pe_new_rope), pe_new_rope_size, 0);
  tpuRtMalloc((void **)(&dev_output), output_size, 0);

  // 拷贝数据到设备
  tpuRtMemcpyS2D(dev_query, query, query_size);
  tpuRtMemcpyS2D(dev_wuq, wuq, wuq_size);
  tpuRtMemcpyS2D(dev_wuq_scale_expanded, wuq_scale_expanded, wuq_scale_expanded_size);
  tpuRtMemcpyS2D(dev_wukv, wukv, wukv_size);
  tpuRtMemcpyS2D(dev_wukv_scale_expanded, wukv_scale_expanded, wukv_scale_expanded_size);
  tpuRtMemcpyS2D(dev_kv_cache, kv_cache, kv_cache_size);
  tpuRtMemcpyS2D(dev_pe_cache, pe_cache, pe_cache_size);
  tpuRtMemcpyS2D(dev_kv_new, kv_new, kv_new_size);
  tpuRtMemcpyS2D(dev_pe_new, pe_new, pe_new_size);
  tpuRtMemcpyS2D(dev_cos, cos, cos_size);
  tpuRtMemcpyS2D(dev_sin, sin, sin_size);
  tpuRtMemcpyS2D(dev_block_table, block_table, block_table_size);
  tpuRtMemcpyS2D(dev_save_slots, save_slots, save_slots_size);
  tpuRtMemcpyS2D(dev_kv_gather, kv_gather, kv_gather_size);
  tpuRtMemcpyS2D(dev_pe_gather, pe_gather, pe_gather_size);
  tpuRtMemcpyS2D(dev_pe_new_rope, pe_new_rope, pe_new_rope_size);
  tpuRtMemcpyS2D(dev_output, output, output_size);

  // 调用内核函数

  auto start = std::chrono::high_resolution_clock::now();  // 开始计时

  int rst = main_kernel((unsigned long long)dev_query, (unsigned long long)dev_wuq, (unsigned long long)dev_wuq_scale_expanded, (unsigned long long)dev_wukv, (unsigned long long)dev_wukv_scale_expanded, (unsigned long long)dev_kv_cache, (unsigned long long)dev_pe_cache, (unsigned long long)dev_kv_new, (unsigned long long)dev_pe_new, (unsigned long long)dev_cos, (unsigned long long)dev_sin, (unsigned long long)dev_block_table, (unsigned long long)dev_save_slots, (unsigned long long)dev_kv_gather, (unsigned long long)dev_pe_gather, (unsigned long long)dev_pe_new_rope, (unsigned long long)dev_output);

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
      rst = main_kernel((unsigned long long)dev_query, (unsigned long long)dev_wuq, (unsigned long long)dev_wuq_scale_expanded, (unsigned long long)dev_wukv, (unsigned long long)dev_wukv_scale_expanded, (unsigned long long)dev_kv_cache, (unsigned long long)dev_pe_cache, (unsigned long long)dev_kv_new, (unsigned long long)dev_pe_new, (unsigned long long)dev_cos, (unsigned long long)dev_sin, (unsigned long long)dev_block_table, (unsigned long long)dev_save_slots, (unsigned long long)dev_kv_gather, (unsigned long long)dev_pe_gather, (unsigned long long)dev_pe_new_rope, (unsigned long long)dev_output);
  }

  printf("Runs: %d\n", measure_runs);

  // 测量运行
  double total_time_us = 0.0;
  double min_time_us = std::numeric_limits<double>::max();
  double max_time_us = 0.0;

  for (int i = 0; i < measure_runs; i++) {
    auto run_start = std::chrono::high_resolution_clock::now();
      rst = main_kernel((unsigned long long)dev_query, (unsigned long long)dev_wuq, (unsigned long long)dev_wuq_scale_expanded, (unsigned long long)dev_wukv, (unsigned long long)dev_wukv_scale_expanded, (unsigned long long)dev_kv_cache, (unsigned long long)dev_pe_cache, (unsigned long long)dev_kv_new, (unsigned long long)dev_pe_new, (unsigned long long)dev_cos, (unsigned long long)dev_sin, (unsigned long long)dev_block_table, (unsigned long long)dev_save_slots, (unsigned long long)dev_kv_gather, (unsigned long long)dev_pe_gather, (unsigned long long)dev_pe_new_rope, (unsigned long long)dev_output);
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
  tpuRtMemcpyD2S(kv_cache, dev_kv_cache, kv_cache_size);
  tpuRtMemcpyD2S(pe_cache, dev_pe_cache, pe_cache_size);
  tpuRtMemcpyD2S(kv_gather, dev_kv_gather, kv_gather_size);
  tpuRtMemcpyD2S(pe_gather, dev_pe_gather, pe_gather_size);
  tpuRtMemcpyD2S(pe_new_rope, dev_pe_new_rope, pe_new_rope_size);
  tpuRtMemcpyD2S(output, dev_output, output_size);
  tpuRtStreamSynchronize(stream);

  // 释放设备内存
  tpuRtFree(&dev_query, 0);
  tpuRtFree(&dev_wuq, 0);
  tpuRtFree(&dev_wuq_scale_expanded, 0);
  tpuRtFree(&dev_wukv, 0);
  tpuRtFree(&dev_wukv_scale_expanded, 0);
  tpuRtFree(&dev_kv_cache, 0);
  tpuRtFree(&dev_pe_cache, 0);
  tpuRtFree(&dev_kv_new, 0);
  tpuRtFree(&dev_pe_new, 0);
  tpuRtFree(&dev_cos, 0);
  tpuRtFree(&dev_sin, 0);
  tpuRtFree(&dev_block_table, 0);
  tpuRtFree(&dev_save_slots, 0);
  tpuRtFree(&dev_kv_gather, 0);
  tpuRtFree(&dev_pe_gather, 0);
  tpuRtFree(&dev_pe_new_rope, 0);
  tpuRtFree(&dev_output, 0);

  post();
  return 0;
}
