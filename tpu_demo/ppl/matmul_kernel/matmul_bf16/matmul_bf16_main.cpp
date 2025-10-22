#if defined(__bm1684x__) || defined(__bm1688__) || defined(__mars3__)
#include "bmlib_runtime.h"
#include "kernel_module_data.h"
#include "tpu_defs.h"
#endif

#ifdef __bm1690__
// 基于 tpuv7-runtime 的runtime的头文件，适用于 bm1690
#include <tpuv7_rt.h>
#endif

// 测试工具，用于随机生成数据和dump数据
#include "host_test_utils.h"
// host端 函数的头文件
#include "matmul.h"
#include <cstring>
#include <string>
#include <vector>

#if defined(__bm1684x__) || defined(__bm1688__)
// 定义设备的句柄
bm_handle_t handle;
// 定义kernel module的句柄
tpu_kernel_module_t tpu_module;
#endif

#ifdef __bm1690__
// 定义 runtime stream
tpuRtStream_t stream;
// 定义 kernel module 的句柄
tpuRtKernelModule_t tpu_module;
#endif

int main() {
  int M = 384;
  int K = 768;
  int N = 768;
  dim4 left_shape = {1, M, 1, K};
  dim4 right_shape = {1, K, 1, N};
  dim4 res_shape = {1, M, 1, N};
  size_t left_size = M * K * DtypeSize(DT_BFP16);
  size_t right_size = K * N * DtypeSize(DT_BFP16);
  size_t res_size = M * N * DtypeSize(DT_BFP16);
#ifdef __bm1690__
  // tpuv7_runimte的状态变量
  tpuRtStatus_t ret;
  // 计算设备runtime初始化
  ret = tpuRtInit();
  if (ret != tpuRtSuccess) {
    printf("tpuRtInit failed\n");
    return -1;
  }
  printf("tpuRtInit success\n");
  // 设置当前进程使用的计算设备
  tpuRtSetDevice(0);
  // 创建一个stream
  tpuRtStreamCreate(&stream);

  // 通过环境变量设置 ppl 代码生成的 deivce 端动态库存放路径
  auto kernel_dir = getenv("PPL_KERNEL_PATH");
  if (!kernel_dir) {
    printf("Please set env PPL_KERNEL_PATH to libkernel.so path\n");
    return -2;
  }

  // 载入 device 端动态库
  tpu_module = tpuRtKernelLoadModuleFile(kernel_dir, stream);
  if (NULL == tpu_module) {
    printf("tpuRtKernelLoadModuleFile failed\n");
    return -2;
  }
  char* left_data  = new char[left_size];
  char* right_data = new char[right_size];
  char* res_data   = new char[res_size];
  memset(res_data, 0x00, res_size);

  // 初始化输入数据
  printf("input_data:\n");
  rand_data("test", "matmul", 0, left_data, left_size, &left_shape, -1.0, 1.0, DT_BFP16);
  rand_data("test", "matmul", 1, right_data, right_size, &right_shape, -1.0, 1.0, DT_BFP16);

  void *dev_left_data;
  void *dev_right_data;
  void *dev_res_data;
  // 向计算设备申请memory，第三个参数为并行度
  tpuRtMalloc((void **)(&dev_left_data), left_size, 0);
  tpuRtMalloc((void **)(&dev_right_data), right_size, 0);
  tpuRtMalloc((void **)(&dev_res_data), res_size, 0);
  //  init input_data
  // 从host memory复制数据到计算设备memory
  tpuRtMemcpyS2D(dev_left_data, left_data, left_size);
  tpuRtMemcpyS2D(dev_right_data, right_data, right_size);

  // 调用自动生成的 host 端函数，此函数内部会调用 device 端的 kernel函数
  int rst = mm2_bfp16((unsigned long long)dev_left_data,
                      (unsigned long long)dev_right_data,
                      (unsigned long long)dev_res_data);
  if (rst) {
    printf("kernel_launch failed\n");
    return 1;
  }
  printf("kernel_launch success\n");

  // 从计算设备memory复制数据到host memory
  tpuRtMemcpyD2S(res_data, dev_res_data, res_size);

  // 保存运行结果
  dump_data("./test", "matmul", "_left", ".in", 0, left_data, left_size, DT_BFP16);
  dump_data("./test", "matmul", "_right", ".in", 1, right_data, right_size, DT_BFP16);
  dump_data("./test", "matmul", "_res", ".out", 2, res_data, res_size, DT_BFP16);

  delete[] left_data;
  delete[] right_data;
  delete[] res_data;
  // 释放计算设备的memory
  tpuRtFree(&dev_left_data, 0);
  tpuRtFree(&dev_right_data, 0);
  tpuRtFree(&dev_res_data, 0);

  tpuRtKernelUnloadModule(tpu_module, stream);
  tpuRtStreamDestroy(stream);
#endif
#if defined(__bm1684x__) || defined(__bm1688__)
  // 返回状态
  bm_status_t ret = BM_SUCCESS;
  // 请求一个设备，得到设备句柄handle
  ret = bm_dev_request(&handle, 0);
  if (ret != BM_SUCCESS)
    throw("bm_dev_request_failed");
  printf("bm_dev_request success\n");
  // kernel_module_data 定义在kernel_module_data.h中
  // 后续在cmake 架构章节中会看到，这是将 kernel
  // 库文件使用hexdump以文本形式打包到头文件中
  const unsigned int *p = kernel_module_data;
  size_t length = sizeof(kernel_module_data);
  // 动态加载 kernel module
  tpu_module = tpu_kernel_load_module(handle, (const char *)p, length);
  if (!tpu_module) {
    printf("tpu_kernel_load_module failed\n");
    return -1;
  }
  printf("tpu_module load success\n");
  // module 句柄
  bm_device_mem_t dev_left_data;
  bm_device_mem_t dev_right_data;
  bm_device_mem_t dev_res_data;
  char *left_data;
  char *right_data;
  char *res_data;
  // 申请指定大小的device mem和host mem，size为device mem的字节大小
  MallocWrap(handle, &dev_left_data, (u64 *)&left_data, left_size);
  MallocWrap(handle, &dev_right_data, (u64 *)&right_data, right_size);
  MallocWrap(handle, &dev_res_data, (u64 *)&res_data, res_size);

  rand_data("test", "matmul", 0, left_data, left_size, &left_shape, -1.0, 1.0, DT_BFP16);
  rand_data("test", "matmul", 1, right_data, right_size, &right_shape, -1.0, 1.0, DT_BFP16);
  /**
   * load_file(std::string file_path, char *data, size_t data_size,
               size_t file_size, int dtype, int file_dtype)
   */

  MemcpyS2D(handle, &dev_left_data, left_data, left_size);
  MemcpyS2D(handle, &dev_right_data, right_data, right_size);
  int WARMUP = 1;
  int REPAET = 2;
  for (int i = 0; i < WARMUP + REPAET; i++){
    bm_profile_t start, end;
    bm_get_profile(handle, &start);
    // 通过自动生成的 host 端封装函数调用 kernel 函数
    int rst = mm2_bfp16(bm_mem_get_device_addr(dev_left_data), bm_mem_get_device_addr(dev_right_data), bm_mem_get_device_addr(dev_res_data));
    bm_get_profile(handle, &end);
    if (rst) {
      printf("kernel_launch failed\n");
      return 1;
    }
    else if (i >= WARMUP) {
      size_t npu_time = end.tpu_process_time - start.tpu_process_time;
      std::cout << "npu time = " << npu_time << "(us) --> ";
      printf("kernel_launch success\n");
    }
  }
  MemcpyD2S(handle, &dev_res_data, res_data, res_size);

  /*
  void dump_data(std::string dir, std::string func_name, std::string pre,
               std::string end, int idx, char *data, size_t data_size,
               int dtype)
  */
  dump_data("./test", "matmul", "_left", ".in", 0, left_data, left_size, DT_BFP16);
  dump_data("./test", "matmul", "_right", ".in", 1, right_data, right_size, DT_BFP16);
  dump_data("./test", "matmul", "_res", ".out", 2, res_data, res_size, DT_BFP16);
  FreeWrap(handle, &dev_left_data, left_data);
  FreeWrap(handle, &dev_right_data, right_data);
  FreeWrap(handle, &dev_res_data, res_data);
  // 释放设备句柄
  tpu_kernel_free_module(handle, tpu_module);
  bm_dev_free(handle);
#endif
  return 0;
}
