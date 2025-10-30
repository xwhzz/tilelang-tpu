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
#include "flashattn.h"
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
  std::string func_name = "flashattn";
  int32_t batch = 4;
  int32_t heads = 8;
  int32_t kv_head_num = 1;
  int32_t seqlen_kv = 512;
  int32_t dim = 32;
  int32_t pe_dim = 4;
  // dim4 input_shape = {N, C, H, W};
  dim4 q_shape = {batch, 1, heads, dim};
  dim4 qpe_shape = {batch, 1, heads, pe_dim};
  dim4 kv_shape = {batch, seqlen_kv, kv_head_num, dim};
  dim4 kpe_shape = {batch, seqlen_kv, kv_head_num, pe_dim};
  dim4 output_shape = {batch, 1, heads, dim};
  // size_t data_size = N * C * H * W * DtypeSize(DT_FP32);
  size_t q_size = batch * 1 * heads * dim * DtypeSize(DT_FP16);
  size_t qpe_size = batch * 1 * heads * pe_dim * DtypeSize(DT_FP16);
  size_t kv_size = batch * seqlen_kv * kv_head_num * dim * DtypeSize(DT_FP16);
  size_t kpe_size = batch * seqlen_kv * kv_head_num * pe_dim * DtypeSize(DT_FP16);
  size_t output_size = batch * 1 * heads * dim * DtypeSize(DT_FP16);
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
  char *q_data = new char[q_size];
  char *qpe_data = new char[qpe_size];
  char *kv_data = new char[kv_size];
  char *kpe_data = new char[kpe_size];
  char *output_data = new char[output_size];
  memset(output_data, 0x00, output_size);

  // 初始化输入数据
  printf("input_data:\n");
  rand_data("test", "flashattn", 0, q_data, q_size, &q_shape, -5.0, 5.0, DT_FP16);
  rand_data("test", "flashattn", 1, qpe_data, qpe_size, &qpe_shape, -5.0, 5.0, DT_FP16);
  rand_data("test", "flashattn", 2, kv_data, kv_size, &kv_shape, -5.0, 5.0, DT_FP16);
  rand_data("test", "flashattn", 3, kpe_data, kpe_size, &kpe_shape, -5.0, 5.0, DT_FP16);
  void *dev_q_data;
  void *dev_qpe_data;
  void *dev_kv_data;
  void *dev_kpe_data;
  void *dev_output_data;
  // 向计算设备申请memory，第三个参数为并行度
  tpuRtMalloc((void **)(&dev_q_data), q_size, 0);
  tpuRtMalloc((void **)(&dev_qpe_data), qpe_size, 0);
  tpuRtMalloc((void **)(&dev_kv_data), kv_size, 0);
  tpuRtMalloc((void **)(&dev_kpe_data), kpe_size, 0);
  tpuRtMalloc((void **)(&dev_output_data), output_size, 0);
  //  init input_data
  // 从host memory复制数据到计算设备memory
  tpuRtMemcpyS2D(dev_q_data, q_data, q_size);
  tpuRtMemcpyS2D(dev_qpe_data, qpe_data, qpe_size);
  tpuRtMemcpyS2D(dev_kv_data, kv_data, kv_size);
  tpuRtMemcpyS2D(dev_kpe_data, kpe_data, kpe_size);
  // 调用自动生成的 host 端函数，此函数内部会调用 device 端的 kernel函数
  int rst = flashattn((unsigned long long)dev_q_data,
                        (unsigned long long)dev_qpe_data,
                        (unsigned long long)dev_kv_data,
                        (unsigned long long)dev_kpe_data,
                        (unsigned long long)dev_output_data);
  if (rst) {
    printf("kernel_launch failed\n");
    return 1;
  }
  printf("kernel_launch success\n");

  // 从计算设备memory复制数据到host memory
  tpuRtMemcpyD2S(output_data, dev_output_data, output_size);

  // 打印运行结果
  printf("output data:\n");
  dump_data("./test", "flashattn", "_q", ".in", 0, q_data, q_size, DT_FP16);
  dump_data("./test", "flashattn", "_qpe", ".in", 1, qpe_data, qpe_size, DT_FP16);
  dump_data("./test", "flashattn", "_kv", ".in", 2, kv_data, kv_size, DT_FP16);
  dump_data("./test", "flashattn", "_kpe", ".in", 3, kpe_data, kpe_size, DT_FP16);
  dump_data("./test", "flashattn", "_output", ".out", 4, output_data, output_size, DT_FP16);
  delete[] q_data;
  delete[] qpe_data;
  delete[] kv_data;
  delete[] kpe_data;
  delete[] output_data;
  // 释放计算设备的memory
  tpuRtFree(&dev_q_data, 0);
  tpuRtFree(&dev_qpe_data, 0);
  tpuRtFree(&dev_kv_data, 0);
  tpuRtFree(&dev_kpe_data, 0);
  tpuRtFree(&dev_output_data, 0);

  tpuRtKernelUnloadModule(tpu_module, stream);
  tpuRtStreamDestroy(stream);
#endif
#if defined(__bm1684x__) || defined(__bm1688__)
 
#endif
  return 0;
}
