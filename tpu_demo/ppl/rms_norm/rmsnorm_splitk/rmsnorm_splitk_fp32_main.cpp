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
#include "rms_norm.h"
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
  std::string func_name = "rms_norm";
  int32_t N = 1;
  int32_t C = 512;
  int32_t H = 1;
  int32_t W = 512;
  // dim4 input_shape = {N, C, H, W};
  dim4 input_shape = {N, C, H, W};
  dim4 output_shape = {N, C, H, W};
  // size_t data_size = N * C * H * W * DtypeSize(DT_FP32);
  size_t input_size = N * C * H * W * DtypeSize(DT_FP32);
  size_t output_size = N * C * H * W * DtypeSize(DT_FP32);
#ifdef __bm1690__
  // tpuv7_runimte的状态变量
  tpuRtStatus_t ret;
  // 计算设备runtime初始化
  ret = tpuRtInit();
  if (ret != tpuRtSuccess) {
    printf("tpuRtInit failed\n)");
    return -1;
  }
  printf("tpuRtInit success\n)");
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
  char *input_data = new char[input_size];
  char *output_data = new char[output_size];
  memset(output_data, 0x00, output_size);

  // 初始化输入数据
  rand_data("test", func_name, 0, input_data, input_size, &input_shape, -5, 5, DT_FP32);

  void *dev_input_data;
  void *dev_output_data;
  // 向计算设备申请memory，第三个参数为并行度
  tpuRtMalloc((void **)(&dev_input_data), input_size, 0);
  tpuRtMalloc((void **)(&dev_output_data), output_size, 0);
  //  init input_data
  // 从host memory复制数据到计算设备memory
  tpuRtMemcpyS2D(dev_input_data, input_data, input_size);

  // 调用自动生成的 host 端函数，此函数内部会调用 device 端的 kernel函数
  int rst = rms_norm((unsigned long long)dev_input_data,
                      (unsigned long long)dev_output_data);
  if (rst) {
    printf("kernel_launch failed\n");
    return 1;
  }
  printf("kernel_launch success\n");

  // 从计算设备memory复制数据到host memory
  tpuRtMemcpyD2S(output_data, dev_output_data, output_size);

  // 保存运行结果
  dump_data("./test", func_name, "_input", ".in", 0, input_data, input_size, DT_FP32);
  dump_data("./test", func_name, "_output", ".out", 1, output_data, output_size, DT_FP32);
  delete[] input_data;
  delete[] output_data;
  // 释放计算设备的memory
  tpuRtFree(&dev_input_data, 0);
  tpuRtFree(&dev_output_data, 0);

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
  bm_device_mem_t dev_input_data;
  bm_device_mem_t dev_output_data;
  char *input_data;
  char *output_data;
  // 申请指定大小的device mem和host mem，size为device mem的字节大小
  MallocWrap(handle, &dev_input_data, (u64 *)&input_data, input_size);
  MallocWrap(handle, &dev_output_data, (u64 *)&output_data, output_size);
  printf("Malloc wrap success\n");
  // 随机生成输入数据
  rand_data("test", func_name, 0, input_data, input_size, &input_shape, -5, 5, DT_FP32);
  // 改为从test中读取输入数据
  //load_file("./test/rms_norm_input0.in", input_data, input_size, input_size, DT_FP32, DT_FP32);
  printf("Input data loaded\n");
  // 将在系统内存上的数据拷贝到device mem
  // MemcpyS2D还有一个默认参数offset，默认为0，从src的offset偏移开始拷贝
  MemcpyS2D(handle, &dev_input_data, input_data, input_size);
  // 预热3次后重复执行10次
  int WARMUP = 3;
  int REPEAT = 10;
  size_t total_time= 0;
  printf("Tests begin\n");
  for (int i = 0; i < WARMUP + REPEAT; i++){
    bm_profile_t start, end;
    bm_get_profile(handle, &start);
    // 通过自动生成的 host 端封装函数调用 kernel 函数
    int rst = rms_norm(bm_mem_get_device_addr(dev_input_data),
                          bm_mem_get_device_addr(dev_output_data));
    bm_get_profile(handle, &end);
    if (rst) {
      printf("kernel_launch failed\n");
      return 1;
    }
    else if (i >= WARMUP) {
      size_t npu_time = end.tpu_process_time - start.tpu_process_time;
      total_time += npu_time;
      std::cout << "npu time = " << npu_time << "(us) --> ";
      printf("kernel_launch success\n");
    }
  }
  std::cout << "total time = " << total_time << "(us)\n";
  // 将在 device mem 上的数据拷贝到 系统内存
  MemcpyD2S(handle, &dev_output_data, output_data, output_size);
  // 打印运行结果
  printf("output data:\n");
  // for (int i = 0; i < N * C * H * W; ++i) {
  //   printf("%f, ", ((float *)output_data)[i]);
  // }
  printf("\n");
  // 保存输入输出结果
  dump_data("./test", func_name, "_input", ".in", 0, input_data, input_size, DT_FP32);
  dump_data("./test", func_name, "_output", ".out", 1, output_data, output_size, DT_FP32);
  // device mem和host mem
  FreeWrap(handle, &dev_input_data, input_data);
  FreeWrap(handle, &dev_output_data, output_data);
  // 释放设备句柄
  tpu_kernel_free_module(handle, tpu_module);
  bm_dev_free(handle);
#endif
  return 0;
}
