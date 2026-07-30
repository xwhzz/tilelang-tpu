#include <tpuv7_rt.h>
#include "host_test_utils.h"
#include "kernel.h"
#include <cstdlib>
#include <mutex>

tpuRtStream_t stream;
tpuRtKernelModule_t tpu_module;

static bool g_runtime_initialized = false;
static bool g_context_initialized = false;
static int g_device_id = -1;
static std::mutex g_init_mutex;

extern "C" int tilelang_tpu_set_device(int device_id) {
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (g_context_initialized && device_id != g_device_id) {
    return -3;
  }
  g_device_id = device_id;
  return 0;
}

extern "C" int tilelang_tpu_synchronize() {
  if (!g_context_initialized) {
    return 0;
  }
  return tpuRtStreamSynchronize(stream);
}

int init() {
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (g_context_initialized) {
    return 0;
  }

  tpuRtStatus_t ret;
  if (!g_runtime_initialized) {
    ret = tpuRtInit();
    if (ret != tpuRtSuccess) {
      return -1;
    }
    g_runtime_initialized = true;
  }
  if (g_device_id < 0) {
    const char *device_env = getenv("TPU_DEVICE_ID");
    g_device_id = device_env ? std::atoi(device_env) : 0;
  }
  tpuRtSetDevice(g_device_id);
  ret = tpuRtStreamCreate(&stream);
  if (ret != tpuRtSuccess) {
    return -1;
  }
  auto kernel_dir = getenv("PPL_KERNEL_PATH");
  if (!kernel_dir) {
    tpuRtStreamDestroy(stream);
    return -2;
  }
  tpu_module = tpuRtKernelLoadModuleFile(kernel_dir, stream);
  if (NULL == tpu_module) {
    printf("tpuRtKernelLoadModuleFile failed\n");
    tpuRtStreamDestroy(stream);
    return -2;
  }
  g_context_initialized = true;
  return 0;
}

void post() {
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (!g_context_initialized) {
    return;
  }
  tpuRtKernelUnloadModule(tpu_module, stream);
  tpuRtStreamDestroy(stream);
  tpu_module = nullptr;
  stream = nullptr;
  g_context_initialized = false;
}

__attribute__((destructor)) static void tilelang_tpu_shutdown() {
  post();
}

extern "C" int tilelang_tpu_run_device(void** args) {
  int32_t Axis = *static_cast<int32_t*>(args[4]);
  float Epsilon = *static_cast<float*>(args[5]);

  int res = init();
  if (res != 0) {
    return res;
  }

  int rst = main_kernel((unsigned long long)args[0], (unsigned long long)args[1], (unsigned long long)args[2], (unsigned long long)args[3], Axis, Epsilon);
  return rst;
}

extern "C" int tilelang_tpu_run(void** args) {
  char* HiddenStates = static_cast<char*>(args[0]);
  size_t HiddenStates_size = 32 * 1 * 1 * 4096 * 2;
  char* Weight = static_cast<char*>(args[1]);
  size_t Weight_size = 1 * 1 * 1 * 4096 * 2;
  char* Bias = static_cast<char*>(args[2]);
  size_t Bias_size = 1 * 1 * 1 * 4096 * 2;
  char* Output = static_cast<char*>(args[3]);
  size_t Output_size = 32 * 1 * 1 * 4096 * 2;
  int32_t Axis = *static_cast<int32_t*>(args[4]);
  float Epsilon = *static_cast<float*>(args[5]);

  int res = init();
  if (res != 0) {
    return res;
  }

  void *dev_HiddenStates = nullptr;
  void *dev_Weight = nullptr;
  void *dev_Bias = nullptr;
  void *dev_Output = nullptr;

  if (HiddenStates != nullptr) tpuRtMalloc((void **)(&dev_HiddenStates), HiddenStates_size, 0);
  if (Weight != nullptr) tpuRtMalloc((void **)(&dev_Weight), Weight_size, 0);
  if (Bias != nullptr) tpuRtMalloc((void **)(&dev_Bias), Bias_size, 0);
  if (Output != nullptr) tpuRtMalloc((void **)(&dev_Output), Output_size, 0);

  if (HiddenStates != nullptr) tpuRtMemcpyS2D(dev_HiddenStates, HiddenStates, HiddenStates_size);
  if (Weight != nullptr) tpuRtMemcpyS2D(dev_Weight, Weight, Weight_size);
  if (Bias != nullptr) tpuRtMemcpyS2D(dev_Bias, Bias, Bias_size);
  if (Output != nullptr) tpuRtMemcpyS2D(dev_Output, Output, Output_size);

  int rst = main_kernel((unsigned long long)dev_HiddenStates, (unsigned long long)dev_Weight, (unsigned long long)dev_Bias, (unsigned long long)dev_Output, Axis, Epsilon);

  if (rst == 0) {
  if (Output != nullptr) tpuRtMemcpyD2S(Output, dev_Output, Output_size);
    tpuRtStreamSynchronize(stream);
  }

  if (dev_HiddenStates != nullptr) tpuRtFree(&dev_HiddenStates, 0);
  if (dev_Weight != nullptr) tpuRtFree(&dev_Weight, 0);
  if (dev_Bias != nullptr) tpuRtFree(&dev_Bias, 0);
  if (dev_Output != nullptr) tpuRtFree(&dev_Output, 0);
  post();
  return rst;
}
