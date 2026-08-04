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

extern "C" int tilelang_tpu_set_device(int device_id) {{
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (g_context_initialized && device_id != g_device_id) {{
    return -3;
  }}
  g_device_id = device_id;
  return 0;
}}

extern "C" int tilelang_tpu_synchronize() {{
  if (!g_context_initialized) {{
    return 0;
  }}
  return tpuRtStreamSynchronize(stream);
}}

int init() {{
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (g_context_initialized) {{
    return 0;
  }}

  tpuRtStatus_t ret;
  if (!g_runtime_initialized) {{
    ret = tpuRtInit();
    if (ret != tpuRtSuccess) {{
      return -1;
    }}
    g_runtime_initialized = true;
  }}
  if (g_device_id < 0) {{
    const char *device_env = getenv("TPU_DEVICE_ID");
    g_device_id = device_env ? std::atoi(device_env) : 0;
  }}
  tpuRtSetDevice(g_device_id);
  ret = tpuRtStreamCreate(&stream);
  if (ret != tpuRtSuccess) {{
    return -1;
  }}
  auto kernel_dir = getenv("PPL_KERNEL_PATH");
  if (!kernel_dir) {{
    tpuRtStreamDestroy(stream);
    return -2;
  }}
  tpu_module = tpuRtKernelLoadModuleFile(kernel_dir, stream);
  if (NULL == tpu_module) {{
    printf("tpuRtKernelLoadModuleFile failed\n");
    tpuRtStreamDestroy(stream);
    return -2;
  }}
  g_context_initialized = true;
  return 0;
}}

void post() {{
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (!g_context_initialized) {{
    return;
  }}
  tpuRtKernelUnloadModule(tpu_module, stream);
  tpuRtStreamDestroy(stream);
  tpu_module = nullptr;
  stream = nullptr;
  g_context_initialized = false;
}}

__attribute__((destructor)) static void tilelang_tpu_shutdown() {{
  post();
}}

extern "C" int tilelang_tpu_run_device(void** args) {{
{device_arg_declarations}

  int res = init();
  if (res != 0) {{
    return res;
  }}

{device_kernel_call}
  return rst;
}}

extern "C" int tilelang_tpu_run(void** args) {{
{arg_declarations}

  int res = init();
  if (res != 0) {{
    return res;
  }}

{device_declarations}

{malloc_statements}

{memcpy_s2d_statements}

{kernel_call}

  if (rst == 0) {{
{memcpy_d2s_statements}
    tpuRtStreamSynchronize(stream);
  }}

{free_statements}
  post();
  return rst;
}}
