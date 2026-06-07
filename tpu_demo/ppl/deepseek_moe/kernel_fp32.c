collect ops num:0
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main_kernel_inner(Param: T.Buffer((128, 64), "float32"), Index: T.Buffer((8, 1), "uint32"), Output: T.Buffer((8, 64), "float32")):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "stackvm", "tag": ""}, "keys": ["cpu"], "kind": "tpu", "tag": ""})})
        # with T.block("root"):
        for bx, by in T.grid(1, 1):
            with T.block(""):
                T.reads(Param[0:128, 0:64], Index[0:8, 0])
                T.writes(Output[0:8, 0:64])
                T.block_attr({"tilelang.is_cpu_kernel_frame": T.bool(True)})
                T.call_extern("handle", "ppl.gather", T.tvm_access_ptr(T.type_annotation("float32"), Output.data, 0, 512, 2), T.tvm_access_ptr(T.type_annotation("float32"), Param.data, 0, 8192, 1), T.tvm_access_ptr(T.type_annotation("uint32"), Index.data, 0, 8, 1), 128)




#include "ppl_helper.h"
static data_type_t __ppl_get_dtype(int type) {
  data_type_t __dtype[] = {DT_FP32,    DT_FP32,    DT_FP16,  DT_BFP16,
    DT_FP8E5M2, DT_FP8E4M3, DT_FP20,  DT_TF32,
    DT_INT32,   DT_UINT32,  DT_INT16, DT_UINT16,
    DT_INT8,    DT_UINT8,   DT_INT4,  DT_UINT4};
  return __dtype[type];
}

typedef struct {
    dim4 shape;
    dim4 stride;
    global_addr_t addr;
    data_type_t dtype;
    int mode;
    int align_mode;
    int size;
    int offset;
    bool unsigned_flag;
    bool default_stride;
} __ppl_tensor_info;

void main_kernel_inner(global_addr_t v1, global_addr_t v2, global_addr_t v3) {
  __ppl_tensor_info v6 = {.shape = {1 ,8, 1, 64}, .stride = {0}, .addr = v3, .dtype = DT_FP32, .mode = 2, .align_mode = 0, .size = 2048, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v5 = {.shape = {1 ,8, 1, 1}, .stride = {0}, .addr = v2, .dtype = DT_UINT32, .mode = 2, .align_mode = 0, .size = 32, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v4 = {.shape = {1 ,128, 1, 64}, .stride = {0}, .addr = v1, .dtype = DT_FP32, .mode = 2, .align_mode = 0, .size = 32768, .unsigned_flag = 0, .default_stride = true};
  {
  dim4 __gather_shape = {1, 1, v6.shape.c, v6.shape.w};
  tpu_gdma_h_gather_S2S(v6.addr, v4.addr, v5.addr, false, (scalar_t){.u32 = 0}, &__gather_shape, 128, NULL, NULL, NULL, DT_FP32);
  }
}

typedef struct {
  global_addr_t v1;
  global_addr_t v2;
  global_addr_t v3;
} tpu_kernel_api_main_inner_args_t;
void main_kernel(const void * args) {
  tpu_kernel_api_main_inner_args_t *api = (tpu_kernel_api_main_inner_args_t*)args;
  main_kernel_inner(api->v1,
    api->v2,
    api->v3);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(main_kernel)
