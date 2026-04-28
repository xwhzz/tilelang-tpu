collect ops num:1
min_expr: ((0) * 128+(0) * 1 ) * 2
min_expr: ((0) * 128+(0) * 1 ) * 2
<class 'tvm.tir.expr.BufferLoad'>
[0, 0]
T.region(X[0, 0], 1, 1, 128)
T.region(local[0, 0], 2, 1, 128)
<class 'tvm.tir.buffer.Buffer'>
[0, 0]
T.region(local[0, 0], 1, 1, 128)
T.region(Y[0, 0], 2, 1, 128)
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main_kernel_inner(X_handle: T.handle, Y_handle: T.handle):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "stackvm", "tag": ""}, "keys": ["cpu"], "kind": "tpu", "tag": ""})})
        M = T.int32()
        X = T.match_buffer(X_handle, (M, 128), "float16")
        Y = T.match_buffer(Y_handle, (M, 128), "float16")
        # with T.block("root"):
        for bx, by in T.grid(1, 1):
            with T.block(""):
                T.reads(X[0, 0])
                T.writes(Y[0, 0])
                T.block_attr({"tilelang.is_cpu_kernel_frame": T.bool(True)})
                local = T.alloc_buffer((1, 128), "float16", scope="shared.dyn")
                T.call_extern("handle", "ppl.copy", T.region(X[0, 0], 1, 1, 128), T.region(local[0, 0], 2, 1, 128))
                T.call_extern("handle", "ppl.copy", T.region(local[0, 0], 1, 1, 128), T.region(Y[0, 0], 2, 1, 128))
=== C source ===
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

void main_kernel_inner(global_addr_t v1, global_addr_t v2, int M) {
  __ppl_tensor_info v4 = {.shape = {1 ,M, 1, 128}, .stride = {0}, .addr = v2, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v3 = {.shape = {1 ,M, 1, 128}, .stride = {0}, .addr = v1, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info local = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info X = {.shape = {1, 1, 1, 128}, .stride = {M * 128, 128, 128, 1} , .addr = v3.addr + ((0) * 128+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((0) * 128+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
  __ppl_tensor_info local_1 = {.shape = {1, 1, 1, 128}, .stride = {0}, .addr = local.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
  tpu_gdma_cpy_S2L(local_1.addr, X.addr, &local_1.shape, (local_1.default_stride ? NULL : &local_1.stride), (X.default_stride ? NULL : &X.stride), DT_FP16);
  __ppl_tensor_info local_2 = {.shape = {1, 1, 1, 128}, .stride = {0}, .addr = local.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info Y = {.shape = {1, 1, 1, 128}, .stride = {M * 128, 128, 128, 1} , .addr = v4.addr + ((0) * 128+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((0) * 128+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
  tpu_gdma_cpy_L2S(Y.addr, local_2.addr, &Y.shape, (Y.default_stride ? NULL : &Y.stride), (local_2.default_stride ? NULL : &local_2.stride), DT_FP16);
}

typedef struct {
  global_addr_t v1;
  global_addr_t v2;
  int M;
} tpu_kernel_api_main_inner_args_t;
void main_kernel(const void * args) {
  tpu_kernel_api_main_inner_args_t *api = (tpu_kernel_api_main_inner_args_t*)args;
  main_kernel_inner(api->v1,
    api->v2,
    api->M);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(main_kernel)

