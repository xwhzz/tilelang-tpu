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

void main(global_addr_t v1, global_addr_t v2, global_addr_t v3, global_addr_t v4) {
  __ppl_tensor_info v8 = {.shape = {1 ,128, 1, 64}, .stride = {0}, .addr = v4, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 16384, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v7 = {.shape = {1 ,128, 1, 64}, .stride = {0}, .addr = v3, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 16384, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v6 = {.shape = {1 ,128, 1, 64}, .stride = {0}, .addr = v2, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 16384, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v5 = {.shape = {1 ,128, 1, 64}, .stride = {0}, .addr = v1, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 16384, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info in_x = {.shape = { 1, 64, 1, 16}, .stride = {0}, .addr = 114688, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 32, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info in_cos = {.shape = { 1, 64, 1, 16}, .stride = {0}, .addr = 98304, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 32, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info in_sin = {.shape = { 1, 64, 1, 16}, .stride = {0}, .addr = 81920, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 32, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info x_cos = {.shape = { 1, 64, 1, 16}, .stride = {0}, .addr = 65536, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 32, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info x_sin = {.shape = { 1, 64, 1, 16}, .stride = {0}, .addr = 49152, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 32, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info x_neg = {.shape = { 1, 64, 1, 16}, .stride = {0}, .addr = 32768, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 32, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info x_neg_sin = {.shape = { 1, 64, 1, 16}, .stride = {0}, .addr = 16384, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 32, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info out = {.shape = { 1, 64, 1, 16}, .stride = {0}, .addr = 0, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 32, .unsigned_flag = 0, .default_stride = true};
  for (int bx = 0; bx < 2; ++bx) {
    for (int by = 0; by < 4; ++by) {
      __ppl_tensor_info G_in = {.shape = {1, 64, 1, 16}, .stride = {8192, 64, 64, 1} , .addr = v6.addr + (((bx * 64)) * 64+((by * 16)) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = (((bx * 64)) * 64+((by * 16)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info in_x_1 = {.shape = {1, 64, 1, 16}, .stride = {0}, .addr = in_x.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      tpu_gdma_cpy_S2L(in_x_1.addr, G_in.addr, &in_x_1.shape, (in_x_1.default_stride ? NULL : &in_x_1.stride), (G_in.default_stride ? NULL : &G_in.stride), DT_FP16);
      __ppl_tensor_info G_cos = {.shape = {1, 64, 1, 16}, .stride = {8192, 64, 64, 1} , .addr = v7.addr + (((bx * 64)) * 64+((by * 16)) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = (((bx * 64)) * 64+((by * 16)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info in_cos_1 = {.shape = {1, 64, 1, 16}, .stride = {0}, .addr = in_cos.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      tpu_gdma_cpy_S2L(in_cos_1.addr, G_cos.addr, &in_cos_1.shape, (in_cos_1.default_stride ? NULL : &in_cos_1.stride), (G_cos.default_stride ? NULL : &G_cos.stride), DT_FP16);
      __ppl_tensor_info G_sin = {.shape = {1, 64, 1, 16}, .stride = {8192, 64, 64, 1} , .addr = v8.addr + (((bx * 64)) * 64+((by * 16)) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = (((bx * 64)) * 64+((by * 16)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info in_sin_1 = {.shape = {1, 64, 1, 16}, .stride = {0}, .addr = in_sin.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      tpu_gdma_cpy_S2L(in_sin_1.addr, G_sin.addr, &in_sin_1.shape, (in_sin_1.default_stride ? NULL : &in_sin_1.stride), (G_sin.default_stride ? NULL : &G_sin.stride), DT_FP16);
      tpu_bdc_fp_mul( x_cos.addr, in_x.addr, in_cos.addr, &x_cos.shape, (x_cos.default_stride ? NULL : &x_cos.stride), (in_x.default_stride ? NULL : &in_x.stride), (in_cos.default_stride ? NULL : &in_cos.stride), DT_FP16);
      tpu_bdc_fp_mul( x_sin.addr, in_x.addr, in_sin.addr, &x_sin.shape, (x_sin.default_stride ? NULL : &x_sin.stride), (in_x.default_stride ? NULL : &in_x.stride), (in_sin.default_stride ? NULL : &in_sin.stride), DT_FP16);
      scalar_t x_neg_scalar_DT_FP16 = {.f32 = (float)-1};
      x_neg_scalar_DT_FP16 = tpu_cast(x_neg_scalar_DT_FP16, DT_FP16, DT_FP32, RM_HALF_TO_EVEN);
      tpu_bdc_fp_mul_C( x_neg.addr, in_x.addr, x_neg_scalar_DT_FP16, &x_neg.shape, (x_neg.default_stride ? NULL : &x_neg.stride), (in_x.default_stride ? NULL : &in_x.stride), DT_FP16);
      tpu_bdc_fp_mul( x_neg_sin.addr, x_neg.addr, in_sin.addr, &x_neg_sin.shape, (x_neg_sin.default_stride ? NULL : &x_neg_sin.stride), (x_neg.default_stride ? NULL : &x_neg.stride), (in_sin.default_stride ? NULL : &in_sin.stride), DT_FP16);
      dim4 half_stride;
      tpu_aligned_stride(&half_stride, 0, &out.shape, DT_FP16);
      half_stride.w *= 2;
      dim4 half_shape = {.n = out.shape.n, .c = out.shape.c, .h = out.shape.h, .w = out.shape.w};
      half_shape.w /= 2;
      tpu_bdc_fp_add( out.addr, x_cos.addr, x_neg_sin.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_FP16);
      tpu_bdc_fp_add( out.addr + 2, x_cos.addr + 2, x_sin.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_FP16);
      __ppl_tensor_info out_1 = {.shape = {1, 64, 1, 16}, .stride = {0}, .addr = out.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      __ppl_tensor_info G_out = {.shape = {1, 64, 1, 16}, .stride = {8192, 64, 64, 1} , .addr = v5.addr + (((bx * 64)) * 64+((by * 16)) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = (((bx * 64)) * 64+((by * 16)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_gdma_cpy_L2S(G_out.addr, out_1.addr, &G_out.shape, (G_out.default_stride ? NULL : &G_out.stride), (out_1.default_stride ? NULL : &out_1.stride), DT_FP16);
    }
  }
}

typedef struct {
  global_addr_t v1;
  global_addr_t v2;
  global_addr_t v3;
  global_addr_t v4;
} tpu_kernel_api_main_args_t;
void rope_kernel(const void * args) {
  tpu_kernel_api_main_args_t *api = (tpu_kernel_api_main_args_t*)args;
  main(api->v1,
    api->v2,
    api->v3,
    api->v4);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(rope_kernel)

