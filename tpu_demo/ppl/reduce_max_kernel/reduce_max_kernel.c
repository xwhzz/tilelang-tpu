#include "ppl_helper.h"
static data_type_t __ppl_get_dtype(int type) {
  data_type_t __dtype[] = {DT_FP32,    DT_FP32,    DT_FP16,  DT_BFP16,
    DT_FP8E5M2, DT_FP8E4M3, DT_FP20,  DT_TF32,
    DT_INT32,   DT_UINT32,  DT_INT16, DT_UINT16,
    DT_INT8,    DT_UINT8,   DT_INT4,  DT_UINT4};
  return __dtype[type];
}

void main(global_addr_t ptr_input_v1, global_addr_t ptr_output_v2) {
  __ppl_tensor_info v4 = {.shape = {1 ,8192, 1, 1}, .stride = NULL, .addr = ptr_output_v2, .dtype = DT_FP32, .mode = 2, .align_mode = 0, .size = 32768, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v3 = {.shape = {1 ,8192, 1, 1020}, .stride = NULL, .addr = ptr_input_v1, .dtype = DT_FP32, .mode = 2, .align_mode = 0, .size = 33423360, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info X_shared = {.shape = { 1, 512, 1, 1020}, .stride = NULL, .addr = 0, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 32640, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info Y_shared = {.shape = { 1, 512, 1, 1}, .stride = NULL, .addr = 147456, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 32, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info tmp_buffer = {.shape = { 1, 512, 1, 32}, .stride = NULL, .addr = 131072, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = true};
  for (int by = 0; by < 16; ++by) {
    __ppl_tensor_info X = {.shape = {1, 512, 1, 1020}, .stride = {8355840, 1020, 1020, 1} , .addr = v3.addr + (((by * 512)) * 1020+(0) * 1 ) * 4, .dtype = DT_FP32, .mode = 2, .size = 1, .offset = (((by * 512)) * 1020+(0) * 1 ) * 4, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info X_shared_1 = {.shape = {1, 512, 1, 1020}, .stride = NULL, .addr = X_shared.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
    tpu_gdma_cpy_S2L(X_shared_1.addr, X.addr, &X_shared_1.shape, (X_shared_1.default_stride ? NULL : &X_shared_1.stride), (X.default_stride ? NULL : &X.stride), DT_FP32);
    scalar_t Y_shared_scalar_f32 = {.f32 = -2147483648};
    tpu_bdc_set_C(Y_shared.addr, Y_shared_scalar_f32, &Y_shared.shape, (Y_shared.default_stride ? NULL : &Y_shared.stride), DT_FP32);
    {
      int eu_num = 32;
      int align_w = 1024;
      scalar_t pad_val = {.f16 = FP_NEG_MAX(DT_FP32)};
      if (align_w > X_shared.shape.w) {
        dim4 fill_shape = {X_shared.shape.n, X_shared.shape.c, 1, align_w - X_shared.shape.w};
        int elem_size = 4;
        int offset = X_shared.shape.w * elem_size;
        dim4 fill_tensor_stride = {8192, align_w, X_shared.shape.w, 1};
        __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = X_shared.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
        tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
      }
      dim4 in_reduce_h = {X_shared.shape.n, X_shared.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {X_shared.shape.n, X_shared.shape.c, 1, eu_num};
      dim4 in_reduce_w = {X_shared.shape.n, X_shared.shape.c, 1, eu_num};
      dim4 out_reduce_w = {X_shared.shape.n, X_shared.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = X_shared.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
      dim2 kernel2 = {1, eu_num};
      __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = Y_shared.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      pad_val.u32 = FP_NEG_MAX(DT_FP32);
      tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
    }  //section redue_max
    __ppl_tensor_info Y_shared_1 = {.shape = {1, 512, 1, 1}, .stride = NULL, .addr = Y_shared.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
    __ppl_tensor_info Y = {.shape = {1, 512, 1, 1}, .stride = {8192, 1, 1, 1} , .addr = v4.addr + (((by * 512)) * 1+(0) * 1 ) * 4, .dtype = DT_FP32, .mode = 2, .size = 1, .offset = (((by * 512)) * 1+(0) * 1 ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(Y.addr, Y_shared_1.addr, &Y.shape, (Y.default_stride ? NULL : &Y.stride), (Y_shared_1.default_stride ? NULL : &Y_shared_1.stride), DT_FP32);
  }
}

typedef struct {
  global_addr_t ptr_input_v1;
  global_addr_t ptr_output_v2;
} tpu_kernel_api_main_args_t;
void main_kernel(const void * args) {
  tpu_kernel_api_main_args_t *api = (tpu_kernel_api_main_args_t*)args;
  main(api->ptr_input_v1,
    api->ptr_output_v2);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(main_kernel)
