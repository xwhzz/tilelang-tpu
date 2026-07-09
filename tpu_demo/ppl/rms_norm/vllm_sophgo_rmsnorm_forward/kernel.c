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
  __ppl_tensor_info v6 = {.shape = {1 ,128, 1, 128}, .stride = {0}, .addr = v3, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 32768, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v5 = {.shape = {1 ,1, 1, 128}, .stride = {0}, .addr = v2, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 256, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v4 = {.shape = {1 ,128, 1, 128}, .stride = {0}, .addr = v1, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 32768, .unsigned_flag = 0, .default_stride = true};
  for (int bx = 0; bx < 4; ++bx) {
    __ppl_tensor_info X_shared = {.shape = { 1, 32, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = true};
    __ppl_tensor_info X_fp32 = {.shape = { 1, 32, 1, 128}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = true};
    __ppl_tensor_info X_pow2 = {.shape = { 1, 32, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = true};
    __ppl_tensor_info X_powsum = {.shape = { 1, 32, 1, 1}, .stride = {0}, .addr = 81920, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
    __ppl_tensor_info Weight_shared = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 65536, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = true};
    __ppl_tensor_info X = {.shape = {1, 32, 1, 128}, .stride = {16384, 128, 128, 1} , .addr = v4.addr + (((bx * 32)) * 128+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = (((bx * 32)) * 128+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info X_shared_1 = {.shape = {1, 32, 1, 128}, .stride = {0}, .addr = X_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
    tpu_gdma_cpy_S2L(X_shared_1.addr, X.addr, &X_shared_1.shape, (X_shared_1.default_stride ? NULL : &X_shared_1.stride), (X.default_stride ? NULL : &X.stride), DT_FP16);
    __ppl_tensor_info X_shared_2 = {.shape = {1, 32, 1, 128}, .stride = {0}, .addr = X_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
    __ppl_tensor_info X_fp32_1 = {.shape = {1, 32, 1, 128}, .stride = {0}, .addr = X_fp32.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
    tpu_bdc_cast(X_fp32_1.addr, X_shared_2.addr, &X_fp32_1.shape, (X_fp32_1.default_stride ? NULL : &X_fp32_1.stride), (X_shared_2.default_stride ? NULL : &X_shared_2.stride), DT_FP32, DT_FP16, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul( X_pow2.addr, X_fp32.addr, X_fp32.addr, &X_pow2.shape, (X_pow2.default_stride ? NULL : &X_pow2.stride), (X_fp32.default_stride ? NULL : &X_fp32.stride), (X_fp32.default_stride ? NULL : &X_fp32.stride), DT_FP32);
    __ppl_tensor_info tmp_buffer_sum = {.shape = { 1, 32, 1, 32}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = true};
    {
      int eu_num = 32;
      int align_w = 128;
      scalar_t pad_val = {.f32 = 0};
      dim4 in_reduce_h = {X_pow2.shape.n, X_pow2.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {X_pow2.shape.n, X_pow2.shape.c, 1, eu_num};
      dim4 in_reduce_w = {X_pow2.shape.n, X_pow2.shape.c, 1, eu_num};
      dim4 out_reduce_w = {X_pow2.shape.n, X_pow2.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      scalar_t scale = {.f32 = (float)1.000000000e+00};
      if (align_w > X_pow2.shape.w && align_w == eu_num) {
        dim4 padded_stride = {128, align_w, align_w, 1};
        dim4 padded_shape = {X_pow2.shape.n, X_pow2.shape.c, 1, align_w};
        dim4 copy_shape = {X_pow2.shape.n, X_pow2.shape.c, 1, X_pow2.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = X_pow2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = X_powsum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      } else {
        if (align_w > X_pow2.shape.w) {
          dim4 fill_shape = {X_pow2.shape.n, X_pow2.shape.c, 1, align_w - X_pow2.shape.w};
          int elem_size = 4;
          int offset = X_pow2.shape.w * elem_size;
          dim4 fill_tensor_stride = {128, align_w, X_pow2.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = X_pow2.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = X_pow2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = X_powsum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
    }
    tpu_bdc_fp_mul_C( X_powsum.addr, X_powsum.addr, (scalar_t){.f32 = 0.0078125}, &X_powsum.shape, (X_powsum.default_stride ? NULL : &X_powsum.stride), (X_powsum.default_stride ? NULL : &X_powsum.stride), DT_FP32);
    tpu_bdc_fp_add_C( X_powsum.addr, X_powsum.addr, (scalar_t){.f32 = 1e-06}, &X_powsum.shape, (X_powsum.default_stride ? NULL : &X_powsum.stride), (X_powsum.default_stride ? NULL : &X_powsum.stride), DT_FP32);
    tpu_bdc_fp32_rsqrt(X_powsum.addr, X_powsum.addr, &X_powsum.shape);
    dim4 X_powsum_stride;
    tpu_aligned_stride(&X_powsum_stride, 0, &X_powsum.shape, DT_FP32);
    X_powsum_stride.w = 0;
    tpu_bdc_fp_mul( X_fp32.addr, X_fp32.addr, X_powsum.addr, &X_fp32.shape, (X_fp32.default_stride ? NULL : &X_fp32.stride), (X_fp32.default_stride ? NULL : &X_fp32.stride), &X_powsum_stride, DT_FP32);
    __ppl_tensor_info X_fp32_2 = {.shape = {1, 32, 1, 128}, .stride = {0}, .addr = X_fp32.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
    __ppl_tensor_info X_shared_3 = {.shape = {1, 32, 1, 128}, .stride = {0}, .addr = X_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
    tpu_bdc_cast(X_shared_3.addr, X_fp32_2.addr, &X_shared_3.shape, (X_shared_3.default_stride ? NULL : &X_shared_3.stride), (X_fp32_2.default_stride ? NULL : &X_fp32_2.stride), DT_FP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info Weight = {.shape = {1, 1, 1, 128}, .stride = {128, 128, 128, 1} , .addr = v5.addr + ((0) * 128+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((0) * 128+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info Weight_shared_1 = {.shape = {1, 1, 1, 128}, .stride = {0}, .addr = Weight_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
    tpu_gdma_cpy_S2L(Weight_shared_1.addr, Weight.addr, &Weight_shared_1.shape, (Weight_shared_1.default_stride ? NULL : &Weight_shared_1.stride), (Weight.default_stride ? NULL : &Weight.stride), DT_FP16);
    dim4 Weight_shared_stride;
    tpu_aligned_stride(&Weight_shared_stride, 0, &Weight_shared.shape, DT_FP16);
    dim4 Weight_shared_broadcast_shape = Weight_shared.shape;
    Weight_shared_broadcast_shape.c = NPU_NUM;
    tpu_bdc_npu_bcast(Weight_shared.addr, Weight_shared.addr, &Weight_shared_broadcast_shape, DT_FP16);
    if (X_shared.shape.c > NPU_NUM) {
      Weight_shared_stride.c = 0;
    }
    tpu_bdc_fp_mul( X_shared.addr, X_shared.addr, Weight_shared.addr, &X_shared.shape, (X_shared.default_stride ? NULL : &X_shared.stride), (X_shared.default_stride ? NULL : &X_shared.stride), &Weight_shared_stride, DT_FP16);
    __ppl_tensor_info X_shared_4 = {.shape = {1, 32, 1, 128}, .stride = {0}, .addr = X_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
    __ppl_tensor_info Output = {.shape = {1, 32, 1, 128}, .stride = {16384, 128, 128, 1} , .addr = v6.addr + (((bx * 32)) * 128+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = (((bx * 32)) * 128+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(Output.addr, X_shared_4.addr, &Output.shape, (Output.default_stride ? NULL : &Output.stride), (X_shared_4.default_stride ? NULL : &X_shared_4.stride), DT_FP16);
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
