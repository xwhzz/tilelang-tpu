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

void main_kernel_inner(global_addr_t v1, global_addr_t v2, global_addr_t v3, global_addr_t v4, int v5, float v6) {
  __ppl_tensor_info v9 = {.shape = {1, 1, 1, 4096}, .stride = {0}, .addr = v3, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 8192, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v8 = {.shape = {1, 1, 1, 4096}, .stride = {0}, .addr = v2, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 8192, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v10 = {.shape = {32, 1, 1, 4096}, .stride = {0}, .addr = v4, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 262144, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v7 = {.shape = {32, 1, 1, 4096}, .stride = {0}, .addr = v1, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 262144, .unsigned_flag = 0, .default_stride = true};
  for (int bx = 0 + tpu_core_index(); bx < 32; bx += tpu_core_num()) {
    __ppl_tensor_info hidden_shared = {.shape = {1, 1, 1, 4096}, .stride = {0}, .addr = 49152, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&hidden_shared.stride, 0, &hidden_shared.shape, DT_FP16);
    __ppl_tensor_info hidden_fp32 = {.shape = {1, 1, 1, 4096}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&hidden_fp32.stride, 0, &hidden_fp32.shape, DT_FP32);
    __ppl_tensor_info hidden_squared = {.shape = {1, 1, 1, 4096}, .stride = {0}, .addr = 0, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&hidden_squared.stride, 0, &hidden_squared.shape, DT_FP32);
    __ppl_tensor_info square_sum = {.shape = {1, 1, 1, 1}, .stride = {0}, .addr = 81920, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&square_sum.stride, 0, &square_sum.shape, DT_FP32);
    __ppl_tensor_info HiddenStates = {.shape = {1, 1, 1, 4096} , .stride = {4096, 4096, 4096, 1} , .addr = v7.addr + ((bx) * 4096+(0) * 4096+(0) * 4096+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((bx) * 4096+(0) * 4096+(0) * 4096+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info hidden_shared_1 = {.shape = {1, 1, 1, 4096} , .stride = hidden_shared.stride, .addr = hidden_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = hidden_shared.default_stride};
    tpu_gdma_cpy_S2L(hidden_shared_1.addr, HiddenStates.addr, &hidden_shared_1.shape, (hidden_shared_1.default_stride ? NULL : &hidden_shared_1.stride), (HiddenStates.default_stride ? NULL : &HiddenStates.stride), DT_FP16);
    __ppl_tensor_info hidden_shared_2 = {.shape = {1, 1, 1, 4096} , .stride = hidden_shared.stride, .addr = hidden_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = hidden_shared.default_stride};
    __ppl_tensor_info hidden_fp32_1 = {.shape = {1, 1, 1, 4096} , .stride = hidden_fp32.stride, .addr = hidden_fp32.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = hidden_fp32.default_stride};
    tpu_bdc_cast(hidden_fp32_1.addr, hidden_shared_2.addr, &hidden_fp32_1.shape, (hidden_fp32_1.default_stride ? NULL : &hidden_fp32_1.stride), (hidden_shared_2.default_stride ? NULL : &hidden_shared_2.stride), DT_FP32, DT_FP16, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul( hidden_squared.addr, hidden_fp32.addr, hidden_fp32.addr, &hidden_squared.shape, (hidden_squared.default_stride ? NULL : &hidden_squared.stride), (hidden_fp32.default_stride ? NULL : &hidden_fp32.stride), (hidden_fp32.default_stride ? NULL : &hidden_fp32.stride), DT_FP32);
    __ppl_tensor_info tmp_buffer_sum = {.shape = {1, 1, 1, 32}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_sum.stride, 0, &tmp_buffer_sum.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 4096;
      scalar_t pad_val = {.f32 = 0};
      dim4 in_reduce_h = {hidden_squared.shape.n, hidden_squared.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {hidden_squared.shape.n, hidden_squared.shape.c, 1, eu_num};
      dim4 in_reduce_w = {hidden_squared.shape.n, hidden_squared.shape.c, 1, eu_num};
      dim4 out_reduce_w = {hidden_squared.shape.n, hidden_squared.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      scalar_t scale = {.f32 = (float)1.000000000e+00};
      if (align_w > hidden_squared.shape.w && align_w == eu_num) {
        dim4 padded_stride = {4096, align_w, align_w, 1};
        dim4 padded_shape = {hidden_squared.shape.n, hidden_squared.shape.c, 1, align_w};
        dim4 copy_shape = {hidden_squared.shape.n, hidden_squared.shape.c, 1, hidden_squared.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = hidden_squared.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = square_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      } else {
        if (align_w > hidden_squared.shape.w) {
          dim4 fill_shape = {hidden_squared.shape.n, hidden_squared.shape.c, 1, align_w - hidden_squared.shape.w};
          int elem_size = 4;
          int offset = hidden_squared.shape.w * elem_size;
          dim4 fill_tensor_stride = {4096, align_w, hidden_squared.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = hidden_squared.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = hidden_squared.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = square_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
    }
    tpu_bdc_fp_mul_C( square_sum.addr, square_sum.addr, (scalar_t){.f32 = 2.441406e-04f}, &square_sum.shape, (square_sum.default_stride ? NULL : &square_sum.stride), (square_sum.default_stride ? NULL : &square_sum.stride), DT_FP32);
    tpu_bdc_fp_add_C( square_sum.addr, square_sum.addr, (scalar_t){.f32 = v6}, &square_sum.shape, (square_sum.default_stride ? NULL : &square_sum.stride), (square_sum.default_stride ? NULL : &square_sum.stride), DT_FP32);
    tpu_bdc_fp32_rsqrt(square_sum.addr, square_sum.addr, &square_sum.shape);
    __ppl_tensor_info weight_shared = {.shape = {1, 1, 1, 4096}, .stride = {0}, .addr = 32768, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&weight_shared.stride, 0, &weight_shared.shape, DT_FP16);
    __ppl_tensor_info HiddenStates_1 = {.shape = {1, 1, 1, 4096} , .stride = {4096, 4096, 4096, 1} , .addr = v7.addr + ((bx) * 4096+(0) * 4096+(0) * 4096+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((bx) * 4096+(0) * 4096+(0) * 4096+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info hidden_shared_3 = {.shape = {1, 1, 1, 4096} , .stride = hidden_shared.stride, .addr = hidden_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = hidden_shared.default_stride};
    tpu_gdma_cpy_S2L(hidden_shared_3.addr, HiddenStates_1.addr, &hidden_shared_3.shape, (hidden_shared_3.default_stride ? NULL : &hidden_shared_3.stride), (HiddenStates_1.default_stride ? NULL : &HiddenStates_1.stride), DT_FP16);
    __ppl_tensor_info hidden_shared_4 = {.shape = {1, 1, 1, 4096} , .stride = hidden_shared.stride, .addr = hidden_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = hidden_shared.default_stride};
    __ppl_tensor_info hidden_fp32_2 = {.shape = {1, 1, 1, 4096} , .stride = hidden_fp32.stride, .addr = hidden_fp32.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = hidden_fp32.default_stride};
    tpu_bdc_cast(hidden_fp32_2.addr, hidden_shared_4.addr, &hidden_fp32_2.shape, (hidden_fp32_2.default_stride ? NULL : &hidden_fp32_2.stride), (hidden_shared_4.default_stride ? NULL : &hidden_shared_4.stride), DT_FP32, DT_FP16, RM_HALF_TO_EVEN);
    dim4 square_sum_stride;
    tpu_aligned_stride(&square_sum_stride, 0, &square_sum.shape, DT_FP32);
    square_sum_stride.w = 0;
    tpu_bdc_fp_mul( hidden_fp32.addr, hidden_fp32.addr, square_sum.addr, &hidden_fp32.shape, (hidden_fp32.default_stride ? NULL : &hidden_fp32.stride), (hidden_fp32.default_stride ? NULL : &hidden_fp32.stride), &square_sum_stride, DT_FP32);
    __ppl_tensor_info hidden_fp32_3 = {.shape = {1, 1, 1, 4096} , .stride = hidden_fp32.stride, .addr = hidden_fp32.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = hidden_fp32.default_stride};
    __ppl_tensor_info hidden_shared_5 = {.shape = {1, 1, 1, 4096} , .stride = hidden_shared.stride, .addr = hidden_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = hidden_shared.default_stride};
    tpu_bdc_cast(hidden_shared_5.addr, hidden_fp32_3.addr, &hidden_shared_5.shape, (hidden_shared_5.default_stride ? NULL : &hidden_shared_5.stride), (hidden_fp32_3.default_stride ? NULL : &hidden_fp32_3.stride), DT_FP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info Weight = {.shape = {1, 1, 1, 4096} , .stride = {4096, 4096, 4096, 1} , .addr = v8.addr + ((0) * 4096+(0) * 4096+(0) * 4096+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((0) * 4096+(0) * 4096+(0) * 4096+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info weight_shared_1 = {.shape = {1, 1, 1, 4096} , .stride = weight_shared.stride, .addr = weight_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = weight_shared.default_stride};
    tpu_gdma_cpy_S2L(weight_shared_1.addr, Weight.addr, &weight_shared_1.shape, (weight_shared_1.default_stride ? NULL : &weight_shared_1.stride), (Weight.default_stride ? NULL : &Weight.stride), DT_FP16);
    tpu_bdc_fp_mul( hidden_shared.addr, hidden_shared.addr, weight_shared.addr, &hidden_shared.shape, (hidden_shared.default_stride ? NULL : &hidden_shared.stride), (hidden_shared.default_stride ? NULL : &hidden_shared.stride), (weight_shared.default_stride ? NULL : &weight_shared.stride), DT_FP16);
    __ppl_tensor_info hidden_shared_6 = {.shape = {1, 1, 1, 4096} , .stride = hidden_shared.stride, .addr = hidden_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = hidden_shared.default_stride};
    __ppl_tensor_info Output = {.shape = {1, 1, 1, 4096} , .stride = {4096, 4096, 4096, 1} , .addr = v10.addr + ((bx) * 4096+(0) * 4096+(0) * 4096+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((bx) * 4096+(0) * 4096+(0) * 4096+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(Output.addr, hidden_shared_6.addr, &Output.shape, (Output.default_stride ? NULL : &Output.stride), (hidden_shared_6.default_stride ? NULL : &hidden_shared_6.stride), DT_FP16);
  }
}

typedef struct {
  global_addr_t v1;
  global_addr_t v2;
  global_addr_t v3;
  global_addr_t v4;
  int v5;
  float v6;
} tpu_kernel_api_main_inner_args_t;
int main_kernel(const void * args) {
  tpu_kernel_api_main_inner_args_t *api = (tpu_kernel_api_main_inner_args_t*)args;
  tpu_initialize();
  main_kernel_inner(api->v1,
    api->v2,
    api->v3,
    api->v4,
    api->v5,
    api->v6);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(main_kernel)
