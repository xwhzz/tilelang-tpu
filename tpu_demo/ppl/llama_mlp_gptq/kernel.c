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

extern int tpu_core_index(void);
void mk(global_addr_t v1, global_addr_t v2, global_addr_t v3, global_addr_t v4, global_addr_t v5, global_addr_t v6, global_addr_t v7, global_addr_t v8, global_addr_t v9, global_addr_t v10, global_addr_t v11, global_addr_t v12, global_addr_t v13) {
  __ppl_tensor_info v26 = {.shape = {1 ,1, 1, 1024}, .stride = {0}, .addr = v13, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 2048, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v23 = {.shape = {1 ,1024, 1, 8}, .stride = {0}, .addr = v10, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 16384, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v25 = {.shape = {}, .stride = {0}, .addr = v12, .dtype = DT_INT32, .mode = 2, .align_mode = 0, .size = 4, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v22 = {.shape = {1 ,1024, 1, 4}, .stride = {0}, .addr = v9, .dtype = DT_UINT8, .mode = 2, .align_mode = 0, .size = 4096, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v20 = {.shape = {1 ,1024, 1, 8}, .stride = {0}, .addr = v7, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 16384, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v19 = {.shape = {1 ,1024, 1, 4}, .stride = {0}, .addr = v6, .dtype = DT_UINT8, .mode = 2, .align_mode = 0, .size = 4096, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v17 = {.shape = {1 ,1024, 1, 8}, .stride = {0}, .addr = v4, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 16384, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v18 = {.shape = {1 ,1024, 1, 512}, .stride = {0}, .addr = v5, .dtype = DT_UINT8, .mode = 2, .align_mode = 0, .size = 524288, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v16 = {.shape = {1 ,1024, 1, 4}, .stride = {0}, .addr = v3, .dtype = DT_UINT8, .mode = 2, .align_mode = 0, .size = 4096, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v21 = {.shape = {1 ,1024, 1, 512}, .stride = {0}, .addr = v8, .dtype = DT_UINT8, .mode = 2, .align_mode = 0, .size = 524288, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v15 = {.shape = {1 ,1024, 1, 512}, .stride = {0}, .addr = v2, .dtype = DT_UINT8, .mode = 2, .align_mode = 0, .size = 524288, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v24 = {.shape = {}, .stride = {0}, .addr = v11, .dtype = DT_INT32, .mode = 2, .align_mode = 0, .size = 4, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v14 = {.shape = {1 ,1, 1, 1024}, .stride = {0}, .addr = v1, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 2048, .unsigned_flag = 0, .default_stride = true};
  {
int bx_total = 8;
int bx_per_core = (bx_total + 8 - 1) / 8;
int bx_start = tpu_core_index() * bx_per_core;
int bx_end = bx_start + bx_per_core;
if (bx_end > bx_total) bx_end = bx_total;
if (tpu_core_index() < 8 && bx_start < bx_total) {
for (int bx = bx_start; bx < bx_end; ++bx) {
    __ppl_tensor_info oa0 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&oa0.stride, 0, &oa0.shape, DT_FP32);
    __ppl_tensor_info oa1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 1024, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&oa1.stride, 0, &oa1.shape, DT_FP32);
    __ppl_tensor_info oc0 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16384, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&oc0.stride, 0, &oc0.shape, DT_FP16);
    __ppl_tensor_info oc1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16512, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&oc1.stride, 0, &oc1.shape, DT_FP16);
    {
    scalar_t oa0_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(oa0.addr, oa0_scalar_f32, &oa0.shape, (oa0.default_stride ? NULL : &oa0.stride), DT_FP32);
    }
    {
    scalar_t oa1_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(oa1.addr, oa1_scalar_f32, &oa1.shape, (oa1.default_stride ? NULL : &oa1.stride), DT_FP32);
    }
    for (int ko = 0; ko < 8; ++ko) {
      __ppl_tensor_info ga0 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 1280, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&ga0.stride, 0, &ga0.shape, DT_FP32);
      __ppl_tensor_info ga1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 1536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&ga1.stride, 0, &ga1.shape, DT_FP32);
      __ppl_tensor_info ua0 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 1792, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&ua0.stride, 0, &ua0.shape, DT_FP32);
      __ppl_tensor_info ua1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 2048, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&ua1.stride, 0, &ua1.shape, DT_FP32);
      __ppl_tensor_info ne0 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&ne0.stride, 0, &ne0.shape, DT_FP32);
      __ppl_tensor_info de0 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 33024, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&de0.stride, 0, &de0.shape, DT_FP32);
      __ppl_tensor_info sw0 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 1280, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&sw0.stride, 0, &sw0.shape, DT_FP32);
      __ppl_tensor_info sw0f = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16448, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&sw0f.stride, 0, &sw0f.shape, DT_FP16);
      __ppl_tensor_info w00 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&w00.stride, 0, &w00.shape, DT_FP32);
      __ppl_tensor_info w01 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&w01.stride, 0, &w01.shape, DT_FP32);
      __ppl_tensor_info ne1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&ne1.stride, 0, &ne1.shape, DT_FP32);
      __ppl_tensor_info de1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 33024, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&de1.stride, 0, &de1.shape, DT_FP32);
      __ppl_tensor_info sw1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 1536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&sw1.stride, 0, &sw1.shape, DT_FP32);
      __ppl_tensor_info sw1f = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16576, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&sw1f.stride, 0, &sw1f.shape, DT_FP16);
      __ppl_tensor_info w10 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&w10.stride, 0, &w10.shape, DT_FP32);
      __ppl_tensor_info w11 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&w11.stride, 0, &w11.shape, DT_FP32);
      __ppl_tensor_info cf = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&cf.stride, 0, &cf.shape, DT_FP32);
      __ppl_tensor_info tb = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 0, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&tb.stride, 0, &tb.shape, DT_FP32);
      __ppl_tensor_info dw0 = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_UINT8, .mode = 2, .align_mode = 1, .size = 32, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&dw0.stride, 0, &dw0.shape, DT_UINT8);
      __ppl_tensor_info dz0 = {.shape = { 1, 64, 1, 1}, .stride = {0}, .addr = 32768, .dtype = DT_UINT8, .mode = 2, .align_mode = 1, .size = 1, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&dz0.stride, 0, &dz0.shape, DT_UINT8);
      __ppl_tensor_info ds0 = {.shape = { 1, 64, 1, 1}, .stride = {0}, .addr = 49152, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&ds0.stride, 0, &ds0.shape, DT_FP16);
      __ppl_tensor_info ddq0 = {.shape = { 1, 64, 1, 64}, .stride = {0}, .addr = 0, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&ddq0.stride, 0, &ddq0.shape, DT_FP16);
      __ppl_tensor_info dwk0 = {.shape = { 1, 64, 1, 2}, .stride = {0}, .addr = 65536, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&dwk0.stride, 0, &dwk0.shape, DT_FP16);
      __ppl_tensor_info dtm0 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16704, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&dtm0.stride, 0, &dtm0.shape, DT_FP32);
      __ppl_tensor_info dw1 = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_UINT8, .mode = 2, .align_mode = 1, .size = 32, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&dw1.stride, 0, &dw1.shape, DT_UINT8);
      __ppl_tensor_info dz1 = {.shape = { 1, 64, 1, 1}, .stride = {0}, .addr = 32768, .dtype = DT_UINT8, .mode = 2, .align_mode = 1, .size = 1, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&dz1.stride, 0, &dz1.shape, DT_UINT8);
      __ppl_tensor_info ds1 = {.shape = { 1, 64, 1, 1}, .stride = {0}, .addr = 49152, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&ds1.stride, 0, &ds1.shape, DT_FP16);
      __ppl_tensor_info ddq1 = {.shape = { 1, 64, 1, 64}, .stride = {0}, .addr = 0, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&ddq1.stride, 0, &ddq1.shape, DT_FP16);
      __ppl_tensor_info dwk1 = {.shape = { 1, 64, 1, 2}, .stride = {0}, .addr = 65536, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&dwk1.stride, 0, &dwk1.shape, DT_FP16);
      __ppl_tensor_info dtm1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16704, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&dtm1.stride, 0, &dtm1.shape, DT_FP32);
      {
      scalar_t ga0_scalar_f32 = {.f32 = 0.000000e+00f};
      tpu_bdc_set_C(ga0.addr, ga0_scalar_f32, &ga0.shape, (ga0.default_stride ? NULL : &ga0.stride), DT_FP32);
      }
      {
      scalar_t ga1_scalar_f32 = {.f32 = 0.000000e+00f};
      tpu_bdc_set_C(ga1.addr, ga1_scalar_f32, &ga1.shape, (ga1.default_stride ? NULL : &ga1.stride), DT_FP32);
      }
      {
      scalar_t ua0_scalar_f32 = {.f32 = 0.000000e+00f};
      tpu_bdc_set_C(ua0.addr, ua0_scalar_f32, &ua0.shape, (ua0.default_stride ? NULL : &ua0.stride), DT_FP32);
      }
      {
      scalar_t ua1_scalar_f32 = {.f32 = 0.000000e+00f};
      tpu_bdc_set_C(ua1.addr, ua1_scalar_f32, &ua1.shape, (ua1.default_stride ? NULL : &ua1.stride), DT_FP32);
      }
      for (int ki = 0; ki < 8; ++ki) {
        __ppl_tensor_info ip = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 16384, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&ip.stride, 0, &ip.shape, DT_FP16);
        __ppl_tensor_info gw0 = {.shape = { 1, 64, 1, 64}, .stride = {0}, .addr = 16640, .dtype = DT_UINT8, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&gw0.stride, 0, &gw0.shape, DT_UINT8);
        __ppl_tensor_info gz0 = {.shape = { 1, 64, 1, 1}, .stride = {0}, .addr = 32768, .dtype = DT_UINT8, .mode = 2, .align_mode = 1, .size = 1, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&gz0.stride, 0, &gz0.shape, DT_UINT8);
        __ppl_tensor_info gs0 = {.shape = { 1, 64, 1, 1}, .stride = {0}, .addr = 49152, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&gs0.stride, 0, &gs0.shape, DT_FP16);
        __ppl_tensor_info gdq0 = {.shape = { 1, 64, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&gdq0.stride, 0, &gdq0.shape, DT_FP16);
        __ppl_tensor_info gwk0 = {.shape = { 1, 64, 1, 2}, .stride = {0}, .addr = 65536, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&gwk0.stride, 0, &gwk0.shape, DT_FP16);
        __ppl_tensor_info gtmp0 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16640, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&gtmp0.stride, 0, &gtmp0.shape, DT_FP32);
        __ppl_tensor_info gw1 = {.shape = { 1, 64, 1, 64}, .stride = {0}, .addr = 16640, .dtype = DT_UINT8, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&gw1.stride, 0, &gw1.shape, DT_UINT8);
        __ppl_tensor_info gz1 = {.shape = { 1, 64, 1, 1}, .stride = {0}, .addr = 32768, .dtype = DT_UINT8, .mode = 2, .align_mode = 1, .size = 1, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&gz1.stride, 0, &gz1.shape, DT_UINT8);
        __ppl_tensor_info gs1 = {.shape = { 1, 64, 1, 1}, .stride = {0}, .addr = 49152, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&gs1.stride, 0, &gs1.shape, DT_FP16);
        __ppl_tensor_info gdq1 = {.shape = { 1, 64, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&gdq1.stride, 0, &gdq1.shape, DT_FP16);
        __ppl_tensor_info gwk1 = {.shape = { 1, 64, 1, 2}, .stride = {0}, .addr = 65536, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&gwk1.stride, 0, &gwk1.shape, DT_FP16);
        __ppl_tensor_info gtmp1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16640, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&gtmp1.stride, 0, &gtmp1.shape, DT_FP32);
        __ppl_tensor_info uw0 = {.shape = { 1, 64, 1, 64}, .stride = {0}, .addr = 16640, .dtype = DT_UINT8, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&uw0.stride, 0, &uw0.shape, DT_UINT8);
        __ppl_tensor_info uz0 = {.shape = { 1, 64, 1, 1}, .stride = {0}, .addr = 32768, .dtype = DT_UINT8, .mode = 2, .align_mode = 1, .size = 1, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&uz0.stride, 0, &uz0.shape, DT_UINT8);
        __ppl_tensor_info us0 = {.shape = { 1, 64, 1, 1}, .stride = {0}, .addr = 49152, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&us0.stride, 0, &us0.shape, DT_FP16);
        __ppl_tensor_info udq0 = {.shape = { 1, 64, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&udq0.stride, 0, &udq0.shape, DT_FP16);
        __ppl_tensor_info uwk0 = {.shape = { 1, 64, 1, 2}, .stride = {0}, .addr = 65536, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&uwk0.stride, 0, &uwk0.shape, DT_FP16);
        __ppl_tensor_info utmp0 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16640, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&utmp0.stride, 0, &utmp0.shape, DT_FP32);
        __ppl_tensor_info uw1 = {.shape = { 1, 64, 1, 64}, .stride = {0}, .addr = 16640, .dtype = DT_UINT8, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&uw1.stride, 0, &uw1.shape, DT_UINT8);
        __ppl_tensor_info uz1 = {.shape = { 1, 64, 1, 1}, .stride = {0}, .addr = 32768, .dtype = DT_UINT8, .mode = 2, .align_mode = 1, .size = 1, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&uz1.stride, 0, &uz1.shape, DT_UINT8);
        __ppl_tensor_info us1 = {.shape = { 1, 64, 1, 1}, .stride = {0}, .addr = 49152, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&us1.stride, 0, &us1.shape, DT_FP16);
        __ppl_tensor_info udq1 = {.shape = { 1, 64, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&udq1.stride, 0, &udq1.shape, DT_FP16);
        __ppl_tensor_info uwk1 = {.shape = { 1, 64, 1, 2}, .stride = {0}, .addr = 65536, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&uwk1.stride, 0, &uwk1.shape, DT_FP16);
        __ppl_tensor_info utmp1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16640, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&utmp1.stride, 0, &utmp1.shape, DT_FP32);
        __ppl_tensor_info it = {.shape = {1, 1, 1, 128}, .stride = {1024, 1024, 1024, 1} , .addr = v14.addr + ((0) * 1024+((ki * 128)) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((0) * 1024+((ki * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info ip_1 = {.shape = {1, 1, 1, 128}, .stride = ip.stride, .addr = ip.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = ip.default_stride};
        tpu_gdma_cpy_S2L(ip_1.addr, it.addr, &ip_1.shape, (ip_1.default_stride ? NULL : &ip_1.stride), (it.default_stride ? NULL : &it.stride), DT_FP16);
        __ppl_tensor_info gw = {.shape = {1, 64, 1, 64}, .stride = {524288, 512, 512, 1} , .addr = v15.addr + (((ko * 128)) * 512+((ki * 64)) * 1 ) * 1, .dtype = DT_UINT8, .mode = 2, .size = 1, .offset = (((ko * 128)) * 512+((ki * 64)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info gw0_1 = {.shape = {1, 64, 1, 64}, .stride = gw0.stride, .addr = gw0.addr, .dtype = DT_UINT8, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = gw0.default_stride};
        tpu_gdma_cpy_S2L(gw0_1.addr, gw.addr, &gw0_1.shape, (gw0_1.default_stride ? NULL : &gw0_1.stride), (gw.default_stride ? NULL : &gw.stride), DT_UINT8);
        __ppl_tensor_info gz = {.shape = {1, 64, 1, 1}, .stride = {4096, 4, 4, 1} , .addr = v16.addr + (((ko * 128)) * 4+((ki / 2)) * 1 ) * 1, .dtype = DT_UINT8, .mode = 2, .size = 1, .offset = (((ko * 128)) * 4+((ki / 2)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info gz0_1 = {.shape = {1, 64, 1, 1}, .stride = gz0.stride, .addr = gz0.addr, .dtype = DT_UINT8, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = gz0.default_stride};
        tpu_gdma_cpy_S2L(gz0_1.addr, gz.addr, &gz0_1.shape, (gz0_1.default_stride ? NULL : &gz0_1.stride), (gz.default_stride ? NULL : &gz.stride), DT_UINT8);
        __ppl_tensor_info gs = {.shape = {1, 64, 1, 1}, .stride = {8192, 8, 8, 1} , .addr = v17.addr + (((ko * 128)) * 8+(ki) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = (((ko * 128)) * 8+(ki) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info gs0_1 = {.shape = {1, 64, 1, 1}, .stride = gs0.stride, .addr = gs0.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = gs0.default_stride};
        tpu_gdma_cpy_S2L(gs0_1.addr, gs.addr, &gs0_1.shape, (gs0_1.default_stride ? NULL : &gs0_1.stride), (gs.default_stride ? NULL : &gs.stride), DT_FP16);
        {
          tpu_bdc_cast(gdq0.addr, gw0.addr, &gdq0.shape, (gdq0.default_stride ? NULL : &gdq0.stride), (gw0.default_stride ? NULL : &gw0.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
          tpu_bdc_cast(gwk0.addr, gz0.addr, &gwk0.shape, (gwk0.default_stride ? NULL : &gwk0.stride), (gz0.default_stride ? NULL : &gz0.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
          dim4 gwk0_bc_stride;
          tpu_aligned_stride(&gwk0_bc_stride, 0, &gwk0.shape, DT_FP16);
          gwk0_bc_stride.w = 0;
          tpu_bdc_fp_sub(gdq0.addr, gdq0.addr, gwk0.addr + ((ki % 2) * 2), &gdq0.shape, (gdq0.default_stride ? NULL : &gdq0.stride), (gdq0.default_stride ? NULL : &gdq0.stride), &gwk0_bc_stride, DT_FP16);
          dim4 gs0_bc_stride;
          tpu_aligned_stride(&gs0_bc_stride, 0, &gs0.shape, DT_FP16);
          gs0_bc_stride.w = 0;
          tpu_bdc_fp_mul(gdq0.addr, gdq0.addr, gs0.addr, &gdq0.shape, (gdq0.default_stride ? NULL : &gdq0.stride), (gdq0.default_stride ? NULL : &gdq0.stride), &gs0_bc_stride, DT_FP16);
        }
        tpu_bdc_fp_mm_R_trans(gtmp0.addr, ip.addr, gdq0.addr, 1, 128, 64, DT_FP32, DT_FP16);
        tpu_bdc_fp_add( ga0.addr, ga0.addr, gtmp0.addr, &ga0.shape, (ga0.default_stride ? NULL : &ga0.stride), (ga0.default_stride ? NULL : &ga0.stride), (gtmp0.default_stride ? NULL : &gtmp0.stride), DT_FP32);
        __ppl_tensor_info gw_1 = {.shape = {1, 64, 1, 64}, .stride = {524288, 512, 512, 1} , .addr = v15.addr + ((((ko * 128) + 64)) * 512+((ki * 64)) * 1 ) * 1, .dtype = DT_UINT8, .mode = 2, .size = 1, .offset = ((((ko * 128) + 64)) * 512+((ki * 64)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info gw1_1 = {.shape = {1, 64, 1, 64}, .stride = gw1.stride, .addr = gw1.addr, .dtype = DT_UINT8, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = gw1.default_stride};
        tpu_gdma_cpy_S2L(gw1_1.addr, gw_1.addr, &gw1_1.shape, (gw1_1.default_stride ? NULL : &gw1_1.stride), (gw_1.default_stride ? NULL : &gw_1.stride), DT_UINT8);
        __ppl_tensor_info gz_1 = {.shape = {1, 64, 1, 1}, .stride = {4096, 4, 4, 1} , .addr = v16.addr + ((((ko * 128) + 64)) * 4+((ki / 2)) * 1 ) * 1, .dtype = DT_UINT8, .mode = 2, .size = 1, .offset = ((((ko * 128) + 64)) * 4+((ki / 2)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info gz1_1 = {.shape = {1, 64, 1, 1}, .stride = gz1.stride, .addr = gz1.addr, .dtype = DT_UINT8, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = gz1.default_stride};
        tpu_gdma_cpy_S2L(gz1_1.addr, gz_1.addr, &gz1_1.shape, (gz1_1.default_stride ? NULL : &gz1_1.stride), (gz_1.default_stride ? NULL : &gz_1.stride), DT_UINT8);
        __ppl_tensor_info gs_1 = {.shape = {1, 64, 1, 1}, .stride = {8192, 8, 8, 1} , .addr = v17.addr + ((((ko * 128) + 64)) * 8+(ki) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((((ko * 128) + 64)) * 8+(ki) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info gs1_1 = {.shape = {1, 64, 1, 1}, .stride = gs1.stride, .addr = gs1.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = gs1.default_stride};
        tpu_gdma_cpy_S2L(gs1_1.addr, gs_1.addr, &gs1_1.shape, (gs1_1.default_stride ? NULL : &gs1_1.stride), (gs_1.default_stride ? NULL : &gs_1.stride), DT_FP16);
        {
          tpu_bdc_cast(gdq1.addr, gw1.addr, &gdq1.shape, (gdq1.default_stride ? NULL : &gdq1.stride), (gw1.default_stride ? NULL : &gw1.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
          tpu_bdc_cast(gwk1.addr, gz1.addr, &gwk1.shape, (gwk1.default_stride ? NULL : &gwk1.stride), (gz1.default_stride ? NULL : &gz1.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
          dim4 gwk1_bc_stride;
          tpu_aligned_stride(&gwk1_bc_stride, 0, &gwk1.shape, DT_FP16);
          gwk1_bc_stride.w = 0;
          tpu_bdc_fp_sub(gdq1.addr, gdq1.addr, gwk1.addr + ((ki % 2) * 2), &gdq1.shape, (gdq1.default_stride ? NULL : &gdq1.stride), (gdq1.default_stride ? NULL : &gdq1.stride), &gwk1_bc_stride, DT_FP16);
          dim4 gs1_bc_stride;
          tpu_aligned_stride(&gs1_bc_stride, 0, &gs1.shape, DT_FP16);
          gs1_bc_stride.w = 0;
          tpu_bdc_fp_mul(gdq1.addr, gdq1.addr, gs1.addr, &gdq1.shape, (gdq1.default_stride ? NULL : &gdq1.stride), (gdq1.default_stride ? NULL : &gdq1.stride), &gs1_bc_stride, DT_FP16);
        }
        tpu_bdc_fp_mm_R_trans(gtmp1.addr, ip.addr, gdq1.addr, 1, 128, 64, DT_FP32, DT_FP16);
        tpu_bdc_fp_add( ga1.addr, ga1.addr, gtmp1.addr, &ga1.shape, (ga1.default_stride ? NULL : &ga1.stride), (ga1.default_stride ? NULL : &ga1.stride), (gtmp1.default_stride ? NULL : &gtmp1.stride), DT_FP32);
        __ppl_tensor_info uw = {.shape = {1, 64, 1, 64}, .stride = {524288, 512, 512, 1} , .addr = v18.addr + (((ko * 128)) * 512+((ki * 64)) * 1 ) * 1, .dtype = DT_UINT8, .mode = 2, .size = 1, .offset = (((ko * 128)) * 512+((ki * 64)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info uw0_1 = {.shape = {1, 64, 1, 64}, .stride = uw0.stride, .addr = uw0.addr, .dtype = DT_UINT8, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = uw0.default_stride};
        tpu_gdma_cpy_S2L(uw0_1.addr, uw.addr, &uw0_1.shape, (uw0_1.default_stride ? NULL : &uw0_1.stride), (uw.default_stride ? NULL : &uw.stride), DT_UINT8);
        __ppl_tensor_info uz = {.shape = {1, 64, 1, 1}, .stride = {4096, 4, 4, 1} , .addr = v19.addr + (((ko * 128)) * 4+((ki / 2)) * 1 ) * 1, .dtype = DT_UINT8, .mode = 2, .size = 1, .offset = (((ko * 128)) * 4+((ki / 2)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info uz0_1 = {.shape = {1, 64, 1, 1}, .stride = uz0.stride, .addr = uz0.addr, .dtype = DT_UINT8, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = uz0.default_stride};
        tpu_gdma_cpy_S2L(uz0_1.addr, uz.addr, &uz0_1.shape, (uz0_1.default_stride ? NULL : &uz0_1.stride), (uz.default_stride ? NULL : &uz.stride), DT_UINT8);
        __ppl_tensor_info us = {.shape = {1, 64, 1, 1}, .stride = {8192, 8, 8, 1} , .addr = v20.addr + (((ko * 128)) * 8+(ki) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = (((ko * 128)) * 8+(ki) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info us0_1 = {.shape = {1, 64, 1, 1}, .stride = us0.stride, .addr = us0.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = us0.default_stride};
        tpu_gdma_cpy_S2L(us0_1.addr, us.addr, &us0_1.shape, (us0_1.default_stride ? NULL : &us0_1.stride), (us.default_stride ? NULL : &us.stride), DT_FP16);
        {
          tpu_bdc_cast(udq0.addr, uw0.addr, &udq0.shape, (udq0.default_stride ? NULL : &udq0.stride), (uw0.default_stride ? NULL : &uw0.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
          tpu_bdc_cast(uwk0.addr, uz0.addr, &uwk0.shape, (uwk0.default_stride ? NULL : &uwk0.stride), (uz0.default_stride ? NULL : &uz0.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
          dim4 uwk0_bc_stride;
          tpu_aligned_stride(&uwk0_bc_stride, 0, &uwk0.shape, DT_FP16);
          uwk0_bc_stride.w = 0;
          tpu_bdc_fp_sub(udq0.addr, udq0.addr, uwk0.addr + ((ki % 2) * 2), &udq0.shape, (udq0.default_stride ? NULL : &udq0.stride), (udq0.default_stride ? NULL : &udq0.stride), &uwk0_bc_stride, DT_FP16);
          dim4 us0_bc_stride;
          tpu_aligned_stride(&us0_bc_stride, 0, &us0.shape, DT_FP16);
          us0_bc_stride.w = 0;
          tpu_bdc_fp_mul(udq0.addr, udq0.addr, us0.addr, &udq0.shape, (udq0.default_stride ? NULL : &udq0.stride), (udq0.default_stride ? NULL : &udq0.stride), &us0_bc_stride, DT_FP16);
        }
        tpu_bdc_fp_mm_R_trans(utmp0.addr, ip.addr, udq0.addr, 1, 128, 64, DT_FP32, DT_FP16);
        tpu_bdc_fp_add( ua0.addr, ua0.addr, utmp0.addr, &ua0.shape, (ua0.default_stride ? NULL : &ua0.stride), (ua0.default_stride ? NULL : &ua0.stride), (utmp0.default_stride ? NULL : &utmp0.stride), DT_FP32);
        __ppl_tensor_info uw_1 = {.shape = {1, 64, 1, 64}, .stride = {524288, 512, 512, 1} , .addr = v18.addr + ((((ko * 128) + 64)) * 512+((ki * 64)) * 1 ) * 1, .dtype = DT_UINT8, .mode = 2, .size = 1, .offset = ((((ko * 128) + 64)) * 512+((ki * 64)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info uw1_1 = {.shape = {1, 64, 1, 64}, .stride = uw1.stride, .addr = uw1.addr, .dtype = DT_UINT8, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = uw1.default_stride};
        tpu_gdma_cpy_S2L(uw1_1.addr, uw_1.addr, &uw1_1.shape, (uw1_1.default_stride ? NULL : &uw1_1.stride), (uw_1.default_stride ? NULL : &uw_1.stride), DT_UINT8);
        __ppl_tensor_info uz_1 = {.shape = {1, 64, 1, 1}, .stride = {4096, 4, 4, 1} , .addr = v19.addr + ((((ko * 128) + 64)) * 4+((ki / 2)) * 1 ) * 1, .dtype = DT_UINT8, .mode = 2, .size = 1, .offset = ((((ko * 128) + 64)) * 4+((ki / 2)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info uz1_1 = {.shape = {1, 64, 1, 1}, .stride = uz1.stride, .addr = uz1.addr, .dtype = DT_UINT8, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = uz1.default_stride};
        tpu_gdma_cpy_S2L(uz1_1.addr, uz_1.addr, &uz1_1.shape, (uz1_1.default_stride ? NULL : &uz1_1.stride), (uz_1.default_stride ? NULL : &uz_1.stride), DT_UINT8);
        __ppl_tensor_info us_1 = {.shape = {1, 64, 1, 1}, .stride = {8192, 8, 8, 1} , .addr = v20.addr + ((((ko * 128) + 64)) * 8+(ki) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((((ko * 128) + 64)) * 8+(ki) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info us1_1 = {.shape = {1, 64, 1, 1}, .stride = us1.stride, .addr = us1.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = us1.default_stride};
        tpu_gdma_cpy_S2L(us1_1.addr, us_1.addr, &us1_1.shape, (us1_1.default_stride ? NULL : &us1_1.stride), (us_1.default_stride ? NULL : &us_1.stride), DT_FP16);
        {
          tpu_bdc_cast(udq1.addr, uw1.addr, &udq1.shape, (udq1.default_stride ? NULL : &udq1.stride), (uw1.default_stride ? NULL : &uw1.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
          tpu_bdc_cast(uwk1.addr, uz1.addr, &uwk1.shape, (uwk1.default_stride ? NULL : &uwk1.stride), (uz1.default_stride ? NULL : &uz1.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
          dim4 uwk1_bc_stride;
          tpu_aligned_stride(&uwk1_bc_stride, 0, &uwk1.shape, DT_FP16);
          uwk1_bc_stride.w = 0;
          tpu_bdc_fp_sub(udq1.addr, udq1.addr, uwk1.addr + ((ki % 2) * 2), &udq1.shape, (udq1.default_stride ? NULL : &udq1.stride), (udq1.default_stride ? NULL : &udq1.stride), &uwk1_bc_stride, DT_FP16);
          dim4 us1_bc_stride;
          tpu_aligned_stride(&us1_bc_stride, 0, &us1.shape, DT_FP16);
          us1_bc_stride.w = 0;
          tpu_bdc_fp_mul(udq1.addr, udq1.addr, us1.addr, &udq1.shape, (udq1.default_stride ? NULL : &udq1.stride), (udq1.default_stride ? NULL : &udq1.stride), &us1_bc_stride, DT_FP16);
        }
        tpu_bdc_fp_mm_R_trans(utmp1.addr, ip.addr, udq1.addr, 1, 128, 64, DT_FP32, DT_FP16);
        tpu_bdc_fp_add( ua1.addr, ua1.addr, utmp1.addr, &ua1.shape, (ua1.default_stride ? NULL : &ua1.stride), (ua1.default_stride ? NULL : &ua1.stride), (utmp1.default_stride ? NULL : &utmp1.stride), DT_FP32);
      }
      tpu_bdc_fp_mul_C( ne0.addr, ga0.addr, (scalar_t){.f32 = -1}, &ne0.shape, (ne0.default_stride ? NULL : &ne0.stride), (ga0.default_stride ? NULL : &ga0.stride), DT_FP32);
      tpu_bdc_load_fp32_exp_coeff(cf.addr);
      tpu_bdc_load_fp32_exp_table(tb.addr);
      tpu_bdc_fp32_exp(ne0.addr, ne0.addr, w00.addr, w01.addr, cf.addr, tb.addr, &ne0.shape);
      tpu_bdc_fp_add_C( de0.addr, ne0.addr, (scalar_t){.f32 = 1}, &de0.shape, (de0.default_stride ? NULL : &de0.stride), (ne0.default_stride ? NULL : &ne0.stride), DT_FP32);
      tpu_bdc_fp_div( ne0.addr, ga0.addr, de0.addr, &ne0.shape, (ne0.default_stride ? NULL : &ne0.stride), (ga0.default_stride ? NULL : &ga0.stride), (de0.default_stride ? NULL : &de0.stride), DT_FP32);
      tpu_bdc_fp_mul( sw0.addr, ua0.addr, ne0.addr, &sw0.shape, (sw0.default_stride ? NULL : &sw0.stride), (ua0.default_stride ? NULL : &ua0.stride), (ne0.default_stride ? NULL : &ne0.stride), DT_FP32);
      tpu_bdc_fp_mul_C( ne1.addr, ga1.addr, (scalar_t){.f32 = -1}, &ne1.shape, (ne1.default_stride ? NULL : &ne1.stride), (ga1.default_stride ? NULL : &ga1.stride), DT_FP32);
      tpu_bdc_load_fp32_exp_coeff(cf.addr);
      tpu_bdc_load_fp32_exp_table(tb.addr);
      tpu_bdc_fp32_exp(ne1.addr, ne1.addr, w10.addr, w11.addr, cf.addr, tb.addr, &ne1.shape);
      tpu_bdc_fp_add_C( de1.addr, ne1.addr, (scalar_t){.f32 = 1}, &de1.shape, (de1.default_stride ? NULL : &de1.stride), (ne1.default_stride ? NULL : &ne1.stride), DT_FP32);
      tpu_bdc_fp_div( ne1.addr, ga1.addr, de1.addr, &ne1.shape, (ne1.default_stride ? NULL : &ne1.stride), (ga1.default_stride ? NULL : &ga1.stride), (de1.default_stride ? NULL : &de1.stride), DT_FP32);
      tpu_bdc_fp_mul( sw1.addr, ua1.addr, ne1.addr, &sw1.shape, (sw1.default_stride ? NULL : &sw1.stride), (ua1.default_stride ? NULL : &ua1.stride), (ne1.default_stride ? NULL : &ne1.stride), DT_FP32);
      __ppl_tensor_info sw0_1 = {.shape = {1, 1, 1, 64}, .stride = sw0.stride, .addr = sw0.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = sw0.default_stride};
      __ppl_tensor_info sw0f_1 = {.shape = {1, 1, 1, 64}, .stride = sw0f.stride, .addr = sw0f.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = sw0f.default_stride};
      tpu_bdc_cast(sw0f_1.addr, sw0_1.addr, &sw0f_1.shape, (sw0f_1.default_stride ? NULL : &sw0f_1.stride), (sw0_1.default_stride ? NULL : &sw0_1.stride), DT_FP16, DT_FP32, RM_HALF_TO_EVEN);
      __ppl_tensor_info sw1_1 = {.shape = {1, 1, 1, 64}, .stride = sw1.stride, .addr = sw1.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = sw1.default_stride};
      __ppl_tensor_info sw1f_1 = {.shape = {1, 1, 1, 64}, .stride = sw1f.stride, .addr = sw1f.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = sw1f.default_stride};
      tpu_bdc_cast(sw1f_1.addr, sw1_1.addr, &sw1f_1.shape, (sw1f_1.default_stride ? NULL : &sw1f_1.stride), (sw1_1.default_stride ? NULL : &sw1_1.stride), DT_FP16, DT_FP32, RM_HALF_TO_EVEN);
      __ppl_tensor_info dw = {.shape = {1, 64, 1, 32}, .stride = {524288, 512, 512, 1} , .addr = v21.addr + (((bx * 128)) * 512+((ko * 64)) * 1 ) * 1, .dtype = DT_UINT8, .mode = 2, .size = 1, .offset = (((bx * 128)) * 512+((ko * 64)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info dw0_1 = {.shape = {1, 64, 1, 32}, .stride = dw0.stride, .addr = dw0.addr, .dtype = DT_UINT8, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = dw0.default_stride};
      tpu_gdma_cpy_S2L(dw0_1.addr, dw.addr, &dw0_1.shape, (dw0_1.default_stride ? NULL : &dw0_1.stride), (dw.default_stride ? NULL : &dw.stride), DT_UINT8);
      __ppl_tensor_info dz = {.shape = {1, 64, 1, 1}, .stride = {4096, 4, 4, 1} , .addr = v22.addr + (((bx * 128)) * 4+((ko / 2)) * 1 ) * 1, .dtype = DT_UINT8, .mode = 2, .size = 1, .offset = (((bx * 128)) * 4+((ko / 2)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info dz0_1 = {.shape = {1, 64, 1, 1}, .stride = dz0.stride, .addr = dz0.addr, .dtype = DT_UINT8, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = dz0.default_stride};
      tpu_gdma_cpy_S2L(dz0_1.addr, dz.addr, &dz0_1.shape, (dz0_1.default_stride ? NULL : &dz0_1.stride), (dz.default_stride ? NULL : &dz.stride), DT_UINT8);
      __ppl_tensor_info ds = {.shape = {1, 64, 1, 1}, .stride = {8192, 8, 8, 1} , .addr = v23.addr + (((bx * 128)) * 8+(ko) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = (((bx * 128)) * 8+(ko) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info ds0_1 = {.shape = {1, 64, 1, 1}, .stride = ds0.stride, .addr = ds0.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = ds0.default_stride};
      tpu_gdma_cpy_S2L(ds0_1.addr, ds.addr, &ds0_1.shape, (ds0_1.default_stride ? NULL : &ds0_1.stride), (ds.default_stride ? NULL : &ds.stride), DT_FP16);
      {
        tpu_bdc_cast(ddq0.addr, dw0.addr, &ddq0.shape, (ddq0.default_stride ? NULL : &ddq0.stride), (dw0.default_stride ? NULL : &dw0.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
        tpu_bdc_cast(dwk0.addr, dz0.addr, &dwk0.shape, (dwk0.default_stride ? NULL : &dwk0.stride), (dz0.default_stride ? NULL : &dz0.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
        dim4 dwk0_bc_stride;
        tpu_aligned_stride(&dwk0_bc_stride, 0, &dwk0.shape, DT_FP16);
        dwk0_bc_stride.w = 0;
        tpu_bdc_fp_sub(ddq0.addr, ddq0.addr, dwk0.addr + ((ko % 2) * 2), &ddq0.shape, (ddq0.default_stride ? NULL : &ddq0.stride), (ddq0.default_stride ? NULL : &ddq0.stride), &dwk0_bc_stride, DT_FP16);
        dim4 ds0_bc_stride;
        tpu_aligned_stride(&ds0_bc_stride, 0, &ds0.shape, DT_FP16);
        ds0_bc_stride.w = 0;
        tpu_bdc_fp_mul(ddq0.addr, ddq0.addr, ds0.addr, &ddq0.shape, (ddq0.default_stride ? NULL : &ddq0.stride), (ddq0.default_stride ? NULL : &ddq0.stride), &ds0_bc_stride, DT_FP16);
      }
      tpu_bdc_fp_mm_R_trans(dtm0.addr, sw0f.addr, ddq0.addr, 1, 64, 64, DT_FP32, DT_FP16);
      tpu_bdc_fp_add( oa0.addr, oa0.addr, dtm0.addr, &oa0.shape, (oa0.default_stride ? NULL : &oa0.stride), (oa0.default_stride ? NULL : &oa0.stride), (dtm0.default_stride ? NULL : &dtm0.stride), DT_FP32);
      __ppl_tensor_info dw_1 = {.shape = {1, 64, 1, 32}, .stride = {524288, 512, 512, 1} , .addr = v21.addr + (((bx * 128)) * 512+(((ko * 64) + 32)) * 1 ) * 1, .dtype = DT_UINT8, .mode = 2, .size = 1, .offset = (((bx * 128)) * 512+(((ko * 64) + 32)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info dw0_2 = {.shape = {1, 64, 1, 32}, .stride = dw0.stride, .addr = dw0.addr, .dtype = DT_UINT8, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = dw0.default_stride};
      tpu_gdma_cpy_S2L(dw0_2.addr, dw_1.addr, &dw0_2.shape, (dw0_2.default_stride ? NULL : &dw0_2.stride), (dw_1.default_stride ? NULL : &dw_1.stride), DT_UINT8);
      {
        tpu_bdc_cast(ddq0.addr, dw0.addr, &ddq0.shape, (ddq0.default_stride ? NULL : &ddq0.stride), (dw0.default_stride ? NULL : &dw0.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
        tpu_bdc_cast(dwk0.addr, dz0.addr, &dwk0.shape, (dwk0.default_stride ? NULL : &dwk0.stride), (dz0.default_stride ? NULL : &dz0.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
        dim4 dwk0_bc_stride;
        tpu_aligned_stride(&dwk0_bc_stride, 0, &dwk0.shape, DT_FP16);
        dwk0_bc_stride.w = 0;
        tpu_bdc_fp_sub(ddq0.addr, ddq0.addr, dwk0.addr + ((ko % 2) * 2), &ddq0.shape, (ddq0.default_stride ? NULL : &ddq0.stride), (ddq0.default_stride ? NULL : &ddq0.stride), &dwk0_bc_stride, DT_FP16);
        dim4 ds0_bc_stride;
        tpu_aligned_stride(&ds0_bc_stride, 0, &ds0.shape, DT_FP16);
        ds0_bc_stride.w = 0;
        tpu_bdc_fp_mul(ddq0.addr, ddq0.addr, ds0.addr, &ddq0.shape, (ddq0.default_stride ? NULL : &ddq0.stride), (ddq0.default_stride ? NULL : &ddq0.stride), &ds0_bc_stride, DT_FP16);
      }
      tpu_bdc_fp_mm_R_trans(dtm0.addr, sw1f.addr, ddq0.addr, 1, 64, 64, DT_FP32, DT_FP16);
      tpu_bdc_fp_add( oa0.addr, oa0.addr, dtm0.addr, &oa0.shape, (oa0.default_stride ? NULL : &oa0.stride), (oa0.default_stride ? NULL : &oa0.stride), (dtm0.default_stride ? NULL : &dtm0.stride), DT_FP32);
      __ppl_tensor_info dw_2 = {.shape = {1, 64, 1, 32}, .stride = {524288, 512, 512, 1} , .addr = v21.addr + ((((bx * 128) + 64)) * 512+((ko * 64)) * 1 ) * 1, .dtype = DT_UINT8, .mode = 2, .size = 1, .offset = ((((bx * 128) + 64)) * 512+((ko * 64)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info dw1_1 = {.shape = {1, 64, 1, 32}, .stride = dw1.stride, .addr = dw1.addr, .dtype = DT_UINT8, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = dw1.default_stride};
      tpu_gdma_cpy_S2L(dw1_1.addr, dw_2.addr, &dw1_1.shape, (dw1_1.default_stride ? NULL : &dw1_1.stride), (dw_2.default_stride ? NULL : &dw_2.stride), DT_UINT8);
      __ppl_tensor_info dz_1 = {.shape = {1, 64, 1, 1}, .stride = {4096, 4, 4, 1} , .addr = v22.addr + ((((bx * 128) + 64)) * 4+((ko / 2)) * 1 ) * 1, .dtype = DT_UINT8, .mode = 2, .size = 1, .offset = ((((bx * 128) + 64)) * 4+((ko / 2)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info dz1_1 = {.shape = {1, 64, 1, 1}, .stride = dz1.stride, .addr = dz1.addr, .dtype = DT_UINT8, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = dz1.default_stride};
      tpu_gdma_cpy_S2L(dz1_1.addr, dz_1.addr, &dz1_1.shape, (dz1_1.default_stride ? NULL : &dz1_1.stride), (dz_1.default_stride ? NULL : &dz_1.stride), DT_UINT8);
      __ppl_tensor_info ds_1 = {.shape = {1, 64, 1, 1}, .stride = {8192, 8, 8, 1} , .addr = v23.addr + ((((bx * 128) + 64)) * 8+(ko) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((((bx * 128) + 64)) * 8+(ko) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info ds1_1 = {.shape = {1, 64, 1, 1}, .stride = ds1.stride, .addr = ds1.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = ds1.default_stride};
      tpu_gdma_cpy_S2L(ds1_1.addr, ds_1.addr, &ds1_1.shape, (ds1_1.default_stride ? NULL : &ds1_1.stride), (ds_1.default_stride ? NULL : &ds_1.stride), DT_FP16);
      {
        tpu_bdc_cast(ddq1.addr, dw1.addr, &ddq1.shape, (ddq1.default_stride ? NULL : &ddq1.stride), (dw1.default_stride ? NULL : &dw1.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
        tpu_bdc_cast(dwk1.addr, dz1.addr, &dwk1.shape, (dwk1.default_stride ? NULL : &dwk1.stride), (dz1.default_stride ? NULL : &dz1.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
        dim4 dwk1_bc_stride;
        tpu_aligned_stride(&dwk1_bc_stride, 0, &dwk1.shape, DT_FP16);
        dwk1_bc_stride.w = 0;
        tpu_bdc_fp_sub(ddq1.addr, ddq1.addr, dwk1.addr + ((ko % 2) * 2), &ddq1.shape, (ddq1.default_stride ? NULL : &ddq1.stride), (ddq1.default_stride ? NULL : &ddq1.stride), &dwk1_bc_stride, DT_FP16);
        dim4 ds1_bc_stride;
        tpu_aligned_stride(&ds1_bc_stride, 0, &ds1.shape, DT_FP16);
        ds1_bc_stride.w = 0;
        tpu_bdc_fp_mul(ddq1.addr, ddq1.addr, ds1.addr, &ddq1.shape, (ddq1.default_stride ? NULL : &ddq1.stride), (ddq1.default_stride ? NULL : &ddq1.stride), &ds1_bc_stride, DT_FP16);
      }
      tpu_bdc_fp_mm_R_trans(dtm1.addr, sw0f.addr, ddq1.addr, 1, 64, 64, DT_FP32, DT_FP16);
      tpu_bdc_fp_add( oa1.addr, oa1.addr, dtm1.addr, &oa1.shape, (oa1.default_stride ? NULL : &oa1.stride), (oa1.default_stride ? NULL : &oa1.stride), (dtm1.default_stride ? NULL : &dtm1.stride), DT_FP32);
      __ppl_tensor_info dw_3 = {.shape = {1, 64, 1, 32}, .stride = {524288, 512, 512, 1} , .addr = v21.addr + ((((bx * 128) + 64)) * 512+(((ko * 64) + 32)) * 1 ) * 1, .dtype = DT_UINT8, .mode = 2, .size = 1, .offset = ((((bx * 128) + 64)) * 512+(((ko * 64) + 32)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info dw1_2 = {.shape = {1, 64, 1, 32}, .stride = dw1.stride, .addr = dw1.addr, .dtype = DT_UINT8, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = dw1.default_stride};
      tpu_gdma_cpy_S2L(dw1_2.addr, dw_3.addr, &dw1_2.shape, (dw1_2.default_stride ? NULL : &dw1_2.stride), (dw_3.default_stride ? NULL : &dw_3.stride), DT_UINT8);
      {
        tpu_bdc_cast(ddq1.addr, dw1.addr, &ddq1.shape, (ddq1.default_stride ? NULL : &ddq1.stride), (dw1.default_stride ? NULL : &dw1.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
        tpu_bdc_cast(dwk1.addr, dz1.addr, &dwk1.shape, (dwk1.default_stride ? NULL : &dwk1.stride), (dz1.default_stride ? NULL : &dz1.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
        dim4 dwk1_bc_stride;
        tpu_aligned_stride(&dwk1_bc_stride, 0, &dwk1.shape, DT_FP16);
        dwk1_bc_stride.w = 0;
        tpu_bdc_fp_sub(ddq1.addr, ddq1.addr, dwk1.addr + ((ko % 2) * 2), &ddq1.shape, (ddq1.default_stride ? NULL : &ddq1.stride), (ddq1.default_stride ? NULL : &ddq1.stride), &dwk1_bc_stride, DT_FP16);
        dim4 ds1_bc_stride;
        tpu_aligned_stride(&ds1_bc_stride, 0, &ds1.shape, DT_FP16);
        ds1_bc_stride.w = 0;
        tpu_bdc_fp_mul(ddq1.addr, ddq1.addr, ds1.addr, &ddq1.shape, (ddq1.default_stride ? NULL : &ddq1.stride), (ddq1.default_stride ? NULL : &ddq1.stride), &ds1_bc_stride, DT_FP16);
      }
      tpu_bdc_fp_mm_R_trans(dtm1.addr, sw1f.addr, ddq1.addr, 1, 64, 64, DT_FP32, DT_FP16);
      tpu_bdc_fp_add( oa1.addr, oa1.addr, dtm1.addr, &oa1.shape, (oa1.default_stride ? NULL : &oa1.stride), (oa1.default_stride ? NULL : &oa1.stride), (dtm1.default_stride ? NULL : &dtm1.stride), DT_FP32);
    }
    __ppl_tensor_info oa0_1 = {.shape = {1, 1, 1, 64}, .stride = oa0.stride, .addr = oa0.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = oa0.default_stride};
    __ppl_tensor_info oc0_1 = {.shape = {1, 1, 1, 64}, .stride = oc0.stride, .addr = oc0.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = oc0.default_stride};
    tpu_bdc_cast(oc0_1.addr, oa0_1.addr, &oc0_1.shape, (oc0_1.default_stride ? NULL : &oc0_1.stride), (oa0_1.default_stride ? NULL : &oa0_1.stride), DT_FP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info oa1_1 = {.shape = {1, 1, 1, 64}, .stride = oa1.stride, .addr = oa1.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = oa1.default_stride};
    __ppl_tensor_info oc1_1 = {.shape = {1, 1, 1, 64}, .stride = oc1.stride, .addr = oc1.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = oc1.default_stride};
    tpu_bdc_cast(oc1_1.addr, oa1_1.addr, &oc1_1.shape, (oc1_1.default_stride ? NULL : &oc1_1.stride), (oa1_1.default_stride ? NULL : &oa1_1.stride), DT_FP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info oc0_2 = {.shape = {1, 1, 1, 64}, .stride = oc0.stride, .addr = oc0.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = oc0.default_stride};
    __ppl_tensor_info ot = {.shape = {1, 1, 1, 64}, .stride = {1024, 1024, 1024, 1} , .addr = v26.addr + ((0) * 1024+((bx * 128)) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((0) * 1024+((bx * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(ot.addr, oc0_2.addr, &ot.shape, (ot.default_stride ? NULL : &ot.stride), (oc0_2.default_stride ? NULL : &oc0_2.stride), DT_FP16);
    __ppl_tensor_info oc1_2 = {.shape = {1, 1, 1, 64}, .stride = oc1.stride, .addr = oc1.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = oc1.default_stride};
    __ppl_tensor_info ot_1 = {.shape = {1, 1, 1, 64}, .stride = {1024, 1024, 1024, 1} , .addr = v26.addr + ((0) * 1024+(((bx * 128) + 64)) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((0) * 1024+(((bx * 128) + 64)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(ot_1.addr, oc1_2.addr, &ot_1.shape, (ot_1.default_stride ? NULL : &ot_1.stride), (oc1_2.default_stride ? NULL : &oc1_2.stride), DT_FP16);
  }
}
}
}

typedef struct {
  global_addr_t v1;
  global_addr_t v2;
  global_addr_t v3;
  global_addr_t v4;
  global_addr_t v5;
  global_addr_t v6;
  global_addr_t v7;
  global_addr_t v8;
  global_addr_t v9;
  global_addr_t v10;
  global_addr_t v11;
  global_addr_t v12;
  global_addr_t v13;
} tpu_kernel_api_main_inner_args_t;
int main_kernel(const void * args) {
  tpu_kernel_api_main_inner_args_t *api = (tpu_kernel_api_main_inner_args_t*)args;
  tpu_initialize();
  mk(api->v1,
    api->v2,
    api->v3,
    api->v4,
    api->v5,
    api->v6,
    api->v7,
    api->v8,
    api->v9,
    api->v10,
    api->v11,
    api->v12,
    api->v13);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(main_kernel)
