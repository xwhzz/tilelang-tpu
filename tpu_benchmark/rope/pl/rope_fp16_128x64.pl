#include "ppl.h"
#include "ppl_wrapper_func.h"

using namespace ppl;

// RoPE fp16: all compute in fp16
// Parameter order: (G_in, G_cos, G_sin, G_out) matching tilelang
__KERNEL__ void main_kernel(fp16 *ptr_in, fp16 *ptr_cos, fp16 *ptr_sin,
                            fp16 *ptr_out) {
  int C = 128;
  int W = 64;
  int block_w = 16;
  int block_c = 64;

  dim4 global_shape = {1, C, 1, W};
  auto g_in = gtensor<fp16>(global_shape, GLOBAL, ptr_in);
  auto g_out = gtensor<fp16>(global_shape, GLOBAL, ptr_out);
  auto g_cos = gtensor<fp16>(global_shape, GLOBAL, ptr_cos);
  auto g_sin = gtensor<fp16>(global_shape, GLOBAL, ptr_sin);

  dim4 block_shape = {1, block_c, 1, block_w};

  for (int idx_c = 0; idx_c < C; idx_c += block_c) {
    int c = min(block_c, C - idx_c);
    for (int idx_w = 0; idx_w < W; idx_w += block_w) {
      ppl::enable_pipeline();
      int w = min(block_w, W - idx_w);

      dim4 in_shape = {1, c, 1, w};
      auto in = make_tensor<fp16>(block_shape, in_shape);
      auto in_cos = make_tensor<fp16>(block_shape, in_shape);
      auto in_sin = make_tensor<fp16>(block_shape, in_shape);

      dim4 in_offset = {0, idx_c, 0, idx_w};
      dma::load(in, g_in.sub_view(in_shape, in_offset));
      dma::load(in_cos, g_cos.sub_view(in_shape, in_offset));
      dma::load(in_sin, g_sin.sub_view(in_shape, in_offset));

      auto x_cos = make_tensor<fp16>(block_shape, in_shape);
      tiu::fmul(x_cos, in, in_cos);

      auto x_sin = make_tensor<fp16>(block_shape, in_shape);
      tiu::fmul(x_sin, in, in_sin);

      auto x_neg = make_tensor<fp16>(block_shape, in_shape);
      tiu::fmul(x_neg, in, -1.0);
      auto x_neg_sin = make_tensor<fp16>(block_shape, in_shape);
      tiu::fmul(x_neg_sin, x_neg, in_sin);

      auto out = make_tensor<fp16>(block_shape, in_shape);

      dim4 half_shape = {1, c, 1, w / 2};
      dim4 half_stride;
      get_stride(&half_stride, &in_shape, TPU_ALIGN, get_eu_num<fp16>());
      half_stride.w = 2;
      dim4 offset = {0, 0, 0, 1};
      auto x_neg_sin_half =
          x_neg_sin.sub_view(half_shape, offset).view(half_shape, half_stride);
      tiu::fadd(out.view(half_shape, half_stride),
                x_cos.view(half_shape, half_stride), x_neg_sin_half);

      tiu::fadd(
          out.sub_view(half_shape, offset).view(half_shape, half_stride),
          x_cos.sub_view(half_shape, offset).view(half_shape, half_stride),
          x_sin.view(half_shape, half_stride));
      dma::store(g_out.sub_view(in_shape, in_offset), out);
    }
  }
}
