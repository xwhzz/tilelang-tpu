#include "ppl.h"
#include "ppl_wrapper_func.h"

using namespace ppl;

// Parameter order matches tilelang: (G_in, G_right, G_out)
// C and W are hardcoded for fair comparison
__KERNEL__ void main_kernel(fp32 *ptr_in, fp32 *ptr_right, fp32 *ptr_out) {
  int C = 64;
  int W = 64;
  int block_w = 16;
  int block_c = 64;

  dim4 global_shape = {1, C, 1, W};
  auto g_in = gtensor<fp32>(global_shape, GLOBAL, ptr_in);
  auto g_right = gtensor<fp32>(global_shape, GLOBAL, ptr_right);
  auto g_out = gtensor<fp32>(global_shape, GLOBAL, ptr_out);

  dim4 block_shape = {1, block_c, 1, block_w};

  for (int idx_c = 0; idx_c < C; idx_c += block_c) {
    int c = min(block_c, C - idx_c);
    for (int idx_w = 0; idx_w < W; idx_w += block_w) {
      ppl::enable_pipeline();
      int w = min(block_w, W - idx_w);

      dim4 in_shape = {1, c, 1, w};
      auto in = make_tensor<fp32>(block_shape, in_shape);
      auto right = make_tensor<fp32>(block_shape, in_shape);

      dim4 in_offset = {0, idx_c, 0, idx_w};
      dma::load(in, g_in.sub_view(in_shape, in_offset));
      dma::load(right, g_right.sub_view(in_shape, in_offset));

      // -x
      auto x_neg = make_tensor<fp32>(block_shape, in_shape);
      tiu::fmul(x_neg, in, -1.0);

      // exp(-x)
      auto x_neg_exp = make_tensor<fp32>(block_shape, in_shape);
      exp_no_overflow(x_neg_exp, x_neg, &block_shape, &in_shape);

      // exp(-x) + 1
      auto x_neg_exp_1 = make_tensor<fp32>(block_shape, in_shape);
      tiu::fadd(x_neg_exp_1, x_neg_exp, 1.0);

      // x / (exp(-x) + 1)
      auto x_neg_exp_1_div = make_tensor<fp32>(block_shape, in_shape);
      tiu::fdiv(x_neg_exp_1_div, in, x_neg_exp_1, 3);

      auto out = make_tensor<fp32>(block_shape, in_shape);
      tiu::fmul(out, x_neg_exp_1_div, right);
      dma::store(g_out.sub_view(in_shape, in_offset), out);
    }
  }
}
