#include "ppl.h"
#include "ppl_wrapper_func.h"

using namespace ppl;

__KERNEL__ void main_kernel(bf16 *ptr_q, bf16 *ptr_k, bf16 *ptr_v,
                            bf16 *ptr_out) {
  int batch = 1;
  int heads = 8;
  int seq_len = 128;
  int dim = 64;
  float sqrt_d = 1.0f / sqrt((float)dim);
  int block_m = 16;
  int block_k = 16;
  int block_h = heads;

  int order[4] = {2, 1, 0, 3};
  dim4 qo_global_shape = {batch, seq_len, heads, dim};
  auto q_global = make_gtensor_permute<bf16>(qo_global_shape, GLOBAL, ptr_q, order);
  auto k_global = make_gtensor_permute<bf16>(qo_global_shape, GLOBAL, ptr_k, order);
  auto v_global = make_gtensor_permute<bf16>(qo_global_shape, GLOBAL, ptr_v, order);
  auto out_global = make_gtensor_permute<bf16>(qo_global_shape, GLOBAL, ptr_out, order);

  dim4 qk_shape = {block_h, block_m, 1, block_k};
  dim4 mi_shape = {block_h, block_m, 1, 1};
  dim4 acc_shape = {block_h, block_m, 1, dim};

  for (int _b = 0; _b < batch; _b++) {
    for (int _h = 0; _h < heads; _h += block_h) {
      int real_h = min(block_h, heads - _h);
      for (int _m = 0; _m < seq_len; _m += block_m) {
        int real_m = min(block_m, seq_len - _m);
        dim4 qi_real_shape = {real_h, real_m, 1, dim};
        dim4 qi_offset = {_h, _m, _b, 0};
        tensor<bf16> qi_tensor;
        dma::load(qi_tensor, q_global.sub_view(qi_real_shape, qi_offset));
        tensor<bf16> qi_scaled;
        tiu::fmul(qi_scaled, qi_tensor, sqrt_d);

        dim4 mi_real = {real_h, real_m, 1, 1};
        dim4 acc_real = {real_h, real_m, 1, dim};
        auto mi_sub = make_tensor<bf16>(mi_shape, mi_real);
        auto li_sub = make_tensor<bf16>(mi_shape, mi_real);
        auto acc_sub = make_tensor<bf16>(acc_shape, acc_real);
        tiu::fill(mi_sub, -15000);
        tiu::zero(li_sub);
        tiu::zero(acc_sub);

        int block_k_iter = max(min(block_k, seq_len / 2), 1);
        for (int _k = 0; _k < seq_len; _k += block_k_iter) {
          ppl::enable_pipeline();
          int real_k = min(block_k_iter, seq_len - _k);
          dim4 kvi_real = {real_h, real_k, 1, dim};
          dim4 kvi_offset = {_h, _k, _b, 0};
          dim4 qk_real = {real_h, real_m, 1, real_k};

          tensor<bf16> ki_tensor, vi_tensor;
          dma::load(ki_tensor, k_global.sub_view(kvi_real, kvi_offset));
          dma::load(vi_tensor, v_global.sub_view(kvi_real, kvi_offset));

          dim4 qk_batch = {1, real_m, 1, real_k};
          dim4 qi_batch = {1, real_m, 1, dim};
          dim4 ki_batch = {1, real_k, 1, dim};
          auto qk_sub = make_tensor<bf16>(qk_shape, qk_real);
          auto ki_sub = ki_tensor.view(kvi_real);
          for (int i = 0; i < real_h; i++) {
            dim4 bo = {i, 0, 0, 0};
            tiu::fmm2(qk_sub.sub_view(qk_batch, bo),
                      qi_scaled.sub_view(qi_batch, bo),
                      ki_sub.sub_view(ki_batch, bo), false, true, false);
          }

          tensor<bf16> max_out, mi_new;
          quick_pooling(max_out, qk_sub, &qk_shape, &qk_real, -15000, 0);
          tiu::fmax(mi_new, mi_sub, max_out);

          tensor<bf16> alpha, sub_out, li_tmp;
          tiu::fsub(sub_out, mi_sub, mi_new);
          tiu::move(mi_sub, mi_new);
          exp_no_overflow(alpha, sub_out, &mi_shape, &mi_real);
          tiu::fmul(acc_sub, acc_sub, alpha);
          tiu::fmul(li_tmp, li_sub, alpha);

          tensor<bf16> sub_out1, p_T, sum_val;
          tiu::fsub(sub_out1, qk_sub, mi_new);
          exp_no_overflow(p_T, sub_out1, &qk_shape, &qk_real);
          quick_pooling(sum_val, p_T, &qk_shape, &qk_real, 0, 1);
          tiu::fadd(li_sub, li_tmp, sum_val);

          auto pv = make_tensor<bf16>(acc_shape, acc_real);
          dim4 pv_batch = {1, real_m, 1, dim};
          dim4 p_batch = {1, real_m, 1, real_k};
          dim4 vi_batch = {1, real_k, 1, dim};
          auto vi_sub = vi_tensor.view(kvi_real);
          for (int i = 0; i < real_h; i++) {
            dim4 bo = {i, 0, 0, 0};
            tiu::fmm2(pv.sub_view(pv_batch, bo),
                      p_T.sub_view(p_batch, bo),
                      vi_sub.sub_view(vi_batch, bo));
          }
          tiu::fadd(acc_sub, acc_sub, pv);
        }

        tensor<fp32> li_fp32, div_li;
        tensor<bf16> div_li_T, final_out;
        tiu::cast(li_fp32, li_sub);
        tiu::fdiv(div_li, 1.0f, li_fp32, 3);
        tiu::cast(div_li_T, div_li);
        tiu::fmul(final_out, acc_sub, div_li_T);
        dma::store(out_global.sub_view(acc_real, qi_offset), final_out);
      }
    }
  }
}
