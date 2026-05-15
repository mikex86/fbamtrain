#pragma once

#include "../declare_kernel.inc.h"

#include "conv2d_bin_kernels_nv.h"

#define DECLARE_ALL_KERNELS(nv_arch)                                                                                   \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_bf16_##nv_arch);                                                             \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_bf16_unaligned_##nv_arch);                                                   \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_fp16_##nv_arch);                                                             \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_fp16_unaligned_##nv_arch);                                                   \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_fp16_acc_fp16_##nv_arch);                                                    \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_fp16_acc_fp16_unaligned_##nv_arch);                                          \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_bf16_cacc_##nv_arch);                                                        \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_bf16_unaligned_cacc_##nv_arch);                                              \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_fp16_cacc_##nv_arch);                                                        \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_fp16_unaligned_cacc_##nv_arch);                                              \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_fp16_acc_fp16_cacc_##nv_arch);                                               \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_fp16_acc_fp16_unaligned_cacc_##nv_arch);                                     \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_preact_bf16_##nv_arch);                                                      \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_preact_bf16_unaligned_##nv_arch);                                            \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_preact_fp16_##nv_arch);                                                      \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_preact_fp16_unaligned_##nv_arch);                                            \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_preact_fp16_acc_fp16_##nv_arch);                                             \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_preact_fp16_acc_fp16_unaligned_##nv_arch);                                   \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_preact_bf16_cacc_##nv_arch);                                                 \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_preact_bf16_unaligned_cacc_##nv_arch);                                       \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_preact_fp16_cacc_##nv_arch);                                                 \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_preact_fp16_unaligned_cacc_##nv_arch);                                       \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_preact_fp16_acc_fp16_cacc_##nv_arch);                                        \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_preact_fp16_acc_fp16_unaligned_cacc_##nv_arch);                              \
    DECLARE_ELEMWISE_KERNEL_BINARY(fill_constant_bf16_##nv_arch);                                                     \
    DECLARE_ELEMWISE_KERNEL_BINARY(fill_constant_fp16_##nv_arch);                                                     \
    DECLARE_ELEMWISE_KERNEL_BINARY(fill_constant_fp32_##nv_arch);                                                     \
    DECLARE_ELEMWISE_KERNEL_BINARY(fill_uniform_bf16_##nv_arch);                                                       \
    DECLARE_ELEMWISE_KERNEL_BINARY(fill_uniform_fp16_##nv_arch);                                                       \
    DECLARE_ELEMWISE_KERNEL_BINARY(fill_uniform_fp32_##nv_arch);                                                       \
    DECLARE_ELEMWISE_KERNEL_BINARY(fill_zeros_##nv_arch);                                                              \
    DECLARE_ELEMWISE_KERNEL_BINARY(adamw_step_bf16_##nv_arch);                                                         \
    DECLARE_ELEMWISE_KERNEL_BINARY(adamw_step_fp16_##nv_arch);                                                         \
    DECLARE_ELEMWISE_KERNEL_BINARY(adamw_step_fp32_##nv_arch);                                                         \
    DECLARE_ELEMWISE_KERNEL_BINARY(sgd_step_bf16_##nv_arch);                                                           \
    DECLARE_ELEMWISE_KERNEL_BINARY(sgd_step_fp16_##nv_arch);                                                           \
    DECLARE_ELEMWISE_KERNEL_BINARY(sgd_step_fp32_##nv_arch);                                                           \
    DECLARE_2D_KERNEL_BINARY(build_cell_embeds_bf16_##nv_arch);                                                        \
    DECLARE_2D_KERNEL_BINARY(build_cell_embeds_fp16_##nv_arch);                                                        \
    DECLARE_ELEMWISE_KERNEL_BINARY(build_cell_embeds_bwd_cp_bf16_##nv_arch);                                           \
    DECLARE_ELEMWISE_KERNEL_BINARY(build_cell_embeds_bwd_cp_fp16_##nv_arch);                                           \
    DECLARE_ELEMWISE_KERNEL_BINARY(build_cell_embeds_bwd_cp_bf16_out_fp32_##nv_arch);                                  \
    DECLARE_ELEMWISE_KERNEL_BINARY(build_cell_embeds_bwd_cp_fp16_out_fp32_##nv_arch);                                  \
    DECLARE_2D_KERNEL_BINARY(build_cell_embeds_bwd_color_bf16_##nv_arch);                                              \
    DECLARE_2D_KERNEL_BINARY(build_cell_embeds_bwd_color_fp16_##nv_arch);                                              \
    DECLARE_2D_KERNEL_BINARY(build_cell_embeds_bwd_color_bf16_out_fp32_##nv_arch);                                     \
    DECLARE_2D_KERNEL_BINARY(build_cell_embeds_bwd_color_fp16_out_fp32_##nv_arch);                                     \
    DECLARE_2D_KERNEL_BINARY(build_cell_embeds_bwd_pos_bf16_##nv_arch);                                                \
    DECLARE_2D_KERNEL_BINARY(build_cell_embeds_bwd_pos_fp16_##nv_arch);                                                \
    DECLARE_2D_KERNEL_BINARY(build_cell_embeds_bwd_pos_bf16_out_fp32_##nv_arch);                                       \
    DECLARE_2D_KERNEL_BINARY(build_cell_embeds_bwd_pos_fp16_out_fp32_##nv_arch);                                       \
    DECLARE_ELEMWISE_KERNEL_BINARY(fill_normal_bf16_##nv_arch);                                                        \
    DECLARE_ELEMWISE_KERNEL_BINARY(fill_normal_fp16_##nv_arch);                                                        \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fwd_hs128_bf16_##nv_arch);                                                  \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fwd_hs128_bf16_nolse_##nv_arch);                                            \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fwd_hs128_bf16_uneven_##nv_arch);                                           \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fwd_hs128_bf16_uneven_nolse_##nv_arch);                                     \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fwd_hs128_fp16_##nv_arch);                                                  \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fwd_hs128_fp16_nolse_##nv_arch);                                            \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fwd_hs128_fp16_uneven_##nv_arch);                                           \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fwd_hs128_fp16_uneven_nolse_##nv_arch);                                     \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fwd_hs128_fp16_acc_fp16_##nv_arch);                                         \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fwd_hs128_fp16_acc_fp16_nolse_##nv_arch);                                   \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fwd_hs128_fp16_acc_fp16_uneven_##nv_arch);                                  \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fwd_hs128_fp16_acc_fp16_uneven_nolse_##nv_arch);                            \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_bwd_pre_hs128_bf16_##nv_arch);                                              \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_bwd_pre_hs128_fp16_##nv_arch);                                              \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_bwd_hs128_bf16_##nv_arch);                                                  \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_bwd_hs128_fp16_##nv_arch);                                                  \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_bwd_hs128_bf16_uneven_##nv_arch);                                           \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_bwd_hs128_fp16_uneven_##nv_arch);                                           \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fa_fwd_hs128_bf16_##nv_arch);                                               \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fa_fwd_hs128_bf16_nolse_##nv_arch);                                         \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fa_fwd_hs128_bf16_even_##nv_arch);                                          \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fa_fwd_hs128_bf16_even_nolse_##nv_arch);                                    \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fa_fwd_hs128_fp16_##nv_arch);                                               \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fa_fwd_hs128_fp16_nolse_##nv_arch);                                         \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fa_fwd_hs128_fp16_even_##nv_arch);                                          \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fa_fwd_hs128_fp16_even_nolse_##nv_arch);                                    \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fa_fwd_hs128_fp16_acc_fp16_##nv_arch);                                      \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fa_fwd_hs128_fp16_acc_fp16_nolse_##nv_arch);                                \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fa_fwd_hs128_fp16_acc_fp16_even_##nv_arch);                                 \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fa_fwd_hs128_fp16_acc_fp16_even_nolse_##nv_arch);                           \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fa_bwd_hs128_bf16_##nv_arch);                                               \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fa_bwd_hs128_fp16_##nv_arch);                                               \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fa_bwd_hs128_bf16_even_##nv_arch);                                          \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fa_bwd_hs128_fp16_even_##nv_arch);                                          \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fa_bwd_dot_do_o_hs128_bf16_##nv_arch);                                      \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fa_bwd_dot_do_o_hs128_fp16_##nv_arch);                                      \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fa_bwd_convert_dq_hs128_bf16_##nv_arch);                                    \
    DECLARE_2D_KERNEL_BINARY(mha_full_attn_fa_bwd_convert_dq_hs128_fp16_##nv_arch);                                    \
    DECLARE_MHA_CUTLASS_KERNEL_BINARY(mha_full_attn_cutlass_bf16_##nv_arch);                                           \
    DECLARE_MHA_CUTLASS_KERNEL_BINARY(mha_full_attn_cutlass_bf16_lse_##nv_arch);                                       \
    DECLARE_MHA_CUTLASS_KERNEL_BINARY(mha_full_attn_cutlass_fp16_##nv_arch);                                           \
    DECLARE_MHA_CUTLASS_KERNEL_BINARY(mha_full_attn_cutlass_fp16_lse_##nv_arch);                                       \
    DECLARE_MHA_CUTLASS_BWD_KERNEL_BINARY(mha_full_attn_bwd_cutlass_bf16_##nv_arch);                                   \
    DECLARE_MHA_CUTLASS_BWD_KERNEL_BINARY(mha_full_attn_bwd_cutlass_fp16_##nv_arch);                                   \
    DECLARE_ELEMWISE_KERNEL_BINARY(layer_norm_fwd_bf16_##nv_arch);                                                     \
    DECLARE_ELEMWISE_KERNEL_BINARY(layer_norm_fwd_fp16_##nv_arch);                                                     \
    DECLARE_ELEMWISE_KERNEL_BINARY(rms_norm_fwd_bf16_##nv_arch);                                                       \
    DECLARE_ELEMWISE_KERNEL_BINARY(rms_norm_fwd_fp16_##nv_arch);                                                       \
    DECLARE_ELEMWISE_KERNEL_BINARY(rms_norm_bwd_bf16_##nv_arch);                                                       \
    DECLARE_ELEMWISE_KERNEL_BINARY(rms_norm_bwd_fp16_##nv_arch);                                                       \
    DECLARE_4D_KERNEL_BINARY(contiguous_4d_bf16_##nv_arch);                                                            \
    DECLARE_4D_KERNEL_BINARY(contiguous_4d_fp16_##nv_arch);                                                            \
    DECLARE_3D_KERNEL_BINARY(contiguous_3d_bf16_##nv_arch);                                                            \
    DECLARE_3D_KERNEL_BINARY(contiguous_3d_fp16_##nv_arch);                                                            \
    DECLARE_2D_KERNEL_BINARY(device_copy_strided_2d_##nv_arch);                                                        \
    DECLARE_ELEMWISE_KERNEL_BINARY(add_trailing_broadcast_bf16_##nv_arch);                                             \
    DECLARE_ELEMWISE_KERNEL_BINARY(add_trailing_broadcast_fp16_##nv_arch);                                             \
    DECLARE_ELEMWISE_KERNEL_BINARY(add_trailing_broadcast_fp32_##nv_arch);                                             \
    DECLARE_ELEMWISE_KERNEL_BINARY(add_elementwise_bf16_##nv_arch);                                                    \
    DECLARE_ELEMWISE_KERNEL_BINARY(add_elementwise_fp16_##nv_arch);                                                    \
    DECLARE_ELEMWISE_KERNEL_BINARY(add_elementwise_fp32_##nv_arch);                                                    \
    DECLARE_ELEMWISE_KERNEL_BINARY(add_elementwise_fp32_out_fp16_##nv_arch);                                           \
    DECLARE_ELEMWISE_KERNEL_BINARY(mul_trailing_broadcast_bf16_##nv_arch);                                             \
    DECLARE_ELEMWISE_KERNEL_BINARY(mul_trailing_broadcast_fp16_##nv_arch);                                             \
    DECLARE_ELEMWISE_KERNEL_BINARY(mul_trailing_broadcast_fp32_##nv_arch);                                             \
    DECLARE_ELEMWISE_KERNEL_BINARY(mul_elementwise_bf16_##nv_arch);                                                    \
    DECLARE_ELEMWISE_KERNEL_BINARY(mul_elementwise_fp16_##nv_arch);                                                    \
    DECLARE_ELEMWISE_KERNEL_BINARY(mul_elementwise_fp32_##nv_arch);                                                    \
    DECLARE_ELEMWISE_KERNEL_BINARY(mul_scalar_bf16_##nv_arch);                                                         \
    DECLARE_ELEMWISE_KERNEL_BINARY(mul_scalar_fp16_##nv_arch);                                                         \
    DECLARE_ELEMWISE_KERNEL_BINARY(mul_scalar_fp32_##nv_arch);                                                         \
    DECLARE_ELEMWISE_KERNEL_BINARY(sqrt_elementwise_bf16_##nv_arch);                                                   \
    DECLARE_ELEMWISE_KERNEL_BINARY(sqrt_elementwise_fp16_##nv_arch);                                                   \
    DECLARE_ELEMWISE_KERNEL_BINARY(sqrt_elementwise_fp32_##nv_arch);                                                   \
    DECLARE_ELEMWISE_KERNEL_BINARY(div_elementwise_bf16_##nv_arch);                                                    \
    DECLARE_ELEMWISE_KERNEL_BINARY(div_elementwise_fp16_##nv_arch);                                                    \
    DECLARE_ELEMWISE_KERNEL_BINARY(div_elementwise_fp32_##nv_arch);                                                    \
    DECLARE_ELEMWISE_KERNEL_BINARY(div_add_bf16_##nv_arch);                                                           \
    DECLARE_ELEMWISE_KERNEL_BINARY(div_add_fp16_##nv_arch);                                                           \
    DECLARE_ELEMWISE_KERNEL_BINARY(div_add_fp32_##nv_arch);                                                           \
    DECLARE_ELEMWISE_KERNEL_BINARY(div_scalar_add_bf16_##nv_arch);                                                    \
    DECLARE_ELEMWISE_KERNEL_BINARY(div_scalar_add_fp16_##nv_arch);                                                    \
    DECLARE_ELEMWISE_KERNEL_BINARY(div_scalar_add_fp32_##nv_arch);                                                    \
    DECLARE_ELEMWISE_KERNEL_BINARY(div_scalar_add_broadcast_bf16_##nv_arch);                                          \
    DECLARE_ELEMWISE_KERNEL_BINARY(div_scalar_add_broadcast_fp16_##nv_arch);                                          \
    DECLARE_ELEMWISE_KERNEL_BINARY(div_scalar_add_broadcast_fp32_##nv_arch);                                          \
    DECLARE_ELEMWISE_KERNEL_BINARY(div_scalar_bf16_##nv_arch);                                                         \
    DECLARE_ELEMWISE_KERNEL_BINARY(div_scalar_fp16_##nv_arch);                                                         \
    DECLARE_ELEMWISE_KERNEL_BINARY(div_scalar_fp32_##nv_arch);                                                         \
    DECLARE_ELEMWISE_KERNEL_BINARY(act_gelu_elementwise_bf16_##nv_arch);                                               \
    DECLARE_ELEMWISE_KERNEL_BINARY(act_gelu_elementwise_fp16_##nv_arch);                                               \
    DECLARE_ELEMWISE_KERNEL_BINARY(act_gelu_bwd_elementwise_bf16_##nv_arch);                                           \
    DECLARE_ELEMWISE_KERNEL_BINARY(act_gelu_bwd_elementwise_fp16_##nv_arch);                                           \
    DECLARE_ELEMWISE_KERNEL_BINARY(act_relu_elementwise_bf16_##nv_arch);                                               \
    DECLARE_ELEMWISE_KERNEL_BINARY(act_relu_elementwise_fp16_##nv_arch);                                               \
    DECLARE_ELEMWISE_KERNEL_BINARY(cross_entropy_on_targets_bf16_##nv_arch);                                           \
    DECLARE_ELEMWISE_KERNEL_BINARY(cross_entropy_on_targets_fp16_##nv_arch);                                           \
    DECLARE_ELEMWISE_KERNEL_BINARY(cross_entropy_on_targets_add_fp32_bf16_##nv_arch);                                  \
    DECLARE_ELEMWISE_KERNEL_BINARY(cross_entropy_on_targets_add_fp32_fp16_##nv_arch);                                  \
    DECLARE_ELEMWISE_KERNEL_BINARY(cross_entropy_on_targets_bwd_bf16_##nv_arch);                                       \
    DECLARE_ELEMWISE_KERNEL_BINARY(cross_entropy_on_targets_bwd_fp16_##nv_arch);                                       \
    DECLARE_ELEMWISE_KERNEL_BINARY(lstm_cell_recompute_out_fp16_##nv_arch);                                            \
    DECLARE_ELEMWISE_KERNEL_BINARY(lstm_cell_recompute_out_bf16_##nv_arch);                                            \
    DECLARE_ELEMWISE_KERNEL_BINARY(lstm_cell_bwd_pointwise_out_fp16_##nv_arch);                                        \
    DECLARE_ELEMWISE_KERNEL_BINARY(lstm_cell_bwd_pointwise_out_bf16_##nv_arch);                                        \
    DECLARE_ELEMWISE_KERNEL_BINARY(reduce_sum_partial_fp32_##nv_arch);                                                 \
    DECLARE_ELEMWISE_KERNEL_BINARY(mean_reduce_contiguous_bf16_##nv_arch);                                             \
    DECLARE_ELEMWISE_KERNEL_BINARY(mean_reduce_contiguous_fp16_##nv_arch);                                             \
    DECLARE_ELEMWISE_KERNEL_BINARY(mean_reduce_column_tiled_bf16_##nv_arch);                                           \
    DECLARE_ELEMWISE_KERNEL_BINARY(mean_reduce_column_tiled_fp16_##nv_arch);                                           \
    DECLARE_ELEMWISE_KERNEL_BINARY(sum_reduce_contiguous_bf16_out_fp32_##nv_arch);                                     \
    DECLARE_ELEMWISE_KERNEL_BINARY(sum_reduce_contiguous_fp16_out_fp32_##nv_arch);                                     \
    DECLARE_ELEMWISE_KERNEL_BINARY(sum_reduce_contiguous_fp32_out_fp32_##nv_arch);                                     \
    DECLARE_ELEMWISE_KERNEL_BINARY(sum_reduce_column_tiled_bf16_out_fp32_##nv_arch);                                   \
    DECLARE_ELEMWISE_KERNEL_BINARY(sum_reduce_column_tiled_fp16_out_fp32_##nv_arch);                                   \
    DECLARE_ELEMWISE_KERNEL_BINARY(sum_reduce_column_tiled_fp32_out_fp32_##nv_arch);                                   \
    DECLARE_2D_KERNEL_BINARY(sum_reduce_column_split_partials_bf16_out_fp32_##nv_arch);                                \
    DECLARE_2D_KERNEL_BINARY(sum_reduce_column_split_partials_fp16_out_fp32_##nv_arch);                                \
    DECLARE_2D_KERNEL_BINARY(sum_reduce_column_split_partials_fp32_out_fp32_##nv_arch);                                \
    DECLARE_ELEMWISE_KERNEL_BINARY(mul_reduce_contiguous_bf16_out_fp32_##nv_arch);                                     \
    DECLARE_ELEMWISE_KERNEL_BINARY(mul_reduce_contiguous_fp16_out_fp32_##nv_arch);                                     \
    DECLARE_ELEMWISE_KERNEL_BINARY(mul_reduce_column_tiled_bf16_out_fp32_##nv_arch);                                   \
    DECLARE_ELEMWISE_KERNEL_BINARY(mul_reduce_column_tiled_fp16_out_fp32_##nv_arch);                                   \
    DECLARE_2D_KERNEL_BINARY(mul_reduce_column_split_partials_bf16_out_fp32_##nv_arch);                                \
    DECLARE_2D_KERNEL_BINARY(mul_reduce_column_split_partials_fp16_out_fp32_##nv_arch);                                \
    DECLARE_POOL1D_KERNEL_BINARY(avg_pool1d_bf16_##nv_arch);                                                           \
    DECLARE_POOL1D_KERNEL_BINARY(avg_pool1d_fp16_##nv_arch);                                                           \
    DECLARE_POOL2D_KERNEL_BINARY(avg_pool2d_bf16_##nv_arch);                                                           \
    DECLARE_POOL2D_KERNEL_BINARY(avg_pool2d_fp16_##nv_arch);                                                           \
    DECLARE_POOL2D_KERNEL_BINARY(avg_pool2d_nhwc_2x2_s2_bf16_##nv_arch);                                               \
    DECLARE_POOL2D_KERNEL_BINARY(avg_pool2d_nhwc_2x2_s2_fp16_##nv_arch);                                               \
    DECLARE_POOL2D_KERNEL_BINARY(avg_pool2d_bwd_bf16_##nv_arch);                                                       \
    DECLARE_POOL2D_KERNEL_BINARY(avg_pool2d_bwd_fp16_##nv_arch);                                                       \
    DECLARE_POOL2D_KERNEL_BINARY(avg_pool2d_bwd_noaccum_bf16_##nv_arch);                                               \
    DECLARE_POOL2D_KERNEL_BINARY(avg_pool2d_bwd_noaccum_fp16_##nv_arch);                                               \
    DECLARE_POOL2D_KERNEL_BINARY(avg_pool2d_bwd_noaccum_nhwc_2x2_s2_bf16_##nv_arch);                                   \
    DECLARE_POOL2D_KERNEL_BINARY(avg_pool2d_bwd_noaccum_nhwc_2x2_s2_fp16_##nv_arch);                                   \
    DECLARE_GEMM_KERNEL_BINARY(addmm_bf16_##nv_arch);                                                                  \
    DECLARE_GEMM_KERNEL_BINARY(addmm_fp16_##nv_arch);                                                                  \
    DECLARE_GEMM_KERNEL_BINARY(addmm_fp16_acc_fp16_##nv_arch);                                                         \
    DECLARE_GEMM_KERNEL_BINARY(addmm_fp16_out_fp32_##nv_arch);                                                         \
    DECLARE_GEMM_KERNEL_BINARY(addmm_fp16_acc_out_fp32_##nv_arch);                                                     \
    DECLARE_GEMM_KERNEL_BINARY(addmm_bf16_unaligned_##nv_arch);                                                       \
    DECLARE_GEMM_KERNEL_BINARY(addmm_fp16_unaligned_##nv_arch);                                                       \
    DECLARE_GEMM_KERNEL_BINARY(addmm_fp16_acc_fp16_unaligned_##nv_arch);                                              \
    DECLARE_GEMM_KERNEL_BINARY(addmm_fp16_out_fp32_unaligned_##nv_arch);                                              \
    DECLARE_GEMM_KERNEL_BINARY(addmm_fp16_acc_out_fp32_unaligned_##nv_arch);                                          \
    DECLARE_GEMM_KERNEL_BINARY(addmm_bf16_cacc_##nv_arch);                                                            \
    DECLARE_GEMM_KERNEL_BINARY(addmm_fp16_cacc_##nv_arch);                                                            \
    DECLARE_GEMM_KERNEL_BINARY(addmm_fp16_acc_fp16_cacc_##nv_arch);                                                   \
    DECLARE_GEMM_KERNEL_BINARY(addmm_fp16_out_fp32_cacc_##nv_arch);                                                   \
    DECLARE_GEMM_KERNEL_BINARY(addmm_fp16_acc_out_fp32_cacc_##nv_arch);                                               \
    DECLARE_GEMM_KERNEL_BINARY(addmm_bf16_unaligned_cacc_##nv_arch);                                                  \
    DECLARE_GEMM_KERNEL_BINARY(addmm_fp16_unaligned_cacc_##nv_arch);                                                  \
    DECLARE_GEMM_KERNEL_BINARY(addmm_fp16_acc_fp16_unaligned_cacc_##nv_arch);                                         \
    DECLARE_GEMM_KERNEL_BINARY(addmm_fp16_out_fp32_unaligned_cacc_##nv_arch);                                         \
    DECLARE_GEMM_KERNEL_BINARY(addmm_fp16_acc_out_fp32_unaligned_cacc_##nv_arch);                                     \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_##nv_arch);                                                                 \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_ta_##nv_arch);                                                              \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_tb_##nv_arch);                                                              \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_tab_##nv_arch);                                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_##nv_arch);                                                                 \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_ta_##nv_arch);                                                              \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_tb_##nv_arch);                                                              \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_tab_##nv_arch);                                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_acc_fp16_##nv_arch);                                                        \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_out_fp32_##nv_arch);                                                       \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_out_fp32_ta_##nv_arch);                                                    \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_out_fp32_tb_##nv_arch);                                                    \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_out_fp32_tab_##nv_arch);                                                   \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_out_fp32_##nv_arch);                                                        \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_acc_out_fp32_##nv_arch);                                                    \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_out_fp32_ta_##nv_arch);                                                     \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_out_fp32_tb_##nv_arch);                                                     \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_out_fp32_tab_##nv_arch);                                                    \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_acc_out_fp32_ta_##nv_arch);                                                 \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_acc_out_fp32_tb_##nv_arch);                                                 \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_acc_out_fp32_tab_##nv_arch);                                                \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_unaligned_##nv_arch);                                                      \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_ta_unaligned_##nv_arch);                                                   \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_tb_unaligned_##nv_arch);                                                   \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_tab_unaligned_##nv_arch);                                                  \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_unaligned_##nv_arch);                                                      \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_ta_unaligned_##nv_arch);                                                   \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_tb_unaligned_##nv_arch);                                                   \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_tab_unaligned_##nv_arch);                                                  \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_acc_fp16_unaligned_##nv_arch);                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_out_fp32_unaligned_##nv_arch);                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_out_fp32_ta_unaligned_##nv_arch);                                          \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_out_fp32_tb_unaligned_##nv_arch);                                          \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_out_fp32_tab_unaligned_##nv_arch);                                         \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_out_fp32_unaligned_##nv_arch);                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_acc_out_fp32_unaligned_##nv_arch);                                         \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_out_fp32_ta_unaligned_##nv_arch);                                          \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_out_fp32_tb_unaligned_##nv_arch);                                          \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_out_fp32_tab_unaligned_##nv_arch);                                         \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_acc_out_fp32_ta_unaligned_##nv_arch);                                      \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_acc_out_fp32_tb_unaligned_##nv_arch);                                      \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_acc_out_fp32_tab_unaligned_##nv_arch);                                     \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bf16_##nv_arch);                                                            \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_fp16_##nv_arch);                                                            \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_fp16_acc_fp16_##nv_arch);                                                   \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bf16_unaligned_##nv_arch);                                                 \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_fp16_unaligned_##nv_arch);                                                 \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_fp16_acc_fp16_unaligned_##nv_arch);                                        \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_bf16_##nv_arch);                                                            \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_bf16_ta_##nv_arch);                                                         \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_bf16_tb_##nv_arch);                                                         \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_bf16_tab_##nv_arch);                                                        \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_##nv_arch);                                                            \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_ta_##nv_arch);                                                         \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_tb_##nv_arch);                                                         \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_tab_##nv_arch);                                                        \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_acc_fp16_##nv_arch);                                                   \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_acc_fp16_ta_##nv_arch);                                                \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_acc_fp16_tb_##nv_arch);                                                \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_acc_fp16_tab_##nv_arch);                                               \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_bf16_unaligned_##nv_arch);                                                 \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_bf16_unaligned_ta_##nv_arch);                                              \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_bf16_unaligned_tb_##nv_arch);                                              \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_bf16_unaligned_tab_##nv_arch);                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_unaligned_##nv_arch);                                                 \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_unaligned_ta_##nv_arch);                                              \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_unaligned_tb_##nv_arch);                                              \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_unaligned_tab_##nv_arch);                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_acc_fp16_unaligned_##nv_arch);                                        \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_acc_fp16_unaligned_ta_##nv_arch);                                     \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_acc_fp16_unaligned_tb_##nv_arch);                                     \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_acc_fp16_unaligned_tab_##nv_arch);                                    \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_cacc_##nv_arch);                                                           \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_ta_cacc_##nv_arch);                                                        \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_tb_cacc_##nv_arch);                                                        \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_tab_cacc_##nv_arch);                                                       \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_cacc_##nv_arch);                                                           \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_ta_cacc_##nv_arch);                                                        \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_tb_cacc_##nv_arch);                                                        \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_tab_cacc_##nv_arch);                                                       \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_acc_fp16_cacc_##nv_arch);                                                  \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_out_fp32_cacc_##nv_arch);                                                  \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_out_fp32_ta_cacc_##nv_arch);                                               \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_out_fp32_tb_cacc_##nv_arch);                                               \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_out_fp32_tab_cacc_##nv_arch);                                              \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_out_fp32_cacc_##nv_arch);                                                  \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_acc_out_fp32_cacc_##nv_arch);                                              \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_out_fp32_ta_cacc_##nv_arch);                                               \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_out_fp32_tb_cacc_##nv_arch);                                               \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_out_fp32_tab_cacc_##nv_arch);                                              \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_acc_out_fp32_ta_cacc_##nv_arch);                                           \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_acc_out_fp32_tb_cacc_##nv_arch);                                           \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_acc_out_fp32_tab_cacc_##nv_arch);                                          \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_unaligned_cacc_##nv_arch);                                                 \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_ta_unaligned_cacc_##nv_arch);                                              \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_tb_unaligned_cacc_##nv_arch);                                              \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_tab_unaligned_cacc_##nv_arch);                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_unaligned_cacc_##nv_arch);                                                 \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_ta_unaligned_cacc_##nv_arch);                                              \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_tb_unaligned_cacc_##nv_arch);                                              \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_tab_unaligned_cacc_##nv_arch);                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_acc_fp16_unaligned_cacc_##nv_arch);                                        \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_out_fp32_unaligned_cacc_##nv_arch);                                        \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_out_fp32_ta_unaligned_cacc_##nv_arch);                                     \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_out_fp32_tb_unaligned_cacc_##nv_arch);                                     \
    DECLARE_GEMM_KERNEL_BINARY(matmul_bf16_out_fp32_tab_unaligned_cacc_##nv_arch);                                    \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_out_fp32_unaligned_cacc_##nv_arch);                                        \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_acc_out_fp32_unaligned_cacc_##nv_arch);                                    \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_out_fp32_ta_unaligned_cacc_##nv_arch);                                     \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_out_fp32_tb_unaligned_cacc_##nv_arch);                                     \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_out_fp32_tab_unaligned_cacc_##nv_arch);                                    \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_acc_out_fp32_ta_unaligned_cacc_##nv_arch);                                 \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_acc_out_fp32_tb_unaligned_cacc_##nv_arch);                                 \
    DECLARE_GEMM_KERNEL_BINARY(matmul_fp16_acc_out_fp32_tab_unaligned_cacc_##nv_arch);                                \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bf16_cacc_##nv_arch);                                                      \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_fp16_cacc_##nv_arch);                                                      \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_fp16_acc_fp16_cacc_##nv_arch);                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bf16_unaligned_cacc_##nv_arch);                                            \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_fp16_unaligned_cacc_##nv_arch);                                            \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_fp16_acc_fp16_unaligned_cacc_##nv_arch);                                   \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_bf16_cacc_##nv_arch);                                                      \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_bf16_ta_cacc_##nv_arch);                                                   \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_bf16_tb_cacc_##nv_arch);                                                   \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_bf16_tab_cacc_##nv_arch);                                                  \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_cacc_##nv_arch);                                                      \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_ta_cacc_##nv_arch);                                                   \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_tb_cacc_##nv_arch);                                                   \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_tab_cacc_##nv_arch);                                                  \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_acc_fp16_cacc_##nv_arch);                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_acc_fp16_ta_cacc_##nv_arch);                                          \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_acc_fp16_tb_cacc_##nv_arch);                                          \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_acc_fp16_tab_cacc_##nv_arch);                                         \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_bf16_unaligned_cacc_##nv_arch);                                            \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_bf16_unaligned_ta_cacc_##nv_arch);                                         \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_bf16_unaligned_tb_cacc_##nv_arch);                                         \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_bf16_unaligned_tab_cacc_##nv_arch);                                        \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_unaligned_cacc_##nv_arch);                                            \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_unaligned_ta_cacc_##nv_arch);                                         \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_unaligned_tb_cacc_##nv_arch);                                         \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_unaligned_tab_cacc_##nv_arch);                                        \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_acc_fp16_unaligned_cacc_##nv_arch);                                   \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_acc_fp16_unaligned_ta_cacc_##nv_arch);                                \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_acc_fp16_unaligned_tb_cacc_##nv_arch);                                \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_fp16_acc_fp16_unaligned_tab_cacc_##nv_arch);                               \
    DECLARE_GEMM_KERNEL_BINARY(addmm_cutlass_bf16_##nv_arch);                                                          \
    DECLARE_GEMM_KERNEL_BINARY(addmm_cutlass_fp16_##nv_arch);                                                          \
    DECLARE_GEMM_KERNEL_BINARY(addmm_cutlass_fp16_acc_fp16_##nv_arch);                                                 \
    DECLARE_GEMM_KERNEL_BINARY(addmm_cutlass_fp16_out_fp32_##nv_arch);                                                 \
    DECLARE_GEMM_KERNEL_BINARY(addmm_cutlass_fp16_acc_fp16_out_fp32_##nv_arch);                                        \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_bf16_##nv_arch);                                                         \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_bf16_ta_##nv_arch);                                                      \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_bf16_tb_##nv_arch);                                                      \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_bf16_tab_##nv_arch);                                                     \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_bf16_out_fp32_##nv_arch);                                                \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_bf16_out_fp32_ta_##nv_arch);                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_bf16_out_fp32_tb_##nv_arch);                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_bf16_out_fp32_tab_##nv_arch);                                            \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_fp16_##nv_arch);                                                         \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_fp16_acc_fp16_##nv_arch);                                                \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_fp16_out_fp32_##nv_arch);                                                \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_fp16_acc_fp16_out_fp32_##nv_arch);                                       \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_fp16_ta_##nv_arch);                                                      \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_fp16_tb_##nv_arch);                                                      \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_fp16_tab_##nv_arch);                                                     \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_fp16_acc_fp16_ta_##nv_arch);                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_fp16_acc_fp16_tb_##nv_arch);                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_fp16_acc_fp16_tab_##nv_arch);                                            \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_fp16_out_fp32_ta_##nv_arch);                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_fp16_out_fp32_tb_##nv_arch);                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_fp16_out_fp32_tab_##nv_arch);                                            \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_fp16_acc_fp16_out_fp32_ta_##nv_arch);                                    \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_fp16_acc_fp16_out_fp32_tb_##nv_arch);                                    \
    DECLARE_GEMM_KERNEL_BINARY(matmul_cutlass_fp16_acc_fp16_out_fp32_tab_##nv_arch);                                   \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_cutlass_bf16_##nv_arch);                                                    \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_cutlass_fp16_##nv_arch);                                                    \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_cutlass_fp16_acc_fp16_##nv_arch);                                           \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_cutlass_bf16_##nv_arch);                                                \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_cutlass_bf16_ta_##nv_arch);                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_cutlass_bf16_tb_##nv_arch);                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_cutlass_bf16_tab_##nv_arch);                                            \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_cutlass_fp16_##nv_arch);                                                \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_cutlass_fp16_ta_##nv_arch);                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_cutlass_fp16_tb_##nv_arch);                                             \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_cutlass_fp16_tab_##nv_arch);                                            \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_cutlass_fp16_acc_fp16_##nv_arch);                                       \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_cutlass_fp16_acc_fp16_ta_##nv_arch);                                    \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_cutlass_fp16_acc_fp16_tb_##nv_arch);                                    \
    DECLARE_GEMM_KERNEL_BINARY(matmul_gelu_bwd_cutlass_fp16_acc_fp16_tab_##nv_arch);                                   \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_cutlass_bf16_##nv_arch);                                                     \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_cutlass_fp16_##nv_arch);                                                     \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_cutlass_fp16_acc_fp16_##nv_arch);                                            \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_preact_cutlass_bf16_##nv_arch);                                              \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_preact_cutlass_fp16_##nv_arch);                                              \
    DECLARE_GEMM_KERNEL_BINARY(addmm_gelu_preact_cutlass_fp16_acc_fp16_##nv_arch);                                     \
    DECLARE_ELEMWISE_KERNEL_BINARY(embedding_lookup_bf16_##nv_arch);                                                   \
    DECLARE_ELEMWISE_KERNEL_BINARY(embedding_lookup_fp16_##nv_arch);                                                   \
    DECLARE_ELEMWISE_KERNEL_BINARY(cast_bf16_to_fp32_##nv_arch);                                                       \
    DECLARE_ELEMWISE_KERNEL_BINARY(cast_fp32_to_bf16_##nv_arch);                                                       \
    DECLARE_ELEMWISE_KERNEL_BINARY(cast_fp16_to_fp32_##nv_arch);                                                       \
    DECLARE_ELEMWISE_KERNEL_BINARY(cast_fp32_to_fp16_##nv_arch);                                                       \
    DECLARE_ELEMWISE_KERNEL_BINARY(cast_bf16_to_fp16_##nv_arch);                                                       \
    DECLARE_ELEMWISE_KERNEL_BINARY(cast_fp16_to_bf16_##nv_arch);                                                       \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_bf16_ic3_oc8_k3_##nv_arch);                                                    \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_bf16_ic3_oc8_k3_dil2_##nv_arch);                                               \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_bf16_ic32_oc64_k3_##nv_arch);                                                  \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_bf16_ic768_oc768_k3_dil2_##nv_arch);                                           \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_bf16_ic768_oc1536_k3_##nv_arch);                                               \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_bf16_ic1024_oc1024_k3_dil2_##nv_arch);                                         \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_bf16_ic1024_oc2048_k3_##nv_arch);                                              \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_fp16_ic3_oc8_k3_##nv_arch);                                                    \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_fp16_ic3_oc8_k3_dil2_##nv_arch);                                               \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_fp16_ic32_oc64_k3_##nv_arch);                                                  \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_fp16_ic768_oc768_k3_dil2_##nv_arch);                                           \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_fp16_ic768_oc1536_k3_##nv_arch);                                               \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_fp16_ic1024_oc1024_k3_dil2_##nv_arch);                                         \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_fp16_ic1024_oc2048_k3_##nv_arch);                                              \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_dgrad_bf16_ic3_oc8_k3_##nv_arch);                                              \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_dgrad_bf16_ic3_oc8_k3_dil2_##nv_arch);                                         \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_dgrad_bf16_ic32_oc64_k3_##nv_arch);                                            \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_dgrad_bf16_ic768_oc768_k3_dil2_##nv_arch);                                     \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_dgrad_bf16_ic768_oc1536_k3_##nv_arch);                                         \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_dgrad_bf16_ic1024_oc1024_k3_dil2_##nv_arch);                                   \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_dgrad_bf16_ic1024_oc2048_k3_##nv_arch);                                        \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_dgrad_fp16_ic3_oc8_k3_##nv_arch);                                              \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_dgrad_fp16_ic3_oc8_k3_dil2_##nv_arch);                                         \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_dgrad_fp16_ic32_oc64_k3_##nv_arch);                                            \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_dgrad_fp16_ic768_oc768_k3_dil2_##nv_arch);                                     \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_dgrad_fp16_ic768_oc1536_k3_##nv_arch);                                         \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_dgrad_fp16_ic1024_oc1024_k3_dil2_##nv_arch);                                   \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_dgrad_fp16_ic1024_oc2048_k3_##nv_arch);                                        \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_wgrad_bf16_ic3_oc8_k3_##nv_arch);                                              \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_wgrad_bf16_ic3_oc8_k3_dil2_##nv_arch);                                         \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_wgrad_bf16_ic32_oc64_k3_##nv_arch);                                            \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_wgrad_bf16_ic768_oc768_k3_dil2_##nv_arch);                                     \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_wgrad_bf16_ic768_oc1536_k3_##nv_arch);                                         \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_wgrad_bf16_ic1024_oc1024_k3_dil2_##nv_arch);                                   \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_wgrad_bf16_ic1024_oc2048_k3_##nv_arch);                                        \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_wgrad_fp16_ic3_oc8_k3_##nv_arch);                                              \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_wgrad_fp16_ic3_oc8_k3_dil2_##nv_arch);                                         \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_wgrad_fp16_ic32_oc64_k3_##nv_arch);                                            \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_wgrad_fp16_ic768_oc768_k3_dil2_##nv_arch);                                     \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_wgrad_fp16_ic768_oc1536_k3_##nv_arch);                                         \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_wgrad_fp16_ic1024_oc1024_k3_dil2_##nv_arch);                                   \
    DECLARE_CONV2D_KERNEL_BINARY(conv2d_wgrad_fp16_ic1024_oc2048_k3_##nv_arch);                                        \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_cutlass_bf16_stride1_pad1_dil1_ic32_oc64_##nv_arch);                   \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_cutlass_bf16_stride1_pad1_dil1_ic768_oc1536_##nv_arch);                \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_cutlass_bf16_stride1_pad1_dil1_ic1024_oc2048_##nv_arch);               \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_cutlass_bf16_stride1_pad2_dil2_ic768_oc768_##nv_arch);                 \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_cutlass_bf16_stride1_pad2_dil2_ic1024_oc1024_##nv_arch);               \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_cutlass_fp16_stride1_pad1_dil1_ic32_oc64_##nv_arch);                   \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_cutlass_fp16_stride1_pad1_dil1_ic768_oc1536_##nv_arch);                \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_cutlass_fp16_stride1_pad1_dil1_ic1024_oc2048_##nv_arch);               \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_cutlass_fp16_stride1_pad2_dil2_ic768_oc768_##nv_arch);                 \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_cutlass_fp16_stride1_pad2_dil2_ic1024_oc1024_##nv_arch);               \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_cutlass_fp16_acc_fp16_stride1_pad1_dil1_ic32_oc64_##nv_arch);          \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_cutlass_fp16_acc_fp16_stride1_pad1_dil1_ic768_oc1536_##nv_arch);       \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_cutlass_fp16_acc_fp16_stride1_pad1_dil1_ic1024_oc2048_##nv_arch);      \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_cutlass_fp16_acc_fp16_stride1_pad2_dil2_ic768_oc768_##nv_arch);        \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_cutlass_fp16_acc_fp16_stride1_pad2_dil2_ic1024_oc1024_##nv_arch);      \
    DECLARE_CONV2D_BIN_KERNELS(nv_arch);                                                                              \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_dgrad_cutlass_bf16_stride1_pad1_dil1_ic32_oc64_##nv_arch);             \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_dgrad_cutlass_bf16_stride1_pad1_dil1_ic768_oc1536_##nv_arch);          \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_dgrad_cutlass_bf16_stride1_pad1_dil1_ic1024_oc2048_##nv_arch);         \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_dgrad_cutlass_bf16_stride1_pad2_dil2_ic768_oc768_##nv_arch);           \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_dgrad_cutlass_bf16_stride1_pad2_dil2_ic1024_oc1024_##nv_arch);         \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_dgrad_cutlass_fp16_stride1_pad1_dil1_ic32_oc64_##nv_arch);             \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_dgrad_cutlass_fp16_stride1_pad1_dil1_ic768_oc1536_##nv_arch);          \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_dgrad_cutlass_fp16_stride1_pad1_dil1_ic1024_oc2048_##nv_arch);         \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_dgrad_cutlass_fp16_stride1_pad2_dil2_ic768_oc768_##nv_arch);           \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_dgrad_cutlass_fp16_stride1_pad2_dil2_ic1024_oc1024_##nv_arch);         \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_wgrad_cutlass_bf16_stride1_pad1_dil1_ic32_oc64_##nv_arch);             \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_wgrad_cutlass_bf16_stride1_pad1_dil1_ic768_oc1536_##nv_arch);          \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_wgrad_cutlass_bf16_stride1_pad1_dil1_ic1024_oc2048_##nv_arch);         \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_wgrad_cutlass_bf16_stride1_pad2_dil2_ic768_oc768_##nv_arch);           \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_wgrad_cutlass_bf16_stride1_pad2_dil2_ic1024_oc1024_##nv_arch);         \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_wgrad_cutlass_fp16_stride1_pad1_dil1_ic32_oc64_##nv_arch);             \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_wgrad_cutlass_fp16_stride1_pad1_dil1_ic768_oc1536_##nv_arch);          \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_wgrad_cutlass_fp16_stride1_pad1_dil1_ic1024_oc2048_##nv_arch);         \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_wgrad_cutlass_fp16_stride1_pad2_dil2_ic768_oc768_##nv_arch);           \
    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(conv2d_wgrad_cutlass_fp16_stride1_pad2_dil2_ic1024_oc1024_##nv_arch);         \
    DECLARE_ELEMWISE_KERNEL_BINARY(lstm_cell_fwd_out_fp16_##nv_arch);                                                  \
    DECLARE_ELEMWISE_KERNEL_BINARY(lstm_cell_fwd_fp32_state_out_fp16_##nv_arch);                                       \
    DECLARE_ELEMWISE_KERNEL_BINARY(lstm_cell_fwd_fp32_state_out_bf16_##nv_arch);

#define DECLARE_KERNEL_ALIAS(name, new_name) inline constexpr auto &k##new_name = k##name

#define DECLARE_KERNEL_ALIASES(nv_arch)                                                                                \
    DECLARE_KERNEL_ALIAS(addmm_gelu_bf16_##nv_arch, addmm_gelu_bf16);                                                  \
    DECLARE_KERNEL_ALIAS(addmm_gelu_bf16_##nv_arch, addmm_gelu_bf16_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_bf16_cacc_##nv_arch, addmm_gelu_bf16_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_bf16_unaligned_##nv_arch, addmm_gelu_bf16_unaligned);                              \
    DECLARE_KERNEL_ALIAS(addmm_gelu_bf16_unaligned_##nv_arch, addmm_gelu_bf16_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_bf16_unaligned_cacc_##nv_arch, addmm_gelu_bf16_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_fp16_##nv_arch, addmm_gelu_fp16);                                                  \
    DECLARE_KERNEL_ALIAS(addmm_gelu_fp16_##nv_arch, addmm_gelu_fp16_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_fp16_cacc_##nv_arch, addmm_gelu_fp16_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_fp16_unaligned_##nv_arch, addmm_gelu_fp16_unaligned);                              \
    DECLARE_KERNEL_ALIAS(addmm_gelu_fp16_unaligned_##nv_arch, addmm_gelu_fp16_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_fp16_unaligned_cacc_##nv_arch, addmm_gelu_fp16_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_fp16_acc_fp16_##nv_arch, addmm_gelu_fp16_acc_fp16);                                \
    DECLARE_KERNEL_ALIAS(addmm_gelu_fp16_acc_fp16_##nv_arch, addmm_gelu_fp16_acc_fp16_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_fp16_acc_fp16_cacc_##nv_arch, addmm_gelu_fp16_acc_fp16_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_fp16_acc_fp16_unaligned_##nv_arch, addmm_gelu_fp16_acc_fp16_unaligned);            \
    DECLARE_KERNEL_ALIAS(addmm_gelu_fp16_acc_fp16_unaligned_##nv_arch, addmm_gelu_fp16_acc_fp16_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_fp16_acc_fp16_unaligned_cacc_##nv_arch, addmm_gelu_fp16_acc_fp16_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_bf16_##nv_arch, addmm_gelu_preact_bf16);                                    \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_bf16_##nv_arch, addmm_gelu_preact_bf16_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_bf16_cacc_##nv_arch, addmm_gelu_preact_bf16_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_bf16_unaligned_##nv_arch, addmm_gelu_preact_bf16_unaligned);                \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_bf16_unaligned_##nv_arch, addmm_gelu_preact_bf16_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_bf16_unaligned_cacc_##nv_arch, addmm_gelu_preact_bf16_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_fp16_##nv_arch, addmm_gelu_preact_fp16);                                    \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_fp16_##nv_arch, addmm_gelu_preact_fp16_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_fp16_cacc_##nv_arch, addmm_gelu_preact_fp16_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_fp16_unaligned_##nv_arch, addmm_gelu_preact_fp16_unaligned);                \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_fp16_unaligned_##nv_arch, addmm_gelu_preact_fp16_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_fp16_unaligned_cacc_##nv_arch, addmm_gelu_preact_fp16_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_fp16_acc_fp16_##nv_arch, addmm_gelu_preact_fp16_acc_fp16);                  \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_fp16_acc_fp16_##nv_arch, addmm_gelu_preact_fp16_acc_fp16_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_fp16_acc_fp16_cacc_##nv_arch, addmm_gelu_preact_fp16_acc_fp16_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_fp16_acc_fp16_unaligned_##nv_arch, addmm_gelu_preact_fp16_acc_fp16_unaligned); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_fp16_acc_fp16_unaligned_##nv_arch, addmm_gelu_preact_fp16_acc_fp16_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_fp16_acc_fp16_unaligned_cacc_##nv_arch, addmm_gelu_preact_fp16_acc_fp16_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(fill_constant_bf16_##nv_arch, fill_constant_bf16);                                            \
    DECLARE_KERNEL_ALIAS(fill_constant_fp16_##nv_arch, fill_constant_fp16);                                            \
    DECLARE_KERNEL_ALIAS(fill_constant_fp32_##nv_arch, fill_constant_fp32);                                            \
    DECLARE_KERNEL_ALIAS(fill_uniform_bf16_##nv_arch, fill_uniform_bf16);                                              \
    DECLARE_KERNEL_ALIAS(fill_uniform_fp16_##nv_arch, fill_uniform_fp16);                                              \
    DECLARE_KERNEL_ALIAS(fill_uniform_fp32_##nv_arch, fill_uniform_fp32);                                              \
    DECLARE_KERNEL_ALIAS(fill_zeros_##nv_arch, fill_zeros);                                                            \
    DECLARE_KERNEL_ALIAS(adamw_step_bf16_##nv_arch, adamw_step_bf16);                                                  \
    DECLARE_KERNEL_ALIAS(adamw_step_fp16_##nv_arch, adamw_step_fp16);                                                  \
    DECLARE_KERNEL_ALIAS(adamw_step_fp32_##nv_arch, adamw_step_fp32);                                                  \
    DECLARE_KERNEL_ALIAS(sgd_step_bf16_##nv_arch, sgd_step_bf16);                                                      \
    DECLARE_KERNEL_ALIAS(sgd_step_fp16_##nv_arch, sgd_step_fp16);                                                      \
    DECLARE_KERNEL_ALIAS(sgd_step_fp32_##nv_arch, sgd_step_fp32);                                                      \
    DECLARE_KERNEL_ALIAS(build_cell_embeds_bf16_##nv_arch, build_cell_embeds_bf16);                                    \
    DECLARE_KERNEL_ALIAS(build_cell_embeds_fp16_##nv_arch, build_cell_embeds_fp16);                                    \
    DECLARE_KERNEL_ALIAS(build_cell_embeds_bwd_cp_bf16_##nv_arch, build_cell_embeds_bwd_cp_bf16);                      \
    DECLARE_KERNEL_ALIAS(build_cell_embeds_bwd_cp_fp16_##nv_arch, build_cell_embeds_bwd_cp_fp16);                      \
    DECLARE_KERNEL_ALIAS(build_cell_embeds_bwd_cp_bf16_out_fp32_##nv_arch, build_cell_embeds_bwd_cp_bf16_out_fp32);    \
    DECLARE_KERNEL_ALIAS(build_cell_embeds_bwd_cp_fp16_out_fp32_##nv_arch, build_cell_embeds_bwd_cp_fp16_out_fp32);    \
    DECLARE_KERNEL_ALIAS(build_cell_embeds_bwd_color_bf16_##nv_arch, build_cell_embeds_bwd_color_bf16);                \
    DECLARE_KERNEL_ALIAS(build_cell_embeds_bwd_color_fp16_##nv_arch, build_cell_embeds_bwd_color_fp16);                \
    DECLARE_KERNEL_ALIAS(build_cell_embeds_bwd_color_bf16_out_fp32_##nv_arch, build_cell_embeds_bwd_color_bf16_out_fp32); \
    DECLARE_KERNEL_ALIAS(build_cell_embeds_bwd_color_fp16_out_fp32_##nv_arch, build_cell_embeds_bwd_color_fp16_out_fp32); \
    DECLARE_KERNEL_ALIAS(build_cell_embeds_bwd_pos_bf16_##nv_arch, build_cell_embeds_bwd_pos_bf16);                    \
    DECLARE_KERNEL_ALIAS(build_cell_embeds_bwd_pos_fp16_##nv_arch, build_cell_embeds_bwd_pos_fp16);                    \
    DECLARE_KERNEL_ALIAS(build_cell_embeds_bwd_pos_bf16_out_fp32_##nv_arch, build_cell_embeds_bwd_pos_bf16_out_fp32);  \
    DECLARE_KERNEL_ALIAS(build_cell_embeds_bwd_pos_fp16_out_fp32_##nv_arch, build_cell_embeds_bwd_pos_fp16_out_fp32);  \
    DECLARE_KERNEL_ALIAS(fill_normal_bf16_##nv_arch, fill_normal_bf16);                                                \
    DECLARE_KERNEL_ALIAS(fill_normal_fp16_##nv_arch, fill_normal_fp16);                                                \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fwd_hs128_bf16_##nv_arch, mha_full_attn_fwd_hs128_bf16);                        \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fwd_hs128_bf16_nolse_##nv_arch, mha_full_attn_fwd_hs128_bf16_nolse);            \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fwd_hs128_fp16_##nv_arch, mha_full_attn_fwd_hs128_fp16);                        \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fwd_hs128_fp16_nolse_##nv_arch, mha_full_attn_fwd_hs128_fp16_nolse);            \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fwd_hs128_fp16_acc_fp16_##nv_arch, mha_full_attn_fwd_hs128_fp16_acc_fp16);      \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fwd_hs128_fp16_acc_fp16_nolse_##nv_arch,                                        \
                         mha_full_attn_fwd_hs128_fp16_acc_fp16_nolse);                                                 \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fwd_hs128_bf16_uneven_##nv_arch, mha_full_attn_fwd_hs128_bf16_uneven);          \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fwd_hs128_bf16_uneven_nolse_##nv_arch,                                          \
                         mha_full_attn_fwd_hs128_bf16_uneven_nolse);                                                    \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fwd_hs128_fp16_uneven_##nv_arch, mha_full_attn_fwd_hs128_fp16_uneven);          \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fwd_hs128_fp16_uneven_nolse_##nv_arch,                                          \
                         mha_full_attn_fwd_hs128_fp16_uneven_nolse);                                                    \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fwd_hs128_fp16_acc_fp16_uneven_##nv_arch,                                       \
                         mha_full_attn_fwd_hs128_fp16_acc_fp16_uneven);                                                 \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fwd_hs128_fp16_acc_fp16_uneven_nolse_##nv_arch,                                 \
                         mha_full_attn_fwd_hs128_fp16_acc_fp16_uneven_nolse);                                           \
    DECLARE_KERNEL_ALIAS(mha_full_attn_bwd_pre_hs128_bf16_##nv_arch, mha_full_attn_bwd_pre_hs128_bf16);                \
    DECLARE_KERNEL_ALIAS(mha_full_attn_bwd_pre_hs128_fp16_##nv_arch, mha_full_attn_bwd_pre_hs128_fp16);                \
    DECLARE_KERNEL_ALIAS(mha_full_attn_bwd_hs128_bf16_##nv_arch, mha_full_attn_bwd_hs128_bf16);                        \
    DECLARE_KERNEL_ALIAS(mha_full_attn_bwd_hs128_fp16_##nv_arch, mha_full_attn_bwd_hs128_fp16);                        \
    DECLARE_KERNEL_ALIAS(mha_full_attn_bwd_hs128_bf16_uneven_##nv_arch, mha_full_attn_bwd_hs128_bf16_uneven);          \
    DECLARE_KERNEL_ALIAS(mha_full_attn_bwd_hs128_fp16_uneven_##nv_arch, mha_full_attn_bwd_hs128_fp16_uneven);          \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fa_fwd_hs128_bf16_##nv_arch, mha_full_attn_fa_fwd_hs128_bf16);                  \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fa_fwd_hs128_bf16_nolse_##nv_arch, mha_full_attn_fa_fwd_hs128_bf16_nolse);      \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fa_fwd_hs128_bf16_even_##nv_arch, mha_full_attn_fa_fwd_hs128_bf16_even);        \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fa_fwd_hs128_bf16_even_nolse_##nv_arch,                                         \
                         mha_full_attn_fa_fwd_hs128_bf16_even_nolse);                                                   \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fa_fwd_hs128_fp16_##nv_arch, mha_full_attn_fa_fwd_hs128_fp16);                  \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fa_fwd_hs128_fp16_nolse_##nv_arch, mha_full_attn_fa_fwd_hs128_fp16_nolse);      \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fa_fwd_hs128_fp16_even_##nv_arch, mha_full_attn_fa_fwd_hs128_fp16_even);        \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fa_fwd_hs128_fp16_even_nolse_##nv_arch,                                         \
                         mha_full_attn_fa_fwd_hs128_fp16_even_nolse);                                                   \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fa_fwd_hs128_fp16_acc_fp16_##nv_arch,                                           \
                         mha_full_attn_fa_fwd_hs128_fp16_acc_fp16);                                                    \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fa_fwd_hs128_fp16_acc_fp16_nolse_##nv_arch,                                     \
                         mha_full_attn_fa_fwd_hs128_fp16_acc_fp16_nolse);                                               \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fa_fwd_hs128_fp16_acc_fp16_even_##nv_arch,                                      \
                         mha_full_attn_fa_fwd_hs128_fp16_acc_fp16_even);                                                \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fa_fwd_hs128_fp16_acc_fp16_even_nolse_##nv_arch,                                \
                         mha_full_attn_fa_fwd_hs128_fp16_acc_fp16_even_nolse);                                          \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fa_bwd_hs128_bf16_##nv_arch, mha_full_attn_fa_bwd_hs128_bf16);                  \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fa_bwd_hs128_fp16_##nv_arch, mha_full_attn_fa_bwd_hs128_fp16);                  \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fa_bwd_hs128_bf16_even_##nv_arch, mha_full_attn_fa_bwd_hs128_bf16_even);        \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fa_bwd_hs128_fp16_even_##nv_arch, mha_full_attn_fa_bwd_hs128_fp16_even);        \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fa_bwd_dot_do_o_hs128_bf16_##nv_arch,                                           \
                         mha_full_attn_fa_bwd_dot_do_o_hs128_bf16);                                                    \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fa_bwd_dot_do_o_hs128_fp16_##nv_arch,                                           \
                         mha_full_attn_fa_bwd_dot_do_o_hs128_fp16);                                                    \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fa_bwd_convert_dq_hs128_bf16_##nv_arch,                                         \
                         mha_full_attn_fa_bwd_convert_dq_hs128_bf16);                                                  \
    DECLARE_KERNEL_ALIAS(mha_full_attn_fa_bwd_convert_dq_hs128_fp16_##nv_arch,                                         \
                         mha_full_attn_fa_bwd_convert_dq_hs128_fp16);                                                  \
    DECLARE_KERNEL_ALIAS(mha_full_attn_cutlass_bf16_##nv_arch, mha_full_attn_cutlass_bf16);                            \
    DECLARE_KERNEL_ALIAS(mha_full_attn_cutlass_bf16_lse_##nv_arch, mha_full_attn_cutlass_bf16_lse);                    \
    DECLARE_KERNEL_ALIAS(mha_full_attn_cutlass_fp16_##nv_arch, mha_full_attn_cutlass_fp16);                            \
    DECLARE_KERNEL_ALIAS(mha_full_attn_cutlass_fp16_lse_##nv_arch, mha_full_attn_cutlass_fp16_lse);                    \
    DECLARE_KERNEL_ALIAS(mha_full_attn_bwd_cutlass_bf16_##nv_arch, mha_full_attn_bwd_cutlass_bf16);                    \
    DECLARE_KERNEL_ALIAS(mha_full_attn_bwd_cutlass_fp16_##nv_arch, mha_full_attn_bwd_cutlass_fp16);                    \
    DECLARE_KERNEL_ALIAS(layer_norm_fwd_bf16_##nv_arch, layer_norm_fwd_bf16);                                          \
    DECLARE_KERNEL_ALIAS(layer_norm_fwd_fp16_##nv_arch, layer_norm_fwd_fp16);                                          \
    DECLARE_KERNEL_ALIAS(rms_norm_fwd_bf16_##nv_arch, rms_norm_fwd_bf16);                                              \
    DECLARE_KERNEL_ALIAS(rms_norm_fwd_fp16_##nv_arch, rms_norm_fwd_fp16);                                              \
    DECLARE_KERNEL_ALIAS(rms_norm_bwd_bf16_##nv_arch, rms_norm_bwd_bf16);                                              \
    DECLARE_KERNEL_ALIAS(rms_norm_bwd_fp16_##nv_arch, rms_norm_bwd_fp16);                                              \
    DECLARE_KERNEL_ALIAS(contiguous_4d_bf16_##nv_arch, contiguous_4d_bf16);                                            \
    DECLARE_KERNEL_ALIAS(contiguous_4d_fp16_##nv_arch, contiguous_4d_fp16);                                            \
    DECLARE_KERNEL_ALIAS(contiguous_3d_bf16_##nv_arch, contiguous_3d_bf16);                                            \
    DECLARE_KERNEL_ALIAS(contiguous_3d_fp16_##nv_arch, contiguous_3d_fp16);                                            \
    DECLARE_KERNEL_ALIAS(device_copy_strided_2d_##nv_arch, device_copy_strided_2d);                                    \
    DECLARE_KERNEL_ALIAS(add_trailing_broadcast_bf16_##nv_arch, add_trailing_broadcast_bf16);                          \
    DECLARE_KERNEL_ALIAS(add_trailing_broadcast_fp16_##nv_arch, add_trailing_broadcast_fp16);                          \
    DECLARE_KERNEL_ALIAS(add_trailing_broadcast_fp32_##nv_arch, add_trailing_broadcast_fp32);                          \
    DECLARE_KERNEL_ALIAS(add_elementwise_bf16_##nv_arch, add_elementwise_bf16);                                        \
    DECLARE_KERNEL_ALIAS(add_elementwise_fp16_##nv_arch, add_elementwise_fp16);                                        \
    DECLARE_KERNEL_ALIAS(add_elementwise_fp32_##nv_arch, add_elementwise_fp32);                                        \
    DECLARE_KERNEL_ALIAS(add_elementwise_fp32_out_fp16_##nv_arch, add_elementwise_fp32_out_fp16);                      \
    DECLARE_KERNEL_ALIAS(mul_trailing_broadcast_bf16_##nv_arch, mul_trailing_broadcast_bf16);                          \
    DECLARE_KERNEL_ALIAS(mul_trailing_broadcast_fp16_##nv_arch, mul_trailing_broadcast_fp16);                          \
    DECLARE_KERNEL_ALIAS(mul_trailing_broadcast_fp32_##nv_arch, mul_trailing_broadcast_fp32);                          \
    DECLARE_KERNEL_ALIAS(mul_elementwise_bf16_##nv_arch, mul_elementwise_bf16);                                        \
    DECLARE_KERNEL_ALIAS(mul_elementwise_fp16_##nv_arch, mul_elementwise_fp16);                                        \
    DECLARE_KERNEL_ALIAS(mul_elementwise_fp32_##nv_arch, mul_elementwise_fp32);                                        \
    DECLARE_KERNEL_ALIAS(mul_scalar_bf16_##nv_arch, mul_scalar_bf16);                                                  \
    DECLARE_KERNEL_ALIAS(mul_scalar_fp16_##nv_arch, mul_scalar_fp16);                                                  \
    DECLARE_KERNEL_ALIAS(mul_scalar_fp32_##nv_arch, mul_scalar_fp32);                                                  \
    DECLARE_KERNEL_ALIAS(sqrt_elementwise_bf16_##nv_arch, sqrt_elementwise_bf16);                                      \
    DECLARE_KERNEL_ALIAS(sqrt_elementwise_fp16_##nv_arch, sqrt_elementwise_fp16);                                      \
    DECLARE_KERNEL_ALIAS(sqrt_elementwise_fp32_##nv_arch, sqrt_elementwise_fp32);                                      \
    DECLARE_KERNEL_ALIAS(div_elementwise_bf16_##nv_arch, div_elementwise_bf16);                                        \
    DECLARE_KERNEL_ALIAS(div_elementwise_fp16_##nv_arch, div_elementwise_fp16);                                        \
    DECLARE_KERNEL_ALIAS(div_elementwise_fp32_##nv_arch, div_elementwise_fp32);                                        \
    DECLARE_KERNEL_ALIAS(div_scalar_bf16_##nv_arch, div_scalar_bf16);                                                  \
    DECLARE_KERNEL_ALIAS(div_scalar_fp16_##nv_arch, div_scalar_fp16);                                                  \
    DECLARE_KERNEL_ALIAS(div_scalar_fp32_##nv_arch, div_scalar_fp32);                                                  \
    DECLARE_KERNEL_ALIAS(div_add_bf16_##nv_arch, div_add_bf16);                                                        \
    DECLARE_KERNEL_ALIAS(div_add_fp16_##nv_arch, div_add_fp16);                                                        \
    DECLARE_KERNEL_ALIAS(div_add_fp32_##nv_arch, div_add_fp32);                                                        \
    DECLARE_KERNEL_ALIAS(div_scalar_add_bf16_##nv_arch, div_scalar_add_bf16);                                          \
    DECLARE_KERNEL_ALIAS(div_scalar_add_fp16_##nv_arch, div_scalar_add_fp16);                                          \
    DECLARE_KERNEL_ALIAS(div_scalar_add_fp32_##nv_arch, div_scalar_add_fp32);                                          \
    DECLARE_KERNEL_ALIAS(div_scalar_add_broadcast_bf16_##nv_arch, div_scalar_add_broadcast_bf16);                    \
    DECLARE_KERNEL_ALIAS(div_scalar_add_broadcast_fp16_##nv_arch, div_scalar_add_broadcast_fp16);                    \
    DECLARE_KERNEL_ALIAS(div_scalar_add_broadcast_fp32_##nv_arch, div_scalar_add_broadcast_fp32);                    \
    DECLARE_KERNEL_ALIAS(act_gelu_elementwise_bf16_##nv_arch, act_gelu_elementwise_bf16);                              \
    DECLARE_KERNEL_ALIAS(act_gelu_elementwise_fp16_##nv_arch, act_gelu_elementwise_fp16);                              \
    DECLARE_KERNEL_ALIAS(act_gelu_bwd_elementwise_bf16_##nv_arch, act_gelu_bwd_elementwise_bf16);                      \
    DECLARE_KERNEL_ALIAS(act_gelu_bwd_elementwise_fp16_##nv_arch, act_gelu_bwd_elementwise_fp16);                      \
    DECLARE_KERNEL_ALIAS(act_relu_elementwise_bf16_##nv_arch, act_relu_elementwise_bf16);                              \
    DECLARE_KERNEL_ALIAS(act_relu_elementwise_fp16_##nv_arch, act_relu_elementwise_fp16);                              \
    DECLARE_KERNEL_ALIAS(cross_entropy_on_targets_bf16_##nv_arch, cross_entropy_on_targets_bf16);                      \
    DECLARE_KERNEL_ALIAS(cross_entropy_on_targets_fp16_##nv_arch, cross_entropy_on_targets_fp16);                      \
    DECLARE_KERNEL_ALIAS(cross_entropy_on_targets_add_fp32_bf16_##nv_arch, cross_entropy_on_targets_add_fp32_bf16);    \
    DECLARE_KERNEL_ALIAS(cross_entropy_on_targets_add_fp32_fp16_##nv_arch, cross_entropy_on_targets_add_fp32_fp16);    \
    DECLARE_KERNEL_ALIAS(cross_entropy_on_targets_bwd_bf16_##nv_arch, cross_entropy_on_targets_bwd_bf16);              \
    DECLARE_KERNEL_ALIAS(cross_entropy_on_targets_bwd_fp16_##nv_arch, cross_entropy_on_targets_bwd_fp16);              \
    DECLARE_KERNEL_ALIAS(lstm_cell_recompute_out_fp16_##nv_arch, lstm_cell_recompute_out_fp16);                        \
    DECLARE_KERNEL_ALIAS(lstm_cell_recompute_out_bf16_##nv_arch, lstm_cell_recompute_out_bf16);                        \
    DECLARE_KERNEL_ALIAS(lstm_cell_bwd_pointwise_out_fp16_##nv_arch, lstm_cell_bwd_pointwise_out_fp16);                \
    DECLARE_KERNEL_ALIAS(lstm_cell_bwd_pointwise_out_bf16_##nv_arch, lstm_cell_bwd_pointwise_out_bf16);                \
    DECLARE_KERNEL_ALIAS(mean_reduce_contiguous_bf16_##nv_arch, mean_reduce_contiguous_bf16);                          \
    DECLARE_KERNEL_ALIAS(mean_reduce_contiguous_fp16_##nv_arch, mean_reduce_contiguous_fp16);                          \
    DECLARE_KERNEL_ALIAS(mean_reduce_column_tiled_bf16_##nv_arch, mean_reduce_column_tiled_bf16);                      \
    DECLARE_KERNEL_ALIAS(mean_reduce_column_tiled_fp16_##nv_arch, mean_reduce_column_tiled_fp16);                      \
    DECLARE_KERNEL_ALIAS(sum_reduce_contiguous_bf16_out_fp32_##nv_arch, sum_reduce_contiguous_bf16_out_fp32);          \
    DECLARE_KERNEL_ALIAS(sum_reduce_contiguous_fp16_out_fp32_##nv_arch, sum_reduce_contiguous_fp16_out_fp32);          \
    DECLARE_KERNEL_ALIAS(sum_reduce_contiguous_fp32_out_fp32_##nv_arch, sum_reduce_contiguous_fp32_out_fp32);          \
    DECLARE_KERNEL_ALIAS(sum_reduce_column_tiled_bf16_out_fp32_##nv_arch, sum_reduce_column_tiled_bf16_out_fp32);      \
    DECLARE_KERNEL_ALIAS(sum_reduce_column_tiled_fp16_out_fp32_##nv_arch, sum_reduce_column_tiled_fp16_out_fp32);      \
    DECLARE_KERNEL_ALIAS(sum_reduce_column_tiled_fp32_out_fp32_##nv_arch, sum_reduce_column_tiled_fp32_out_fp32);      \
    DECLARE_KERNEL_ALIAS(sum_reduce_column_split_partials_bf16_out_fp32_##nv_arch,                                    \
                         sum_reduce_column_split_partials_bf16_out_fp32);                                             \
    DECLARE_KERNEL_ALIAS(sum_reduce_column_split_partials_fp16_out_fp32_##nv_arch,                                    \
                         sum_reduce_column_split_partials_fp16_out_fp32);                                             \
    DECLARE_KERNEL_ALIAS(sum_reduce_column_split_partials_fp32_out_fp32_##nv_arch,                                    \
                         sum_reduce_column_split_partials_fp32_out_fp32);                                             \
    DECLARE_KERNEL_ALIAS(mul_reduce_contiguous_bf16_out_fp32_##nv_arch, mul_reduce_contiguous_bf16_out_fp32);          \
    DECLARE_KERNEL_ALIAS(mul_reduce_contiguous_fp16_out_fp32_##nv_arch, mul_reduce_contiguous_fp16_out_fp32);          \
    DECLARE_KERNEL_ALIAS(mul_reduce_column_tiled_bf16_out_fp32_##nv_arch, mul_reduce_column_tiled_bf16_out_fp32);      \
    DECLARE_KERNEL_ALIAS(mul_reduce_column_tiled_fp16_out_fp32_##nv_arch, mul_reduce_column_tiled_fp16_out_fp32);      \
    DECLARE_KERNEL_ALIAS(mul_reduce_column_split_partials_bf16_out_fp32_##nv_arch,                                    \
                         mul_reduce_column_split_partials_bf16_out_fp32);                                             \
    DECLARE_KERNEL_ALIAS(mul_reduce_column_split_partials_fp16_out_fp32_##nv_arch,                                    \
                         mul_reduce_column_split_partials_fp16_out_fp32);                                             \
    DECLARE_KERNEL_ALIAS(reduce_sum_partial_fp32_##nv_arch, reduce_sum_partial_fp32);                                  \
    DECLARE_KERNEL_ALIAS(avg_pool1d_bf16_##nv_arch, avg_pool1d_bf16);                                                  \
    DECLARE_KERNEL_ALIAS(avg_pool1d_fp16_##nv_arch, avg_pool1d_fp16);                                                  \
    DECLARE_KERNEL_ALIAS(avg_pool2d_bf16_##nv_arch, avg_pool2d_bf16);                                                  \
    DECLARE_KERNEL_ALIAS(avg_pool2d_fp16_##nv_arch, avg_pool2d_fp16);                                                  \
    DECLARE_KERNEL_ALIAS(avg_pool2d_nhwc_2x2_s2_bf16_##nv_arch, avg_pool2d_nhwc_2x2_s2_bf16);                          \
    DECLARE_KERNEL_ALIAS(avg_pool2d_nhwc_2x2_s2_fp16_##nv_arch, avg_pool2d_nhwc_2x2_s2_fp16);                          \
    DECLARE_KERNEL_ALIAS(avg_pool2d_bwd_bf16_##nv_arch, avg_pool2d_bwd_bf16);                                          \
    DECLARE_KERNEL_ALIAS(avg_pool2d_bwd_fp16_##nv_arch, avg_pool2d_bwd_fp16);                                          \
    DECLARE_KERNEL_ALIAS(avg_pool2d_bwd_noaccum_bf16_##nv_arch, avg_pool2d_bwd_noaccum_bf16);                          \
    DECLARE_KERNEL_ALIAS(avg_pool2d_bwd_noaccum_fp16_##nv_arch, avg_pool2d_bwd_noaccum_fp16);                          \
    DECLARE_KERNEL_ALIAS(avg_pool2d_bwd_noaccum_nhwc_2x2_s2_bf16_##nv_arch,                                            \
                         avg_pool2d_bwd_noaccum_nhwc_2x2_s2_bf16);                                                     \
    DECLARE_KERNEL_ALIAS(avg_pool2d_bwd_noaccum_nhwc_2x2_s2_fp16_##nv_arch,                                            \
                         avg_pool2d_bwd_noaccum_nhwc_2x2_s2_fp16);                                                     \
    DECLARE_KERNEL_ALIAS(addmm_bf16_##nv_arch, addmm_bf16);                                                            \
    DECLARE_KERNEL_ALIAS(addmm_bf16_##nv_arch, addmm_bf16_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_bf16_cacc_##nv_arch, addmm_bf16_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_bf16_unaligned_##nv_arch, addmm_bf16_unaligned);                                        \
    DECLARE_KERNEL_ALIAS(addmm_bf16_unaligned_##nv_arch, addmm_bf16_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_bf16_unaligned_cacc_##nv_arch, addmm_bf16_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_fp16_##nv_arch, addmm_fp16);                                                            \
    DECLARE_KERNEL_ALIAS(addmm_fp16_##nv_arch, addmm_fp16_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_fp16_cacc_##nv_arch, addmm_fp16_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_fp16_unaligned_##nv_arch, addmm_fp16_unaligned);                                        \
    DECLARE_KERNEL_ALIAS(addmm_fp16_unaligned_##nv_arch, addmm_fp16_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_fp16_unaligned_cacc_##nv_arch, addmm_fp16_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_fp16_acc_fp16_##nv_arch, addmm_fp16_acc_fp16);                                          \
    DECLARE_KERNEL_ALIAS(addmm_fp16_acc_fp16_##nv_arch, addmm_fp16_acc_fp16_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_fp16_acc_fp16_cacc_##nv_arch, addmm_fp16_acc_fp16_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_fp16_acc_fp16_unaligned_##nv_arch, addmm_fp16_acc_fp16_unaligned);                      \
    DECLARE_KERNEL_ALIAS(addmm_fp16_acc_fp16_unaligned_##nv_arch, addmm_fp16_acc_fp16_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_fp16_acc_fp16_unaligned_cacc_##nv_arch, addmm_fp16_acc_fp16_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_fp16_out_fp32_##nv_arch, addmm_fp16_out_fp32);                                          \
    DECLARE_KERNEL_ALIAS(addmm_fp16_out_fp32_##nv_arch, addmm_fp16_out_fp32_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_fp16_out_fp32_cacc_##nv_arch, addmm_fp16_out_fp32_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_fp16_out_fp32_unaligned_##nv_arch, addmm_fp16_out_fp32_unaligned);                      \
    DECLARE_KERNEL_ALIAS(addmm_fp16_out_fp32_unaligned_##nv_arch, addmm_fp16_out_fp32_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_fp16_out_fp32_unaligned_cacc_##nv_arch, addmm_fp16_out_fp32_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_fp16_acc_out_fp32_##nv_arch, addmm_fp16_acc_out_fp32);                                  \
    DECLARE_KERNEL_ALIAS(addmm_fp16_acc_out_fp32_##nv_arch, addmm_fp16_acc_out_fp32_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_fp16_acc_out_fp32_cacc_##nv_arch, addmm_fp16_acc_out_fp32_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_fp16_acc_out_fp32_unaligned_##nv_arch, addmm_fp16_acc_out_fp32_unaligned);              \
    DECLARE_KERNEL_ALIAS(addmm_fp16_acc_out_fp32_unaligned_##nv_arch, addmm_fp16_acc_out_fp32_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(addmm_fp16_acc_out_fp32_unaligned_cacc_##nv_arch, addmm_fp16_acc_out_fp32_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_##nv_arch, matmul_bf16);                                                          \
    DECLARE_KERNEL_ALIAS(matmul_bf16_##nv_arch, matmul_bf16_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_ta_##nv_arch, matmul_bf16_ta);                                                    \
    DECLARE_KERNEL_ALIAS(matmul_bf16_ta_##nv_arch, matmul_bf16_ta_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_tb_##nv_arch, matmul_bf16_tb);                                                    \
    DECLARE_KERNEL_ALIAS(matmul_bf16_tb_##nv_arch, matmul_bf16_tb_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_tab_##nv_arch, matmul_bf16_tab);                                                  \
    DECLARE_KERNEL_ALIAS(matmul_bf16_tab_##nv_arch, matmul_bf16_tab_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_cacc_##nv_arch, matmul_bf16_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_ta_cacc_##nv_arch, matmul_bf16_ta_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_tb_cacc_##nv_arch, matmul_bf16_tb_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_tab_cacc_##nv_arch, matmul_bf16_tab_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_unaligned_##nv_arch, matmul_bf16_unaligned);                                      \
    DECLARE_KERNEL_ALIAS(matmul_bf16_unaligned_##nv_arch, matmul_bf16_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_ta_unaligned_##nv_arch, matmul_bf16_ta_unaligned);                                \
    DECLARE_KERNEL_ALIAS(matmul_bf16_ta_unaligned_##nv_arch, matmul_bf16_ta_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_tb_unaligned_##nv_arch, matmul_bf16_tb_unaligned);                                \
    DECLARE_KERNEL_ALIAS(matmul_bf16_tb_unaligned_##nv_arch, matmul_bf16_tb_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_tab_unaligned_##nv_arch, matmul_bf16_tab_unaligned);                              \
    DECLARE_KERNEL_ALIAS(matmul_bf16_tab_unaligned_##nv_arch, matmul_bf16_tab_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_unaligned_cacc_##nv_arch, matmul_bf16_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_ta_unaligned_cacc_##nv_arch, matmul_bf16_ta_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_tb_unaligned_cacc_##nv_arch, matmul_bf16_tb_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_tab_unaligned_cacc_##nv_arch, matmul_bf16_tab_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_##nv_arch, matmul_fp16);                                                          \
    DECLARE_KERNEL_ALIAS(matmul_fp16_##nv_arch, matmul_fp16_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_ta_##nv_arch, matmul_fp16_ta);                                                    \
    DECLARE_KERNEL_ALIAS(matmul_fp16_ta_##nv_arch, matmul_fp16_ta_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_tb_##nv_arch, matmul_fp16_tb);                                                    \
    DECLARE_KERNEL_ALIAS(matmul_fp16_tb_##nv_arch, matmul_fp16_tb_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_tab_##nv_arch, matmul_fp16_tab);                                                  \
    DECLARE_KERNEL_ALIAS(matmul_fp16_tab_##nv_arch, matmul_fp16_tab_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_cacc_##nv_arch, matmul_fp16_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_ta_cacc_##nv_arch, matmul_fp16_ta_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_tb_cacc_##nv_arch, matmul_fp16_tb_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_tab_cacc_##nv_arch, matmul_fp16_tab_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_unaligned_##nv_arch, matmul_fp16_unaligned);                                      \
    DECLARE_KERNEL_ALIAS(matmul_fp16_unaligned_##nv_arch, matmul_fp16_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_ta_unaligned_##nv_arch, matmul_fp16_ta_unaligned);                                \
    DECLARE_KERNEL_ALIAS(matmul_fp16_ta_unaligned_##nv_arch, matmul_fp16_ta_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_tb_unaligned_##nv_arch, matmul_fp16_tb_unaligned);                                \
    DECLARE_KERNEL_ALIAS(matmul_fp16_tb_unaligned_##nv_arch, matmul_fp16_tb_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_tab_unaligned_##nv_arch, matmul_fp16_tab_unaligned);                              \
    DECLARE_KERNEL_ALIAS(matmul_fp16_tab_unaligned_##nv_arch, matmul_fp16_tab_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_unaligned_cacc_##nv_arch, matmul_fp16_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_ta_unaligned_cacc_##nv_arch, matmul_fp16_ta_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_tb_unaligned_cacc_##nv_arch, matmul_fp16_tb_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_tab_unaligned_cacc_##nv_arch, matmul_fp16_tab_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_fp16_##nv_arch, matmul_fp16_acc_fp16);                                        \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_fp16_##nv_arch, matmul_fp16_acc_fp16_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_fp16_cacc_##nv_arch, matmul_fp16_acc_fp16_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_fp16_unaligned_##nv_arch, matmul_fp16_acc_fp16_unaligned);                    \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_fp16_unaligned_##nv_arch, matmul_fp16_acc_fp16_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_fp16_unaligned_cacc_##nv_arch, matmul_fp16_acc_fp16_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_##nv_arch, matmul_bf16_out_fp32);                                        \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_##nv_arch, matmul_bf16_out_fp32_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_cacc_##nv_arch, matmul_bf16_out_fp32_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_unaligned_##nv_arch, matmul_bf16_out_fp32_unaligned);                    \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_unaligned_##nv_arch, matmul_bf16_out_fp32_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_unaligned_cacc_##nv_arch, matmul_bf16_out_fp32_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_ta_##nv_arch, matmul_bf16_out_fp32_ta);                                  \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_ta_##nv_arch, matmul_bf16_out_fp32_ta_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_tb_##nv_arch, matmul_bf16_out_fp32_tb);                                  \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_tb_##nv_arch, matmul_bf16_out_fp32_tb_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_tab_##nv_arch, matmul_bf16_out_fp32_tab);                                \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_tab_##nv_arch, matmul_bf16_out_fp32_tab_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_ta_cacc_##nv_arch, matmul_bf16_out_fp32_ta_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_tb_cacc_##nv_arch, matmul_bf16_out_fp32_tb_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_tab_cacc_##nv_arch, matmul_bf16_out_fp32_tab_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_ta_unaligned_##nv_arch, matmul_bf16_out_fp32_ta_unaligned);              \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_ta_unaligned_##nv_arch, matmul_bf16_out_fp32_ta_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_tb_unaligned_##nv_arch, matmul_bf16_out_fp32_tb_unaligned);              \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_tb_unaligned_##nv_arch, matmul_bf16_out_fp32_tb_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_tab_unaligned_##nv_arch, matmul_bf16_out_fp32_tab_unaligned);            \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_tab_unaligned_##nv_arch, matmul_bf16_out_fp32_tab_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_ta_unaligned_cacc_##nv_arch, matmul_bf16_out_fp32_ta_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_tb_unaligned_cacc_##nv_arch, matmul_bf16_out_fp32_tb_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_bf16_out_fp32_tab_unaligned_cacc_##nv_arch, matmul_bf16_out_fp32_tab_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_##nv_arch, matmul_fp16_out_fp32);                                        \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_##nv_arch, matmul_fp16_out_fp32_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_cacc_##nv_arch, matmul_fp16_out_fp32_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_unaligned_##nv_arch, matmul_fp16_out_fp32_unaligned);                    \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_unaligned_##nv_arch, matmul_fp16_out_fp32_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_unaligned_cacc_##nv_arch, matmul_fp16_out_fp32_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_ta_##nv_arch, matmul_fp16_out_fp32_ta);                                  \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_ta_##nv_arch, matmul_fp16_out_fp32_ta_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_tb_##nv_arch, matmul_fp16_out_fp32_tb);                                  \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_tb_##nv_arch, matmul_fp16_out_fp32_tb_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_tab_##nv_arch, matmul_fp16_out_fp32_tab);                                \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_tab_##nv_arch, matmul_fp16_out_fp32_tab_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_ta_cacc_##nv_arch, matmul_fp16_out_fp32_ta_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_tb_cacc_##nv_arch, matmul_fp16_out_fp32_tb_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_tab_cacc_##nv_arch, matmul_fp16_out_fp32_tab_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_ta_unaligned_##nv_arch, matmul_fp16_out_fp32_ta_unaligned);              \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_ta_unaligned_##nv_arch, matmul_fp16_out_fp32_ta_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_tb_unaligned_##nv_arch, matmul_fp16_out_fp32_tb_unaligned);              \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_tb_unaligned_##nv_arch, matmul_fp16_out_fp32_tb_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_tab_unaligned_##nv_arch, matmul_fp16_out_fp32_tab_unaligned);            \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_tab_unaligned_##nv_arch, matmul_fp16_out_fp32_tab_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_ta_unaligned_cacc_##nv_arch, matmul_fp16_out_fp32_ta_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_tb_unaligned_cacc_##nv_arch, matmul_fp16_out_fp32_tb_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_out_fp32_tab_unaligned_cacc_##nv_arch, matmul_fp16_out_fp32_tab_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_##nv_arch, matmul_fp16_acc_out_fp32);                                \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_##nv_arch, matmul_fp16_acc_out_fp32_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_cacc_##nv_arch, matmul_fp16_acc_out_fp32_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_unaligned_##nv_arch, matmul_fp16_acc_out_fp32_unaligned);            \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_unaligned_##nv_arch, matmul_fp16_acc_out_fp32_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_unaligned_cacc_##nv_arch, matmul_fp16_acc_out_fp32_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_ta_##nv_arch, matmul_fp16_acc_out_fp32_ta);                          \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_ta_##nv_arch, matmul_fp16_acc_out_fp32_ta_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_tb_##nv_arch, matmul_fp16_acc_out_fp32_tb);                          \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_tb_##nv_arch, matmul_fp16_acc_out_fp32_tb_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_tab_##nv_arch, matmul_fp16_acc_out_fp32_tab);                        \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_tab_##nv_arch, matmul_fp16_acc_out_fp32_tab_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_ta_cacc_##nv_arch, matmul_fp16_acc_out_fp32_ta_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_tb_cacc_##nv_arch, matmul_fp16_acc_out_fp32_tb_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_tab_cacc_##nv_arch, matmul_fp16_acc_out_fp32_tab_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_ta_unaligned_##nv_arch, matmul_fp16_acc_out_fp32_ta_unaligned);      \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_ta_unaligned_##nv_arch, matmul_fp16_acc_out_fp32_ta_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_tb_unaligned_##nv_arch, matmul_fp16_acc_out_fp32_tb_unaligned);      \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_tb_unaligned_##nv_arch, matmul_fp16_acc_out_fp32_tb_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_tab_unaligned_##nv_arch, matmul_fp16_acc_out_fp32_tab_unaligned);    \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_tab_unaligned_##nv_arch, matmul_fp16_acc_out_fp32_tab_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_ta_unaligned_cacc_##nv_arch, matmul_fp16_acc_out_fp32_ta_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_tb_unaligned_cacc_##nv_arch, matmul_fp16_acc_out_fp32_tb_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_fp16_acc_out_fp32_tab_unaligned_cacc_##nv_arch, matmul_fp16_acc_out_fp32_tab_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_cutlass_bf16_##nv_arch, matmul_gelu_cutlass_bf16);                                \
    DECLARE_KERNEL_ALIAS(matmul_gelu_cutlass_fp16_##nv_arch, matmul_gelu_cutlass_fp16);                                \
    DECLARE_KERNEL_ALIAS(matmul_gelu_cutlass_fp16_acc_fp16_##nv_arch, matmul_gelu_cutlass_fp16_acc_fp16);              \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_cutlass_bf16_##nv_arch, matmul_gelu_bwd_cutlass_bf16);                        \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_cutlass_bf16_ta_##nv_arch, matmul_gelu_bwd_cutlass_bf16_ta);                  \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_cutlass_bf16_tb_##nv_arch, matmul_gelu_bwd_cutlass_bf16_tb);                  \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_cutlass_bf16_tab_##nv_arch, matmul_gelu_bwd_cutlass_bf16_tab);                \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_cutlass_fp16_##nv_arch, matmul_gelu_bwd_cutlass_fp16);                        \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_cutlass_fp16_ta_##nv_arch, matmul_gelu_bwd_cutlass_fp16_ta);                  \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_cutlass_fp16_tb_##nv_arch, matmul_gelu_bwd_cutlass_fp16_tb);                  \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_cutlass_fp16_tab_##nv_arch, matmul_gelu_bwd_cutlass_fp16_tab);                \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_cutlass_fp16_acc_fp16_##nv_arch, matmul_gelu_bwd_cutlass_fp16_acc_fp16);      \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_cutlass_fp16_acc_fp16_ta_##nv_arch, matmul_gelu_bwd_cutlass_fp16_acc_fp16_ta); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_cutlass_fp16_acc_fp16_tb_##nv_arch, matmul_gelu_bwd_cutlass_fp16_acc_fp16_tb); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_cutlass_fp16_acc_fp16_tab_##nv_arch, matmul_gelu_bwd_cutlass_fp16_acc_fp16_tab); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bf16_##nv_arch, matmul_gelu_bf16);                                                \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bf16_##nv_arch, matmul_gelu_bf16_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bf16_cacc_##nv_arch, matmul_gelu_bf16_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bf16_unaligned_##nv_arch, matmul_gelu_bf16_unaligned);                            \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bf16_unaligned_##nv_arch, matmul_gelu_bf16_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bf16_unaligned_cacc_##nv_arch, matmul_gelu_bf16_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_fp16_##nv_arch, matmul_gelu_fp16);                                                \
    DECLARE_KERNEL_ALIAS(matmul_gelu_fp16_##nv_arch, matmul_gelu_fp16_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_fp16_cacc_##nv_arch, matmul_gelu_fp16_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_fp16_unaligned_##nv_arch, matmul_gelu_fp16_unaligned);                            \
    DECLARE_KERNEL_ALIAS(matmul_gelu_fp16_unaligned_##nv_arch, matmul_gelu_fp16_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_fp16_unaligned_cacc_##nv_arch, matmul_gelu_fp16_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_fp16_acc_fp16_##nv_arch, matmul_gelu_fp16_acc_fp16);                              \
    DECLARE_KERNEL_ALIAS(matmul_gelu_fp16_acc_fp16_##nv_arch, matmul_gelu_fp16_acc_fp16_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_fp16_acc_fp16_cacc_##nv_arch, matmul_gelu_fp16_acc_fp16_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_fp16_acc_fp16_unaligned_##nv_arch, matmul_gelu_fp16_acc_fp16_unaligned);          \
    DECLARE_KERNEL_ALIAS(matmul_gelu_fp16_acc_fp16_unaligned_##nv_arch, matmul_gelu_fp16_acc_fp16_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_fp16_acc_fp16_unaligned_cacc_##nv_arch, matmul_gelu_fp16_acc_fp16_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_##nv_arch, matmul_gelu_bwd_bf16);                                                \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_##nv_arch, matmul_gelu_bwd_bf16_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_cacc_##nv_arch, matmul_gelu_bwd_bf16_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_ta_##nv_arch, matmul_gelu_bwd_bf16_ta);                                              \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_ta_##nv_arch, matmul_gelu_bwd_bf16_ta_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_ta_cacc_##nv_arch, matmul_gelu_bwd_bf16_ta_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_tb_##nv_arch, matmul_gelu_bwd_bf16_tb);                                              \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_tb_##nv_arch, matmul_gelu_bwd_bf16_tb_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_tb_cacc_##nv_arch, matmul_gelu_bwd_bf16_tb_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_tab_##nv_arch, matmul_gelu_bwd_bf16_tab);                                            \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_tab_##nv_arch, matmul_gelu_bwd_bf16_tab_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_tab_cacc_##nv_arch, matmul_gelu_bwd_bf16_tab_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_unaligned_##nv_arch, matmul_gelu_bwd_bf16_unaligned);                            \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_unaligned_##nv_arch, matmul_gelu_bwd_bf16_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_unaligned_cacc_##nv_arch, matmul_gelu_bwd_bf16_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_unaligned_ta_##nv_arch, matmul_gelu_bwd_bf16_unaligned_ta);                      \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_unaligned_ta_##nv_arch, matmul_gelu_bwd_bf16_unaligned_ta_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_unaligned_ta_cacc_##nv_arch, matmul_gelu_bwd_bf16_unaligned_ta_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_unaligned_tb_##nv_arch, matmul_gelu_bwd_bf16_unaligned_tb);                      \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_unaligned_tb_##nv_arch, matmul_gelu_bwd_bf16_unaligned_tb_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_unaligned_tb_cacc_##nv_arch, matmul_gelu_bwd_bf16_unaligned_tb_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_unaligned_tab_##nv_arch, matmul_gelu_bwd_bf16_unaligned_tab);                    \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_unaligned_tab_##nv_arch, matmul_gelu_bwd_bf16_unaligned_tab_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_bf16_unaligned_tab_cacc_##nv_arch, matmul_gelu_bwd_bf16_unaligned_tab_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_##nv_arch, matmul_gelu_bwd_fp16);                                                \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_##nv_arch, matmul_gelu_bwd_fp16_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_cacc_##nv_arch, matmul_gelu_bwd_fp16_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_ta_##nv_arch, matmul_gelu_bwd_fp16_ta);                                              \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_ta_##nv_arch, matmul_gelu_bwd_fp16_ta_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_ta_cacc_##nv_arch, matmul_gelu_bwd_fp16_ta_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_tb_##nv_arch, matmul_gelu_bwd_fp16_tb);                                              \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_tb_##nv_arch, matmul_gelu_bwd_fp16_tb_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_tb_cacc_##nv_arch, matmul_gelu_bwd_fp16_tb_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_tab_##nv_arch, matmul_gelu_bwd_fp16_tab);                                            \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_tab_##nv_arch, matmul_gelu_bwd_fp16_tab_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_tab_cacc_##nv_arch, matmul_gelu_bwd_fp16_tab_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_unaligned_##nv_arch, matmul_gelu_bwd_fp16_unaligned);                            \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_unaligned_##nv_arch, matmul_gelu_bwd_fp16_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_unaligned_cacc_##nv_arch, matmul_gelu_bwd_fp16_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_unaligned_ta_##nv_arch, matmul_gelu_bwd_fp16_unaligned_ta);                      \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_unaligned_ta_##nv_arch, matmul_gelu_bwd_fp16_unaligned_ta_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_unaligned_ta_cacc_##nv_arch, matmul_gelu_bwd_fp16_unaligned_ta_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_unaligned_tb_##nv_arch, matmul_gelu_bwd_fp16_unaligned_tb);                      \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_unaligned_tb_##nv_arch, matmul_gelu_bwd_fp16_unaligned_tb_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_unaligned_tb_cacc_##nv_arch, matmul_gelu_bwd_fp16_unaligned_tb_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_unaligned_tab_##nv_arch, matmul_gelu_bwd_fp16_unaligned_tab);                    \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_unaligned_tab_##nv_arch, matmul_gelu_bwd_fp16_unaligned_tab_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_unaligned_tab_cacc_##nv_arch, matmul_gelu_bwd_fp16_unaligned_tab_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16);                              \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_cacc_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_ta_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_ta);                              \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_ta_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_ta_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_ta_cacc_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_ta_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_tb_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_tb);                              \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_tb_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_tb_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_tb_cacc_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_tb_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_tab_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_tab);                            \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_tab_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_tab_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_tab_cacc_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_tab_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_unaligned_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_unaligned);          \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_unaligned_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_unaligned_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_unaligned_cacc_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_unaligned_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_unaligned_ta_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_unaligned_ta);          \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_unaligned_ta_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_unaligned_ta_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_unaligned_ta_cacc_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_unaligned_ta_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_unaligned_tb_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_unaligned_tb);          \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_unaligned_tb_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_unaligned_tb_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_unaligned_tb_cacc_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_unaligned_tb_cacc); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_unaligned_tab_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_unaligned_tab);        \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_unaligned_tab_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_unaligned_tab_cstore); \
    DECLARE_KERNEL_ALIAS(matmul_gelu_bwd_fp16_acc_fp16_unaligned_tab_cacc_##nv_arch, matmul_gelu_bwd_fp16_acc_fp16_unaligned_tab_cacc); \
    DECLARE_KERNEL_ALIAS(addmm_cutlass_bf16_##nv_arch, addmm_cutlass_bf16);                                            \
    DECLARE_KERNEL_ALIAS(addmm_cutlass_fp16_##nv_arch, addmm_cutlass_fp16);                                            \
    DECLARE_KERNEL_ALIAS(addmm_cutlass_fp16_acc_fp16_##nv_arch, addmm_cutlass_fp16_acc_fp16);                          \
    DECLARE_KERNEL_ALIAS(addmm_cutlass_fp16_out_fp32_##nv_arch, addmm_cutlass_fp16_out_fp32);                          \
    DECLARE_KERNEL_ALIAS(addmm_cutlass_fp16_acc_fp16_out_fp32_##nv_arch, addmm_cutlass_fp16_acc_fp16_out_fp32);        \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_bf16_##nv_arch, matmul_cutlass_bf16);                                          \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_bf16_ta_##nv_arch, matmul_cutlass_bf16_ta);                                    \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_bf16_tb_##nv_arch, matmul_cutlass_bf16_tb);                                    \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_bf16_tab_##nv_arch, matmul_cutlass_bf16_tab);                                  \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_bf16_out_fp32_##nv_arch, matmul_cutlass_bf16_out_fp32);                        \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_bf16_out_fp32_ta_##nv_arch, matmul_cutlass_bf16_out_fp32_ta);                  \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_bf16_out_fp32_tb_##nv_arch, matmul_cutlass_bf16_out_fp32_tb);                  \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_bf16_out_fp32_tab_##nv_arch, matmul_cutlass_bf16_out_fp32_tab);                \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_fp16_##nv_arch, matmul_cutlass_fp16);                                          \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_fp16_acc_fp16_##nv_arch, matmul_cutlass_fp16_acc_fp16);                        \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_fp16_out_fp32_##nv_arch, matmul_cutlass_fp16_out_fp32);                        \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_fp16_acc_fp16_out_fp32_##nv_arch, matmul_cutlass_fp16_acc_fp16_out_fp32);      \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_fp16_ta_##nv_arch, matmul_cutlass_fp16_ta);                                    \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_fp16_tb_##nv_arch, matmul_cutlass_fp16_tb);                                    \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_fp16_tab_##nv_arch, matmul_cutlass_fp16_tab);                                  \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_fp16_acc_fp16_ta_##nv_arch, matmul_cutlass_fp16_acc_fp16_ta);                  \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_fp16_acc_fp16_tb_##nv_arch, matmul_cutlass_fp16_acc_fp16_tb);                  \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_fp16_acc_fp16_tab_##nv_arch, matmul_cutlass_fp16_acc_fp16_tab);                \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_fp16_out_fp32_ta_##nv_arch, matmul_cutlass_fp16_out_fp32_ta);                  \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_fp16_out_fp32_tb_##nv_arch, matmul_cutlass_fp16_out_fp32_tb);                  \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_fp16_out_fp32_tab_##nv_arch, matmul_cutlass_fp16_out_fp32_tab);                \
    DECLARE_KERNEL_ALIAS(matmul_cutlass_fp16_acc_fp16_out_fp32_ta_##nv_arch, matmul_cutlass_fp16_acc_fp16_out_fp32_ta);\
    DECLARE_KERNEL_ALIAS(matmul_cutlass_fp16_acc_fp16_out_fp32_tb_##nv_arch, matmul_cutlass_fp16_acc_fp16_out_fp32_tb);\
    DECLARE_KERNEL_ALIAS(matmul_cutlass_fp16_acc_fp16_out_fp32_tab_##nv_arch, matmul_cutlass_fp16_acc_fp16_out_fp32_tab);\
    DECLARE_KERNEL_ALIAS(addmm_gelu_cutlass_bf16_##nv_arch, addmm_gelu_cutlass_bf16);                                  \
    DECLARE_KERNEL_ALIAS(addmm_gelu_cutlass_fp16_##nv_arch, addmm_gelu_cutlass_fp16);                                  \
    DECLARE_KERNEL_ALIAS(addmm_gelu_cutlass_fp16_acc_fp16_##nv_arch, addmm_gelu_cutlass_fp16_acc_fp16);                \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_cutlass_bf16_##nv_arch, addmm_gelu_preact_cutlass_bf16);                    \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_cutlass_fp16_##nv_arch, addmm_gelu_preact_cutlass_fp16);                    \
    DECLARE_KERNEL_ALIAS(addmm_gelu_preact_cutlass_fp16_acc_fp16_##nv_arch, addmm_gelu_preact_cutlass_fp16_acc_fp16);  \
    DECLARE_KERNEL_ALIAS(embedding_lookup_bf16_##nv_arch, embedding_lookup_bf16);                                      \
    DECLARE_KERNEL_ALIAS(embedding_lookup_fp16_##nv_arch, embedding_lookup_fp16);                                      \
    DECLARE_KERNEL_ALIAS(cast_bf16_to_fp32_##nv_arch, cast_bf16_to_fp32);                                              \
    DECLARE_KERNEL_ALIAS(cast_fp32_to_bf16_##nv_arch, cast_fp32_to_bf16);                                              \
    DECLARE_KERNEL_ALIAS(cast_fp16_to_fp32_##nv_arch, cast_fp16_to_fp32);                                              \
    DECLARE_KERNEL_ALIAS(cast_fp32_to_fp16_##nv_arch, cast_fp32_to_fp16);                                              \
    DECLARE_KERNEL_ALIAS(cast_bf16_to_fp16_##nv_arch, cast_bf16_to_fp16);                                              \
    DECLARE_KERNEL_ALIAS(cast_fp16_to_bf16_##nv_arch, cast_fp16_to_bf16);                                              \
    DECLARE_KERNEL_ALIAS(conv2d_bf16_ic3_oc8_k3_##nv_arch, conv2d_bf16_ic3_oc8_k3);                                    \
    DECLARE_KERNEL_ALIAS(conv2d_bf16_ic3_oc8_k3_dil2_##nv_arch, conv2d_bf16_ic3_oc8_k3_dil2);                          \
    DECLARE_KERNEL_ALIAS(conv2d_bf16_ic32_oc64_k3_##nv_arch, conv2d_bf16_ic32_oc64_k3);                                \
    DECLARE_KERNEL_ALIAS(conv2d_bf16_ic768_oc768_k3_dil2_##nv_arch, conv2d_bf16_ic768_oc768_k3_dil2);                  \
    DECLARE_KERNEL_ALIAS(conv2d_bf16_ic768_oc1536_k3_##nv_arch, conv2d_bf16_ic768_oc1536_k3);                          \
    DECLARE_KERNEL_ALIAS(conv2d_bf16_ic1024_oc1024_k3_dil2_##nv_arch, conv2d_bf16_ic1024_oc1024_k3_dil2);              \
    DECLARE_KERNEL_ALIAS(conv2d_bf16_ic1024_oc2048_k3_##nv_arch, conv2d_bf16_ic1024_oc2048_k3);                        \
    DECLARE_KERNEL_ALIAS(conv2d_fp16_ic3_oc8_k3_##nv_arch, conv2d_fp16_ic3_oc8_k3);                                    \
    DECLARE_KERNEL_ALIAS(conv2d_fp16_ic3_oc8_k3_dil2_##nv_arch, conv2d_fp16_ic3_oc8_k3_dil2);                          \
    DECLARE_KERNEL_ALIAS(conv2d_fp16_ic32_oc64_k3_##nv_arch, conv2d_fp16_ic32_oc64_k3);                                \
    DECLARE_KERNEL_ALIAS(conv2d_fp16_ic768_oc768_k3_dil2_##nv_arch, conv2d_fp16_ic768_oc768_k3_dil2);                  \
    DECLARE_KERNEL_ALIAS(conv2d_fp16_ic768_oc1536_k3_##nv_arch, conv2d_fp16_ic768_oc1536_k3);                          \
    DECLARE_KERNEL_ALIAS(conv2d_fp16_ic1024_oc1024_k3_dil2_##nv_arch, conv2d_fp16_ic1024_oc1024_k3_dil2);              \
    DECLARE_KERNEL_ALIAS(conv2d_fp16_ic1024_oc2048_k3_##nv_arch, conv2d_fp16_ic1024_oc2048_k3);                        \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_bf16_ic3_oc8_k3_##nv_arch, conv2d_dgrad_bf16_ic3_oc8_k3);                        \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_bf16_ic3_oc8_k3_dil2_##nv_arch, conv2d_dgrad_bf16_ic3_oc8_k3_dil2);              \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_bf16_ic32_oc64_k3_##nv_arch, conv2d_dgrad_bf16_ic32_oc64_k3);                    \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_bf16_ic768_oc768_k3_dil2_##nv_arch,                                              \
                         conv2d_dgrad_bf16_ic768_oc768_k3_dil2);                                                       \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_bf16_ic768_oc1536_k3_##nv_arch, conv2d_dgrad_bf16_ic768_oc1536_k3);              \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_bf16_ic1024_oc1024_k3_dil2_##nv_arch,                                            \
                         conv2d_dgrad_bf16_ic1024_oc1024_k3_dil2);                                                     \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_bf16_ic1024_oc2048_k3_##nv_arch, conv2d_dgrad_bf16_ic1024_oc2048_k3);            \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_fp16_ic3_oc8_k3_##nv_arch, conv2d_dgrad_fp16_ic3_oc8_k3);                        \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_fp16_ic3_oc8_k3_dil2_##nv_arch, conv2d_dgrad_fp16_ic3_oc8_k3_dil2);              \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_fp16_ic32_oc64_k3_##nv_arch, conv2d_dgrad_fp16_ic32_oc64_k3);                    \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_fp16_ic768_oc768_k3_dil2_##nv_arch,                                              \
                         conv2d_dgrad_fp16_ic768_oc768_k3_dil2);                                                       \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_fp16_ic768_oc1536_k3_##nv_arch, conv2d_dgrad_fp16_ic768_oc1536_k3);              \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_fp16_ic1024_oc1024_k3_dil2_##nv_arch,                                            \
                         conv2d_dgrad_fp16_ic1024_oc1024_k3_dil2);                                                     \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_fp16_ic1024_oc2048_k3_##nv_arch, conv2d_dgrad_fp16_ic1024_oc2048_k3);            \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_bf16_ic3_oc8_k3_##nv_arch, conv2d_wgrad_bf16_ic3_oc8_k3);                        \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_bf16_ic3_oc8_k3_dil2_##nv_arch, conv2d_wgrad_bf16_ic3_oc8_k3_dil2);              \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_bf16_ic32_oc64_k3_##nv_arch, conv2d_wgrad_bf16_ic32_oc64_k3);                    \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_bf16_ic768_oc768_k3_dil2_##nv_arch,                                              \
                         conv2d_wgrad_bf16_ic768_oc768_k3_dil2);                                                       \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_bf16_ic768_oc1536_k3_##nv_arch, conv2d_wgrad_bf16_ic768_oc1536_k3);              \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_bf16_ic1024_oc1024_k3_dil2_##nv_arch,                                            \
                         conv2d_wgrad_bf16_ic1024_oc1024_k3_dil2);                                                     \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_bf16_ic1024_oc2048_k3_##nv_arch, conv2d_wgrad_bf16_ic1024_oc2048_k3);            \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_fp16_ic3_oc8_k3_##nv_arch, conv2d_wgrad_fp16_ic3_oc8_k3);                        \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_fp16_ic3_oc8_k3_dil2_##nv_arch, conv2d_wgrad_fp16_ic3_oc8_k3_dil2);              \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_fp16_ic32_oc64_k3_##nv_arch, conv2d_wgrad_fp16_ic32_oc64_k3);                    \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_fp16_ic768_oc768_k3_dil2_##nv_arch,                                              \
                         conv2d_wgrad_fp16_ic768_oc768_k3_dil2);                                                       \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_fp16_ic768_oc1536_k3_##nv_arch, conv2d_wgrad_fp16_ic768_oc1536_k3);              \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_fp16_ic1024_oc1024_k3_dil2_##nv_arch,                                            \
                         conv2d_wgrad_fp16_ic1024_oc1024_k3_dil2);                                                     \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_fp16_ic1024_oc2048_k3_##nv_arch, conv2d_wgrad_fp16_ic1024_oc2048_k3);            \
    DECLARE_KERNEL_ALIAS(conv2d_cutlass_bf16_stride1_pad1_dil1_ic32_oc64_##nv_arch,                                    \
                         conv2d_cutlass_bf16_stride1_pad1_dil1_ic32_oc64);                                             \
    DECLARE_KERNEL_ALIAS(conv2d_cutlass_bf16_stride1_pad1_dil1_ic768_oc1536_##nv_arch,                                 \
                         conv2d_cutlass_bf16_stride1_pad1_dil1_ic768_oc1536);                                          \
    DECLARE_KERNEL_ALIAS(conv2d_cutlass_bf16_stride1_pad1_dil1_ic1024_oc2048_##nv_arch,                                \
                         conv2d_cutlass_bf16_stride1_pad1_dil1_ic1024_oc2048);                                         \
    DECLARE_KERNEL_ALIAS(conv2d_cutlass_bf16_stride1_pad2_dil2_ic768_oc768_##nv_arch,                                  \
                         conv2d_cutlass_bf16_stride1_pad2_dil2_ic768_oc768);                                           \
    DECLARE_KERNEL_ALIAS(conv2d_cutlass_bf16_stride1_pad2_dil2_ic1024_oc1024_##nv_arch,                                \
                         conv2d_cutlass_bf16_stride1_pad2_dil2_ic1024_oc1024);                                         \
    DECLARE_KERNEL_ALIAS(conv2d_cutlass_fp16_stride1_pad1_dil1_ic32_oc64_##nv_arch,                                    \
                         conv2d_cutlass_fp16_stride1_pad1_dil1_ic32_oc64);                                             \
    DECLARE_KERNEL_ALIAS(conv2d_cutlass_fp16_stride1_pad1_dil1_ic768_oc1536_##nv_arch,                                 \
                         conv2d_cutlass_fp16_stride1_pad1_dil1_ic768_oc1536);                                          \
    DECLARE_KERNEL_ALIAS(conv2d_cutlass_fp16_stride1_pad1_dil1_ic1024_oc2048_##nv_arch,                                \
                         conv2d_cutlass_fp16_stride1_pad1_dil1_ic1024_oc2048);                                         \
    DECLARE_KERNEL_ALIAS(conv2d_cutlass_fp16_stride1_pad2_dil2_ic768_oc768_##nv_arch,                                  \
                         conv2d_cutlass_fp16_stride1_pad2_dil2_ic768_oc768);                                           \
    DECLARE_KERNEL_ALIAS(conv2d_cutlass_fp16_stride1_pad2_dil2_ic1024_oc1024_##nv_arch,                                \
                         conv2d_cutlass_fp16_stride1_pad2_dil2_ic1024_oc1024);                                         \
    DECLARE_KERNEL_ALIAS(conv2d_cutlass_fp16_acc_fp16_stride1_pad1_dil1_ic32_oc64_##nv_arch,                           \
                         conv2d_cutlass_fp16_acc_fp16_stride1_pad1_dil1_ic32_oc64);                                    \
    DECLARE_KERNEL_ALIAS(conv2d_cutlass_fp16_acc_fp16_stride1_pad1_dil1_ic768_oc1536_##nv_arch,                        \
                         conv2d_cutlass_fp16_acc_fp16_stride1_pad1_dil1_ic768_oc1536);                                 \
    DECLARE_KERNEL_ALIAS(conv2d_cutlass_fp16_acc_fp16_stride1_pad1_dil1_ic1024_oc2048_##nv_arch,                       \
                         conv2d_cutlass_fp16_acc_fp16_stride1_pad1_dil1_ic1024_oc2048);                                \
    DECLARE_KERNEL_ALIAS(conv2d_cutlass_fp16_acc_fp16_stride1_pad2_dil2_ic768_oc768_##nv_arch,                         \
                         conv2d_cutlass_fp16_acc_fp16_stride1_pad2_dil2_ic768_oc768);                                  \
    DECLARE_KERNEL_ALIAS(conv2d_cutlass_fp16_acc_fp16_stride1_pad2_dil2_ic1024_oc1024_##nv_arch,                       \
                         conv2d_cutlass_fp16_acc_fp16_stride1_pad2_dil2_ic1024_oc1024);                                \
    DECLARE_CONV2D_BIN_KERNEL_ALIASES(nv_arch);                                                                        \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_cutlass_bf16_stride1_pad1_dil1_ic32_oc64_##nv_arch,                              \
                         conv2d_dgrad_cutlass_bf16_stride1_pad1_dil1_ic32_oc64);                                       \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_cutlass_bf16_stride1_pad1_dil1_ic768_oc1536_##nv_arch,                           \
                         conv2d_dgrad_cutlass_bf16_stride1_pad1_dil1_ic768_oc1536);                                    \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_cutlass_bf16_stride1_pad1_dil1_ic1024_oc2048_##nv_arch,                          \
                         conv2d_dgrad_cutlass_bf16_stride1_pad1_dil1_ic1024_oc2048);                                   \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_cutlass_bf16_stride1_pad2_dil2_ic768_oc768_##nv_arch,                            \
                         conv2d_dgrad_cutlass_bf16_stride1_pad2_dil2_ic768_oc768);                                     \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_cutlass_bf16_stride1_pad2_dil2_ic1024_oc1024_##nv_arch,                          \
                         conv2d_dgrad_cutlass_bf16_stride1_pad2_dil2_ic1024_oc1024);                                   \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_cutlass_fp16_stride1_pad1_dil1_ic32_oc64_##nv_arch,                              \
                         conv2d_dgrad_cutlass_fp16_stride1_pad1_dil1_ic32_oc64);                                       \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_cutlass_fp16_stride1_pad1_dil1_ic768_oc1536_##nv_arch,                           \
                         conv2d_dgrad_cutlass_fp16_stride1_pad1_dil1_ic768_oc1536);                                    \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_cutlass_fp16_stride1_pad1_dil1_ic1024_oc2048_##nv_arch,                          \
                         conv2d_dgrad_cutlass_fp16_stride1_pad1_dil1_ic1024_oc2048);                                   \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_cutlass_fp16_stride1_pad2_dil2_ic768_oc768_##nv_arch,                            \
                         conv2d_dgrad_cutlass_fp16_stride1_pad2_dil2_ic768_oc768);                                     \
    DECLARE_KERNEL_ALIAS(conv2d_dgrad_cutlass_fp16_stride1_pad2_dil2_ic1024_oc1024_##nv_arch,                          \
                         conv2d_dgrad_cutlass_fp16_stride1_pad2_dil2_ic1024_oc1024);                                   \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_cutlass_bf16_stride1_pad1_dil1_ic32_oc64_##nv_arch,                              \
                         conv2d_wgrad_cutlass_bf16_stride1_pad1_dil1_ic32_oc64);                                       \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_cutlass_bf16_stride1_pad1_dil1_ic768_oc1536_##nv_arch,                           \
                         conv2d_wgrad_cutlass_bf16_stride1_pad1_dil1_ic768_oc1536);                                    \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_cutlass_bf16_stride1_pad1_dil1_ic1024_oc2048_##nv_arch,                          \
                         conv2d_wgrad_cutlass_bf16_stride1_pad1_dil1_ic1024_oc2048);                                   \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_cutlass_bf16_stride1_pad2_dil2_ic768_oc768_##nv_arch,                            \
                         conv2d_wgrad_cutlass_bf16_stride1_pad2_dil2_ic768_oc768);                                     \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_cutlass_bf16_stride1_pad2_dil2_ic1024_oc1024_##nv_arch,                          \
                         conv2d_wgrad_cutlass_bf16_stride1_pad2_dil2_ic1024_oc1024);                                   \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_cutlass_fp16_stride1_pad1_dil1_ic32_oc64_##nv_arch,                              \
                         conv2d_wgrad_cutlass_fp16_stride1_pad1_dil1_ic32_oc64);                                       \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_cutlass_fp16_stride1_pad1_dil1_ic768_oc1536_##nv_arch,                           \
                         conv2d_wgrad_cutlass_fp16_stride1_pad1_dil1_ic768_oc1536);                                    \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_cutlass_fp16_stride1_pad1_dil1_ic1024_oc2048_##nv_arch,                          \
                         conv2d_wgrad_cutlass_fp16_stride1_pad1_dil1_ic1024_oc2048);                                   \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_cutlass_fp16_stride1_pad2_dil2_ic768_oc768_##nv_arch,                            \
                         conv2d_wgrad_cutlass_fp16_stride1_pad2_dil2_ic768_oc768);                                     \
    DECLARE_KERNEL_ALIAS(conv2d_wgrad_cutlass_fp16_stride1_pad2_dil2_ic1024_oc1024_##nv_arch,                          \
                         conv2d_wgrad_cutlass_fp16_stride1_pad2_dil2_ic1024_oc1024);                                   \
    DECLARE_KERNEL_ALIAS(lstm_cell_fwd_out_fp16_##nv_arch, lstm_cell_fwd_out_fp16);                                    \
    DECLARE_KERNEL_ALIAS(lstm_cell_fwd_fp32_state_out_fp16_##nv_arch, lstm_cell_fwd_fp32_state_out_fp16);              \
    DECLARE_KERNEL_ALIAS(lstm_cell_fwd_fp32_state_out_bf16_##nv_arch, lstm_cell_fwd_fp32_state_out_bf16);
