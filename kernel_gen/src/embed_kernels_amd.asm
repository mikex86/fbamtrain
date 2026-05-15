// embed_kernels_amd.asm
// This file contains the assembly code for embedding AMD GPU kernels as binary blobs to be launched by the host application.
#include "declare_asset.inc.asm"

#if defined(__ELF__)
    .section .note.GNU-stack,"",@progbits
#endif


#if AMD_KERNEL_ARCH == 942
#include "../kernel_gen/output/act_gelu_bwd_elementwise_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO act_gelu_bwd_elementwise_bf16_gfx942
EMBED_ASSET act_gelu_bwd_elementwise_bf16_gfx942, act_gelu_bwd_elementwise_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/act_gelu_bwd_elementwise_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO act_gelu_bwd_elementwise_fp16_gfx942
EMBED_ASSET act_gelu_bwd_elementwise_fp16_gfx942, act_gelu_bwd_elementwise_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/act_gelu_elementwise_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO act_gelu_elementwise_bf16_gfx942
EMBED_ASSET act_gelu_elementwise_bf16_gfx942, act_gelu_elementwise_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/act_gelu_elementwise_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO act_gelu_elementwise_fp16_gfx942
EMBED_ASSET act_gelu_elementwise_fp16_gfx942, act_gelu_elementwise_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/act_relu_elementwise_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO act_relu_elementwise_bf16_gfx942
EMBED_ASSET act_relu_elementwise_bf16_gfx942, act_relu_elementwise_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/act_relu_elementwise_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO act_relu_elementwise_fp16_gfx942
EMBED_ASSET act_relu_elementwise_fp16_gfx942, act_relu_elementwise_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/adamw_step_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO adamw_step_bf16_gfx942
EMBED_ASSET adamw_step_bf16_gfx942, adamw_step_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/adamw_step_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO adamw_step_fp16_gfx942
EMBED_ASSET adamw_step_fp16_gfx942, adamw_step_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/adamw_step_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO adamw_step_fp32_gfx942
EMBED_ASSET adamw_step_fp32_gfx942, adamw_step_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/add_elementwise_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO add_elementwise_bf16_gfx942
EMBED_ASSET add_elementwise_bf16_gfx942, add_elementwise_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/add_elementwise_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO add_elementwise_fp16_gfx942
EMBED_ASSET add_elementwise_fp16_gfx942, add_elementwise_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/add_elementwise_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO add_elementwise_fp32_gfx942
EMBED_ASSET add_elementwise_fp32_gfx942, add_elementwise_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/add_elementwise_fp32_out_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO add_elementwise_fp32_out_fp16_gfx942
EMBED_ASSET add_elementwise_fp32_out_fp16_gfx942, add_elementwise_fp32_out_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/add_trailing_broadcast_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO add_trailing_broadcast_bf16_gfx942
EMBED_ASSET add_trailing_broadcast_bf16_gfx942, add_trailing_broadcast_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/add_trailing_broadcast_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO add_trailing_broadcast_fp16_gfx942
EMBED_ASSET add_trailing_broadcast_fp16_gfx942, add_trailing_broadcast_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/add_trailing_broadcast_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO add_trailing_broadcast_fp32_gfx942
EMBED_ASSET add_trailing_broadcast_fp32_gfx942, add_trailing_broadcast_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_bf16_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_bf16_cacc_gfx942
EMBED_ASSET addmm_bf16_cacc_gfx942, addmm_bf16_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_bf16_gfx942
EMBED_ASSET addmm_bf16_gfx942, addmm_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_bf16_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_bf16_unaligned_cacc_gfx942
EMBED_ASSET addmm_bf16_unaligned_cacc_gfx942, addmm_bf16_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_bf16_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_bf16_unaligned_gfx942
EMBED_ASSET addmm_bf16_unaligned_gfx942, addmm_bf16_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_fp16_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_fp16_cacc_gfx942
EMBED_ASSET addmm_fp16_cacc_gfx942, addmm_fp16_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_fp16_gfx942
EMBED_ASSET addmm_fp16_gfx942, addmm_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_fp16_out_fp32_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_fp16_out_fp32_cacc_gfx942
EMBED_ASSET addmm_fp16_out_fp32_cacc_gfx942, addmm_fp16_out_fp32_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_fp16_out_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_fp16_out_fp32_gfx942
EMBED_ASSET addmm_fp16_out_fp32_gfx942, addmm_fp16_out_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_fp16_out_fp32_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_fp16_out_fp32_unaligned_cacc_gfx942
EMBED_ASSET addmm_fp16_out_fp32_unaligned_cacc_gfx942, addmm_fp16_out_fp32_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_fp16_out_fp32_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_fp16_out_fp32_unaligned_gfx942
EMBED_ASSET addmm_fp16_out_fp32_unaligned_gfx942, addmm_fp16_out_fp32_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_fp16_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_fp16_unaligned_cacc_gfx942
EMBED_ASSET addmm_fp16_unaligned_cacc_gfx942, addmm_fp16_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_fp16_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_fp16_unaligned_gfx942
EMBED_ASSET addmm_fp16_unaligned_gfx942, addmm_fp16_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_bf16_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_bf16_cacc_gfx942
EMBED_ASSET addmm_gelu_bf16_cacc_gfx942, addmm_gelu_bf16_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_bf16_gfx942
EMBED_ASSET addmm_gelu_bf16_gfx942, addmm_gelu_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_bf16_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_bf16_unaligned_cacc_gfx942
EMBED_ASSET addmm_gelu_bf16_unaligned_cacc_gfx942, addmm_gelu_bf16_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_bf16_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_bf16_unaligned_gfx942
EMBED_ASSET addmm_gelu_bf16_unaligned_gfx942, addmm_gelu_bf16_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_fp16_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_fp16_cacc_gfx942
EMBED_ASSET addmm_gelu_fp16_cacc_gfx942, addmm_gelu_fp16_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_fp16_gfx942
EMBED_ASSET addmm_gelu_fp16_gfx942, addmm_gelu_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_fp16_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_fp16_unaligned_cacc_gfx942
EMBED_ASSET addmm_gelu_fp16_unaligned_cacc_gfx942, addmm_gelu_fp16_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_fp16_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_fp16_unaligned_gfx942
EMBED_ASSET addmm_gelu_fp16_unaligned_gfx942, addmm_gelu_fp16_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_preact_bf16_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_preact_bf16_cacc_gfx942
EMBED_ASSET addmm_gelu_preact_bf16_cacc_gfx942, addmm_gelu_preact_bf16_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_preact_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_preact_bf16_gfx942
EMBED_ASSET addmm_gelu_preact_bf16_gfx942, addmm_gelu_preact_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_preact_bf16_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_preact_bf16_unaligned_cacc_gfx942
EMBED_ASSET addmm_gelu_preact_bf16_unaligned_cacc_gfx942, addmm_gelu_preact_bf16_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_preact_bf16_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_preact_bf16_unaligned_gfx942
EMBED_ASSET addmm_gelu_preact_bf16_unaligned_gfx942, addmm_gelu_preact_bf16_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_preact_fp16_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_preact_fp16_cacc_gfx942
EMBED_ASSET addmm_gelu_preact_fp16_cacc_gfx942, addmm_gelu_preact_fp16_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_preact_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_preact_fp16_gfx942
EMBED_ASSET addmm_gelu_preact_fp16_gfx942, addmm_gelu_preact_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_preact_fp16_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_preact_fp16_unaligned_cacc_gfx942
EMBED_ASSET addmm_gelu_preact_fp16_unaligned_cacc_gfx942, addmm_gelu_preact_fp16_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_preact_fp16_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_preact_fp16_unaligned_gfx942
EMBED_ASSET addmm_gelu_preact_fp16_unaligned_gfx942, addmm_gelu_preact_fp16_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/avg_pool1d_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO avg_pool1d_bf16_gfx942
EMBED_ASSET avg_pool1d_bf16_gfx942, avg_pool1d_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/avg_pool1d_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO avg_pool1d_fp16_gfx942
EMBED_ASSET avg_pool1d_fp16_gfx942, avg_pool1d_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/avg_pool2d_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO avg_pool2d_bf16_gfx942
EMBED_ASSET avg_pool2d_bf16_gfx942, avg_pool2d_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/avg_pool2d_bwd_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO avg_pool2d_bwd_bf16_gfx942
EMBED_ASSET avg_pool2d_bwd_bf16_gfx942, avg_pool2d_bwd_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/avg_pool2d_bwd_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO avg_pool2d_bwd_fp16_gfx942
EMBED_ASSET avg_pool2d_bwd_fp16_gfx942, avg_pool2d_bwd_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/avg_pool2d_bwd_noaccum_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO avg_pool2d_bwd_noaccum_bf16_gfx942
EMBED_ASSET avg_pool2d_bwd_noaccum_bf16_gfx942, avg_pool2d_bwd_noaccum_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/avg_pool2d_bwd_noaccum_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO avg_pool2d_bwd_noaccum_fp16_gfx942
EMBED_ASSET avg_pool2d_bwd_noaccum_fp16_gfx942, avg_pool2d_bwd_noaccum_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/avg_pool2d_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO avg_pool2d_fp16_gfx942
EMBED_ASSET avg_pool2d_fp16_gfx942, avg_pool2d_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bf16_gfx942
EMBED_ASSET build_cell_embeds_bf16_gfx942, build_cell_embeds_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_color_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_color_bf16_gfx942
EMBED_ASSET build_cell_embeds_bwd_color_bf16_gfx942, build_cell_embeds_bwd_color_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_color_bf16_out_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_color_bf16_out_fp32_gfx942
EMBED_ASSET build_cell_embeds_bwd_color_bf16_out_fp32_gfx942, build_cell_embeds_bwd_color_bf16_out_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_color_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_color_fp16_gfx942
EMBED_ASSET build_cell_embeds_bwd_color_fp16_gfx942, build_cell_embeds_bwd_color_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_color_fp16_out_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_color_fp16_out_fp32_gfx942
EMBED_ASSET build_cell_embeds_bwd_color_fp16_out_fp32_gfx942, build_cell_embeds_bwd_color_fp16_out_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_cp_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_cp_bf16_gfx942
EMBED_ASSET build_cell_embeds_bwd_cp_bf16_gfx942, build_cell_embeds_bwd_cp_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_cp_bf16_out_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_cp_bf16_out_fp32_gfx942
EMBED_ASSET build_cell_embeds_bwd_cp_bf16_out_fp32_gfx942, build_cell_embeds_bwd_cp_bf16_out_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_cp_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_cp_fp16_gfx942
EMBED_ASSET build_cell_embeds_bwd_cp_fp16_gfx942, build_cell_embeds_bwd_cp_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_cp_fp16_out_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_cp_fp16_out_fp32_gfx942
EMBED_ASSET build_cell_embeds_bwd_cp_fp16_out_fp32_gfx942, build_cell_embeds_bwd_cp_fp16_out_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_pos_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_pos_bf16_gfx942
EMBED_ASSET build_cell_embeds_bwd_pos_bf16_gfx942, build_cell_embeds_bwd_pos_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_pos_bf16_out_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_pos_bf16_out_fp32_gfx942
EMBED_ASSET build_cell_embeds_bwd_pos_bf16_out_fp32_gfx942, build_cell_embeds_bwd_pos_bf16_out_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_pos_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_pos_fp16_gfx942
EMBED_ASSET build_cell_embeds_bwd_pos_fp16_gfx942, build_cell_embeds_bwd_pos_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_pos_fp16_out_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_pos_fp16_out_fp32_gfx942
EMBED_ASSET build_cell_embeds_bwd_pos_fp16_out_fp32_gfx942, build_cell_embeds_bwd_pos_fp16_out_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_fp16_gfx942
EMBED_ASSET build_cell_embeds_fp16_gfx942, build_cell_embeds_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/cast_bf16_to_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO cast_bf16_to_fp16_gfx942
EMBED_ASSET cast_bf16_to_fp16_gfx942, cast_bf16_to_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/cast_bf16_to_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO cast_bf16_to_fp32_gfx942
EMBED_ASSET cast_bf16_to_fp32_gfx942, cast_bf16_to_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/cast_fp16_to_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO cast_fp16_to_bf16_gfx942
EMBED_ASSET cast_fp16_to_bf16_gfx942, cast_fp16_to_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/cast_fp16_to_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO cast_fp16_to_fp32_gfx942
EMBED_ASSET cast_fp16_to_fp32_gfx942, cast_fp16_to_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/cast_fp32_to_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO cast_fp32_to_bf16_gfx942
EMBED_ASSET cast_fp32_to_bf16_gfx942, cast_fp32_to_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/cast_fp32_to_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO cast_fp32_to_fp16_gfx942
EMBED_ASSET cast_fp32_to_fp16_gfx942, cast_fp32_to_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/contiguous_3d_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO contiguous_3d_bf16_gfx942
EMBED_ASSET contiguous_3d_bf16_gfx942, contiguous_3d_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/contiguous_3d_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO contiguous_3d_fp16_gfx942
EMBED_ASSET contiguous_3d_fp16_gfx942, contiguous_3d_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/contiguous_4d_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO contiguous_4d_bf16_gfx942
EMBED_ASSET contiguous_4d_bf16_gfx942, contiguous_4d_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/contiguous_4d_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO contiguous_4d_fp16_gfx942
EMBED_ASSET contiguous_4d_fp16_gfx942, contiguous_4d_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_bf16_ic1024_oc1024_k3_dil2_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_bf16_ic1024_oc1024_k3_dil2_gfx942
EMBED_ASSET conv2d_bf16_ic1024_oc1024_k3_dil2_gfx942, conv2d_bf16_ic1024_oc1024_k3_dil2_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_bf16_ic1024_oc2048_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_bf16_ic1024_oc2048_k3_gfx942
EMBED_ASSET conv2d_bf16_ic1024_oc2048_k3_gfx942, conv2d_bf16_ic1024_oc2048_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_bf16_ic32_oc64_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_bf16_ic32_oc64_k3_gfx942
EMBED_ASSET conv2d_bf16_ic32_oc64_k3_gfx942, conv2d_bf16_ic32_oc64_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_bf16_ic3_oc8_k3_dil2_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_bf16_ic3_oc8_k3_dil2_gfx942
EMBED_ASSET conv2d_bf16_ic3_oc8_k3_dil2_gfx942, conv2d_bf16_ic3_oc8_k3_dil2_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_bf16_ic3_oc8_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_bf16_ic3_oc8_k3_gfx942
EMBED_ASSET conv2d_bf16_ic3_oc8_k3_gfx942, conv2d_bf16_ic3_oc8_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_bf16_ic768_oc1536_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_bf16_ic768_oc1536_k3_gfx942
EMBED_ASSET conv2d_bf16_ic768_oc1536_k3_gfx942, conv2d_bf16_ic768_oc1536_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_bf16_ic768_oc768_k3_dil2_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_bf16_ic768_oc768_k3_dil2_gfx942
EMBED_ASSET conv2d_bf16_ic768_oc768_k3_dil2_gfx942, conv2d_bf16_ic768_oc768_k3_dil2_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_bf16_ic1024_oc1024_k3_dil2_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_bf16_ic1024_oc1024_k3_dil2_gfx942
EMBED_ASSET conv2d_dgrad_bf16_ic1024_oc1024_k3_dil2_gfx942, conv2d_dgrad_bf16_ic1024_oc1024_k3_dil2_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_bf16_ic1024_oc2048_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_bf16_ic1024_oc2048_k3_gfx942
EMBED_ASSET conv2d_dgrad_bf16_ic1024_oc2048_k3_gfx942, conv2d_dgrad_bf16_ic1024_oc2048_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_bf16_ic32_oc64_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_bf16_ic32_oc64_k3_gfx942
EMBED_ASSET conv2d_dgrad_bf16_ic32_oc64_k3_gfx942, conv2d_dgrad_bf16_ic32_oc64_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_bf16_ic3_oc8_k3_dil2_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_bf16_ic3_oc8_k3_dil2_gfx942
EMBED_ASSET conv2d_dgrad_bf16_ic3_oc8_k3_dil2_gfx942, conv2d_dgrad_bf16_ic3_oc8_k3_dil2_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_bf16_ic3_oc8_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_bf16_ic3_oc8_k3_gfx942
EMBED_ASSET conv2d_dgrad_bf16_ic3_oc8_k3_gfx942, conv2d_dgrad_bf16_ic3_oc8_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_bf16_ic768_oc1536_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_bf16_ic768_oc1536_k3_gfx942
EMBED_ASSET conv2d_dgrad_bf16_ic768_oc1536_k3_gfx942, conv2d_dgrad_bf16_ic768_oc1536_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_bf16_ic768_oc768_k3_dil2_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_bf16_ic768_oc768_k3_dil2_gfx942
EMBED_ASSET conv2d_dgrad_bf16_ic768_oc768_k3_dil2_gfx942, conv2d_dgrad_bf16_ic768_oc768_k3_dil2_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_fp16_ic1024_oc1024_k3_dil2_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_fp16_ic1024_oc1024_k3_dil2_gfx942
EMBED_ASSET conv2d_dgrad_fp16_ic1024_oc1024_k3_dil2_gfx942, conv2d_dgrad_fp16_ic1024_oc1024_k3_dil2_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_fp16_ic1024_oc2048_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_fp16_ic1024_oc2048_k3_gfx942
EMBED_ASSET conv2d_dgrad_fp16_ic1024_oc2048_k3_gfx942, conv2d_dgrad_fp16_ic1024_oc2048_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_fp16_ic32_oc64_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_fp16_ic32_oc64_k3_gfx942
EMBED_ASSET conv2d_dgrad_fp16_ic32_oc64_k3_gfx942, conv2d_dgrad_fp16_ic32_oc64_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_fp16_ic3_oc8_k3_dil2_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_fp16_ic3_oc8_k3_dil2_gfx942
EMBED_ASSET conv2d_dgrad_fp16_ic3_oc8_k3_dil2_gfx942, conv2d_dgrad_fp16_ic3_oc8_k3_dil2_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_fp16_ic3_oc8_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_fp16_ic3_oc8_k3_gfx942
EMBED_ASSET conv2d_dgrad_fp16_ic3_oc8_k3_gfx942, conv2d_dgrad_fp16_ic3_oc8_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_fp16_ic768_oc1536_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_fp16_ic768_oc1536_k3_gfx942
EMBED_ASSET conv2d_dgrad_fp16_ic768_oc1536_k3_gfx942, conv2d_dgrad_fp16_ic768_oc1536_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_fp16_ic768_oc768_k3_dil2_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_fp16_ic768_oc768_k3_dil2_gfx942
EMBED_ASSET conv2d_dgrad_fp16_ic768_oc768_k3_dil2_gfx942, conv2d_dgrad_fp16_ic768_oc768_k3_dil2_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_fp16_ic1024_oc1024_k3_dil2_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_fp16_ic1024_oc1024_k3_dil2_gfx942
EMBED_ASSET conv2d_fp16_ic1024_oc1024_k3_dil2_gfx942, conv2d_fp16_ic1024_oc1024_k3_dil2_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_fp16_ic1024_oc2048_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_fp16_ic1024_oc2048_k3_gfx942
EMBED_ASSET conv2d_fp16_ic1024_oc2048_k3_gfx942, conv2d_fp16_ic1024_oc2048_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_fp16_ic32_oc64_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_fp16_ic32_oc64_k3_gfx942
EMBED_ASSET conv2d_fp16_ic32_oc64_k3_gfx942, conv2d_fp16_ic32_oc64_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_fp16_ic3_oc8_k3_dil2_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_fp16_ic3_oc8_k3_dil2_gfx942
EMBED_ASSET conv2d_fp16_ic3_oc8_k3_dil2_gfx942, conv2d_fp16_ic3_oc8_k3_dil2_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_fp16_ic3_oc8_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_fp16_ic3_oc8_k3_gfx942
EMBED_ASSET conv2d_fp16_ic3_oc8_k3_gfx942, conv2d_fp16_ic3_oc8_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_fp16_ic768_oc1536_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_fp16_ic768_oc1536_k3_gfx942
EMBED_ASSET conv2d_fp16_ic768_oc1536_k3_gfx942, conv2d_fp16_ic768_oc1536_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_fp16_ic768_oc768_k3_dil2_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_fp16_ic768_oc768_k3_dil2_gfx942
EMBED_ASSET conv2d_fp16_ic768_oc768_k3_dil2_gfx942, conv2d_fp16_ic768_oc768_k3_dil2_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_bf16_ic1024_oc1024_k3_dil2_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_bf16_ic1024_oc1024_k3_dil2_gfx942
EMBED_ASSET conv2d_wgrad_bf16_ic1024_oc1024_k3_dil2_gfx942, conv2d_wgrad_bf16_ic1024_oc1024_k3_dil2_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_bf16_ic1024_oc2048_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_bf16_ic1024_oc2048_k3_gfx942
EMBED_ASSET conv2d_wgrad_bf16_ic1024_oc2048_k3_gfx942, conv2d_wgrad_bf16_ic1024_oc2048_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_bf16_ic32_oc64_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_bf16_ic32_oc64_k3_gfx942
EMBED_ASSET conv2d_wgrad_bf16_ic32_oc64_k3_gfx942, conv2d_wgrad_bf16_ic32_oc64_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_bf16_ic3_oc8_k3_dil2_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_bf16_ic3_oc8_k3_dil2_gfx942
EMBED_ASSET conv2d_wgrad_bf16_ic3_oc8_k3_dil2_gfx942, conv2d_wgrad_bf16_ic3_oc8_k3_dil2_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_bf16_ic3_oc8_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_bf16_ic3_oc8_k3_gfx942
EMBED_ASSET conv2d_wgrad_bf16_ic3_oc8_k3_gfx942, conv2d_wgrad_bf16_ic3_oc8_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_bf16_ic768_oc1536_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_bf16_ic768_oc1536_k3_gfx942
EMBED_ASSET conv2d_wgrad_bf16_ic768_oc1536_k3_gfx942, conv2d_wgrad_bf16_ic768_oc1536_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_bf16_ic768_oc768_k3_dil2_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_bf16_ic768_oc768_k3_dil2_gfx942
EMBED_ASSET conv2d_wgrad_bf16_ic768_oc768_k3_dil2_gfx942, conv2d_wgrad_bf16_ic768_oc768_k3_dil2_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_fp16_ic1024_oc1024_k3_dil2_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_fp16_ic1024_oc1024_k3_dil2_gfx942
EMBED_ASSET conv2d_wgrad_fp16_ic1024_oc1024_k3_dil2_gfx942, conv2d_wgrad_fp16_ic1024_oc1024_k3_dil2_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_fp16_ic1024_oc2048_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_fp16_ic1024_oc2048_k3_gfx942
EMBED_ASSET conv2d_wgrad_fp16_ic1024_oc2048_k3_gfx942, conv2d_wgrad_fp16_ic1024_oc2048_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_fp16_ic32_oc64_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_fp16_ic32_oc64_k3_gfx942
EMBED_ASSET conv2d_wgrad_fp16_ic32_oc64_k3_gfx942, conv2d_wgrad_fp16_ic32_oc64_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_fp16_ic3_oc8_k3_dil2_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_fp16_ic3_oc8_k3_dil2_gfx942
EMBED_ASSET conv2d_wgrad_fp16_ic3_oc8_k3_dil2_gfx942, conv2d_wgrad_fp16_ic3_oc8_k3_dil2_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_fp16_ic3_oc8_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_fp16_ic3_oc8_k3_gfx942
EMBED_ASSET conv2d_wgrad_fp16_ic3_oc8_k3_gfx942, conv2d_wgrad_fp16_ic3_oc8_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_fp16_ic768_oc1536_k3_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_fp16_ic768_oc1536_k3_gfx942
EMBED_ASSET conv2d_wgrad_fp16_ic768_oc1536_k3_gfx942, conv2d_wgrad_fp16_ic768_oc1536_k3_gfx942.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_fp16_ic768_oc768_k3_dil2_gfx942.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_fp16_ic768_oc768_k3_dil2_gfx942
EMBED_ASSET conv2d_wgrad_fp16_ic768_oc768_k3_dil2_gfx942, conv2d_wgrad_fp16_ic768_oc768_k3_dil2_gfx942.hsaco, 4
#include "../kernel_gen/output/cross_entropy_on_targets_add_fp32_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO cross_entropy_on_targets_add_fp32_bf16_gfx942
EMBED_ASSET cross_entropy_on_targets_add_fp32_bf16_gfx942, cross_entropy_on_targets_add_fp32_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/cross_entropy_on_targets_add_fp32_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO cross_entropy_on_targets_add_fp32_fp16_gfx942
EMBED_ASSET cross_entropy_on_targets_add_fp32_fp16_gfx942, cross_entropy_on_targets_add_fp32_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/cross_entropy_on_targets_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO cross_entropy_on_targets_bf16_gfx942
EMBED_ASSET cross_entropy_on_targets_bf16_gfx942, cross_entropy_on_targets_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/cross_entropy_on_targets_bwd_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO cross_entropy_on_targets_bwd_bf16_gfx942
EMBED_ASSET cross_entropy_on_targets_bwd_bf16_gfx942, cross_entropy_on_targets_bwd_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/cross_entropy_on_targets_bwd_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO cross_entropy_on_targets_bwd_fp16_gfx942
EMBED_ASSET cross_entropy_on_targets_bwd_fp16_gfx942, cross_entropy_on_targets_bwd_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/cross_entropy_on_targets_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO cross_entropy_on_targets_fp16_gfx942
EMBED_ASSET cross_entropy_on_targets_fp16_gfx942, cross_entropy_on_targets_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/device_copy_strided_2d_gfx942.kinfo.inc.asm"
DECLARE_KINFO device_copy_strided_2d_gfx942
EMBED_ASSET device_copy_strided_2d_gfx942, device_copy_strided_2d_gfx942.hsaco, 4
#include "../kernel_gen/output/div_add_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO div_add_bf16_gfx942
EMBED_ASSET div_add_bf16_gfx942, div_add_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/div_add_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO div_add_fp16_gfx942
EMBED_ASSET div_add_fp16_gfx942, div_add_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/div_add_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO div_add_fp32_gfx942
EMBED_ASSET div_add_fp32_gfx942, div_add_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/div_elementwise_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO div_elementwise_bf16_gfx942
EMBED_ASSET div_elementwise_bf16_gfx942, div_elementwise_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/div_elementwise_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO div_elementwise_fp16_gfx942
EMBED_ASSET div_elementwise_fp16_gfx942, div_elementwise_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/div_elementwise_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO div_elementwise_fp32_gfx942
EMBED_ASSET div_elementwise_fp32_gfx942, div_elementwise_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/div_scalar_add_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO div_scalar_add_bf16_gfx942
EMBED_ASSET div_scalar_add_bf16_gfx942, div_scalar_add_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/div_scalar_add_broadcast_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO div_scalar_add_broadcast_bf16_gfx942
EMBED_ASSET div_scalar_add_broadcast_bf16_gfx942, div_scalar_add_broadcast_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/div_scalar_add_broadcast_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO div_scalar_add_broadcast_fp16_gfx942
EMBED_ASSET div_scalar_add_broadcast_fp16_gfx942, div_scalar_add_broadcast_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/div_scalar_add_broadcast_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO div_scalar_add_broadcast_fp32_gfx942
EMBED_ASSET div_scalar_add_broadcast_fp32_gfx942, div_scalar_add_broadcast_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/div_scalar_add_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO div_scalar_add_fp16_gfx942
EMBED_ASSET div_scalar_add_fp16_gfx942, div_scalar_add_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/div_scalar_add_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO div_scalar_add_fp32_gfx942
EMBED_ASSET div_scalar_add_fp32_gfx942, div_scalar_add_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/div_scalar_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO div_scalar_bf16_gfx942
EMBED_ASSET div_scalar_bf16_gfx942, div_scalar_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/div_scalar_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO div_scalar_fp16_gfx942
EMBED_ASSET div_scalar_fp16_gfx942, div_scalar_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/div_scalar_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO div_scalar_fp32_gfx942
EMBED_ASSET div_scalar_fp32_gfx942, div_scalar_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/embedding_lookup_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO embedding_lookup_bf16_gfx942
EMBED_ASSET embedding_lookup_bf16_gfx942, embedding_lookup_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/embedding_lookup_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO embedding_lookup_fp16_gfx942
EMBED_ASSET embedding_lookup_fp16_gfx942, embedding_lookup_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/fill_constant_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO fill_constant_bf16_gfx942
EMBED_ASSET fill_constant_bf16_gfx942, fill_constant_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/fill_constant_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO fill_constant_fp16_gfx942
EMBED_ASSET fill_constant_fp16_gfx942, fill_constant_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/fill_constant_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO fill_constant_fp32_gfx942
EMBED_ASSET fill_constant_fp32_gfx942, fill_constant_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/fill_normal_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO fill_normal_bf16_gfx942
EMBED_ASSET fill_normal_bf16_gfx942, fill_normal_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/fill_normal_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO fill_normal_fp16_gfx942
EMBED_ASSET fill_normal_fp16_gfx942, fill_normal_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/fill_uniform_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO fill_uniform_bf16_gfx942
EMBED_ASSET fill_uniform_bf16_gfx942, fill_uniform_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/fill_uniform_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO fill_uniform_fp16_gfx942
EMBED_ASSET fill_uniform_fp16_gfx942, fill_uniform_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/fill_uniform_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO fill_uniform_fp32_gfx942
EMBED_ASSET fill_uniform_fp32_gfx942, fill_uniform_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/fill_zeros_gfx942.kinfo.inc.asm"
DECLARE_KINFO fill_zeros_gfx942
EMBED_ASSET fill_zeros_gfx942, fill_zeros_gfx942.hsaco, 4
#include "../kernel_gen/output/layer_norm_fwd_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO layer_norm_fwd_bf16_gfx942
EMBED_ASSET layer_norm_fwd_bf16_gfx942, layer_norm_fwd_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/layer_norm_fwd_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO layer_norm_fwd_fp16_gfx942
EMBED_ASSET layer_norm_fwd_fp16_gfx942, layer_norm_fwd_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/lstm_cell_bwd_pointwise_out_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO lstm_cell_bwd_pointwise_out_bf16_gfx942
EMBED_ASSET lstm_cell_bwd_pointwise_out_bf16_gfx942, lstm_cell_bwd_pointwise_out_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/lstm_cell_bwd_pointwise_out_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO lstm_cell_bwd_pointwise_out_fp16_gfx942
EMBED_ASSET lstm_cell_bwd_pointwise_out_fp16_gfx942, lstm_cell_bwd_pointwise_out_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/lstm_cell_fwd_fp32_state_out_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO lstm_cell_fwd_fp32_state_out_bf16_gfx942
EMBED_ASSET lstm_cell_fwd_fp32_state_out_bf16_gfx942, lstm_cell_fwd_fp32_state_out_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/lstm_cell_fwd_fp32_state_out_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO lstm_cell_fwd_fp32_state_out_fp16_gfx942
EMBED_ASSET lstm_cell_fwd_fp32_state_out_fp16_gfx942, lstm_cell_fwd_fp32_state_out_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/lstm_cell_fwd_out_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO lstm_cell_fwd_out_fp16_gfx942
EMBED_ASSET lstm_cell_fwd_out_fp16_gfx942, lstm_cell_fwd_out_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/lstm_cell_recompute_out_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO lstm_cell_recompute_out_bf16_gfx942
EMBED_ASSET lstm_cell_recompute_out_bf16_gfx942, lstm_cell_recompute_out_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/lstm_cell_recompute_out_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO lstm_cell_recompute_out_fp16_gfx942
EMBED_ASSET lstm_cell_recompute_out_fp16_gfx942, lstm_cell_recompute_out_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_cacc_gfx942
EMBED_ASSET matmul_bf16_cacc_gfx942, matmul_bf16_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_gfx942
EMBED_ASSET matmul_bf16_gfx942, matmul_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_cacc_gfx942
EMBED_ASSET matmul_bf16_out_fp32_cacc_gfx942, matmul_bf16_out_fp32_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_gfx942
EMBED_ASSET matmul_bf16_out_fp32_gfx942, matmul_bf16_out_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_ta_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_ta_cacc_gfx942
EMBED_ASSET matmul_bf16_out_fp32_ta_cacc_gfx942, matmul_bf16_out_fp32_ta_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_ta_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_ta_gfx942
EMBED_ASSET matmul_bf16_out_fp32_ta_gfx942, matmul_bf16_out_fp32_ta_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_tab_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_tab_cacc_gfx942
EMBED_ASSET matmul_bf16_out_fp32_tab_cacc_gfx942, matmul_bf16_out_fp32_tab_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_tab_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_tab_gfx942
EMBED_ASSET matmul_bf16_out_fp32_tab_gfx942, matmul_bf16_out_fp32_tab_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_tb_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_tb_cacc_gfx942
EMBED_ASSET matmul_bf16_out_fp32_tb_cacc_gfx942, matmul_bf16_out_fp32_tb_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_tb_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_tb_gfx942
EMBED_ASSET matmul_bf16_out_fp32_tb_gfx942, matmul_bf16_out_fp32_tb_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_unaligned_cacc_gfx942
EMBED_ASSET matmul_bf16_out_fp32_unaligned_cacc_gfx942, matmul_bf16_out_fp32_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_unaligned_gfx942
EMBED_ASSET matmul_bf16_out_fp32_unaligned_gfx942, matmul_bf16_out_fp32_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_ta_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_ta_unaligned_cacc_gfx942
EMBED_ASSET matmul_bf16_out_fp32_ta_unaligned_cacc_gfx942, matmul_bf16_out_fp32_ta_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_ta_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_ta_unaligned_gfx942
EMBED_ASSET matmul_bf16_out_fp32_ta_unaligned_gfx942, matmul_bf16_out_fp32_ta_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_tab_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_tab_unaligned_cacc_gfx942
EMBED_ASSET matmul_bf16_out_fp32_tab_unaligned_cacc_gfx942, matmul_bf16_out_fp32_tab_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_tab_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_tab_unaligned_gfx942
EMBED_ASSET matmul_bf16_out_fp32_tab_unaligned_gfx942, matmul_bf16_out_fp32_tab_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_tb_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_tb_unaligned_cacc_gfx942
EMBED_ASSET matmul_bf16_out_fp32_tb_unaligned_cacc_gfx942, matmul_bf16_out_fp32_tb_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_tb_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_tb_unaligned_gfx942
EMBED_ASSET matmul_bf16_out_fp32_tb_unaligned_gfx942, matmul_bf16_out_fp32_tb_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_ta_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_ta_cacc_gfx942
EMBED_ASSET matmul_bf16_ta_cacc_gfx942, matmul_bf16_ta_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_ta_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_ta_gfx942
EMBED_ASSET matmul_bf16_ta_gfx942, matmul_bf16_ta_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_tab_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_tab_cacc_gfx942
EMBED_ASSET matmul_bf16_tab_cacc_gfx942, matmul_bf16_tab_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_tab_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_tab_gfx942
EMBED_ASSET matmul_bf16_tab_gfx942, matmul_bf16_tab_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_tb_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_tb_cacc_gfx942
EMBED_ASSET matmul_bf16_tb_cacc_gfx942, matmul_bf16_tb_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_tb_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_tb_gfx942
EMBED_ASSET matmul_bf16_tb_gfx942, matmul_bf16_tb_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_unaligned_cacc_gfx942
EMBED_ASSET matmul_bf16_unaligned_cacc_gfx942, matmul_bf16_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_unaligned_gfx942
EMBED_ASSET matmul_bf16_unaligned_gfx942, matmul_bf16_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_ta_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_ta_unaligned_cacc_gfx942
EMBED_ASSET matmul_bf16_ta_unaligned_cacc_gfx942, matmul_bf16_ta_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_ta_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_ta_unaligned_gfx942
EMBED_ASSET matmul_bf16_ta_unaligned_gfx942, matmul_bf16_ta_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_tab_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_tab_unaligned_cacc_gfx942
EMBED_ASSET matmul_bf16_tab_unaligned_cacc_gfx942, matmul_bf16_tab_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_tab_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_tab_unaligned_gfx942
EMBED_ASSET matmul_bf16_tab_unaligned_gfx942, matmul_bf16_tab_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_tb_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_tb_unaligned_cacc_gfx942
EMBED_ASSET matmul_bf16_tb_unaligned_cacc_gfx942, matmul_bf16_tb_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_tb_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_tb_unaligned_gfx942
EMBED_ASSET matmul_bf16_tb_unaligned_gfx942, matmul_bf16_tb_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_cacc_gfx942
EMBED_ASSET matmul_fp16_cacc_gfx942, matmul_fp16_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_gfx942
EMBED_ASSET matmul_fp16_gfx942, matmul_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_cacc_gfx942
EMBED_ASSET matmul_fp16_out_fp32_cacc_gfx942, matmul_fp16_out_fp32_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_gfx942
EMBED_ASSET matmul_fp16_out_fp32_gfx942, matmul_fp16_out_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_ta_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_ta_cacc_gfx942
EMBED_ASSET matmul_fp16_out_fp32_ta_cacc_gfx942, matmul_fp16_out_fp32_ta_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_ta_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_ta_gfx942
EMBED_ASSET matmul_fp16_out_fp32_ta_gfx942, matmul_fp16_out_fp32_ta_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_tab_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_tab_cacc_gfx942
EMBED_ASSET matmul_fp16_out_fp32_tab_cacc_gfx942, matmul_fp16_out_fp32_tab_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_tab_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_tab_gfx942
EMBED_ASSET matmul_fp16_out_fp32_tab_gfx942, matmul_fp16_out_fp32_tab_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_tb_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_tb_cacc_gfx942
EMBED_ASSET matmul_fp16_out_fp32_tb_cacc_gfx942, matmul_fp16_out_fp32_tb_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_tb_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_tb_gfx942
EMBED_ASSET matmul_fp16_out_fp32_tb_gfx942, matmul_fp16_out_fp32_tb_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_unaligned_cacc_gfx942
EMBED_ASSET matmul_fp16_out_fp32_unaligned_cacc_gfx942, matmul_fp16_out_fp32_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_unaligned_gfx942
EMBED_ASSET matmul_fp16_out_fp32_unaligned_gfx942, matmul_fp16_out_fp32_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_ta_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_ta_unaligned_cacc_gfx942
EMBED_ASSET matmul_fp16_out_fp32_ta_unaligned_cacc_gfx942, matmul_fp16_out_fp32_ta_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_ta_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_ta_unaligned_gfx942
EMBED_ASSET matmul_fp16_out_fp32_ta_unaligned_gfx942, matmul_fp16_out_fp32_ta_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_tab_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_tab_unaligned_cacc_gfx942
EMBED_ASSET matmul_fp16_out_fp32_tab_unaligned_cacc_gfx942, matmul_fp16_out_fp32_tab_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_tab_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_tab_unaligned_gfx942
EMBED_ASSET matmul_fp16_out_fp32_tab_unaligned_gfx942, matmul_fp16_out_fp32_tab_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_tb_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_tb_unaligned_cacc_gfx942
EMBED_ASSET matmul_fp16_out_fp32_tb_unaligned_cacc_gfx942, matmul_fp16_out_fp32_tb_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_tb_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_tb_unaligned_gfx942
EMBED_ASSET matmul_fp16_out_fp32_tb_unaligned_gfx942, matmul_fp16_out_fp32_tb_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_ta_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_ta_cacc_gfx942
EMBED_ASSET matmul_fp16_ta_cacc_gfx942, matmul_fp16_ta_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_ta_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_ta_gfx942
EMBED_ASSET matmul_fp16_ta_gfx942, matmul_fp16_ta_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_tab_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_tab_cacc_gfx942
EMBED_ASSET matmul_fp16_tab_cacc_gfx942, matmul_fp16_tab_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_tab_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_tab_gfx942
EMBED_ASSET matmul_fp16_tab_gfx942, matmul_fp16_tab_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_tb_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_tb_cacc_gfx942
EMBED_ASSET matmul_fp16_tb_cacc_gfx942, matmul_fp16_tb_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_tb_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_tb_gfx942
EMBED_ASSET matmul_fp16_tb_gfx942, matmul_fp16_tb_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_unaligned_cacc_gfx942
EMBED_ASSET matmul_fp16_unaligned_cacc_gfx942, matmul_fp16_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_unaligned_gfx942
EMBED_ASSET matmul_fp16_unaligned_gfx942, matmul_fp16_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_ta_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_ta_unaligned_cacc_gfx942
EMBED_ASSET matmul_fp16_ta_unaligned_cacc_gfx942, matmul_fp16_ta_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_ta_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_ta_unaligned_gfx942
EMBED_ASSET matmul_fp16_ta_unaligned_gfx942, matmul_fp16_ta_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_tab_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_tab_unaligned_cacc_gfx942
EMBED_ASSET matmul_fp16_tab_unaligned_cacc_gfx942, matmul_fp16_tab_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_tab_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_tab_unaligned_gfx942
EMBED_ASSET matmul_fp16_tab_unaligned_gfx942, matmul_fp16_tab_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_tb_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_tb_unaligned_cacc_gfx942
EMBED_ASSET matmul_fp16_tb_unaligned_cacc_gfx942, matmul_fp16_tb_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_tb_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_tb_unaligned_gfx942
EMBED_ASSET matmul_fp16_tb_unaligned_gfx942, matmul_fp16_tb_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bf16_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bf16_cacc_gfx942
EMBED_ASSET matmul_gelu_bf16_cacc_gfx942, matmul_gelu_bf16_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bf16_gfx942
EMBED_ASSET matmul_gelu_bf16_gfx942, matmul_gelu_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bf16_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bf16_unaligned_cacc_gfx942
EMBED_ASSET matmul_gelu_bf16_unaligned_cacc_gfx942, matmul_gelu_bf16_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bf16_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bf16_unaligned_gfx942
EMBED_ASSET matmul_gelu_bf16_unaligned_gfx942, matmul_gelu_bf16_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_cacc_gfx942
EMBED_ASSET matmul_gelu_bwd_bf16_cacc_gfx942, matmul_gelu_bwd_bf16_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_gfx942
EMBED_ASSET matmul_gelu_bwd_bf16_gfx942, matmul_gelu_bwd_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_ta_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_ta_cacc_gfx942
EMBED_ASSET matmul_gelu_bwd_bf16_ta_cacc_gfx942, matmul_gelu_bwd_bf16_ta_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_ta_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_ta_gfx942
EMBED_ASSET matmul_gelu_bwd_bf16_ta_gfx942, matmul_gelu_bwd_bf16_ta_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_tab_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_tab_cacc_gfx942
EMBED_ASSET matmul_gelu_bwd_bf16_tab_cacc_gfx942, matmul_gelu_bwd_bf16_tab_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_tab_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_tab_gfx942
EMBED_ASSET matmul_gelu_bwd_bf16_tab_gfx942, matmul_gelu_bwd_bf16_tab_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_tb_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_tb_cacc_gfx942
EMBED_ASSET matmul_gelu_bwd_bf16_tb_cacc_gfx942, matmul_gelu_bwd_bf16_tb_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_tb_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_tb_gfx942
EMBED_ASSET matmul_gelu_bwd_bf16_tb_gfx942, matmul_gelu_bwd_bf16_tb_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_unaligned_cacc_gfx942
EMBED_ASSET matmul_gelu_bwd_bf16_unaligned_cacc_gfx942, matmul_gelu_bwd_bf16_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_unaligned_gfx942
EMBED_ASSET matmul_gelu_bwd_bf16_unaligned_gfx942, matmul_gelu_bwd_bf16_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_unaligned_ta_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_unaligned_ta_cacc_gfx942
EMBED_ASSET matmul_gelu_bwd_bf16_unaligned_ta_cacc_gfx942, matmul_gelu_bwd_bf16_unaligned_ta_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_unaligned_ta_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_unaligned_ta_gfx942
EMBED_ASSET matmul_gelu_bwd_bf16_unaligned_ta_gfx942, matmul_gelu_bwd_bf16_unaligned_ta_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_unaligned_tab_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_unaligned_tab_cacc_gfx942
EMBED_ASSET matmul_gelu_bwd_bf16_unaligned_tab_cacc_gfx942, matmul_gelu_bwd_bf16_unaligned_tab_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_unaligned_tab_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_unaligned_tab_gfx942
EMBED_ASSET matmul_gelu_bwd_bf16_unaligned_tab_gfx942, matmul_gelu_bwd_bf16_unaligned_tab_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_unaligned_tb_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_unaligned_tb_cacc_gfx942
EMBED_ASSET matmul_gelu_bwd_bf16_unaligned_tb_cacc_gfx942, matmul_gelu_bwd_bf16_unaligned_tb_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_unaligned_tb_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_unaligned_tb_gfx942
EMBED_ASSET matmul_gelu_bwd_bf16_unaligned_tb_gfx942, matmul_gelu_bwd_bf16_unaligned_tb_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_cacc_gfx942
EMBED_ASSET matmul_gelu_bwd_fp16_cacc_gfx942, matmul_gelu_bwd_fp16_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_gfx942
EMBED_ASSET matmul_gelu_bwd_fp16_gfx942, matmul_gelu_bwd_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_ta_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_ta_cacc_gfx942
EMBED_ASSET matmul_gelu_bwd_fp16_ta_cacc_gfx942, matmul_gelu_bwd_fp16_ta_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_ta_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_ta_gfx942
EMBED_ASSET matmul_gelu_bwd_fp16_ta_gfx942, matmul_gelu_bwd_fp16_ta_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_tab_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_tab_cacc_gfx942
EMBED_ASSET matmul_gelu_bwd_fp16_tab_cacc_gfx942, matmul_gelu_bwd_fp16_tab_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_tab_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_tab_gfx942
EMBED_ASSET matmul_gelu_bwd_fp16_tab_gfx942, matmul_gelu_bwd_fp16_tab_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_tb_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_tb_cacc_gfx942
EMBED_ASSET matmul_gelu_bwd_fp16_tb_cacc_gfx942, matmul_gelu_bwd_fp16_tb_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_tb_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_tb_gfx942
EMBED_ASSET matmul_gelu_bwd_fp16_tb_gfx942, matmul_gelu_bwd_fp16_tb_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_unaligned_cacc_gfx942
EMBED_ASSET matmul_gelu_bwd_fp16_unaligned_cacc_gfx942, matmul_gelu_bwd_fp16_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_unaligned_gfx942
EMBED_ASSET matmul_gelu_bwd_fp16_unaligned_gfx942, matmul_gelu_bwd_fp16_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_unaligned_ta_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_unaligned_ta_cacc_gfx942
EMBED_ASSET matmul_gelu_bwd_fp16_unaligned_ta_cacc_gfx942, matmul_gelu_bwd_fp16_unaligned_ta_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_unaligned_ta_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_unaligned_ta_gfx942
EMBED_ASSET matmul_gelu_bwd_fp16_unaligned_ta_gfx942, matmul_gelu_bwd_fp16_unaligned_ta_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_unaligned_tab_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_unaligned_tab_cacc_gfx942
EMBED_ASSET matmul_gelu_bwd_fp16_unaligned_tab_cacc_gfx942, matmul_gelu_bwd_fp16_unaligned_tab_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_unaligned_tab_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_unaligned_tab_gfx942
EMBED_ASSET matmul_gelu_bwd_fp16_unaligned_tab_gfx942, matmul_gelu_bwd_fp16_unaligned_tab_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_unaligned_tb_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_unaligned_tb_cacc_gfx942
EMBED_ASSET matmul_gelu_bwd_fp16_unaligned_tb_cacc_gfx942, matmul_gelu_bwd_fp16_unaligned_tb_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_unaligned_tb_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_unaligned_tb_gfx942
EMBED_ASSET matmul_gelu_bwd_fp16_unaligned_tb_gfx942, matmul_gelu_bwd_fp16_unaligned_tb_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_fp16_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_fp16_cacc_gfx942
EMBED_ASSET matmul_gelu_fp16_cacc_gfx942, matmul_gelu_fp16_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_fp16_gfx942
EMBED_ASSET matmul_gelu_fp16_gfx942, matmul_gelu_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_fp16_unaligned_cacc_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_fp16_unaligned_cacc_gfx942
EMBED_ASSET matmul_gelu_fp16_unaligned_cacc_gfx942, matmul_gelu_fp16_unaligned_cacc_gfx942.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_fp16_unaligned_gfx942.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_fp16_unaligned_gfx942
EMBED_ASSET matmul_gelu_fp16_unaligned_gfx942, matmul_gelu_fp16_unaligned_gfx942.hsaco, 4
#include "../kernel_gen/output/mean_reduce_column_tiled_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO mean_reduce_column_tiled_bf16_gfx942
EMBED_ASSET mean_reduce_column_tiled_bf16_gfx942, mean_reduce_column_tiled_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/mean_reduce_column_tiled_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO mean_reduce_column_tiled_fp16_gfx942
EMBED_ASSET mean_reduce_column_tiled_fp16_gfx942, mean_reduce_column_tiled_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/mean_reduce_contiguous_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO mean_reduce_contiguous_bf16_gfx942
EMBED_ASSET mean_reduce_contiguous_bf16_gfx942, mean_reduce_contiguous_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/mean_reduce_contiguous_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO mean_reduce_contiguous_fp16_gfx942
EMBED_ASSET mean_reduce_contiguous_fp16_gfx942, mean_reduce_contiguous_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_bwd_hs128_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_bwd_hs128_bf16_gfx942
EMBED_ASSET mha_full_attn_bwd_hs128_bf16_gfx942, mha_full_attn_bwd_hs128_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_bwd_hs128_bf16_uneven_gfx942.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_bwd_hs128_bf16_uneven_gfx942
EMBED_ASSET mha_full_attn_bwd_hs128_bf16_uneven_gfx942, mha_full_attn_bwd_hs128_bf16_uneven_gfx942.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_bwd_hs128_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_bwd_hs128_fp16_gfx942
EMBED_ASSET mha_full_attn_bwd_hs128_fp16_gfx942, mha_full_attn_bwd_hs128_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_bwd_hs128_fp16_uneven_gfx942.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_bwd_hs128_fp16_uneven_gfx942
EMBED_ASSET mha_full_attn_bwd_hs128_fp16_uneven_gfx942, mha_full_attn_bwd_hs128_fp16_uneven_gfx942.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_bwd_pre_hs128_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_bwd_pre_hs128_bf16_gfx942
EMBED_ASSET mha_full_attn_bwd_pre_hs128_bf16_gfx942, mha_full_attn_bwd_pre_hs128_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_bwd_pre_hs128_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_bwd_pre_hs128_fp16_gfx942
EMBED_ASSET mha_full_attn_bwd_pre_hs128_fp16_gfx942, mha_full_attn_bwd_pre_hs128_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_fwd_hs128_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_fwd_hs128_bf16_gfx942
EMBED_ASSET mha_full_attn_fwd_hs128_bf16_gfx942, mha_full_attn_fwd_hs128_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_fwd_hs128_bf16_nolse_gfx942.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_fwd_hs128_bf16_nolse_gfx942
EMBED_ASSET mha_full_attn_fwd_hs128_bf16_nolse_gfx942, mha_full_attn_fwd_hs128_bf16_nolse_gfx942.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_fwd_hs128_bf16_uneven_gfx942.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_fwd_hs128_bf16_uneven_gfx942
EMBED_ASSET mha_full_attn_fwd_hs128_bf16_uneven_gfx942, mha_full_attn_fwd_hs128_bf16_uneven_gfx942.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_fwd_hs128_bf16_uneven_nolse_gfx942.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_fwd_hs128_bf16_uneven_nolse_gfx942
EMBED_ASSET mha_full_attn_fwd_hs128_bf16_uneven_nolse_gfx942, mha_full_attn_fwd_hs128_bf16_uneven_nolse_gfx942.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_fwd_hs128_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_fwd_hs128_fp16_gfx942
EMBED_ASSET mha_full_attn_fwd_hs128_fp16_gfx942, mha_full_attn_fwd_hs128_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_fwd_hs128_fp16_nolse_gfx942.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_fwd_hs128_fp16_nolse_gfx942
EMBED_ASSET mha_full_attn_fwd_hs128_fp16_nolse_gfx942, mha_full_attn_fwd_hs128_fp16_nolse_gfx942.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_fwd_hs128_fp16_uneven_gfx942.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_fwd_hs128_fp16_uneven_gfx942
EMBED_ASSET mha_full_attn_fwd_hs128_fp16_uneven_gfx942, mha_full_attn_fwd_hs128_fp16_uneven_gfx942.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_fwd_hs128_fp16_uneven_nolse_gfx942.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_fwd_hs128_fp16_uneven_nolse_gfx942
EMBED_ASSET mha_full_attn_fwd_hs128_fp16_uneven_nolse_gfx942, mha_full_attn_fwd_hs128_fp16_uneven_nolse_gfx942.hsaco, 4
#include "../kernel_gen/output/mul_elementwise_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO mul_elementwise_bf16_gfx942
EMBED_ASSET mul_elementwise_bf16_gfx942, mul_elementwise_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/mul_elementwise_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO mul_elementwise_fp16_gfx942
EMBED_ASSET mul_elementwise_fp16_gfx942, mul_elementwise_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/mul_elementwise_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO mul_elementwise_fp32_gfx942
EMBED_ASSET mul_elementwise_fp32_gfx942, mul_elementwise_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/mul_reduce_column_tiled_bf16_out_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO mul_reduce_column_tiled_bf16_out_fp32_gfx942
EMBED_ASSET mul_reduce_column_tiled_bf16_out_fp32_gfx942, mul_reduce_column_tiled_bf16_out_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/mul_reduce_column_tiled_fp16_out_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO mul_reduce_column_tiled_fp16_out_fp32_gfx942
EMBED_ASSET mul_reduce_column_tiled_fp16_out_fp32_gfx942, mul_reduce_column_tiled_fp16_out_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/mul_reduce_contiguous_bf16_out_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO mul_reduce_contiguous_bf16_out_fp32_gfx942
EMBED_ASSET mul_reduce_contiguous_bf16_out_fp32_gfx942, mul_reduce_contiguous_bf16_out_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/mul_reduce_contiguous_fp16_out_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO mul_reduce_contiguous_fp16_out_fp32_gfx942
EMBED_ASSET mul_reduce_contiguous_fp16_out_fp32_gfx942, mul_reduce_contiguous_fp16_out_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/mul_scalar_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO mul_scalar_bf16_gfx942
EMBED_ASSET mul_scalar_bf16_gfx942, mul_scalar_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/mul_scalar_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO mul_scalar_fp16_gfx942
EMBED_ASSET mul_scalar_fp16_gfx942, mul_scalar_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/mul_scalar_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO mul_scalar_fp32_gfx942
EMBED_ASSET mul_scalar_fp32_gfx942, mul_scalar_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/sqrt_elementwise_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO sqrt_elementwise_bf16_gfx942
EMBED_ASSET sqrt_elementwise_bf16_gfx942, sqrt_elementwise_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/sqrt_elementwise_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO sqrt_elementwise_fp16_gfx942
EMBED_ASSET sqrt_elementwise_fp16_gfx942, sqrt_elementwise_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/sqrt_elementwise_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO sqrt_elementwise_fp32_gfx942
EMBED_ASSET sqrt_elementwise_fp32_gfx942, sqrt_elementwise_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/mul_trailing_broadcast_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO mul_trailing_broadcast_bf16_gfx942
EMBED_ASSET mul_trailing_broadcast_bf16_gfx942, mul_trailing_broadcast_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/mul_trailing_broadcast_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO mul_trailing_broadcast_fp16_gfx942
EMBED_ASSET mul_trailing_broadcast_fp16_gfx942, mul_trailing_broadcast_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/mul_trailing_broadcast_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO mul_trailing_broadcast_fp32_gfx942
EMBED_ASSET mul_trailing_broadcast_fp32_gfx942, mul_trailing_broadcast_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/reduce_sum_partial_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO reduce_sum_partial_fp32_gfx942
EMBED_ASSET reduce_sum_partial_fp32_gfx942, reduce_sum_partial_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/rms_norm_bwd_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO rms_norm_bwd_bf16_gfx942
EMBED_ASSET rms_norm_bwd_bf16_gfx942, rms_norm_bwd_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/rms_norm_bwd_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO rms_norm_bwd_fp16_gfx942
EMBED_ASSET rms_norm_bwd_fp16_gfx942, rms_norm_bwd_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/rms_norm_fwd_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO rms_norm_fwd_bf16_gfx942
EMBED_ASSET rms_norm_fwd_bf16_gfx942, rms_norm_fwd_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/rms_norm_fwd_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO rms_norm_fwd_fp16_gfx942
EMBED_ASSET rms_norm_fwd_fp16_gfx942, rms_norm_fwd_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/sgd_step_bf16_gfx942.kinfo.inc.asm"
DECLARE_KINFO sgd_step_bf16_gfx942
EMBED_ASSET sgd_step_bf16_gfx942, sgd_step_bf16_gfx942.hsaco, 4
#include "../kernel_gen/output/sgd_step_fp16_gfx942.kinfo.inc.asm"
DECLARE_KINFO sgd_step_fp16_gfx942
EMBED_ASSET sgd_step_fp16_gfx942, sgd_step_fp16_gfx942.hsaco, 4
#include "../kernel_gen/output/sgd_step_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO sgd_step_fp32_gfx942
EMBED_ASSET sgd_step_fp32_gfx942, sgd_step_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/sum_reduce_column_tiled_bf16_out_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO sum_reduce_column_tiled_bf16_out_fp32_gfx942
EMBED_ASSET sum_reduce_column_tiled_bf16_out_fp32_gfx942, sum_reduce_column_tiled_bf16_out_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/sum_reduce_column_tiled_fp16_out_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO sum_reduce_column_tiled_fp16_out_fp32_gfx942
EMBED_ASSET sum_reduce_column_tiled_fp16_out_fp32_gfx942, sum_reduce_column_tiled_fp16_out_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/sum_reduce_column_tiled_fp32_out_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO sum_reduce_column_tiled_fp32_out_fp32_gfx942
EMBED_ASSET sum_reduce_column_tiled_fp32_out_fp32_gfx942, sum_reduce_column_tiled_fp32_out_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/sum_reduce_contiguous_bf16_out_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO sum_reduce_contiguous_bf16_out_fp32_gfx942
EMBED_ASSET sum_reduce_contiguous_bf16_out_fp32_gfx942, sum_reduce_contiguous_bf16_out_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/sum_reduce_contiguous_fp16_out_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO sum_reduce_contiguous_fp16_out_fp32_gfx942
EMBED_ASSET sum_reduce_contiguous_fp16_out_fp32_gfx942, sum_reduce_contiguous_fp16_out_fp32_gfx942.hsaco, 4
#include "../kernel_gen/output/sum_reduce_contiguous_fp32_out_fp32_gfx942.kinfo.inc.asm"
DECLARE_KINFO sum_reduce_contiguous_fp32_out_fp32_gfx942
EMBED_ASSET sum_reduce_contiguous_fp32_out_fp32_gfx942, sum_reduce_contiguous_fp32_out_fp32_gfx942.hsaco, 4

#elif AMD_KERNEL_ARCH == 1101
#include "../kernel_gen/output/act_gelu_bwd_elementwise_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO act_gelu_bwd_elementwise_bf16_gfx1101
EMBED_ASSET act_gelu_bwd_elementwise_bf16_gfx1101, act_gelu_bwd_elementwise_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/act_gelu_bwd_elementwise_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO act_gelu_bwd_elementwise_fp16_gfx1101
EMBED_ASSET act_gelu_bwd_elementwise_fp16_gfx1101, act_gelu_bwd_elementwise_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/act_gelu_elementwise_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO act_gelu_elementwise_bf16_gfx1101
EMBED_ASSET act_gelu_elementwise_bf16_gfx1101, act_gelu_elementwise_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/act_gelu_elementwise_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO act_gelu_elementwise_fp16_gfx1101
EMBED_ASSET act_gelu_elementwise_fp16_gfx1101, act_gelu_elementwise_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/act_relu_elementwise_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO act_relu_elementwise_bf16_gfx1101
EMBED_ASSET act_relu_elementwise_bf16_gfx1101, act_relu_elementwise_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/act_relu_elementwise_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO act_relu_elementwise_fp16_gfx1101
EMBED_ASSET act_relu_elementwise_fp16_gfx1101, act_relu_elementwise_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/adamw_step_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO adamw_step_bf16_gfx1101
EMBED_ASSET adamw_step_bf16_gfx1101, adamw_step_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/adamw_step_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO adamw_step_fp16_gfx1101
EMBED_ASSET adamw_step_fp16_gfx1101, adamw_step_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/adamw_step_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO adamw_step_fp32_gfx1101
EMBED_ASSET adamw_step_fp32_gfx1101, adamw_step_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/add_elementwise_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO add_elementwise_bf16_gfx1101
EMBED_ASSET add_elementwise_bf16_gfx1101, add_elementwise_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/add_elementwise_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO add_elementwise_fp16_gfx1101
EMBED_ASSET add_elementwise_fp16_gfx1101, add_elementwise_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/add_elementwise_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO add_elementwise_fp32_gfx1101
EMBED_ASSET add_elementwise_fp32_gfx1101, add_elementwise_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/add_elementwise_fp32_out_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO add_elementwise_fp32_out_fp16_gfx1101
EMBED_ASSET add_elementwise_fp32_out_fp16_gfx1101, add_elementwise_fp32_out_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/add_trailing_broadcast_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO add_trailing_broadcast_bf16_gfx1101
EMBED_ASSET add_trailing_broadcast_bf16_gfx1101, add_trailing_broadcast_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/add_trailing_broadcast_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO add_trailing_broadcast_fp16_gfx1101
EMBED_ASSET add_trailing_broadcast_fp16_gfx1101, add_trailing_broadcast_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/add_trailing_broadcast_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO add_trailing_broadcast_fp32_gfx1101
EMBED_ASSET add_trailing_broadcast_fp32_gfx1101, add_trailing_broadcast_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_bf16_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_bf16_cacc_gfx1101
EMBED_ASSET addmm_bf16_cacc_gfx1101, addmm_bf16_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_bf16_gfx1101
EMBED_ASSET addmm_bf16_gfx1101, addmm_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_bf16_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_bf16_unaligned_cacc_gfx1101
EMBED_ASSET addmm_bf16_unaligned_cacc_gfx1101, addmm_bf16_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_bf16_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_bf16_unaligned_gfx1101
EMBED_ASSET addmm_bf16_unaligned_gfx1101, addmm_bf16_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_fp16_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_fp16_cacc_gfx1101
EMBED_ASSET addmm_fp16_cacc_gfx1101, addmm_fp16_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_fp16_gfx1101
EMBED_ASSET addmm_fp16_gfx1101, addmm_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_fp16_out_fp32_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_fp16_out_fp32_cacc_gfx1101
EMBED_ASSET addmm_fp16_out_fp32_cacc_gfx1101, addmm_fp16_out_fp32_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_fp16_out_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_fp16_out_fp32_gfx1101
EMBED_ASSET addmm_fp16_out_fp32_gfx1101, addmm_fp16_out_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_fp16_out_fp32_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_fp16_out_fp32_unaligned_cacc_gfx1101
EMBED_ASSET addmm_fp16_out_fp32_unaligned_cacc_gfx1101, addmm_fp16_out_fp32_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_fp16_out_fp32_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_fp16_out_fp32_unaligned_gfx1101
EMBED_ASSET addmm_fp16_out_fp32_unaligned_gfx1101, addmm_fp16_out_fp32_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_fp16_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_fp16_unaligned_cacc_gfx1101
EMBED_ASSET addmm_fp16_unaligned_cacc_gfx1101, addmm_fp16_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_fp16_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_fp16_unaligned_gfx1101
EMBED_ASSET addmm_fp16_unaligned_gfx1101, addmm_fp16_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_bf16_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_bf16_cacc_gfx1101
EMBED_ASSET addmm_gelu_bf16_cacc_gfx1101, addmm_gelu_bf16_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_bf16_gfx1101
EMBED_ASSET addmm_gelu_bf16_gfx1101, addmm_gelu_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_bf16_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_bf16_unaligned_cacc_gfx1101
EMBED_ASSET addmm_gelu_bf16_unaligned_cacc_gfx1101, addmm_gelu_bf16_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_bf16_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_bf16_unaligned_gfx1101
EMBED_ASSET addmm_gelu_bf16_unaligned_gfx1101, addmm_gelu_bf16_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_fp16_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_fp16_cacc_gfx1101
EMBED_ASSET addmm_gelu_fp16_cacc_gfx1101, addmm_gelu_fp16_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_fp16_gfx1101
EMBED_ASSET addmm_gelu_fp16_gfx1101, addmm_gelu_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_fp16_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_fp16_unaligned_cacc_gfx1101
EMBED_ASSET addmm_gelu_fp16_unaligned_cacc_gfx1101, addmm_gelu_fp16_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_fp16_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_fp16_unaligned_gfx1101
EMBED_ASSET addmm_gelu_fp16_unaligned_gfx1101, addmm_gelu_fp16_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_preact_bf16_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_preact_bf16_cacc_gfx1101
EMBED_ASSET addmm_gelu_preact_bf16_cacc_gfx1101, addmm_gelu_preact_bf16_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_preact_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_preact_bf16_gfx1101
EMBED_ASSET addmm_gelu_preact_bf16_gfx1101, addmm_gelu_preact_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_preact_bf16_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_preact_bf16_unaligned_cacc_gfx1101
EMBED_ASSET addmm_gelu_preact_bf16_unaligned_cacc_gfx1101, addmm_gelu_preact_bf16_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_preact_bf16_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_preact_bf16_unaligned_gfx1101
EMBED_ASSET addmm_gelu_preact_bf16_unaligned_gfx1101, addmm_gelu_preact_bf16_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_preact_fp16_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_preact_fp16_cacc_gfx1101
EMBED_ASSET addmm_gelu_preact_fp16_cacc_gfx1101, addmm_gelu_preact_fp16_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_preact_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_preact_fp16_gfx1101
EMBED_ASSET addmm_gelu_preact_fp16_gfx1101, addmm_gelu_preact_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_preact_fp16_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_preact_fp16_unaligned_cacc_gfx1101
EMBED_ASSET addmm_gelu_preact_fp16_unaligned_cacc_gfx1101, addmm_gelu_preact_fp16_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/addmm_gelu_preact_fp16_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO addmm_gelu_preact_fp16_unaligned_gfx1101
EMBED_ASSET addmm_gelu_preact_fp16_unaligned_gfx1101, addmm_gelu_preact_fp16_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/avg_pool1d_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO avg_pool1d_bf16_gfx1101
EMBED_ASSET avg_pool1d_bf16_gfx1101, avg_pool1d_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/avg_pool1d_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO avg_pool1d_fp16_gfx1101
EMBED_ASSET avg_pool1d_fp16_gfx1101, avg_pool1d_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/avg_pool2d_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO avg_pool2d_bf16_gfx1101
EMBED_ASSET avg_pool2d_bf16_gfx1101, avg_pool2d_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/avg_pool2d_bwd_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO avg_pool2d_bwd_bf16_gfx1101
EMBED_ASSET avg_pool2d_bwd_bf16_gfx1101, avg_pool2d_bwd_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/avg_pool2d_bwd_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO avg_pool2d_bwd_fp16_gfx1101
EMBED_ASSET avg_pool2d_bwd_fp16_gfx1101, avg_pool2d_bwd_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/avg_pool2d_bwd_noaccum_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO avg_pool2d_bwd_noaccum_bf16_gfx1101
EMBED_ASSET avg_pool2d_bwd_noaccum_bf16_gfx1101, avg_pool2d_bwd_noaccum_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/avg_pool2d_bwd_noaccum_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO avg_pool2d_bwd_noaccum_fp16_gfx1101
EMBED_ASSET avg_pool2d_bwd_noaccum_fp16_gfx1101, avg_pool2d_bwd_noaccum_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/avg_pool2d_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO avg_pool2d_fp16_gfx1101
EMBED_ASSET avg_pool2d_fp16_gfx1101, avg_pool2d_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bf16_gfx1101
EMBED_ASSET build_cell_embeds_bf16_gfx1101, build_cell_embeds_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_color_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_color_bf16_gfx1101
EMBED_ASSET build_cell_embeds_bwd_color_bf16_gfx1101, build_cell_embeds_bwd_color_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_color_bf16_out_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_color_bf16_out_fp32_gfx1101
EMBED_ASSET build_cell_embeds_bwd_color_bf16_out_fp32_gfx1101, build_cell_embeds_bwd_color_bf16_out_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_color_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_color_fp16_gfx1101
EMBED_ASSET build_cell_embeds_bwd_color_fp16_gfx1101, build_cell_embeds_bwd_color_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_color_fp16_out_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_color_fp16_out_fp32_gfx1101
EMBED_ASSET build_cell_embeds_bwd_color_fp16_out_fp32_gfx1101, build_cell_embeds_bwd_color_fp16_out_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_cp_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_cp_bf16_gfx1101
EMBED_ASSET build_cell_embeds_bwd_cp_bf16_gfx1101, build_cell_embeds_bwd_cp_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_cp_bf16_out_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_cp_bf16_out_fp32_gfx1101
EMBED_ASSET build_cell_embeds_bwd_cp_bf16_out_fp32_gfx1101, build_cell_embeds_bwd_cp_bf16_out_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_cp_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_cp_fp16_gfx1101
EMBED_ASSET build_cell_embeds_bwd_cp_fp16_gfx1101, build_cell_embeds_bwd_cp_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_cp_fp16_out_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_cp_fp16_out_fp32_gfx1101
EMBED_ASSET build_cell_embeds_bwd_cp_fp16_out_fp32_gfx1101, build_cell_embeds_bwd_cp_fp16_out_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_pos_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_pos_bf16_gfx1101
EMBED_ASSET build_cell_embeds_bwd_pos_bf16_gfx1101, build_cell_embeds_bwd_pos_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_pos_bf16_out_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_pos_bf16_out_fp32_gfx1101
EMBED_ASSET build_cell_embeds_bwd_pos_bf16_out_fp32_gfx1101, build_cell_embeds_bwd_pos_bf16_out_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_pos_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_pos_fp16_gfx1101
EMBED_ASSET build_cell_embeds_bwd_pos_fp16_gfx1101, build_cell_embeds_bwd_pos_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_bwd_pos_fp16_out_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_bwd_pos_fp16_out_fp32_gfx1101
EMBED_ASSET build_cell_embeds_bwd_pos_fp16_out_fp32_gfx1101, build_cell_embeds_bwd_pos_fp16_out_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/build_cell_embeds_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO build_cell_embeds_fp16_gfx1101
EMBED_ASSET build_cell_embeds_fp16_gfx1101, build_cell_embeds_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/cast_bf16_to_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO cast_bf16_to_fp16_gfx1101
EMBED_ASSET cast_bf16_to_fp16_gfx1101, cast_bf16_to_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/cast_bf16_to_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO cast_bf16_to_fp32_gfx1101
EMBED_ASSET cast_bf16_to_fp32_gfx1101, cast_bf16_to_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/cast_fp16_to_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO cast_fp16_to_bf16_gfx1101
EMBED_ASSET cast_fp16_to_bf16_gfx1101, cast_fp16_to_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/cast_fp16_to_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO cast_fp16_to_fp32_gfx1101
EMBED_ASSET cast_fp16_to_fp32_gfx1101, cast_fp16_to_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/cast_fp32_to_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO cast_fp32_to_bf16_gfx1101
EMBED_ASSET cast_fp32_to_bf16_gfx1101, cast_fp32_to_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/cast_fp32_to_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO cast_fp32_to_fp16_gfx1101
EMBED_ASSET cast_fp32_to_fp16_gfx1101, cast_fp32_to_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/contiguous_3d_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO contiguous_3d_bf16_gfx1101
EMBED_ASSET contiguous_3d_bf16_gfx1101, contiguous_3d_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/contiguous_3d_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO contiguous_3d_fp16_gfx1101
EMBED_ASSET contiguous_3d_fp16_gfx1101, contiguous_3d_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/contiguous_4d_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO contiguous_4d_bf16_gfx1101
EMBED_ASSET contiguous_4d_bf16_gfx1101, contiguous_4d_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/contiguous_4d_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO contiguous_4d_fp16_gfx1101
EMBED_ASSET contiguous_4d_fp16_gfx1101, contiguous_4d_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_bf16_ic1024_oc1024_k3_dil2_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_bf16_ic1024_oc1024_k3_dil2_gfx1101
EMBED_ASSET conv2d_bf16_ic1024_oc1024_k3_dil2_gfx1101, conv2d_bf16_ic1024_oc1024_k3_dil2_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_bf16_ic1024_oc2048_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_bf16_ic1024_oc2048_k3_gfx1101
EMBED_ASSET conv2d_bf16_ic1024_oc2048_k3_gfx1101, conv2d_bf16_ic1024_oc2048_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_bf16_ic32_oc64_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_bf16_ic32_oc64_k3_gfx1101
EMBED_ASSET conv2d_bf16_ic32_oc64_k3_gfx1101, conv2d_bf16_ic32_oc64_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_bf16_ic3_oc8_k3_dil2_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_bf16_ic3_oc8_k3_dil2_gfx1101
EMBED_ASSET conv2d_bf16_ic3_oc8_k3_dil2_gfx1101, conv2d_bf16_ic3_oc8_k3_dil2_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_bf16_ic3_oc8_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_bf16_ic3_oc8_k3_gfx1101
EMBED_ASSET conv2d_bf16_ic3_oc8_k3_gfx1101, conv2d_bf16_ic3_oc8_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_bf16_ic768_oc1536_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_bf16_ic768_oc1536_k3_gfx1101
EMBED_ASSET conv2d_bf16_ic768_oc1536_k3_gfx1101, conv2d_bf16_ic768_oc1536_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_bf16_ic768_oc768_k3_dil2_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_bf16_ic768_oc768_k3_dil2_gfx1101
EMBED_ASSET conv2d_bf16_ic768_oc768_k3_dil2_gfx1101, conv2d_bf16_ic768_oc768_k3_dil2_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_bf16_ic1024_oc1024_k3_dil2_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_bf16_ic1024_oc1024_k3_dil2_gfx1101
EMBED_ASSET conv2d_dgrad_bf16_ic1024_oc1024_k3_dil2_gfx1101, conv2d_dgrad_bf16_ic1024_oc1024_k3_dil2_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_bf16_ic1024_oc2048_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_bf16_ic1024_oc2048_k3_gfx1101
EMBED_ASSET conv2d_dgrad_bf16_ic1024_oc2048_k3_gfx1101, conv2d_dgrad_bf16_ic1024_oc2048_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_bf16_ic32_oc64_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_bf16_ic32_oc64_k3_gfx1101
EMBED_ASSET conv2d_dgrad_bf16_ic32_oc64_k3_gfx1101, conv2d_dgrad_bf16_ic32_oc64_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_bf16_ic3_oc8_k3_dil2_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_bf16_ic3_oc8_k3_dil2_gfx1101
EMBED_ASSET conv2d_dgrad_bf16_ic3_oc8_k3_dil2_gfx1101, conv2d_dgrad_bf16_ic3_oc8_k3_dil2_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_bf16_ic3_oc8_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_bf16_ic3_oc8_k3_gfx1101
EMBED_ASSET conv2d_dgrad_bf16_ic3_oc8_k3_gfx1101, conv2d_dgrad_bf16_ic3_oc8_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_bf16_ic768_oc1536_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_bf16_ic768_oc1536_k3_gfx1101
EMBED_ASSET conv2d_dgrad_bf16_ic768_oc1536_k3_gfx1101, conv2d_dgrad_bf16_ic768_oc1536_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_bf16_ic768_oc768_k3_dil2_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_bf16_ic768_oc768_k3_dil2_gfx1101
EMBED_ASSET conv2d_dgrad_bf16_ic768_oc768_k3_dil2_gfx1101, conv2d_dgrad_bf16_ic768_oc768_k3_dil2_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_fp16_ic1024_oc1024_k3_dil2_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_fp16_ic1024_oc1024_k3_dil2_gfx1101
EMBED_ASSET conv2d_dgrad_fp16_ic1024_oc1024_k3_dil2_gfx1101, conv2d_dgrad_fp16_ic1024_oc1024_k3_dil2_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_fp16_ic1024_oc2048_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_fp16_ic1024_oc2048_k3_gfx1101
EMBED_ASSET conv2d_dgrad_fp16_ic1024_oc2048_k3_gfx1101, conv2d_dgrad_fp16_ic1024_oc2048_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_fp16_ic32_oc64_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_fp16_ic32_oc64_k3_gfx1101
EMBED_ASSET conv2d_dgrad_fp16_ic32_oc64_k3_gfx1101, conv2d_dgrad_fp16_ic32_oc64_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_fp16_ic3_oc8_k3_dil2_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_fp16_ic3_oc8_k3_dil2_gfx1101
EMBED_ASSET conv2d_dgrad_fp16_ic3_oc8_k3_dil2_gfx1101, conv2d_dgrad_fp16_ic3_oc8_k3_dil2_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_fp16_ic3_oc8_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_fp16_ic3_oc8_k3_gfx1101
EMBED_ASSET conv2d_dgrad_fp16_ic3_oc8_k3_gfx1101, conv2d_dgrad_fp16_ic3_oc8_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_fp16_ic768_oc1536_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_fp16_ic768_oc1536_k3_gfx1101
EMBED_ASSET conv2d_dgrad_fp16_ic768_oc1536_k3_gfx1101, conv2d_dgrad_fp16_ic768_oc1536_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_dgrad_fp16_ic768_oc768_k3_dil2_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_dgrad_fp16_ic768_oc768_k3_dil2_gfx1101
EMBED_ASSET conv2d_dgrad_fp16_ic768_oc768_k3_dil2_gfx1101, conv2d_dgrad_fp16_ic768_oc768_k3_dil2_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_fp16_ic1024_oc1024_k3_dil2_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_fp16_ic1024_oc1024_k3_dil2_gfx1101
EMBED_ASSET conv2d_fp16_ic1024_oc1024_k3_dil2_gfx1101, conv2d_fp16_ic1024_oc1024_k3_dil2_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_fp16_ic1024_oc2048_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_fp16_ic1024_oc2048_k3_gfx1101
EMBED_ASSET conv2d_fp16_ic1024_oc2048_k3_gfx1101, conv2d_fp16_ic1024_oc2048_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_fp16_ic32_oc64_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_fp16_ic32_oc64_k3_gfx1101
EMBED_ASSET conv2d_fp16_ic32_oc64_k3_gfx1101, conv2d_fp16_ic32_oc64_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_fp16_ic3_oc8_k3_dil2_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_fp16_ic3_oc8_k3_dil2_gfx1101
EMBED_ASSET conv2d_fp16_ic3_oc8_k3_dil2_gfx1101, conv2d_fp16_ic3_oc8_k3_dil2_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_fp16_ic3_oc8_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_fp16_ic3_oc8_k3_gfx1101
EMBED_ASSET conv2d_fp16_ic3_oc8_k3_gfx1101, conv2d_fp16_ic3_oc8_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_fp16_ic768_oc1536_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_fp16_ic768_oc1536_k3_gfx1101
EMBED_ASSET conv2d_fp16_ic768_oc1536_k3_gfx1101, conv2d_fp16_ic768_oc1536_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_fp16_ic768_oc768_k3_dil2_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_fp16_ic768_oc768_k3_dil2_gfx1101
EMBED_ASSET conv2d_fp16_ic768_oc768_k3_dil2_gfx1101, conv2d_fp16_ic768_oc768_k3_dil2_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_bf16_ic1024_oc1024_k3_dil2_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_bf16_ic1024_oc1024_k3_dil2_gfx1101
EMBED_ASSET conv2d_wgrad_bf16_ic1024_oc1024_k3_dil2_gfx1101, conv2d_wgrad_bf16_ic1024_oc1024_k3_dil2_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_bf16_ic1024_oc2048_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_bf16_ic1024_oc2048_k3_gfx1101
EMBED_ASSET conv2d_wgrad_bf16_ic1024_oc2048_k3_gfx1101, conv2d_wgrad_bf16_ic1024_oc2048_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_bf16_ic32_oc64_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_bf16_ic32_oc64_k3_gfx1101
EMBED_ASSET conv2d_wgrad_bf16_ic32_oc64_k3_gfx1101, conv2d_wgrad_bf16_ic32_oc64_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_bf16_ic3_oc8_k3_dil2_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_bf16_ic3_oc8_k3_dil2_gfx1101
EMBED_ASSET conv2d_wgrad_bf16_ic3_oc8_k3_dil2_gfx1101, conv2d_wgrad_bf16_ic3_oc8_k3_dil2_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_bf16_ic3_oc8_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_bf16_ic3_oc8_k3_gfx1101
EMBED_ASSET conv2d_wgrad_bf16_ic3_oc8_k3_gfx1101, conv2d_wgrad_bf16_ic3_oc8_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_bf16_ic768_oc1536_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_bf16_ic768_oc1536_k3_gfx1101
EMBED_ASSET conv2d_wgrad_bf16_ic768_oc1536_k3_gfx1101, conv2d_wgrad_bf16_ic768_oc1536_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_bf16_ic768_oc768_k3_dil2_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_bf16_ic768_oc768_k3_dil2_gfx1101
EMBED_ASSET conv2d_wgrad_bf16_ic768_oc768_k3_dil2_gfx1101, conv2d_wgrad_bf16_ic768_oc768_k3_dil2_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_fp16_ic1024_oc1024_k3_dil2_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_fp16_ic1024_oc1024_k3_dil2_gfx1101
EMBED_ASSET conv2d_wgrad_fp16_ic1024_oc1024_k3_dil2_gfx1101, conv2d_wgrad_fp16_ic1024_oc1024_k3_dil2_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_fp16_ic1024_oc2048_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_fp16_ic1024_oc2048_k3_gfx1101
EMBED_ASSET conv2d_wgrad_fp16_ic1024_oc2048_k3_gfx1101, conv2d_wgrad_fp16_ic1024_oc2048_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_fp16_ic32_oc64_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_fp16_ic32_oc64_k3_gfx1101
EMBED_ASSET conv2d_wgrad_fp16_ic32_oc64_k3_gfx1101, conv2d_wgrad_fp16_ic32_oc64_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_fp16_ic3_oc8_k3_dil2_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_fp16_ic3_oc8_k3_dil2_gfx1101
EMBED_ASSET conv2d_wgrad_fp16_ic3_oc8_k3_dil2_gfx1101, conv2d_wgrad_fp16_ic3_oc8_k3_dil2_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_fp16_ic3_oc8_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_fp16_ic3_oc8_k3_gfx1101
EMBED_ASSET conv2d_wgrad_fp16_ic3_oc8_k3_gfx1101, conv2d_wgrad_fp16_ic3_oc8_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_fp16_ic768_oc1536_k3_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_fp16_ic768_oc1536_k3_gfx1101
EMBED_ASSET conv2d_wgrad_fp16_ic768_oc1536_k3_gfx1101, conv2d_wgrad_fp16_ic768_oc1536_k3_gfx1101.hsaco, 4
#include "../kernel_gen/output/conv2d_wgrad_fp16_ic768_oc768_k3_dil2_gfx1101.kinfo.inc.asm"
DECLARE_KINFO conv2d_wgrad_fp16_ic768_oc768_k3_dil2_gfx1101
EMBED_ASSET conv2d_wgrad_fp16_ic768_oc768_k3_dil2_gfx1101, conv2d_wgrad_fp16_ic768_oc768_k3_dil2_gfx1101.hsaco, 4
#include "../kernel_gen/output/cross_entropy_on_targets_add_fp32_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO cross_entropy_on_targets_add_fp32_bf16_gfx1101
EMBED_ASSET cross_entropy_on_targets_add_fp32_bf16_gfx1101, cross_entropy_on_targets_add_fp32_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/cross_entropy_on_targets_add_fp32_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO cross_entropy_on_targets_add_fp32_fp16_gfx1101
EMBED_ASSET cross_entropy_on_targets_add_fp32_fp16_gfx1101, cross_entropy_on_targets_add_fp32_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/cross_entropy_on_targets_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO cross_entropy_on_targets_bf16_gfx1101
EMBED_ASSET cross_entropy_on_targets_bf16_gfx1101, cross_entropy_on_targets_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/cross_entropy_on_targets_bwd_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO cross_entropy_on_targets_bwd_bf16_gfx1101
EMBED_ASSET cross_entropy_on_targets_bwd_bf16_gfx1101, cross_entropy_on_targets_bwd_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/cross_entropy_on_targets_bwd_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO cross_entropy_on_targets_bwd_fp16_gfx1101
EMBED_ASSET cross_entropy_on_targets_bwd_fp16_gfx1101, cross_entropy_on_targets_bwd_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/cross_entropy_on_targets_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO cross_entropy_on_targets_fp16_gfx1101
EMBED_ASSET cross_entropy_on_targets_fp16_gfx1101, cross_entropy_on_targets_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/device_copy_strided_2d_gfx1101.kinfo.inc.asm"
DECLARE_KINFO device_copy_strided_2d_gfx1101
EMBED_ASSET device_copy_strided_2d_gfx1101, device_copy_strided_2d_gfx1101.hsaco, 4
#include "../kernel_gen/output/div_add_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO div_add_bf16_gfx1101
EMBED_ASSET div_add_bf16_gfx1101, div_add_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/div_add_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO div_add_fp16_gfx1101
EMBED_ASSET div_add_fp16_gfx1101, div_add_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/div_add_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO div_add_fp32_gfx1101
EMBED_ASSET div_add_fp32_gfx1101, div_add_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/div_elementwise_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO div_elementwise_bf16_gfx1101
EMBED_ASSET div_elementwise_bf16_gfx1101, div_elementwise_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/div_elementwise_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO div_elementwise_fp16_gfx1101
EMBED_ASSET div_elementwise_fp16_gfx1101, div_elementwise_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/div_elementwise_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO div_elementwise_fp32_gfx1101
EMBED_ASSET div_elementwise_fp32_gfx1101, div_elementwise_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/div_scalar_add_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO div_scalar_add_bf16_gfx1101
EMBED_ASSET div_scalar_add_bf16_gfx1101, div_scalar_add_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/div_scalar_add_broadcast_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO div_scalar_add_broadcast_bf16_gfx1101
EMBED_ASSET div_scalar_add_broadcast_bf16_gfx1101, div_scalar_add_broadcast_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/div_scalar_add_broadcast_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO div_scalar_add_broadcast_fp16_gfx1101
EMBED_ASSET div_scalar_add_broadcast_fp16_gfx1101, div_scalar_add_broadcast_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/div_scalar_add_broadcast_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO div_scalar_add_broadcast_fp32_gfx1101
EMBED_ASSET div_scalar_add_broadcast_fp32_gfx1101, div_scalar_add_broadcast_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/div_scalar_add_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO div_scalar_add_fp16_gfx1101
EMBED_ASSET div_scalar_add_fp16_gfx1101, div_scalar_add_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/div_scalar_add_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO div_scalar_add_fp32_gfx1101
EMBED_ASSET div_scalar_add_fp32_gfx1101, div_scalar_add_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/div_scalar_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO div_scalar_bf16_gfx1101
EMBED_ASSET div_scalar_bf16_gfx1101, div_scalar_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/div_scalar_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO div_scalar_fp16_gfx1101
EMBED_ASSET div_scalar_fp16_gfx1101, div_scalar_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/div_scalar_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO div_scalar_fp32_gfx1101
EMBED_ASSET div_scalar_fp32_gfx1101, div_scalar_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/embedding_lookup_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO embedding_lookup_bf16_gfx1101
EMBED_ASSET embedding_lookup_bf16_gfx1101, embedding_lookup_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/embedding_lookup_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO embedding_lookup_fp16_gfx1101
EMBED_ASSET embedding_lookup_fp16_gfx1101, embedding_lookup_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/fill_constant_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO fill_constant_bf16_gfx1101
EMBED_ASSET fill_constant_bf16_gfx1101, fill_constant_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/fill_constant_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO fill_constant_fp16_gfx1101
EMBED_ASSET fill_constant_fp16_gfx1101, fill_constant_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/fill_constant_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO fill_constant_fp32_gfx1101
EMBED_ASSET fill_constant_fp32_gfx1101, fill_constant_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/fill_normal_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO fill_normal_bf16_gfx1101
EMBED_ASSET fill_normal_bf16_gfx1101, fill_normal_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/fill_normal_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO fill_normal_fp16_gfx1101
EMBED_ASSET fill_normal_fp16_gfx1101, fill_normal_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/fill_uniform_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO fill_uniform_bf16_gfx1101
EMBED_ASSET fill_uniform_bf16_gfx1101, fill_uniform_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/fill_uniform_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO fill_uniform_fp16_gfx1101
EMBED_ASSET fill_uniform_fp16_gfx1101, fill_uniform_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/fill_uniform_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO fill_uniform_fp32_gfx1101
EMBED_ASSET fill_uniform_fp32_gfx1101, fill_uniform_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/fill_zeros_gfx1101.kinfo.inc.asm"
DECLARE_KINFO fill_zeros_gfx1101
EMBED_ASSET fill_zeros_gfx1101, fill_zeros_gfx1101.hsaco, 4
#include "../kernel_gen/output/layer_norm_fwd_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO layer_norm_fwd_bf16_gfx1101
EMBED_ASSET layer_norm_fwd_bf16_gfx1101, layer_norm_fwd_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/layer_norm_fwd_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO layer_norm_fwd_fp16_gfx1101
EMBED_ASSET layer_norm_fwd_fp16_gfx1101, layer_norm_fwd_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/lstm_cell_bwd_pointwise_out_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO lstm_cell_bwd_pointwise_out_bf16_gfx1101
EMBED_ASSET lstm_cell_bwd_pointwise_out_bf16_gfx1101, lstm_cell_bwd_pointwise_out_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/lstm_cell_bwd_pointwise_out_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO lstm_cell_bwd_pointwise_out_fp16_gfx1101
EMBED_ASSET lstm_cell_bwd_pointwise_out_fp16_gfx1101, lstm_cell_bwd_pointwise_out_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/lstm_cell_fwd_fp32_state_out_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO lstm_cell_fwd_fp32_state_out_bf16_gfx1101
EMBED_ASSET lstm_cell_fwd_fp32_state_out_bf16_gfx1101, lstm_cell_fwd_fp32_state_out_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/lstm_cell_fwd_fp32_state_out_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO lstm_cell_fwd_fp32_state_out_fp16_gfx1101
EMBED_ASSET lstm_cell_fwd_fp32_state_out_fp16_gfx1101, lstm_cell_fwd_fp32_state_out_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/lstm_cell_fwd_out_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO lstm_cell_fwd_out_fp16_gfx1101
EMBED_ASSET lstm_cell_fwd_out_fp16_gfx1101, lstm_cell_fwd_out_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/lstm_cell_recompute_out_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO lstm_cell_recompute_out_bf16_gfx1101
EMBED_ASSET lstm_cell_recompute_out_bf16_gfx1101, lstm_cell_recompute_out_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/lstm_cell_recompute_out_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO lstm_cell_recompute_out_fp16_gfx1101
EMBED_ASSET lstm_cell_recompute_out_fp16_gfx1101, lstm_cell_recompute_out_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_cacc_gfx1101
EMBED_ASSET matmul_bf16_cacc_gfx1101, matmul_bf16_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_gfx1101
EMBED_ASSET matmul_bf16_gfx1101, matmul_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_cacc_gfx1101
EMBED_ASSET matmul_bf16_out_fp32_cacc_gfx1101, matmul_bf16_out_fp32_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_gfx1101
EMBED_ASSET matmul_bf16_out_fp32_gfx1101, matmul_bf16_out_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_ta_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_ta_cacc_gfx1101
EMBED_ASSET matmul_bf16_out_fp32_ta_cacc_gfx1101, matmul_bf16_out_fp32_ta_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_ta_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_ta_gfx1101
EMBED_ASSET matmul_bf16_out_fp32_ta_gfx1101, matmul_bf16_out_fp32_ta_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_tab_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_tab_cacc_gfx1101
EMBED_ASSET matmul_bf16_out_fp32_tab_cacc_gfx1101, matmul_bf16_out_fp32_tab_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_tab_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_tab_gfx1101
EMBED_ASSET matmul_bf16_out_fp32_tab_gfx1101, matmul_bf16_out_fp32_tab_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_tb_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_tb_cacc_gfx1101
EMBED_ASSET matmul_bf16_out_fp32_tb_cacc_gfx1101, matmul_bf16_out_fp32_tb_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_tb_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_tb_gfx1101
EMBED_ASSET matmul_bf16_out_fp32_tb_gfx1101, matmul_bf16_out_fp32_tb_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_unaligned_cacc_gfx1101
EMBED_ASSET matmul_bf16_out_fp32_unaligned_cacc_gfx1101, matmul_bf16_out_fp32_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_unaligned_gfx1101
EMBED_ASSET matmul_bf16_out_fp32_unaligned_gfx1101, matmul_bf16_out_fp32_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_ta_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_ta_unaligned_cacc_gfx1101
EMBED_ASSET matmul_bf16_out_fp32_ta_unaligned_cacc_gfx1101, matmul_bf16_out_fp32_ta_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_ta_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_ta_unaligned_gfx1101
EMBED_ASSET matmul_bf16_out_fp32_ta_unaligned_gfx1101, matmul_bf16_out_fp32_ta_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_tab_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_tab_unaligned_cacc_gfx1101
EMBED_ASSET matmul_bf16_out_fp32_tab_unaligned_cacc_gfx1101, matmul_bf16_out_fp32_tab_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_tab_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_tab_unaligned_gfx1101
EMBED_ASSET matmul_bf16_out_fp32_tab_unaligned_gfx1101, matmul_bf16_out_fp32_tab_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_tb_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_tb_unaligned_cacc_gfx1101
EMBED_ASSET matmul_bf16_out_fp32_tb_unaligned_cacc_gfx1101, matmul_bf16_out_fp32_tb_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_out_fp32_tb_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_out_fp32_tb_unaligned_gfx1101
EMBED_ASSET matmul_bf16_out_fp32_tb_unaligned_gfx1101, matmul_bf16_out_fp32_tb_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_ta_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_ta_cacc_gfx1101
EMBED_ASSET matmul_bf16_ta_cacc_gfx1101, matmul_bf16_ta_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_ta_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_ta_gfx1101
EMBED_ASSET matmul_bf16_ta_gfx1101, matmul_bf16_ta_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_tab_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_tab_cacc_gfx1101
EMBED_ASSET matmul_bf16_tab_cacc_gfx1101, matmul_bf16_tab_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_tab_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_tab_gfx1101
EMBED_ASSET matmul_bf16_tab_gfx1101, matmul_bf16_tab_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_tb_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_tb_cacc_gfx1101
EMBED_ASSET matmul_bf16_tb_cacc_gfx1101, matmul_bf16_tb_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_tb_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_tb_gfx1101
EMBED_ASSET matmul_bf16_tb_gfx1101, matmul_bf16_tb_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_unaligned_cacc_gfx1101
EMBED_ASSET matmul_bf16_unaligned_cacc_gfx1101, matmul_bf16_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_unaligned_gfx1101
EMBED_ASSET matmul_bf16_unaligned_gfx1101, matmul_bf16_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_ta_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_ta_unaligned_cacc_gfx1101
EMBED_ASSET matmul_bf16_ta_unaligned_cacc_gfx1101, matmul_bf16_ta_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_ta_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_ta_unaligned_gfx1101
EMBED_ASSET matmul_bf16_ta_unaligned_gfx1101, matmul_bf16_ta_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_tab_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_tab_unaligned_cacc_gfx1101
EMBED_ASSET matmul_bf16_tab_unaligned_cacc_gfx1101, matmul_bf16_tab_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_tab_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_tab_unaligned_gfx1101
EMBED_ASSET matmul_bf16_tab_unaligned_gfx1101, matmul_bf16_tab_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_tb_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_tb_unaligned_cacc_gfx1101
EMBED_ASSET matmul_bf16_tb_unaligned_cacc_gfx1101, matmul_bf16_tb_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_bf16_tb_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_bf16_tb_unaligned_gfx1101
EMBED_ASSET matmul_bf16_tb_unaligned_gfx1101, matmul_bf16_tb_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_cacc_gfx1101
EMBED_ASSET matmul_fp16_cacc_gfx1101, matmul_fp16_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_gfx1101
EMBED_ASSET matmul_fp16_gfx1101, matmul_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_cacc_gfx1101
EMBED_ASSET matmul_fp16_out_fp32_cacc_gfx1101, matmul_fp16_out_fp32_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_gfx1101
EMBED_ASSET matmul_fp16_out_fp32_gfx1101, matmul_fp16_out_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_ta_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_ta_cacc_gfx1101
EMBED_ASSET matmul_fp16_out_fp32_ta_cacc_gfx1101, matmul_fp16_out_fp32_ta_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_ta_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_ta_gfx1101
EMBED_ASSET matmul_fp16_out_fp32_ta_gfx1101, matmul_fp16_out_fp32_ta_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_tab_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_tab_cacc_gfx1101
EMBED_ASSET matmul_fp16_out_fp32_tab_cacc_gfx1101, matmul_fp16_out_fp32_tab_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_tab_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_tab_gfx1101
EMBED_ASSET matmul_fp16_out_fp32_tab_gfx1101, matmul_fp16_out_fp32_tab_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_tb_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_tb_cacc_gfx1101
EMBED_ASSET matmul_fp16_out_fp32_tb_cacc_gfx1101, matmul_fp16_out_fp32_tb_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_tb_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_tb_gfx1101
EMBED_ASSET matmul_fp16_out_fp32_tb_gfx1101, matmul_fp16_out_fp32_tb_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_unaligned_cacc_gfx1101
EMBED_ASSET matmul_fp16_out_fp32_unaligned_cacc_gfx1101, matmul_fp16_out_fp32_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_unaligned_gfx1101
EMBED_ASSET matmul_fp16_out_fp32_unaligned_gfx1101, matmul_fp16_out_fp32_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_ta_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_ta_unaligned_cacc_gfx1101
EMBED_ASSET matmul_fp16_out_fp32_ta_unaligned_cacc_gfx1101, matmul_fp16_out_fp32_ta_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_ta_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_ta_unaligned_gfx1101
EMBED_ASSET matmul_fp16_out_fp32_ta_unaligned_gfx1101, matmul_fp16_out_fp32_ta_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_tab_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_tab_unaligned_cacc_gfx1101
EMBED_ASSET matmul_fp16_out_fp32_tab_unaligned_cacc_gfx1101, matmul_fp16_out_fp32_tab_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_tab_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_tab_unaligned_gfx1101
EMBED_ASSET matmul_fp16_out_fp32_tab_unaligned_gfx1101, matmul_fp16_out_fp32_tab_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_tb_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_tb_unaligned_cacc_gfx1101
EMBED_ASSET matmul_fp16_out_fp32_tb_unaligned_cacc_gfx1101, matmul_fp16_out_fp32_tb_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_out_fp32_tb_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_out_fp32_tb_unaligned_gfx1101
EMBED_ASSET matmul_fp16_out_fp32_tb_unaligned_gfx1101, matmul_fp16_out_fp32_tb_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_ta_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_ta_cacc_gfx1101
EMBED_ASSET matmul_fp16_ta_cacc_gfx1101, matmul_fp16_ta_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_ta_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_ta_gfx1101
EMBED_ASSET matmul_fp16_ta_gfx1101, matmul_fp16_ta_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_tab_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_tab_cacc_gfx1101
EMBED_ASSET matmul_fp16_tab_cacc_gfx1101, matmul_fp16_tab_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_tab_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_tab_gfx1101
EMBED_ASSET matmul_fp16_tab_gfx1101, matmul_fp16_tab_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_tb_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_tb_cacc_gfx1101
EMBED_ASSET matmul_fp16_tb_cacc_gfx1101, matmul_fp16_tb_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_tb_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_tb_gfx1101
EMBED_ASSET matmul_fp16_tb_gfx1101, matmul_fp16_tb_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_unaligned_cacc_gfx1101
EMBED_ASSET matmul_fp16_unaligned_cacc_gfx1101, matmul_fp16_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_unaligned_gfx1101
EMBED_ASSET matmul_fp16_unaligned_gfx1101, matmul_fp16_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_ta_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_ta_unaligned_cacc_gfx1101
EMBED_ASSET matmul_fp16_ta_unaligned_cacc_gfx1101, matmul_fp16_ta_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_ta_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_ta_unaligned_gfx1101
EMBED_ASSET matmul_fp16_ta_unaligned_gfx1101, matmul_fp16_ta_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_tab_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_tab_unaligned_cacc_gfx1101
EMBED_ASSET matmul_fp16_tab_unaligned_cacc_gfx1101, matmul_fp16_tab_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_tab_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_tab_unaligned_gfx1101
EMBED_ASSET matmul_fp16_tab_unaligned_gfx1101, matmul_fp16_tab_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_tb_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_tb_unaligned_cacc_gfx1101
EMBED_ASSET matmul_fp16_tb_unaligned_cacc_gfx1101, matmul_fp16_tb_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_fp16_tb_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_fp16_tb_unaligned_gfx1101
EMBED_ASSET matmul_fp16_tb_unaligned_gfx1101, matmul_fp16_tb_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bf16_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bf16_cacc_gfx1101
EMBED_ASSET matmul_gelu_bf16_cacc_gfx1101, matmul_gelu_bf16_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bf16_gfx1101
EMBED_ASSET matmul_gelu_bf16_gfx1101, matmul_gelu_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bf16_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bf16_unaligned_cacc_gfx1101
EMBED_ASSET matmul_gelu_bf16_unaligned_cacc_gfx1101, matmul_gelu_bf16_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bf16_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bf16_unaligned_gfx1101
EMBED_ASSET matmul_gelu_bf16_unaligned_gfx1101, matmul_gelu_bf16_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_cacc_gfx1101
EMBED_ASSET matmul_gelu_bwd_bf16_cacc_gfx1101, matmul_gelu_bwd_bf16_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_gfx1101
EMBED_ASSET matmul_gelu_bwd_bf16_gfx1101, matmul_gelu_bwd_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_ta_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_ta_cacc_gfx1101
EMBED_ASSET matmul_gelu_bwd_bf16_ta_cacc_gfx1101, matmul_gelu_bwd_bf16_ta_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_ta_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_ta_gfx1101
EMBED_ASSET matmul_gelu_bwd_bf16_ta_gfx1101, matmul_gelu_bwd_bf16_ta_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_tab_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_tab_cacc_gfx1101
EMBED_ASSET matmul_gelu_bwd_bf16_tab_cacc_gfx1101, matmul_gelu_bwd_bf16_tab_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_tab_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_tab_gfx1101
EMBED_ASSET matmul_gelu_bwd_bf16_tab_gfx1101, matmul_gelu_bwd_bf16_tab_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_tb_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_tb_cacc_gfx1101
EMBED_ASSET matmul_gelu_bwd_bf16_tb_cacc_gfx1101, matmul_gelu_bwd_bf16_tb_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_tb_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_tb_gfx1101
EMBED_ASSET matmul_gelu_bwd_bf16_tb_gfx1101, matmul_gelu_bwd_bf16_tb_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_unaligned_cacc_gfx1101
EMBED_ASSET matmul_gelu_bwd_bf16_unaligned_cacc_gfx1101, matmul_gelu_bwd_bf16_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_unaligned_gfx1101
EMBED_ASSET matmul_gelu_bwd_bf16_unaligned_gfx1101, matmul_gelu_bwd_bf16_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_unaligned_ta_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_unaligned_ta_cacc_gfx1101
EMBED_ASSET matmul_gelu_bwd_bf16_unaligned_ta_cacc_gfx1101, matmul_gelu_bwd_bf16_unaligned_ta_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_unaligned_ta_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_unaligned_ta_gfx1101
EMBED_ASSET matmul_gelu_bwd_bf16_unaligned_ta_gfx1101, matmul_gelu_bwd_bf16_unaligned_ta_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_unaligned_tab_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_unaligned_tab_cacc_gfx1101
EMBED_ASSET matmul_gelu_bwd_bf16_unaligned_tab_cacc_gfx1101, matmul_gelu_bwd_bf16_unaligned_tab_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_unaligned_tab_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_unaligned_tab_gfx1101
EMBED_ASSET matmul_gelu_bwd_bf16_unaligned_tab_gfx1101, matmul_gelu_bwd_bf16_unaligned_tab_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_unaligned_tb_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_unaligned_tb_cacc_gfx1101
EMBED_ASSET matmul_gelu_bwd_bf16_unaligned_tb_cacc_gfx1101, matmul_gelu_bwd_bf16_unaligned_tb_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_bf16_unaligned_tb_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_bf16_unaligned_tb_gfx1101
EMBED_ASSET matmul_gelu_bwd_bf16_unaligned_tb_gfx1101, matmul_gelu_bwd_bf16_unaligned_tb_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_cacc_gfx1101
EMBED_ASSET matmul_gelu_bwd_fp16_cacc_gfx1101, matmul_gelu_bwd_fp16_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_gfx1101
EMBED_ASSET matmul_gelu_bwd_fp16_gfx1101, matmul_gelu_bwd_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_ta_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_ta_cacc_gfx1101
EMBED_ASSET matmul_gelu_bwd_fp16_ta_cacc_gfx1101, matmul_gelu_bwd_fp16_ta_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_ta_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_ta_gfx1101
EMBED_ASSET matmul_gelu_bwd_fp16_ta_gfx1101, matmul_gelu_bwd_fp16_ta_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_tab_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_tab_cacc_gfx1101
EMBED_ASSET matmul_gelu_bwd_fp16_tab_cacc_gfx1101, matmul_gelu_bwd_fp16_tab_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_tab_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_tab_gfx1101
EMBED_ASSET matmul_gelu_bwd_fp16_tab_gfx1101, matmul_gelu_bwd_fp16_tab_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_tb_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_tb_cacc_gfx1101
EMBED_ASSET matmul_gelu_bwd_fp16_tb_cacc_gfx1101, matmul_gelu_bwd_fp16_tb_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_tb_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_tb_gfx1101
EMBED_ASSET matmul_gelu_bwd_fp16_tb_gfx1101, matmul_gelu_bwd_fp16_tb_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_unaligned_cacc_gfx1101
EMBED_ASSET matmul_gelu_bwd_fp16_unaligned_cacc_gfx1101, matmul_gelu_bwd_fp16_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_unaligned_gfx1101
EMBED_ASSET matmul_gelu_bwd_fp16_unaligned_gfx1101, matmul_gelu_bwd_fp16_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_unaligned_ta_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_unaligned_ta_cacc_gfx1101
EMBED_ASSET matmul_gelu_bwd_fp16_unaligned_ta_cacc_gfx1101, matmul_gelu_bwd_fp16_unaligned_ta_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_unaligned_ta_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_unaligned_ta_gfx1101
EMBED_ASSET matmul_gelu_bwd_fp16_unaligned_ta_gfx1101, matmul_gelu_bwd_fp16_unaligned_ta_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_unaligned_tab_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_unaligned_tab_cacc_gfx1101
EMBED_ASSET matmul_gelu_bwd_fp16_unaligned_tab_cacc_gfx1101, matmul_gelu_bwd_fp16_unaligned_tab_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_unaligned_tab_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_unaligned_tab_gfx1101
EMBED_ASSET matmul_gelu_bwd_fp16_unaligned_tab_gfx1101, matmul_gelu_bwd_fp16_unaligned_tab_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_unaligned_tb_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_unaligned_tb_cacc_gfx1101
EMBED_ASSET matmul_gelu_bwd_fp16_unaligned_tb_cacc_gfx1101, matmul_gelu_bwd_fp16_unaligned_tb_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_bwd_fp16_unaligned_tb_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_bwd_fp16_unaligned_tb_gfx1101
EMBED_ASSET matmul_gelu_bwd_fp16_unaligned_tb_gfx1101, matmul_gelu_bwd_fp16_unaligned_tb_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_fp16_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_fp16_cacc_gfx1101
EMBED_ASSET matmul_gelu_fp16_cacc_gfx1101, matmul_gelu_fp16_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_fp16_gfx1101
EMBED_ASSET matmul_gelu_fp16_gfx1101, matmul_gelu_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_fp16_unaligned_cacc_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_fp16_unaligned_cacc_gfx1101
EMBED_ASSET matmul_gelu_fp16_unaligned_cacc_gfx1101, matmul_gelu_fp16_unaligned_cacc_gfx1101.hsaco, 4
#include "../kernel_gen/output/matmul_gelu_fp16_unaligned_gfx1101.kinfo.inc.asm"
DECLARE_KINFO matmul_gelu_fp16_unaligned_gfx1101
EMBED_ASSET matmul_gelu_fp16_unaligned_gfx1101, matmul_gelu_fp16_unaligned_gfx1101.hsaco, 4
#include "../kernel_gen/output/mean_reduce_column_tiled_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mean_reduce_column_tiled_bf16_gfx1101
EMBED_ASSET mean_reduce_column_tiled_bf16_gfx1101, mean_reduce_column_tiled_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/mean_reduce_column_tiled_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mean_reduce_column_tiled_fp16_gfx1101
EMBED_ASSET mean_reduce_column_tiled_fp16_gfx1101, mean_reduce_column_tiled_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/mean_reduce_contiguous_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mean_reduce_contiguous_bf16_gfx1101
EMBED_ASSET mean_reduce_contiguous_bf16_gfx1101, mean_reduce_contiguous_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/mean_reduce_contiguous_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mean_reduce_contiguous_fp16_gfx1101
EMBED_ASSET mean_reduce_contiguous_fp16_gfx1101, mean_reduce_contiguous_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_bwd_hs128_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_bwd_hs128_bf16_gfx1101
EMBED_ASSET mha_full_attn_bwd_hs128_bf16_gfx1101, mha_full_attn_bwd_hs128_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_bwd_hs128_bf16_uneven_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_bwd_hs128_bf16_uneven_gfx1101
EMBED_ASSET mha_full_attn_bwd_hs128_bf16_uneven_gfx1101, mha_full_attn_bwd_hs128_bf16_uneven_gfx1101.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_bwd_hs128_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_bwd_hs128_fp16_gfx1101
EMBED_ASSET mha_full_attn_bwd_hs128_fp16_gfx1101, mha_full_attn_bwd_hs128_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_bwd_hs128_fp16_uneven_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_bwd_hs128_fp16_uneven_gfx1101
EMBED_ASSET mha_full_attn_bwd_hs128_fp16_uneven_gfx1101, mha_full_attn_bwd_hs128_fp16_uneven_gfx1101.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_bwd_pre_hs128_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_bwd_pre_hs128_bf16_gfx1101
EMBED_ASSET mha_full_attn_bwd_pre_hs128_bf16_gfx1101, mha_full_attn_bwd_pre_hs128_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_bwd_pre_hs128_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_bwd_pre_hs128_fp16_gfx1101
EMBED_ASSET mha_full_attn_bwd_pre_hs128_fp16_gfx1101, mha_full_attn_bwd_pre_hs128_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_fwd_hs128_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_fwd_hs128_bf16_gfx1101
EMBED_ASSET mha_full_attn_fwd_hs128_bf16_gfx1101, mha_full_attn_fwd_hs128_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_fwd_hs128_bf16_nolse_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_fwd_hs128_bf16_nolse_gfx1101
EMBED_ASSET mha_full_attn_fwd_hs128_bf16_nolse_gfx1101, mha_full_attn_fwd_hs128_bf16_nolse_gfx1101.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_fwd_hs128_bf16_uneven_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_fwd_hs128_bf16_uneven_gfx1101
EMBED_ASSET mha_full_attn_fwd_hs128_bf16_uneven_gfx1101, mha_full_attn_fwd_hs128_bf16_uneven_gfx1101.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_fwd_hs128_bf16_uneven_nolse_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_fwd_hs128_bf16_uneven_nolse_gfx1101
EMBED_ASSET mha_full_attn_fwd_hs128_bf16_uneven_nolse_gfx1101, mha_full_attn_fwd_hs128_bf16_uneven_nolse_gfx1101.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_fwd_hs128_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_fwd_hs128_fp16_gfx1101
EMBED_ASSET mha_full_attn_fwd_hs128_fp16_gfx1101, mha_full_attn_fwd_hs128_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_fwd_hs128_fp16_nolse_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_fwd_hs128_fp16_nolse_gfx1101
EMBED_ASSET mha_full_attn_fwd_hs128_fp16_nolse_gfx1101, mha_full_attn_fwd_hs128_fp16_nolse_gfx1101.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_fwd_hs128_fp16_uneven_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_fwd_hs128_fp16_uneven_gfx1101
EMBED_ASSET mha_full_attn_fwd_hs128_fp16_uneven_gfx1101, mha_full_attn_fwd_hs128_fp16_uneven_gfx1101.hsaco, 4
#include "../kernel_gen/output/mha_full_attn_fwd_hs128_fp16_uneven_nolse_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mha_full_attn_fwd_hs128_fp16_uneven_nolse_gfx1101
EMBED_ASSET mha_full_attn_fwd_hs128_fp16_uneven_nolse_gfx1101, mha_full_attn_fwd_hs128_fp16_uneven_nolse_gfx1101.hsaco, 4
#include "../kernel_gen/output/mul_elementwise_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mul_elementwise_bf16_gfx1101
EMBED_ASSET mul_elementwise_bf16_gfx1101, mul_elementwise_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/mul_elementwise_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mul_elementwise_fp16_gfx1101
EMBED_ASSET mul_elementwise_fp16_gfx1101, mul_elementwise_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/mul_elementwise_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mul_elementwise_fp32_gfx1101
EMBED_ASSET mul_elementwise_fp32_gfx1101, mul_elementwise_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/mul_reduce_column_tiled_bf16_out_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mul_reduce_column_tiled_bf16_out_fp32_gfx1101
EMBED_ASSET mul_reduce_column_tiled_bf16_out_fp32_gfx1101, mul_reduce_column_tiled_bf16_out_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/mul_reduce_column_tiled_fp16_out_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mul_reduce_column_tiled_fp16_out_fp32_gfx1101
EMBED_ASSET mul_reduce_column_tiled_fp16_out_fp32_gfx1101, mul_reduce_column_tiled_fp16_out_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/mul_reduce_contiguous_bf16_out_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mul_reduce_contiguous_bf16_out_fp32_gfx1101
EMBED_ASSET mul_reduce_contiguous_bf16_out_fp32_gfx1101, mul_reduce_contiguous_bf16_out_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/mul_reduce_contiguous_fp16_out_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mul_reduce_contiguous_fp16_out_fp32_gfx1101
EMBED_ASSET mul_reduce_contiguous_fp16_out_fp32_gfx1101, mul_reduce_contiguous_fp16_out_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/mul_scalar_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mul_scalar_bf16_gfx1101
EMBED_ASSET mul_scalar_bf16_gfx1101, mul_scalar_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/mul_scalar_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mul_scalar_fp16_gfx1101
EMBED_ASSET mul_scalar_fp16_gfx1101, mul_scalar_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/mul_scalar_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mul_scalar_fp32_gfx1101
EMBED_ASSET mul_scalar_fp32_gfx1101, mul_scalar_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/sqrt_elementwise_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO sqrt_elementwise_bf16_gfx1101
EMBED_ASSET sqrt_elementwise_bf16_gfx1101, sqrt_elementwise_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/sqrt_elementwise_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO sqrt_elementwise_fp16_gfx1101
EMBED_ASSET sqrt_elementwise_fp16_gfx1101, sqrt_elementwise_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/sqrt_elementwise_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO sqrt_elementwise_fp32_gfx1101
EMBED_ASSET sqrt_elementwise_fp32_gfx1101, sqrt_elementwise_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/mul_trailing_broadcast_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mul_trailing_broadcast_bf16_gfx1101
EMBED_ASSET mul_trailing_broadcast_bf16_gfx1101, mul_trailing_broadcast_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/mul_trailing_broadcast_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mul_trailing_broadcast_fp16_gfx1101
EMBED_ASSET mul_trailing_broadcast_fp16_gfx1101, mul_trailing_broadcast_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/mul_trailing_broadcast_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO mul_trailing_broadcast_fp32_gfx1101
EMBED_ASSET mul_trailing_broadcast_fp32_gfx1101, mul_trailing_broadcast_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/reduce_sum_partial_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO reduce_sum_partial_fp32_gfx1101
EMBED_ASSET reduce_sum_partial_fp32_gfx1101, reduce_sum_partial_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/rms_norm_bwd_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO rms_norm_bwd_bf16_gfx1101
EMBED_ASSET rms_norm_bwd_bf16_gfx1101, rms_norm_bwd_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/rms_norm_bwd_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO rms_norm_bwd_fp16_gfx1101
EMBED_ASSET rms_norm_bwd_fp16_gfx1101, rms_norm_bwd_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/rms_norm_fwd_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO rms_norm_fwd_bf16_gfx1101
EMBED_ASSET rms_norm_fwd_bf16_gfx1101, rms_norm_fwd_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/rms_norm_fwd_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO rms_norm_fwd_fp16_gfx1101
EMBED_ASSET rms_norm_fwd_fp16_gfx1101, rms_norm_fwd_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/sgd_step_bf16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO sgd_step_bf16_gfx1101
EMBED_ASSET sgd_step_bf16_gfx1101, sgd_step_bf16_gfx1101.hsaco, 4
#include "../kernel_gen/output/sgd_step_fp16_gfx1101.kinfo.inc.asm"
DECLARE_KINFO sgd_step_fp16_gfx1101
EMBED_ASSET sgd_step_fp16_gfx1101, sgd_step_fp16_gfx1101.hsaco, 4
#include "../kernel_gen/output/sgd_step_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO sgd_step_fp32_gfx1101
EMBED_ASSET sgd_step_fp32_gfx1101, sgd_step_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/sum_reduce_column_tiled_bf16_out_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO sum_reduce_column_tiled_bf16_out_fp32_gfx1101
EMBED_ASSET sum_reduce_column_tiled_bf16_out_fp32_gfx1101, sum_reduce_column_tiled_bf16_out_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/sum_reduce_column_tiled_fp16_out_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO sum_reduce_column_tiled_fp16_out_fp32_gfx1101
EMBED_ASSET sum_reduce_column_tiled_fp16_out_fp32_gfx1101, sum_reduce_column_tiled_fp16_out_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/sum_reduce_column_tiled_fp32_out_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO sum_reduce_column_tiled_fp32_out_fp32_gfx1101
EMBED_ASSET sum_reduce_column_tiled_fp32_out_fp32_gfx1101, sum_reduce_column_tiled_fp32_out_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/sum_reduce_contiguous_bf16_out_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO sum_reduce_contiguous_bf16_out_fp32_gfx1101
EMBED_ASSET sum_reduce_contiguous_bf16_out_fp32_gfx1101, sum_reduce_contiguous_bf16_out_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/sum_reduce_contiguous_fp16_out_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO sum_reduce_contiguous_fp16_out_fp32_gfx1101
EMBED_ASSET sum_reduce_contiguous_fp16_out_fp32_gfx1101, sum_reduce_contiguous_fp16_out_fp32_gfx1101.hsaco, 4
#include "../kernel_gen/output/sum_reduce_contiguous_fp32_out_fp32_gfx1101.kinfo.inc.asm"
DECLARE_KINFO sum_reduce_contiguous_fp32_out_fp32_gfx1101
EMBED_ASSET sum_reduce_contiguous_fp32_out_fp32_gfx1101, sum_reduce_contiguous_fp32_out_fp32_gfx1101.hsaco, 4

#endif
