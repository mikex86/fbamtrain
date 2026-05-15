#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1
# Tuned on RTX 4090: BLOCK_SIZE=512, NUM_WARPS=8, NUM_STAGES=2
# signature: act_gelu_elementwise(out_ptr, in_ptr,
#             n_elements,
#             BLOCK_SIZE: tl.constexpr)
BLOCK_SIZE_GELU=512
NUM_WARPS_GELU=8
NUM_STAGES_GELU=2
SIGNATURE_GELU_BF16="*bf16, *bf16, i32, $BLOCK_SIZE_GELU"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_elementwise --signature "$SIGNATURE_GELU_BF16" --target cuda:80:32 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_elementwise_bf16_sm80 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_elementwise --signature "$SIGNATURE_GELU_BF16" --target cuda:89:32 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_elementwise_bf16_sm89 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_elementwise --signature "$SIGNATURE_GELU_BF16" --target cuda:90:32 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_elementwise_bf16_sm90 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_elementwise --signature "$SIGNATURE_GELU_BF16" --target cuda:100:32 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_elementwise_bf16_sm100 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_elementwise --signature "$SIGNATURE_GELU_BF16" --target hip:gfx942:64 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_elementwise_bf16_gfx942 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_elementwise --signature "$SIGNATURE_GELU_BF16" --target hip:gfx1101:32 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_elementwise_bf16_gfx1101 kernels/act.py

SIGNATURE_GELU_FP16="*fp16, *fp16, i32, $BLOCK_SIZE_GELU"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_elementwise --signature "$SIGNATURE_GELU_FP16" --target cuda:80:32 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_elementwise_fp16_sm80 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_elementwise --signature "$SIGNATURE_GELU_FP16" --target cuda:89:32 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_elementwise_fp16_sm89 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_elementwise --signature "$SIGNATURE_GELU_FP16" --target cuda:90:32 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_elementwise_fp16_sm90 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_elementwise --signature "$SIGNATURE_GELU_FP16" --target cuda:100:32 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_elementwise_fp16_sm100 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_elementwise --signature "$SIGNATURE_GELU_FP16" --target hip:gfx942:64 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_elementwise_fp16_gfx942 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_elementwise --signature "$SIGNATURE_GELU_FP16" --target hip:gfx1101:32 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_elementwise_fp16_gfx1101 kernels/act.py

# signature: act_gelu_bwd_elementwise(out_ptr, in_ptr, upstream_ptr,
#             n_elements,
#             BLOCK_SIZE: tl.constexpr)
SIGNATURE_GELU_BWD_BF16="*bf16, *bf16, *bf16, i32, $BLOCK_SIZE_GELU"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_bwd_elementwise --signature "$SIGNATURE_GELU_BWD_BF16" --target cuda:80:32 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_bwd_elementwise_bf16_sm80 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_bwd_elementwise --signature "$SIGNATURE_GELU_BWD_BF16" --target cuda:89:32 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_bwd_elementwise_bf16_sm89 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_bwd_elementwise --signature "$SIGNATURE_GELU_BWD_BF16" --target cuda:90:32 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_bwd_elementwise_bf16_sm90 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_bwd_elementwise --signature "$SIGNATURE_GELU_BWD_BF16" --target cuda:100:32 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_bwd_elementwise_bf16_sm100 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_bwd_elementwise --signature "$SIGNATURE_GELU_BWD_BF16" --target hip:gfx942:64 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_bwd_elementwise_bf16_gfx942 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_bwd_elementwise --signature "$SIGNATURE_GELU_BWD_BF16" --target hip:gfx1101:32 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_bwd_elementwise_bf16_gfx1101 kernels/act.py

SIGNATURE_GELU_BWD_FP16="*fp16, *fp16, *fp16, i32, $BLOCK_SIZE_GELU"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_bwd_elementwise --signature "$SIGNATURE_GELU_BWD_FP16" --target cuda:80:32 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_bwd_elementwise_fp16_sm80 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_bwd_elementwise --signature "$SIGNATURE_GELU_BWD_FP16" --target cuda:89:32 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_bwd_elementwise_fp16_sm89 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_bwd_elementwise --signature "$SIGNATURE_GELU_BWD_FP16" --target cuda:90:32 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_bwd_elementwise_fp16_sm90 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_bwd_elementwise --signature "$SIGNATURE_GELU_BWD_FP16" --target cuda:100:32 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_bwd_elementwise_fp16_sm100 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_bwd_elementwise --signature "$SIGNATURE_GELU_BWD_FP16" --target hip:gfx942:64 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_bwd_elementwise_fp16_gfx942 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_gelu_bwd_elementwise --signature "$SIGNATURE_GELU_BWD_FP16" --target hip:gfx1101:32 --num-stages $NUM_STAGES_GELU --num-warps $NUM_WARPS_GELU --out-name output/act_gelu_bwd_elementwise_fp16_gfx1101 kernels/act.py

# Tuned on RTX 4090: BLOCK_SIZE=1024, NUM_WARPS=8, NUM_STAGES=2
# signature: act_relu_elementwise(out_ptr, in_ptr,
#             n_elements,
#             BLOCK_SIZE: tl.constexpr)
BLOCK_SIZE_RELU=1024
NUM_WARPS_RELU=8
NUM_STAGES_RELU=2
SIGNATURE_RELU_BF16="*bf16, *bf16, i32, $BLOCK_SIZE_RELU"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_relu_elementwise --signature "$SIGNATURE_RELU_BF16" --target cuda:80:32 --num-stages $NUM_STAGES_RELU --num-warps $NUM_WARPS_RELU --out-name output/act_relu_elementwise_bf16_sm80 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_relu_elementwise --signature "$SIGNATURE_RELU_BF16" --target cuda:89:32 --num-stages $NUM_STAGES_RELU --num-warps $NUM_WARPS_RELU --out-name output/act_relu_elementwise_bf16_sm89 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_relu_elementwise --signature "$SIGNATURE_RELU_BF16" --target cuda:90:32 --num-stages $NUM_STAGES_RELU --num-warps $NUM_WARPS_RELU --out-name output/act_relu_elementwise_bf16_sm90 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_relu_elementwise --signature "$SIGNATURE_RELU_BF16" --target cuda:100:32 --num-stages $NUM_STAGES_RELU --num-warps $NUM_WARPS_RELU --out-name output/act_relu_elementwise_bf16_sm100 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_relu_elementwise --signature "$SIGNATURE_RELU_BF16" --target hip:gfx942:64 --num-stages $NUM_STAGES_RELU --num-warps $NUM_WARPS_RELU --out-name output/act_relu_elementwise_bf16_gfx942 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_relu_elementwise --signature "$SIGNATURE_RELU_BF16" --target hip:gfx1101:32 --num-stages $NUM_STAGES_RELU --num-warps $NUM_WARPS_RELU --out-name output/act_relu_elementwise_bf16_gfx1101 kernels/act.py

SIGNATURE_RELU_FP16="*fp16, *fp16, i32, $BLOCK_SIZE_RELU"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_relu_elementwise --signature "$SIGNATURE_RELU_FP16" --target cuda:80:32 --num-stages $NUM_STAGES_RELU --num-warps $NUM_WARPS_RELU --out-name output/act_relu_elementwise_fp16_sm80 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_relu_elementwise --signature "$SIGNATURE_RELU_FP16" --target cuda:89:32 --num-stages $NUM_STAGES_RELU --num-warps $NUM_WARPS_RELU --out-name output/act_relu_elementwise_fp16_sm89 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_relu_elementwise --signature "$SIGNATURE_RELU_FP16" --target cuda:90:32 --num-stages $NUM_STAGES_RELU --num-warps $NUM_WARPS_RELU --out-name output/act_relu_elementwise_fp16_sm90 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_relu_elementwise --signature "$SIGNATURE_RELU_FP16" --target cuda:100:32 --num-stages $NUM_STAGES_RELU --num-warps $NUM_WARPS_RELU --out-name output/act_relu_elementwise_fp16_sm100 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_relu_elementwise --signature "$SIGNATURE_RELU_FP16" --target hip:gfx942:64 --num-stages $NUM_STAGES_RELU --num-warps $NUM_WARPS_RELU --out-name output/act_relu_elementwise_fp16_gfx942 kernels/act.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name act_relu_elementwise --signature "$SIGNATURE_RELU_FP16" --target hip:gfx1101:32 --num-stages $NUM_STAGES_RELU --num-warps $NUM_WARPS_RELU --out-name output/act_relu_elementwise_fp16_gfx1101 kernels/act.py
