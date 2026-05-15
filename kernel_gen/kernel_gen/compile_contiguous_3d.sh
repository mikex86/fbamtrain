#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
export TRITON_ALWAYS_COMPILE=1
# BLOCK_SIZE_X: 32, BLOCK_SIZE_Y: 1, BLOCK_SIZE_Z: 512, GROUP_X: 8, num_warps: 8, num_ctas: 1, num_stages: 3, maxnreg: None ; best on RTX 4090
# contiguous_3d(
#        out_ptr, in_ptr,
#        stride_x, stride_y, stride_z,
#        X, Y, Z,
#        BLOCK_SIZE_X: tl.constexpr,
#        BLOCK_SIZE_Y: tl.constexpr,
#        BLOCK_SIZE_Z: tl.constexpr,
#        GROUP_X: tl.constexpr,
# )
BLOCK_SIZE_X=32
BLOCK_SIZE_Y=1
BLOCK_SIZE_Z=512
GROUP_X=8
CONTIGUOUS_SIGNATURE_BF16="*bf16:16, *bf16:16, i32, i32, i32, i32, i32, i32, $BLOCK_SIZE_X, $BLOCK_SIZE_Y, $BLOCK_SIZE_Z, $GROUP_X"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name contiguous_3d --signature "$CONTIGUOUS_SIGNATURE_BF16" --target cuda:80:32 --num-stages 3 --num-warps 8 --out-name output/contiguous_3d_bf16_sm80 kernels/contiguous_3d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name contiguous_3d --signature "$CONTIGUOUS_SIGNATURE_BF16" --target cuda:89:32 --num-stages 3 --num-warps 8 --out-name output/contiguous_3d_bf16_sm89 kernels/contiguous_3d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name contiguous_3d --signature "$CONTIGUOUS_SIGNATURE_BF16" --target cuda:90:32 --num-stages 3 --num-warps 8 --out-name output/contiguous_3d_bf16_sm90 kernels/contiguous_3d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name contiguous_3d --signature "$CONTIGUOUS_SIGNATURE_BF16" --target cuda:100:32 --num-stages 3 --num-warps 8 --out-name output/contiguous_3d_bf16_sm100 kernels/contiguous_3d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name contiguous_3d --signature "$CONTIGUOUS_SIGNATURE_BF16" --target hip:gfx942:64 --num-stages 3 --num-warps 8 --out-name output/contiguous_3d_bf16_gfx942 kernels/contiguous_3d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name contiguous_3d --signature "$CONTIGUOUS_SIGNATURE_BF16" --target hip:gfx1101:32 --num-stages 3 --num-warps 8 --out-name output/contiguous_3d_bf16_gfx1101 kernels/contiguous_3d.py

CONTIGUOUS_SIGNATURE_FP16="*fp16:16, *fp16:16, i32, i32, i32, i32, i32, i32, $BLOCK_SIZE_X, $BLOCK_SIZE_Y, $BLOCK_SIZE_Z, $GROUP_X"

$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name contiguous_3d --signature "$CONTIGUOUS_SIGNATURE_FP16" --target cuda:80:32 --num-stages 3 --num-warps 8 --out-name output/contiguous_3d_fp16_sm80 kernels/contiguous_3d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name contiguous_3d --signature "$CONTIGUOUS_SIGNATURE_FP16" --target cuda:89:32 --num-stages 3 --num-warps 8 --out-name output/contiguous_3d_fp16_sm89 kernels/contiguous_3d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name contiguous_3d --signature "$CONTIGUOUS_SIGNATURE_FP16" --target cuda:90:32 --num-stages 3 --num-warps 8 --out-name output/contiguous_3d_fp16_sm90 kernels/contiguous_3d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name contiguous_3d --signature "$CONTIGUOUS_SIGNATURE_FP16" --target cuda:100:32 --num-stages 3 --num-warps 8 --out-name output/contiguous_3d_fp16_sm100 kernels/contiguous_3d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name contiguous_3d --signature "$CONTIGUOUS_SIGNATURE_FP16" --target hip:gfx942:64 --num-stages 3 --num-warps 8 --out-name output/contiguous_3d_fp16_gfx942 kernels/contiguous_3d.py
$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name contiguous_3d --signature "$CONTIGUOUS_SIGNATURE_FP16" --target hip:gfx1101:32 --num-stages 3 --num-warps 8 --out-name output/contiguous_3d_fp16_gfx1101 kernels/contiguous_3d.py
