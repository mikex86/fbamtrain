echo "GEMM: triton, CONV2D: triton"
FBAMTRAIN_PREFER_GEMM_BACKEND=triton;FBAMTRAIN_PREFER_CONV2D_BACKEND=triton bash validate_numerics.sh
echo "GEMM: cutlass, CONV2D: cutlass"
FBAMTRAIN_PREFER_GEMM_BACKEND=cutlass;FBAMTRAIN_PREFER_CONV2D_BACKEND=cutlass bash validate_numerics.sh
echo "MHA: flash, GEMM: cutlass, CONV2D: cutlass"
FBAMTRAIN_PREFER_MHA_BACKEND=flash;FBAMTRAIN_PREFER_GEMM_BACKEND=triton;FBAMTRAIN_PREFER_GEMM_BACKEND=cutlass;FBAMTRAIN_PREFER_CONV2D_BACKEND=cutlass bash validate_numerics.sh
echo "MHA: triton, GEMM: cutlass, CONV2D: cutlass"
FBAMTRAIN_PREFER_MHA_BACKEND=triton;FBAMTRAIN_PREFER_GEMM_BACKEND=triton;FBAMTRAIN_PREFER_GEMM_BACKEND=cutlass;FBAMTRAIN_PREFER_CONV2D_BACKEND=cutlass bash validate_numerics.sh