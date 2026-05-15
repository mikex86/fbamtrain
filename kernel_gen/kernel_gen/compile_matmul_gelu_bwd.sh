#!/usr/bin/env bash
set -euo pipefail
export TRITON_ALWAYS_COMPILE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
TRITON_JOBS=${TRITON_JOBS:-"$(nproc)"}

GEMM_BLOCK_SIZE_M=128
GEMM_BLOCK_SIZE_N=128
GEMM_BLOCK_SIZE_K=32
GEMM_GROUP_SIZE_M=8
ACTIVATION=3
ACCUM_FP16_FALSE=0
ACCUM_FP16_TRUE=1
HAS_BIAS_FALSE=0
HAS_BIAS_TRUE=1
ACCUM_OUT_FALSE=0
ACCUM_OUT_TRUE=1
DEFAULT_NUM_STAGES=4
DEFAULT_NUM_WARPS=4

TARGETS=(
  "cuda:80:32 sm80"
  "cuda:89:32 sm89"
  "cuda:90:32 sm90"
  "cuda:100:32 sm100"
  "hip:gfx942:64 gfx942"
  "hip:gfx1101:32 gfx1101"
)

join_signature() {
  local out=""
  for part in "$@"; do
    if [[ -n "${out}" ]]; then
      out+=", "
    fi
    out+="${part}"
  done
  echo "${out}"
}

make_matmul_signature_base() {
  local a_ptr_type="$1"
  local b_ptr_type="$2"
  local c_ptr_type="$3"
  local d_ptr_type="$4"
  local m_sig="$5"
  local n_sig="$6"
  local k_sig="$7"
  local stride_am="$8"
  local stride_ak="$9"
  local stride_bk="${10}"
  local stride_bn="${11}"
  local stride_cm="${12}"
  local stride_cn="${13}"
  local stride_dm="${14}"
  local stride_dn="${15}"
  local accum_fp16="${16}"
  local has_bias="${17}"

  join_signature \
    "${a_ptr_type}" "${b_ptr_type}" "${c_ptr_type}" "${d_ptr_type}" \
    "${m_sig}" "${n_sig}" "${k_sig}" \
    "${stride_am}" "${stride_ak}" "${stride_bk}" "${stride_bn}" \
    "${stride_cm}" "${stride_cn}" "${stride_dm}" "${stride_dn}" \
    "${GEMM_BLOCK_SIZE_M}" "${GEMM_BLOCK_SIZE_N}" "${GEMM_BLOCK_SIZE_K}" "${GEMM_GROUP_SIZE_M}" \
    "${ACTIVATION}" "${accum_fp16}" "${has_bias}"
}

make_matmul_signature() {
  local a_ptr_type="$1"
  local b_ptr_type="$2"
  local c_ptr_type="$3"
  local d_ptr_type="$4"
  local aligned="$5"
  local transpose_state="$6" # none, ta, tb, tab
  local accum_fp16="$7"
  local has_bias="$8"

  local m_sig n_sig k_sig
  local stride_am stride_ak stride_bk stride_bn stride_cm stride_cn stride_dm stride_dn
  if [[ "${aligned}" == "true" ]]; then
    m_sig="i32:16"
    n_sig="i32:16"
    k_sig="i32:16"
    case "${transpose_state}" in
      ta)
        stride_am="1"
        stride_ak="i32:16"
        stride_bk="i32:16"
        stride_bn="1"
        ;;
      tb)
        stride_am="i32:16"
        stride_ak="1"
        stride_bk="1"
        stride_bn="i32:16"
        ;;
      tab)
        stride_am="1"
        stride_ak="i32:16"
        stride_bk="1"
        stride_bn="i32:16"
        ;;
      *)
        stride_am="i32:16"
        stride_ak="1"
        stride_bk="i32:16"
        stride_bn="1"
        ;;
    esac
    stride_cm="i32"
    stride_cn="1"
    stride_dm="i32:16"
    stride_dn="1"
  else
    m_sig="i32"
    n_sig="i32"
    k_sig="i32"
    stride_am="i32"
    stride_ak="i32"
    stride_bk="i32"
    stride_bn="i32"
    stride_cm="i32"
    stride_cn="i32"
    stride_dm="i32"
    stride_dn="i32"
  fi

  make_matmul_signature_base "${a_ptr_type}" "${b_ptr_type}" "${c_ptr_type}" "${d_ptr_type}" \
    "${m_sig}" "${n_sig}" "${k_sig}" \
    "${stride_am}" "${stride_ak}" "${stride_bk}" "${stride_bn}" \
    "${stride_cm}" "${stride_cn}" "${stride_dm}" "${stride_dn}" \
    "${accum_fp16}" "${has_bias}"
}

compile_triton_variants() {
  local signature_base="$1"
  local out_base="$2"
  local num_stages="${3:-$DEFAULT_NUM_STAGES}"
  local num_warps="${4:-$DEFAULT_NUM_WARPS}"
  local cmds=()

  for target_entry in "${TARGETS[@]}"; do
    read -r target arch <<<"${target_entry}"
    for accum_mode in cstore cacc; do
      local accum_value="${ACCUM_OUT_FALSE}"
      local accum_suffix=""
      if [[ "${accum_mode}" == "cacc" ]]; then
        accum_value="${ACCUM_OUT_TRUE}"
        accum_suffix="_cacc"
      fi
      cmds+=("$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name addmm_act --signature \"${signature_base}, ${accum_value}\" --target \"${target}\" --num-stages \"${num_stages}\" --num-warps \"${num_warps}\" --out-name \"output/${out_base}${accum_suffix}_${arch}\" kernels/addmm_act.py")
    done
  done
  printf '%s\0' "${cmds[@]}" | xargs -0 -P "${TRITON_JOBS}" -I {} bash -c "{}"
}

compile_triton_variants_cuda_only() {
  local signature_base="$1"
  local out_base="$2"
  local num_stages="${3:-$DEFAULT_NUM_STAGES}"
  local num_warps="${4:-$DEFAULT_NUM_WARPS}"
  local cmds=()

  for target_entry in "${TARGETS[@]}"; do
    read -r target arch <<<"${target_entry}"
    if [[ "${target}" != cuda* ]]; then
      continue
    fi
    for accum_mode in cstore cacc; do
      local accum_value="${ACCUM_OUT_FALSE}"
      local accum_suffix=""
      if [[ "${accum_mode}" == "cacc" ]]; then
        accum_value="${ACCUM_OUT_TRUE}"
        accum_suffix="_cacc"
      fi
      cmds+=("$PYTHON_BIN -m pysrc.triton_aot_compiler --kernel-name addmm_act --signature \"${signature_base}, ${accum_value}\" --target \"${target}\" --num-stages \"${num_stages}\" --num-warps \"${num_warps}\" --out-name \"output/${out_base}${accum_suffix}_${arch}\" kernels/addmm_act.py")
    done
  done
  if ((${#cmds[@]})); then
    printf '%s\0' "${cmds[@]}" | xargs -0 -P "${TRITON_JOBS}" -I {} bash -c "{}"
  fi
}

MATMUL_GELU_BWD_BF16_SIGNATURE_BASE=$(make_matmul_signature "*bf16:16" "*bf16:16" "*bf16" "*bf16:16" true "none" "${ACCUM_FP16_FALSE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_BF16_SIGNATURE_UNALIGNED_BASE=$(make_matmul_signature "*bf16:16" "*bf16:16" "*bf16" "*bf16:16" false "none" "${ACCUM_FP16_FALSE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_BF16_TA_SIGNATURE_BASE=$(make_matmul_signature "*bf16:16" "*bf16:16" "*bf16" "*bf16:16" true "ta" "${ACCUM_FP16_FALSE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_BF16_TA_SIGNATURE_UNALIGNED_BASE=$(make_matmul_signature "*bf16:16" "*bf16:16" "*bf16" "*bf16:16" false "ta" "${ACCUM_FP16_FALSE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_BF16_TB_SIGNATURE_BASE=$(make_matmul_signature "*bf16:16" "*bf16:16" "*bf16" "*bf16:16" true "tb" "${ACCUM_FP16_FALSE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_BF16_TB_SIGNATURE_UNALIGNED_BASE=$(make_matmul_signature "*bf16:16" "*bf16:16" "*bf16" "*bf16:16" false "tb" "${ACCUM_FP16_FALSE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_BF16_TAB_SIGNATURE_BASE=$(make_matmul_signature "*bf16:16" "*bf16:16" "*bf16" "*bf16:16" true "tab" "${ACCUM_FP16_FALSE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_BF16_TAB_SIGNATURE_UNALIGNED_BASE=$(make_matmul_signature "*bf16:16" "*bf16:16" "*bf16" "*bf16:16" false "tab" "${ACCUM_FP16_FALSE}" "${HAS_BIAS_TRUE}")

MATMUL_GELU_BWD_FP16_SIGNATURE_BASE=$(make_matmul_signature "*fp16:16" "*fp16:16" "*fp16" "*fp16:16" true "none" "${ACCUM_FP16_FALSE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_FP16_SIGNATURE_UNALIGNED_BASE=$(make_matmul_signature "*fp16:16" "*fp16:16" "*fp16" "*fp16:16" false "none" "${ACCUM_FP16_FALSE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_FP16_TA_SIGNATURE_BASE=$(make_matmul_signature "*fp16:16" "*fp16:16" "*fp16" "*fp16:16" true "ta" "${ACCUM_FP16_FALSE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_FP16_TA_SIGNATURE_UNALIGNED_BASE=$(make_matmul_signature "*fp16:16" "*fp16:16" "*fp16" "*fp16:16" false "ta" "${ACCUM_FP16_FALSE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_FP16_TB_SIGNATURE_BASE=$(make_matmul_signature "*fp16:16" "*fp16:16" "*fp16" "*fp16:16" true "tb" "${ACCUM_FP16_FALSE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_FP16_TB_SIGNATURE_UNALIGNED_BASE=$(make_matmul_signature "*fp16:16" "*fp16:16" "*fp16" "*fp16:16" false "tb" "${ACCUM_FP16_FALSE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_FP16_TAB_SIGNATURE_BASE=$(make_matmul_signature "*fp16:16" "*fp16:16" "*fp16" "*fp16:16" true "tab" "${ACCUM_FP16_FALSE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_FP16_TAB_SIGNATURE_UNALIGNED_BASE=$(make_matmul_signature "*fp16:16" "*fp16:16" "*fp16" "*fp16:16" false "tab" "${ACCUM_FP16_FALSE}" "${HAS_BIAS_TRUE}")

MATMUL_GELU_BWD_FP16_ACC_SIGNATURE_BASE=$(make_matmul_signature "*fp16:16" "*fp16:16" "*fp16" "*fp16:16" true "none" "${ACCUM_FP16_TRUE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_FP16_ACC_SIGNATURE_UNALIGNED_BASE=$(make_matmul_signature "*fp16:16" "*fp16:16" "*fp16" "*fp16:16" false "none" "${ACCUM_FP16_TRUE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_FP16_ACC_TA_SIGNATURE_BASE=$(make_matmul_signature "*fp16:16" "*fp16:16" "*fp16" "*fp16:16" true "ta" "${ACCUM_FP16_TRUE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_FP16_ACC_TA_SIGNATURE_UNALIGNED_BASE=$(make_matmul_signature "*fp16:16" "*fp16:16" "*fp16" "*fp16:16" false "ta" "${ACCUM_FP16_TRUE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_FP16_ACC_TB_SIGNATURE_BASE=$(make_matmul_signature "*fp16:16" "*fp16:16" "*fp16" "*fp16:16" true "tb" "${ACCUM_FP16_TRUE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_FP16_ACC_TB_SIGNATURE_UNALIGNED_BASE=$(make_matmul_signature "*fp16:16" "*fp16:16" "*fp16" "*fp16:16" false "tb" "${ACCUM_FP16_TRUE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_FP16_ACC_TAB_SIGNATURE_BASE=$(make_matmul_signature "*fp16:16" "*fp16:16" "*fp16" "*fp16:16" true "tab" "${ACCUM_FP16_TRUE}" "${HAS_BIAS_TRUE}")
MATMUL_GELU_BWD_FP16_ACC_TAB_SIGNATURE_UNALIGNED_BASE=$(make_matmul_signature "*fp16:16" "*fp16:16" "*fp16" "*fp16:16" false "tab" "${ACCUM_FP16_TRUE}" "${HAS_BIAS_TRUE}")

compile_triton_variants "${MATMUL_GELU_BWD_BF16_SIGNATURE_BASE}" "matmul_gelu_bwd_bf16"
compile_triton_variants "${MATMUL_GELU_BWD_BF16_SIGNATURE_UNALIGNED_BASE}" "matmul_gelu_bwd_bf16_unaligned"
compile_triton_variants "${MATMUL_GELU_BWD_BF16_TA_SIGNATURE_BASE}" "matmul_gelu_bwd_bf16_ta"
compile_triton_variants "${MATMUL_GELU_BWD_BF16_TA_SIGNATURE_UNALIGNED_BASE}" "matmul_gelu_bwd_bf16_unaligned_ta"
compile_triton_variants "${MATMUL_GELU_BWD_BF16_TB_SIGNATURE_BASE}" "matmul_gelu_bwd_bf16_tb"
compile_triton_variants "${MATMUL_GELU_BWD_BF16_TB_SIGNATURE_UNALIGNED_BASE}" "matmul_gelu_bwd_bf16_unaligned_tb"
compile_triton_variants "${MATMUL_GELU_BWD_BF16_TAB_SIGNATURE_BASE}" "matmul_gelu_bwd_bf16_tab"
compile_triton_variants "${MATMUL_GELU_BWD_BF16_TAB_SIGNATURE_UNALIGNED_BASE}" "matmul_gelu_bwd_bf16_unaligned_tab"
compile_triton_variants "${MATMUL_GELU_BWD_FP16_SIGNATURE_BASE}" "matmul_gelu_bwd_fp16"
compile_triton_variants "${MATMUL_GELU_BWD_FP16_SIGNATURE_UNALIGNED_BASE}" "matmul_gelu_bwd_fp16_unaligned"
compile_triton_variants "${MATMUL_GELU_BWD_FP16_TA_SIGNATURE_BASE}" "matmul_gelu_bwd_fp16_ta"
compile_triton_variants "${MATMUL_GELU_BWD_FP16_TA_SIGNATURE_UNALIGNED_BASE}" "matmul_gelu_bwd_fp16_unaligned_ta"
compile_triton_variants "${MATMUL_GELU_BWD_FP16_TB_SIGNATURE_BASE}" "matmul_gelu_bwd_fp16_tb"
compile_triton_variants "${MATMUL_GELU_BWD_FP16_TB_SIGNATURE_UNALIGNED_BASE}" "matmul_gelu_bwd_fp16_unaligned_tb"
compile_triton_variants "${MATMUL_GELU_BWD_FP16_TAB_SIGNATURE_BASE}" "matmul_gelu_bwd_fp16_tab"
compile_triton_variants "${MATMUL_GELU_BWD_FP16_TAB_SIGNATURE_UNALIGNED_BASE}" "matmul_gelu_bwd_fp16_unaligned_tab"
compile_triton_variants_cuda_only "${MATMUL_GELU_BWD_FP16_ACC_SIGNATURE_BASE}" "matmul_gelu_bwd_fp16_acc_fp16"
compile_triton_variants_cuda_only "${MATMUL_GELU_BWD_FP16_ACC_SIGNATURE_UNALIGNED_BASE}" "matmul_gelu_bwd_fp16_acc_fp16_unaligned"
compile_triton_variants_cuda_only "${MATMUL_GELU_BWD_FP16_ACC_TA_SIGNATURE_BASE}" "matmul_gelu_bwd_fp16_acc_fp16_ta"
compile_triton_variants_cuda_only "${MATMUL_GELU_BWD_FP16_ACC_TA_SIGNATURE_UNALIGNED_BASE}" "matmul_gelu_bwd_fp16_acc_fp16_unaligned_ta"
compile_triton_variants_cuda_only "${MATMUL_GELU_BWD_FP16_ACC_TB_SIGNATURE_BASE}" "matmul_gelu_bwd_fp16_acc_fp16_tb"
compile_triton_variants_cuda_only "${MATMUL_GELU_BWD_FP16_ACC_TB_SIGNATURE_UNALIGNED_BASE}" "matmul_gelu_bwd_fp16_acc_fp16_unaligned_tb"
compile_triton_variants_cuda_only "${MATMUL_GELU_BWD_FP16_ACC_TAB_SIGNATURE_BASE}" "matmul_gelu_bwd_fp16_acc_fp16_tab"
compile_triton_variants_cuda_only "${MATMUL_GELU_BWD_FP16_ACC_TAB_SIGNATURE_UNALIGNED_BASE}" "matmul_gelu_bwd_fp16_acc_fp16_unaligned_tab"
