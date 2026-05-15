#include "passes.h"
#include "shape_utils.h"

#include "passes/pass_utils.h"
#include <kernels/kernel_binaries.h>

#include <activation.h>
#include <utils.h>

#include <algorithm>
#include <any>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <ranges>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#if PI_TENSORLIB_ENABLE_HIP

// Ensure that CUDA and HIP are not both enabled
#if PI_TENSORLIB_ENABLE_CUDA
#error "PI_TENSORLIB_ENABLE_HIP and PI_TENSORLIB_ENABLE_CUDA cannot both be enabled"
#endif

#define KERNEL_ONLY_AVAILABLE_ON_NV(ptr) nullptr
#else
#define KERNEL_ONLY_AVAILABLE_ON_NV(ptr) ptr
#endif

enum class GemmBackendPreference
{
    Cutlass,
    Triton,
};

static GemmBackendPreference GetGemmBackendPreference()
{
    const auto env = GetEnvValue("FBAMTRAIN_PREFER_GEMM_BACKEND");
#if !PI_TENSORLIB_ENABLE_CUDA
    (void)env;
    return GemmBackendPreference::Triton;
#else
    if (!env.has_value())
    {
        return GemmBackendPreference::Cutlass;
    }
    if (*env == "cutlass")
    {
        return GemmBackendPreference::Cutlass;
    }
    if (*env == "triton")
    {
        return GemmBackendPreference::Triton;
    }
    throw std::runtime_error("FBAMTRAIN_PREFER_GEMM_BACKEND must be one of: cutlass, triton");
#endif
}

struct GemmKernelSet
{
    const kernel_bin_t<kernel_meta_gemm_t> *bf16_kernel;
    const kernel_bin_t<kernel_meta_gemm_t> *fp16_kernel;
    const kernel_bin_t<kernel_meta_gemm_t> *fp16_acc_fp16_kernel;
    const kernel_bin_t<kernel_meta_gemm_t> *fp16_out_fp32_kernel;
    const kernel_bin_t<kernel_meta_gemm_t> *fp16_acc_fp16_out_fp32_kernel;
};

#if PI_TENSORLIB_ENABLE_CUDA
static constexpr GemmKernelSet kCutlassAddmmKernels{
    &kaddmm_cutlass_bf16,
    &kaddmm_cutlass_fp16,
    &kaddmm_cutlass_fp16_acc_fp16,
    &kaddmm_cutlass_fp16_out_fp32,
    &kaddmm_cutlass_fp16_acc_fp16_out_fp32,
};

static constexpr GemmKernelSet kCutlassAddmmGeluKernels{
    &kaddmm_gelu_cutlass_bf16, &kaddmm_gelu_cutlass_fp16, &kaddmm_gelu_cutlass_fp16_acc_fp16, nullptr, nullptr,
};

static constexpr GemmKernelSet kCutlassAddmmGeluPreactKernels{
    &kaddmm_gelu_preact_cutlass_bf16,
    &kaddmm_gelu_preact_cutlass_fp16,
    &kaddmm_gelu_preact_cutlass_fp16_acc_fp16,
    nullptr,
    nullptr,
};

static constexpr GemmKernelSet kCutlassMatmulGeluBwdKernels{
    &kmatmul_gelu_bwd_cutlass_bf16,
    &kmatmul_gelu_bwd_cutlass_fp16,
    &kmatmul_gelu_bwd_cutlass_fp16_acc_fp16,
    nullptr,
    nullptr,
};
static constexpr GemmKernelSet kCutlassMatmulGeluBwdKernelsTA{
    &kmatmul_gelu_bwd_cutlass_bf16_ta,
    &kmatmul_gelu_bwd_cutlass_fp16_ta,
    &kmatmul_gelu_bwd_cutlass_fp16_acc_fp16_ta,
    nullptr,
    nullptr,
};
static constexpr GemmKernelSet kCutlassMatmulGeluBwdKernelsTB{
    &kmatmul_gelu_bwd_cutlass_bf16_tb,
    &kmatmul_gelu_bwd_cutlass_fp16_tb,
    &kmatmul_gelu_bwd_cutlass_fp16_acc_fp16_tb,
    nullptr,
    nullptr,
};
static constexpr GemmKernelSet kCutlassMatmulGeluBwdKernelsTAB{
    &kmatmul_gelu_bwd_cutlass_bf16_tab,
    &kmatmul_gelu_bwd_cutlass_fp16_tab,
    &kmatmul_gelu_bwd_cutlass_fp16_acc_fp16_tab,
    nullptr,
    nullptr,
};

static constexpr GemmKernelSet kCutlassMatmulKernels{
    &kmatmul_cutlass_bf16,
    &kmatmul_cutlass_fp16,
    &kmatmul_cutlass_fp16_acc_fp16,
    &kmatmul_cutlass_fp16_out_fp32,
    &kmatmul_cutlass_fp16_acc_fp16_out_fp32,
};
static constexpr GemmKernelSet kCutlassMatmulKernelsTA{
    &kmatmul_cutlass_bf16_ta,
    &kmatmul_cutlass_fp16_ta,
    &kmatmul_cutlass_fp16_acc_fp16_ta,
    &kmatmul_cutlass_fp16_out_fp32_ta,
    &kmatmul_cutlass_fp16_acc_fp16_out_fp32_ta,
};
static constexpr GemmKernelSet kCutlassMatmulKernelsTB{
    &kmatmul_cutlass_bf16_tb,
    &kmatmul_cutlass_fp16_tb,
    &kmatmul_cutlass_fp16_acc_fp16_tb,
    &kmatmul_cutlass_fp16_out_fp32_tb,
    &kmatmul_cutlass_fp16_acc_fp16_out_fp32_tb,
};
static constexpr GemmKernelSet kCutlassMatmulKernelsTAB{
    &kmatmul_cutlass_bf16_tab,
    &kmatmul_cutlass_fp16_tab,
    &kmatmul_cutlass_fp16_acc_fp16_tab,
    &kmatmul_cutlass_fp16_out_fp32_tab,
    &kmatmul_cutlass_fp16_acc_fp16_out_fp32_tab,
};
static constexpr GemmKernelSet kCutlassMatmulFp32OutKernels{
    &kmatmul_cutlass_bf16_out_fp32,
    nullptr,
    nullptr,
    &kmatmul_cutlass_fp16_out_fp32,
    &kmatmul_cutlass_fp16_acc_fp16_out_fp32,
};
static constexpr GemmKernelSet kCutlassMatmulFp32OutKernelsTA{
    &kmatmul_cutlass_bf16_out_fp32_ta,
    nullptr,
    nullptr,
    &kmatmul_cutlass_fp16_out_fp32_ta,
    &kmatmul_cutlass_fp16_acc_fp16_out_fp32_ta,
};
static constexpr GemmKernelSet kCutlassMatmulFp32OutKernelsTB{
    &kmatmul_cutlass_bf16_out_fp32_tb,
    nullptr,
    nullptr,
    &kmatmul_cutlass_fp16_out_fp32_tb,
    &kmatmul_cutlass_fp16_acc_fp16_out_fp32_tb,
};
static constexpr GemmKernelSet kCutlassMatmulFp32OutKernelsTAB{
    &kmatmul_cutlass_bf16_out_fp32_tab,
    nullptr,
    nullptr,
    &kmatmul_cutlass_fp16_out_fp32_tab,
    &kmatmul_cutlass_fp16_acc_fp16_out_fp32_tab,
};
#else
static constexpr GemmKernelSet kCutlassAddmmKernels{nullptr, nullptr, nullptr, nullptr, nullptr};
static constexpr GemmKernelSet kCutlassAddmmGeluKernels{nullptr, nullptr, nullptr, nullptr, nullptr};
static constexpr GemmKernelSet kCutlassAddmmGeluPreactKernels{nullptr, nullptr, nullptr, nullptr, nullptr};
static constexpr GemmKernelSet kCutlassMatmulGeluBwdKernels{nullptr, nullptr, nullptr, nullptr, nullptr};
static constexpr GemmKernelSet kCutlassMatmulGeluBwdKernelsTA{nullptr, nullptr, nullptr, nullptr, nullptr};
static constexpr GemmKernelSet kCutlassMatmulGeluBwdKernelsTB{nullptr, nullptr, nullptr, nullptr, nullptr};
static constexpr GemmKernelSet kCutlassMatmulGeluBwdKernelsTAB{nullptr, nullptr, nullptr, nullptr, nullptr};
static constexpr GemmKernelSet kCutlassMatmulKernels{nullptr, nullptr, nullptr, nullptr, nullptr};
static constexpr GemmKernelSet kCutlassMatmulKernelsTA{nullptr, nullptr, nullptr, nullptr, nullptr};
static constexpr GemmKernelSet kCutlassMatmulKernelsTB{nullptr, nullptr, nullptr, nullptr, nullptr};
static constexpr GemmKernelSet kCutlassMatmulKernelsTAB{nullptr, nullptr, nullptr, nullptr, nullptr};
static constexpr GemmKernelSet kCutlassMatmulFp32OutKernels{nullptr, nullptr, nullptr, nullptr, nullptr};
static constexpr GemmKernelSet kCutlassMatmulFp32OutKernelsTA{nullptr, nullptr, nullptr, nullptr, nullptr};
static constexpr GemmKernelSet kCutlassMatmulFp32OutKernelsTB{nullptr, nullptr, nullptr, nullptr, nullptr};
static constexpr GemmKernelSet kCutlassMatmulFp32OutKernelsTAB{nullptr, nullptr, nullptr, nullptr, nullptr};
#endif
static constexpr GemmKernelSet kTritonMatmulKernels{
    &kmatmul_bf16, &kmatmul_fp16, KERNEL_ONLY_AVAILABLE_ON_NV(&kmatmul_fp16_acc_fp16), nullptr, nullptr,
};

static constexpr GemmKernelSet kTritonMatmulKernelsCacc{
    &kmatmul_bf16_cacc, &kmatmul_fp16_cacc, KERNEL_ONLY_AVAILABLE_ON_NV(&kmatmul_fp16_acc_fp16_cacc), nullptr, nullptr,
};

static constexpr GemmKernelSet kTritonMatmulUnalignedKernels{
    &kmatmul_bf16_unaligned,
    &kmatmul_fp16_unaligned,
    KERNEL_ONLY_AVAILABLE_ON_NV(&kmatmul_fp16_acc_fp16_unaligned),
    nullptr,
    nullptr,
};

static constexpr GemmKernelSet kTritonMatmulUnalignedKernelsCacc{
    &kmatmul_bf16_unaligned_cacc,
    &kmatmul_fp16_unaligned_cacc,
    KERNEL_ONLY_AVAILABLE_ON_NV(&kmatmul_fp16_acc_fp16_unaligned_cacc),
    nullptr,
    nullptr,
};

enum class MatmulTransposeState : uint8_t
{
    kNone = 0,
    kTransposeA = 1,
    kTransposeB = 2,
    kTransposeAB = 3,
};

static constexpr GemmKernelSet kTritonMatmulKernelsTA{
    &kmatmul_bf16_ta, &kmatmul_fp16_ta, nullptr, nullptr, nullptr,
};
static constexpr GemmKernelSet kTritonMatmulKernelsTB{
    &kmatmul_bf16_tb, &kmatmul_fp16_tb, nullptr, nullptr, nullptr,
};
static constexpr GemmKernelSet kTritonMatmulKernelsTAB{
    &kmatmul_bf16_tab, &kmatmul_fp16_tab, nullptr, nullptr, nullptr,
};
static constexpr GemmKernelSet kTritonMatmulKernelsCACCTA{
    &kmatmul_bf16_ta_cacc, &kmatmul_fp16_ta_cacc, nullptr, nullptr, nullptr,
};
static constexpr GemmKernelSet kTritonMatmulKernelsCACCTB{
    &kmatmul_bf16_tb_cacc, &kmatmul_fp16_tb_cacc, nullptr, nullptr, nullptr,
};
static constexpr GemmKernelSet kTritonMatmulKernelsCACCTAB{
    &kmatmul_bf16_tab_cacc, &kmatmul_fp16_tab_cacc, nullptr, nullptr, nullptr,
};
static constexpr GemmKernelSet kTritonMatmulUnalignedKernelsTA{
    &kmatmul_bf16_ta_unaligned, &kmatmul_fp16_ta_unaligned, nullptr, nullptr, nullptr,
};
static constexpr GemmKernelSet kTritonMatmulUnalignedKernelsTB{
    &kmatmul_bf16_tb_unaligned, &kmatmul_fp16_tb_unaligned, nullptr, nullptr, nullptr,
};
static constexpr GemmKernelSet kTritonMatmulUnalignedKernelsTAB{
    &kmatmul_bf16_tab_unaligned, &kmatmul_fp16_tab_unaligned, nullptr, nullptr, nullptr,
};
static constexpr GemmKernelSet kTritonMatmulUnalignedKernelsCACCTA{
    &kmatmul_bf16_ta_unaligned_cacc, &kmatmul_fp16_ta_unaligned_cacc, nullptr, nullptr, nullptr,
};
static constexpr GemmKernelSet kTritonMatmulUnalignedKernelsCACCTB{
    &kmatmul_bf16_tb_unaligned_cacc, &kmatmul_fp16_tb_unaligned_cacc, nullptr, nullptr, nullptr,
};
static constexpr GemmKernelSet kTritonMatmulUnalignedKernelsCACCTAB{
    &kmatmul_bf16_tab_unaligned_cacc, &kmatmul_fp16_tab_unaligned_cacc, nullptr, nullptr, nullptr,
};

static constexpr GemmKernelSet kTritonMatmulFp32OutKernels{
    &kmatmul_bf16_out_fp32,
    nullptr,
    nullptr,
    &kmatmul_fp16_out_fp32,
    KERNEL_ONLY_AVAILABLE_ON_NV(&kmatmul_fp16_acc_out_fp32),
};

static constexpr GemmKernelSet kTritonMatmulFp32OutKernelsTA{
    &kmatmul_bf16_out_fp32_ta,
    nullptr,
    nullptr,
    &kmatmul_fp16_out_fp32_ta,
    KERNEL_ONLY_AVAILABLE_ON_NV(&kmatmul_fp16_acc_out_fp32_ta),
};

static constexpr GemmKernelSet kTritonMatmulFp32OutKernelsTB{
    &kmatmul_bf16_out_fp32_tb,
    nullptr,
    nullptr,
    &kmatmul_fp16_out_fp32_tb,
    KERNEL_ONLY_AVAILABLE_ON_NV(&kmatmul_fp16_acc_out_fp32_tb),
};

static constexpr GemmKernelSet kTritonMatmulFp32OutKernelsTAB{
    &kmatmul_bf16_out_fp32_tab,
    nullptr,
    nullptr,
    &kmatmul_fp16_out_fp32_tab,
    KERNEL_ONLY_AVAILABLE_ON_NV(&kmatmul_fp16_acc_out_fp32_tab),
};

static constexpr GemmKernelSet kTritonMatmulFp32OutKernelsCacc{
    &kmatmul_bf16_out_fp32_cacc,
    nullptr,
    nullptr,
    &kmatmul_fp16_out_fp32_cacc,
    KERNEL_ONLY_AVAILABLE_ON_NV(&kmatmul_fp16_acc_out_fp32_cacc),
};

static constexpr GemmKernelSet kTritonMatmulFp32OutKernelsCACCTA{
    &kmatmul_bf16_out_fp32_ta_cacc,
    nullptr,
    nullptr,
    &kmatmul_fp16_out_fp32_ta_cacc,
    KERNEL_ONLY_AVAILABLE_ON_NV(&kmatmul_fp16_acc_out_fp32_ta_cacc),
};

static constexpr GemmKernelSet kTritonMatmulFp32OutKernelsCACCTB{
    &kmatmul_bf16_out_fp32_tb_cacc,
    nullptr,
    nullptr,
    &kmatmul_fp16_out_fp32_tb_cacc,
    KERNEL_ONLY_AVAILABLE_ON_NV(&kmatmul_fp16_acc_out_fp32_tb_cacc),
};

static constexpr GemmKernelSet kTritonMatmulFp32OutKernelsCACCTAB{
    &kmatmul_bf16_out_fp32_tab_cacc,
    nullptr,
    nullptr,
    &kmatmul_fp16_out_fp32_tab_cacc,
    KERNEL_ONLY_AVAILABLE_ON_NV(&kmatmul_fp16_acc_out_fp32_tab_cacc),
};

static constexpr GemmKernelSet kTritonMatmulFp32OutUnalignedKernels{
    &kmatmul_bf16_out_fp32_unaligned,
    nullptr,
    nullptr,
    &kmatmul_fp16_out_fp32_unaligned,
    KERNEL_ONLY_AVAILABLE_ON_NV(&kmatmul_fp16_acc_out_fp32_unaligned),
};

static constexpr GemmKernelSet kTritonMatmulFp32OutUnalignedKernelsTA{
    &kmatmul_bf16_out_fp32_ta_unaligned,
    nullptr,
    nullptr,
    &kmatmul_fp16_out_fp32_ta_unaligned,
    KERNEL_ONLY_AVAILABLE_ON_NV(&kmatmul_fp16_acc_out_fp32_ta_unaligned),
};

static constexpr GemmKernelSet kTritonMatmulFp32OutUnalignedKernelsTB{
    &kmatmul_bf16_out_fp32_tb_unaligned,
    nullptr,
    nullptr,
    &kmatmul_fp16_out_fp32_tb_unaligned,
    KERNEL_ONLY_AVAILABLE_ON_NV(&kmatmul_fp16_acc_out_fp32_tb_unaligned),
};

static constexpr GemmKernelSet kTritonMatmulFp32OutUnalignedKernelsTAB{
    &kmatmul_bf16_out_fp32_tab_unaligned,
    nullptr,
    nullptr,
    &kmatmul_fp16_out_fp32_tab_unaligned,
    KERNEL_ONLY_AVAILABLE_ON_NV(&kmatmul_fp16_acc_out_fp32_tab_unaligned),
};

static constexpr GemmKernelSet kTritonMatmulFp32OutUnalignedKernelsCacc{
    &kmatmul_bf16_out_fp32_unaligned_cacc,
    nullptr,
    nullptr,
    &kmatmul_fp16_out_fp32_unaligned_cacc,
    KERNEL_ONLY_AVAILABLE_ON_NV(&kmatmul_fp16_acc_out_fp32_unaligned_cacc),
};

static constexpr GemmKernelSet kTritonMatmulFp32OutUnalignedKernelsCACCTA{
    &kmatmul_bf16_out_fp32_ta_unaligned_cacc,
    nullptr,
    nullptr,
    &kmatmul_fp16_out_fp32_ta_unaligned_cacc,
    KERNEL_ONLY_AVAILABLE_ON_NV(&kmatmul_fp16_acc_out_fp32_ta_unaligned_cacc),
};

static constexpr GemmKernelSet kTritonMatmulFp32OutUnalignedKernelsCACCTB{
    &kmatmul_bf16_out_fp32_tb_unaligned_cacc,
    nullptr,
    nullptr,
    &kmatmul_fp16_out_fp32_tb_unaligned_cacc,
    KERNEL_ONLY_AVAILABLE_ON_NV(&kmatmul_fp16_acc_out_fp32_tb_unaligned_cacc),
};

static constexpr GemmKernelSet kTritonMatmulFp32OutUnalignedKernelsCACCTAB{
    &kmatmul_bf16_out_fp32_tab_unaligned_cacc,
    nullptr,
    nullptr,
    &kmatmul_fp16_out_fp32_tab_unaligned_cacc,
    KERNEL_ONLY_AVAILABLE_ON_NV(&kmatmul_fp16_acc_out_fp32_tab_unaligned_cacc),
};

struct CutlassGemmConfig
{
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t lda;
    int32_t ldb;
    int32_t ldd;
    int32_t bias_stride;
    int32_t split_k_slices;
};

static std::pair<uint32_t, uint32_t> ComputeCutlassGridShape(const uint32_t grid_tiles_m, const uint32_t grid_tiles_n,
                                                             const uint32_t swizzle_size)
{
    if (grid_tiles_m == 0 || grid_tiles_n == 0)
    {
        return {grid_tiles_m, grid_tiles_n};
    }

    const uint32_t normalized_swizzle = std::max(swizzle_size, 1u);
    uint32_t log_tile = 0;
    if (normalized_swizzle >= 8 && grid_tiles_n >= 6)
    {
        log_tile = 3;
    }
    else if (normalized_swizzle >= 4 && grid_tiles_n >= 3)
    {
        log_tile = 2;
    }
    else if (normalized_swizzle >= 2 && grid_tiles_n >= 2)
    {
        log_tile = 1;
    }

    const uint32_t tile = 1u << log_tile;
    const uint32_t grid_dim_x = grid_tiles_m * tile;
    const uint32_t grid_dim_y = grid_tiles_n;
    return {grid_dim_x, std::max(grid_dim_y, 1u)};
}

static bool IsTritonGemmAligned(const uint64_t m_dim, const uint64_t n_dim, const uint64_t k_dim,
                                const uint64_t stride_am, const uint64_t stride_ak, const uint64_t stride_bk,
                                const uint64_t stride_bn, const uint64_t stride_dm,
                                const MatmulTransposeState transpose_state)
{
    auto aligned_16 = [](const uint64_t value) { return (value % 16u) == 0u; };
    if (const bool dims_aligned = aligned_16(m_dim) && aligned_16(n_dim) && aligned_16(k_dim); !dims_aligned)
    {
        return false;
    }

    switch (transpose_state)
    {
        case MatmulTransposeState::kTransposeA:
            return stride_am == 1 && aligned_16(stride_ak) && aligned_16(stride_bk) && stride_bn == 1 &&
                   aligned_16(stride_dm);
        case MatmulTransposeState::kTransposeB:
            return aligned_16(stride_am) && stride_ak == 1 && stride_bk == 1 && aligned_16(stride_bn) &&
                   aligned_16(stride_dm);
        case MatmulTransposeState::kTransposeAB:
            return stride_am == 1 && aligned_16(stride_ak) && stride_bk == 1 && aligned_16(stride_bn) &&
                   aligned_16(stride_dm);
        case MatmulTransposeState::kNone:
        default:
            return aligned_16(stride_am) && aligned_16(stride_bk) && aligned_16(stride_dm);
    }
}

static MatmulTransposeState GetMatmulTransposeState(const bool transpose_a, const bool transpose_b)
{
    if (transpose_a && transpose_b)
    {
        return MatmulTransposeState::kTransposeAB;
    }
    if (transpose_a)
    {
        return MatmulTransposeState::kTransposeA;
    }
    if (transpose_b)
    {
        return MatmulTransposeState::kTransposeB;
    }
    return MatmulTransposeState::kNone;
}

struct MatmulArgLayout
{
    bool stride_am_runtime{true};
    bool stride_ak_runtime{true};
    bool stride_bk_runtime{true};
    bool stride_bn_runtime{true};
    bool stride_cm_runtime{true};
    bool stride_cn_runtime{true};
    bool stride_dm_runtime{true};
    bool stride_dn_runtime{true};
};

static MatmulArgLayout GetMatmulArgLayout(const bool aligned, const MatmulTransposeState transpose_state)
{
    if (!aligned)
    {
        return {};
    }

    MatmulArgLayout layout{};
    switch (transpose_state)
    {
        case MatmulTransposeState::kTransposeA:
        {
            layout.stride_am_runtime = false;
            layout.stride_bn_runtime = false;
            layout.stride_cn_runtime = false;
            layout.stride_dn_runtime = false;
            break;
        }
        case MatmulTransposeState::kTransposeB:
        {
            layout.stride_ak_runtime = false;
            layout.stride_bk_runtime = false;
            layout.stride_cn_runtime = false;
            layout.stride_dn_runtime = false;
            break;
        }
        case MatmulTransposeState::kTransposeAB:
        {
            layout.stride_am_runtime = false;
            layout.stride_bk_runtime = false;
            layout.stride_cn_runtime = false;
            layout.stride_dn_runtime = false;
            break;
        }
        case MatmulTransposeState::kNone:
        default:
            layout.stride_ak_runtime = false;
            layout.stride_bn_runtime = false;
            layout.stride_cn_runtime = false;
            layout.stride_dn_runtime = false;
            break;
    }
    return layout;
}

struct AddmmArgLayout
{
    bool stride_am_runtime{true};
    bool stride_ak_runtime{true};
    bool stride_bk_runtime{true};
    bool stride_bn_runtime{true};
    bool stride_cm_runtime{true};
    bool stride_cn_runtime{true};
    bool stride_dm_runtime{true};
    bool stride_dn_runtime{true};
};

static AddmmArgLayout GetAddmmArgLayout(const bool aligned)
{
    if (!aligned)
    {
        return {};
    }
    AddmmArgLayout layout{};
    layout.stride_ak_runtime = false;
    layout.stride_bn_runtime = false;
    layout.stride_cn_runtime = false;
    layout.stride_dn_runtime = false;
    return layout;
}

static void MoveCreateTensorBefore(pi::tensorlib::ExecutionPlan &execution_plan, const uint64_t tensor_id,
                                   const size_t target_index)
{
    size_t create_idx = execution_plan.entries.size();
    for (size_t idx = 0; idx < execution_plan.entries.size(); ++idx)
    {
        const auto &entry = execution_plan.entries[idx];
        if (entry.op_type != pi::tensorlib::OpType::CREATE_TENSOR || entry.outputs.size() != 1 ||
            entry.outputs[0] == nullptr)
        {
            continue;
        }
        if (entry.outputs[0]->id() == tensor_id)
        {
            create_idx = idx;
            break;
        }
    }

    if (create_idx == execution_plan.entries.size() || create_idx <= target_index)
    {
        return;
    }

    auto entry = execution_plan.entries[create_idx];
    execution_plan.entries.erase(execution_plan.entries.begin() + static_cast<std::ptrdiff_t>(create_idx));
    execution_plan.entries.insert(execution_plan.entries.begin() + static_cast<std::ptrdiff_t>(target_index),
                                  std::move(entry));
}

static bool EntryConsumesTensor(const pi::tensorlib::ExecutionEntry &entry, const uint64_t tensor_id)
{
    for (const auto &input : entry.inputs)
    {
        if (input != nullptr && input->id() == tensor_id)
        {
            return true;
        }
    }
    return false;
}

static bool EntryProducesTensor(const pi::tensorlib::ExecutionEntry &entry, const uint64_t tensor_id)
{
    for (const auto &output : entry.outputs)
    {
        if (output != nullptr && output->id() == tensor_id)
        {
            return true;
        }
    }
    return false;
}

static void RemoveCreateTensorsIfUnused(pi::tensorlib::ExecutionPlan &execution_plan,
                                        const std::unordered_set<uint64_t> &tensor_ids)
{
    if (tensor_ids.empty())
    {
        return;
    }

    std::unordered_map<uint64_t, bool> is_used{};
    is_used.reserve(tensor_ids.size() * 2);
    for (const uint64_t tensor_id : tensor_ids)
    {
        is_used.emplace(tensor_id, false);
    }

    const auto mark_if_tracked = [&is_used](const std::shared_ptr<pi::tensorlib::RealTensor> &tensor)
    {
        if (tensor == nullptr)
        {
            return;
        }
        if (auto it = is_used.find(tensor->id()); it != is_used.end())
        {
            it->second = true;
        }
    };

    for (const auto &entry : execution_plan.entries)
    {
        if (entry.op_type == pi::tensorlib::OpType::CREATE_TENSOR)
        {
            continue;
        }
        for (const auto &input : entry.inputs)
        {
            mark_if_tracked(input);
        }
        for (const auto &output : entry.outputs)
        {
            mark_if_tracked(output);
        }
    }

    std::vector<size_t> create_indices_to_remove{};
    for (size_t i = 0; i < execution_plan.entries.size(); ++i)
    {
        const auto &entry = execution_plan.entries[i];
        if (entry.op_type != pi::tensorlib::OpType::CREATE_TENSOR || entry.outputs.size() != 1 ||
            entry.outputs[0] == nullptr)
        {
            continue;
        }
        const uint64_t output_id = entry.outputs[0]->id();
        if (const auto it = is_used.find(output_id); it != is_used.end() && !it->second)
        {
            create_indices_to_remove.push_back(i);
        }
    }

    for (auto it = create_indices_to_remove.rbegin(); it != create_indices_to_remove.rend(); ++it)
    {
        execution_plan.entries.erase(execution_plan.entries.begin() + static_cast<std::ptrdiff_t>(*it));
    }
}

static const kernel_bin_t<kernel_meta_gemm_t> *SelectCutlassGemmKernel(const GemmKernelSet &kernels,
                                                                       const pi::tensorlib::DataType dtype,
                                                                       const bool use_fp16_accumulation)
{
    switch (dtype)
    {
        case pi::tensorlib::DataType::FLOAT32:
            return use_fp16_accumulation ? kernels.fp16_acc_fp16_out_fp32_kernel : kernels.fp16_out_fp32_kernel;
        case pi::tensorlib::DataType::BFLOAT16:
            return kernels.bf16_kernel;
        case pi::tensorlib::DataType::FLOAT16:
            if (use_fp16_accumulation && kernels.fp16_acc_fp16_kernel != nullptr)
            {
                return kernels.fp16_acc_fp16_kernel;
            }
            return kernels.fp16_kernel;
        default:
            return nullptr;
    }
}

static const kernel_bin_t<kernel_meta_gemm_t> *SelectFp32OutKernel(const GemmKernelSet &kernels,
                                                                   const pi::tensorlib::DataType input_dtype,
                                                                   const bool use_fp16_accumulation)
{
    switch (input_dtype)
    {
        case pi::tensorlib::DataType::BFLOAT16:
            return kernels.bf16_kernel;
        case pi::tensorlib::DataType::FLOAT16:
            return use_fp16_accumulation ? kernels.fp16_acc_fp16_out_fp32_kernel : kernels.fp16_out_fp32_kernel;
        default:
            return nullptr;
    }
}

static pi::tensorlib::ComputeKernelDescriptor CreateCutlassGemmComputeKernelDescriptor(
    const pi::tensorlib::DataType dtype, const kernel_bin_t<kernel_meta_gemm_t> &kernel_bin,
    const bool use_fp16_accumulation, const bool accumulate_output, const bool write_out_preact = false)
{
    const auto dtype_name = std::string(pi::tensorlib::GetDataTypeName(dtype));
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_bin.kernel_name,
        .function_name = kernel_bin.function_name,
        .expected_arg_count = kernel_bin.arg_count,
        .argument_provider = [dtype, dtype_name, &kernel_bin, use_fp16_accumulation, accumulate_output,
                              write_out_preact](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                                const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            const size_t expected_outputs = write_out_preact ? 2u : 1u;
            if (inputs.size() != 3 || outputs.size() != expected_outputs)
            {
                throw std::runtime_error("addmm kernels expect three inputs and " + std::to_string(expected_outputs) +
                                         " output(s)");
            }

            const auto &a = inputs[0];
            const auto &b = inputs[1];
            const auto &bias = inputs[2];
            const auto &result = outputs[0];
            const std::shared_ptr<pi::tensorlib::RealTensor> pre_act =
                write_out_preact ? outputs[1] : std::shared_ptr<pi::tensorlib::RealTensor>{};

            const auto expected_input_dtype =
                (dtype == pi::tensorlib::DataType::FLOAT32) ? pi::tensorlib::DataType::FLOAT16 : dtype;
            if (a->dtype() != expected_input_dtype || b->dtype() != expected_input_dtype || bias->dtype() != dtype ||
                result->dtype() != dtype || (write_out_preact && pre_act->dtype() != dtype))
            {
                if (dtype == pi::tensorlib::DataType::FLOAT32)
                {
                    throw std::runtime_error("addmm requires fp16 inputs and fp32 output");
                }
                throw std::runtime_error("addmm requires " + dtype_name + " tensors");
            }

            if (a->device().device_type != pi::tensorlib::DeviceType::GPU ||
                b->device().device_type != pi::tensorlib::DeviceType::GPU ||
                bias->device().device_type != pi::tensorlib::DeviceType::GPU ||
                result->device().device_type != pi::tensorlib::DeviceType::GPU ||
                (write_out_preact && pre_act->device().device_type != pi::tensorlib::DeviceType::GPU))
            {
                throw std::runtime_error("addmm currently supports GPU tensors only");
            }

            const auto device_ordinal = a->device().ordinal;
            if (b->device().ordinal != device_ordinal || bias->device().ordinal != device_ordinal ||
                result->device().ordinal != device_ordinal ||
                (write_out_preact && pre_act->device().ordinal != device_ordinal))
            {
                throw std::runtime_error("addmm requires all tensors on the same GPU");
            }

            if (a->shape().ndims() != 2 || b->shape().ndims() != 2 || result->shape().ndims() != 2)
            {
                throw std::runtime_error("addmm expects 2D input, weight, and output tensors");
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(a) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(b) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(result))
            {
                throw std::runtime_error("addmm cutlass kernels require contiguous row-major tensors");
            }
            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(bias))
            {
                throw std::runtime_error("addmm cutlass kernels require contiguous bias tensors");
            }

            const auto m64 = a->shape()[0];
            const auto k64 = a->shape()[1];
            const auto k_b64 = b->shape()[0];
            const auto n64 = b->shape()[1];

            if (k64 != k_b64)
            {
                throw std::runtime_error("Cannot multiply matrices: incompatible shapes");
            }

            if (result->shape()[0] != m64 || result->shape()[1] != n64)
            {
                throw std::runtime_error("addmm result tensor has unexpected shape");
            }
            if (write_out_preact)
            {
                if (pre_act->shape()[0] != m64 || pre_act->shape()[1] != n64)
                {
                    throw std::runtime_error("addmm pre-activation output has unexpected shape");
                }
                if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(pre_act))
                {
                    throw std::runtime_error("addmm cutlass kernels require contiguous pre-activation output tensors");
                }
            }

            if (bias->shape().ndims() != 1 || bias->shape()[0] != n64)
            {
                throw std::runtime_error("addmm cutlass kernels currently require 1D bias vectors");
            }

            auto to_i32 = [](const uint64_t value, const char *what) -> int32_t
            {
                if (value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
                {
                    std::stringstream ss;
                    ss << what << " exceeds supported range";
                    throw std::runtime_error(ss.str());
                }
                return static_cast<int32_t>(value);
            };

            const int32_t m = to_i32(m64, "addmm rows");
            const int32_t n = to_i32(n64, "addmm columns");
            const int32_t k = to_i32(k64, "addmm shared dimension");

            const auto &a_strides = a->strides();
            const auto &b_strides = b->strides();
            const auto &out_strides = result->strides();

            const int32_t lda = to_i32(a_strides[0], "addmm A leading dimension");
            const int32_t ldb = to_i32(b_strides[0], "addmm B leading dimension");
            const int32_t ldd = to_i32(out_strides[0], "addmm output leading dimension");
            constexpr int32_t bias_stride = 0;

            const CutlassGemmConfig cfg{
                .M = m,
                .N = n,
                .K = k,
                .lda = lda,
                .ldb = ldb,
                .ldd = ldd,
                .bias_stride = bias_stride,
                .split_k_slices = 1,
            };

            pi::tensorlib::KernelLaunchArguments arguments{};

            pi::tensorlib::KernelDataArg cfg_storage{};
            cfg_storage.bytes.resize(sizeof(CutlassGemmConfig));
            std::memcpy(cfg_storage.bytes.data(), &cfg, sizeof(CutlassGemmConfig));

            constexpr float alpha_f32 = 1.0f;
            const float beta_f32 = accumulate_output ? 1.0f : 0.0f;

            arguments.args.emplace_back(std::move(cfg_storage));
            arguments.args.emplace_back(a->dataptr());
            arguments.args.emplace_back(b->dataptr());
            arguments.args.emplace_back(bias->dataptr());
            arguments.args.emplace_back(result->dataptr());
            if (write_out_preact)
            {
                arguments.args.emplace_back(pre_act->dataptr());
            }

            if (dtype == pi::tensorlib::DataType::FLOAT16 && use_fp16_accumulation)
            {
                const uint16_t alpha_half = pi::tensorlib::utils::Fp16FromFp32(alpha_f32);
                const uint16_t beta_half = pi::tensorlib::utils::Fp16FromFp32(beta_f32);
                arguments.args.emplace_back(alpha_half);
                arguments.args.emplace_back(beta_half);
            }
            else
            {
                arguments.args.emplace_back(alpha_f32);
                arguments.args.emplace_back(beta_f32);
            }

            arguments.args.emplace_back(static_cast<void *>(nullptr));

            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_GEMM"); env != nullptr)
            {
                std::clog << "[Gemm] CUTLASS kernel " << kernel_bin.function_name << " launch params: M=" << m
                          << ", N=" << n << ", K=" << k << ", lda=" << lda << ", ldb=" << ldb << ", ldd=" << ldd
                          << ", bias_stride=" << bias_stride << ", block_m=" << kernel_bin.meta.block_size_m
                          << ", block_n=" << kernel_bin.meta.block_size_n
                          << ", block_k=" << kernel_bin.meta.block_size_k << '\n';
            }

            const uint32_t block_threads = kernel_bin.num_warps * CUDA_WARP_SIZE;
            const uint32_t grid_tiles_m = CEIL_DIV(static_cast<uint32_t>(m), kernel_bin.meta.block_size_m);
            const uint32_t grid_tiles_n = CEIL_DIV(static_cast<uint32_t>(n), kernel_bin.meta.block_size_n);
            const auto [grid_dim_x, grid_dim_y] =
                ComputeCutlassGridShape(grid_tiles_m, grid_tiles_n, kernel_bin.meta.swizzle_size);
            arguments.grid_dim_x = grid_dim_x;
            arguments.grid_dim_y = grid_dim_y;
            arguments.grid_dim_z = 1;
            arguments.block_dim_x = block_threads;
            arguments.block_dim_y = 1;
            arguments.block_dim_z = 1;
            arguments.shared_mem_bytes = kernel_bin.shared_mem_bytes;
            arguments.device_ordinal = static_cast<int>(device_ordinal);
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel_bin.data, kernel_bin.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel_bin.data, kernel_bin.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor CreateCutlassMatmulGeluBwdComputeKernelDescriptor(
    const pi::tensorlib::DataType dtype, const kernel_bin_t<kernel_meta_gemm_t> &kernel_bin,
    const bool use_fp16_accumulation, const bool accumulate_output, const MatmulTransposeState transpose_state)
{
    const auto dtype_name = std::string(pi::tensorlib::GetDataTypeName(dtype));
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_bin.kernel_name,
        .function_name = kernel_bin.function_name,
        .expected_arg_count = kernel_bin.arg_count,
        .argument_provider = [dtype, dtype_name, &kernel_bin, use_fp16_accumulation, accumulate_output,
                              transpose_state](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                               const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 3 || outputs.size() != 1)
            {
                throw std::runtime_error("matmul_gelu_bwd cutlass kernels expect three inputs and one output");
            }

            const auto &a = inputs[0];
            const auto &b = inputs[1];
            const auto &pre_act = inputs[2];
            const auto &result = outputs[0];

            if (a->dtype() != dtype || b->dtype() != dtype || pre_act->dtype() != dtype || result->dtype() != dtype)
            {
                throw std::runtime_error("matmul_gelu_bwd requires " + dtype_name + " tensors");
            }

            if (a->device().device_type != pi::tensorlib::DeviceType::GPU ||
                b->device().device_type != pi::tensorlib::DeviceType::GPU ||
                pre_act->device().device_type != pi::tensorlib::DeviceType::GPU ||
                result->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("matmul_gelu_bwd currently supports GPU tensors only");
            }

            const auto device_ordinal = a->device().ordinal;
            if (b->device().ordinal != device_ordinal || pre_act->device().ordinal != device_ordinal ||
                result->device().ordinal != device_ordinal)
            {
                throw std::runtime_error("matmul_gelu_bwd requires all tensors on the same GPU");
            }

            if (a->shape().ndims() != 2 || b->shape().ndims() != 2 || pre_act->shape().ndims() != 2 ||
                result->shape().ndims() != 2)
            {
                throw std::runtime_error("matmul_gelu_bwd expects 2D input and output tensors");
            }

            auto is_transposed_contiguous = [](const std::shared_ptr<pi::tensorlib::RealTensor> &tensor) -> bool
            {
                if (tensor->shape().ndims() != 2)
                {
                    return false;
                }
                const auto &shape = tensor->shape();
                const auto &strides = tensor->strides();
                return strides[0] == 1 && strides[1] == shape[0];
            };

            const bool a_row_major = pi::tensorlib::shape_utils::IsRowMajorContiguous(a);
            const bool b_row_major = pi::tensorlib::shape_utils::IsRowMajorContiguous(b);
            const bool result_row_major = pi::tensorlib::shape_utils::IsRowMajorContiguous(result);
            const bool pre_row_major = pi::tensorlib::shape_utils::IsRowMajorContiguous(pre_act);
            const bool a_transposed_ok = (transpose_state == MatmulTransposeState::kTransposeA ||
                                          transpose_state == MatmulTransposeState::kTransposeAB) &&
                                         is_transposed_contiguous(a);
            const bool b_transposed_ok = (transpose_state == MatmulTransposeState::kTransposeB ||
                                          transpose_state == MatmulTransposeState::kTransposeAB) &&
                                         is_transposed_contiguous(b);

            const bool a_layout_ok = (transpose_state == MatmulTransposeState::kTransposeA ||
                                      transpose_state == MatmulTransposeState::kTransposeAB)
                                         ? a_transposed_ok
                                         : a_row_major;
            const bool b_layout_ok = (transpose_state == MatmulTransposeState::kTransposeB ||
                                      transpose_state == MatmulTransposeState::kTransposeAB)
                                         ? b_transposed_ok
                                         : b_row_major;

            if (!result_row_major || !pre_row_major || !a_layout_ok || !b_layout_ok)
            {
                throw std::runtime_error(
                    "matmul_gelu_bwd cutlass kernels require contiguous inputs (row-major or transposed views)");
            }

            const auto m64 = a->shape()[0];
            const auto k64 = a->shape()[1];
            const auto k_b64 = b->shape()[0];
            const auto n64 = b->shape()[1];

            if (k64 != k_b64)
            {
                throw std::runtime_error("Cannot multiply matrices: incompatible shapes");
            }

            if (result->shape()[0] != m64 || result->shape()[1] != n64)
            {
                throw std::runtime_error("matmul_gelu_bwd result tensor has unexpected shape");
            }
            if (pre_act->shape()[0] != m64 || pre_act->shape()[1] != n64)
            {
                throw std::runtime_error("matmul_gelu_bwd expects pre-activation to match output");
            }

            if (pre_act->strides()[0] != result->strides()[0] || pre_act->strides()[1] != result->strides()[1])
            {
                throw std::runtime_error("matmul_gelu_bwd pre-activation strides must match output strides");
            }

            auto to_i32 = [](const uint64_t value, const char *what) -> int32_t
            {
                if (value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
                {
                    std::stringstream ss;
                    ss << what << " exceeds supported range";
                    throw std::runtime_error(ss.str());
                }
                return static_cast<int32_t>(value);
            };

            const int32_t m = to_i32(m64, "matmul_gelu_bwd rows");
            const int32_t n = to_i32(n64, "matmul_gelu_bwd columns");
            const int32_t k = to_i32(k64, "matmul_gelu_bwd shared dimension");

            const auto &a_strides = a->strides();
            const auto &b_strides = b->strides();
            const auto &out_strides = result->strides();

            const int32_t lda = to_i32((transpose_state == MatmulTransposeState::kTransposeA ||
                                        transpose_state == MatmulTransposeState::kTransposeAB)
                                           ? a_strides[1]
                                           : a_strides[0],
                                       "matmul_gelu_bwd A leading dimension");
            const int32_t ldb = to_i32((transpose_state == MatmulTransposeState::kTransposeB ||
                                        transpose_state == MatmulTransposeState::kTransposeAB)
                                           ? b_strides[1]
                                           : b_strides[0],
                                       "matmul_gelu_bwd B leading dimension");
            const int32_t ldd = to_i32(out_strides[0], "matmul_gelu_bwd output leading dimension");
            const int32_t bias_stride = to_i32(pre_act->strides()[0], "matmul_gelu_bwd pre-activation stride");

            const CutlassGemmConfig cfg{
                .M = m,
                .N = n,
                .K = k,
                .lda = lda,
                .ldb = ldb,
                .ldd = ldd,
                .bias_stride = bias_stride,
                .split_k_slices = 1,
            };

            pi::tensorlib::KernelLaunchArguments arguments{};

            pi::tensorlib::KernelDataArg cfg_storage{};
            cfg_storage.bytes.resize(sizeof(CutlassGemmConfig));
            std::memcpy(cfg_storage.bytes.data(), &cfg, sizeof(CutlassGemmConfig));

            constexpr float alpha_f32 = 1.0f;
            const float beta_f32 = accumulate_output ? 1.0f : 0.0f;

            arguments.args.emplace_back(std::move(cfg_storage));
            arguments.args.emplace_back(a->dataptr());
            arguments.args.emplace_back(b->dataptr());
            arguments.args.emplace_back(pre_act->dataptr());
            arguments.args.emplace_back(result->dataptr());

            if (dtype == pi::tensorlib::DataType::FLOAT16 && use_fp16_accumulation)
            {
                const uint16_t alpha_half = pi::tensorlib::utils::Fp16FromFp32(alpha_f32);
                const uint16_t beta_half = pi::tensorlib::utils::Fp16FromFp32(beta_f32);
                arguments.args.emplace_back(alpha_half);
                arguments.args.emplace_back(beta_half);
            }
            else
            {
                arguments.args.emplace_back(alpha_f32);
                arguments.args.emplace_back(beta_f32);
            }

            arguments.args.emplace_back(static_cast<void *>(nullptr));

            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_GEMM"); env != nullptr)
            {
                std::clog << "[Gemm] CUTLASS kernel " << kernel_bin.function_name << " launch params: M=" << m
                          << ", N=" << n << ", K=" << k << ", lda=" << lda << ", ldb=" << ldb << ", ldd=" << ldd
                          << ", bias_stride=" << bias_stride << ", block_m=" << kernel_bin.meta.block_size_m
                          << ", block_n=" << kernel_bin.meta.block_size_n
                          << ", block_k=" << kernel_bin.meta.block_size_k << '\n';
            }

            const uint32_t block_threads = kernel_bin.num_warps * CUDA_WARP_SIZE;
            const uint32_t grid_tiles_m = CEIL_DIV(static_cast<uint32_t>(m), kernel_bin.meta.block_size_m);
            const uint32_t grid_tiles_n = CEIL_DIV(static_cast<uint32_t>(n), kernel_bin.meta.block_size_n);
            const auto [grid_dim_x, grid_dim_y] =
                ComputeCutlassGridShape(grid_tiles_m, grid_tiles_n, kernel_bin.meta.swizzle_size);
            arguments.grid_dim_x = grid_dim_x;
            arguments.grid_dim_y = grid_dim_y;
            arguments.grid_dim_z = 1;
            arguments.block_dim_x = block_threads;
            arguments.block_dim_y = 1;
            arguments.block_dim_z = 1;
            arguments.shared_mem_bytes = kernel_bin.shared_mem_bytes;
            arguments.device_ordinal = static_cast<int>(device_ordinal);
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel_bin.data, kernel_bin.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel_bin.data, kernel_bin.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor CreateCutlassMatmulComputeKernelDescriptor(
    const pi::tensorlib::DataType dtype, const pi::tensorlib::DataType input_dtype,
    const kernel_bin_t<kernel_meta_gemm_t> &kernel_bin, const bool use_fp16_accumulation,
    const bool accumulate_output, const MatmulTransposeState transpose_state)
{
    const std::string dtype_name = std::string(pi::tensorlib::GetDataTypeName(dtype));
    const std::string input_dtype_name = std::string(pi::tensorlib::GetDataTypeName(input_dtype));
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_bin.kernel_name,
        .function_name = kernel_bin.function_name,
        .expected_arg_count = kernel_bin.arg_count,
        .argument_provider =
            [dtype, input_dtype, dtype_name, input_dtype_name, &kernel_bin, use_fp16_accumulation, accumulate_output,
             transpose_state](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                              const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 2 || outputs.size() != 1)
            {
                throw std::runtime_error("matmul kernels expect two inputs and one output");
            }

            const auto &a = inputs[0];
            const auto &b = inputs[1];
            const auto &result = outputs[0];

            const auto expected_input_dtype = (dtype == pi::tensorlib::DataType::FLOAT32) ? input_dtype : dtype;
            if (dtype == pi::tensorlib::DataType::FLOAT32 && expected_input_dtype != pi::tensorlib::DataType::FLOAT16 &&
                expected_input_dtype != pi::tensorlib::DataType::BFLOAT16)
            {
                throw std::runtime_error("matmul fp32 output expects fp16 or bf16 inputs");
            }
            if (a->dtype() != expected_input_dtype || b->dtype() != expected_input_dtype || result->dtype() != dtype)
            {
                throw std::runtime_error("matmul requires input dtype " + input_dtype_name + " and output dtype " +
                                         dtype_name);
            }

            if (a->device().device_type != pi::tensorlib::DeviceType::GPU ||
                b->device().device_type != pi::tensorlib::DeviceType::GPU ||
                result->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("matmul currently supports GPU tensors only");
            }

            const auto device_ordinal = a->device().ordinal;
            if (b->device().ordinal != device_ordinal || result->device().ordinal != device_ordinal)
            {
                throw std::runtime_error("matmul requires all tensors on the same GPU");
            }

            if (a->shape().ndims() != 2 || b->shape().ndims() != 2 || result->shape().ndims() != 2)
            {
                throw std::runtime_error("matmul expects 2D input, weight, and output tensors");
            }

            auto is_transposed_contiguous = [](const std::shared_ptr<pi::tensorlib::RealTensor> &tensor) -> bool
            {
                if (tensor->shape().ndims() != 2)
                {
                    return false;
                }
                const auto &shape = tensor->shape();
                const auto &strides = tensor->strides();
                return strides[0] == 1 && strides[1] == shape[0];
            };

            const bool a_row_major = pi::tensorlib::shape_utils::IsRowMajorContiguous(a);
            const bool b_row_major = pi::tensorlib::shape_utils::IsRowMajorContiguous(b);
            const bool result_row_major = pi::tensorlib::shape_utils::IsRowMajorContiguous(result);
            const bool a_transposed_ok = (transpose_state == MatmulTransposeState::kTransposeA ||
                                          transpose_state == MatmulTransposeState::kTransposeAB) &&
                                         is_transposed_contiguous(a);
            const bool b_transposed_ok = (transpose_state == MatmulTransposeState::kTransposeB ||
                                          transpose_state == MatmulTransposeState::kTransposeAB) &&
                                         is_transposed_contiguous(b);

            if (!result_row_major || (!a_row_major && !a_transposed_ok) || (!b_row_major && !b_transposed_ok))
            {
                throw std::runtime_error(
                    "matmul cutlass kernels require contiguous inputs (row-major or transposed views)");
            }

            const auto m64 = a->shape()[0];
            const auto k64 = a->shape()[1];
            const auto k_b64 = b->shape()[0];
            const auto n64 = b->shape()[1];

            if (k64 != k_b64)
            {
                throw std::runtime_error("Cannot multiply matrices: incompatible shapes");
            }

            if (result->shape()[0] != m64 || result->shape()[1] != n64)
            {
                throw std::runtime_error("matmul result tensor has unexpected shape");
            }

            auto to_i32 = [](const uint64_t value, const char *what) -> int32_t
            {
                if (value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
                {
                    std::stringstream ss;
                    ss << what << " exceeds supported range";
                    throw std::runtime_error(ss.str());
                }
                return static_cast<int32_t>(value);
            };

            const int32_t m = to_i32(m64, "matmul rows");
            const int32_t n = to_i32(n64, "matmul columns");
            const int32_t k = to_i32(k64, "matmul shared dimension");

            const auto &a_strides = a->strides();
            const auto &b_strides = b->strides();
            const auto &out_strides = result->strides();

            const bool transpose_a = transpose_state == MatmulTransposeState::kTransposeA ||
                                     transpose_state == MatmulTransposeState::kTransposeAB;
            const bool transpose_b = transpose_state == MatmulTransposeState::kTransposeB ||
                                     transpose_state == MatmulTransposeState::kTransposeAB;

            const int32_t lda = to_i32(transpose_a ? a_strides[1] : a_strides[0], "matmul A leading dimension");
            const int32_t ldb = to_i32(transpose_b ? b_strides[1] : b_strides[0], "matmul B leading dimension");
            const int32_t ldd = to_i32(out_strides[0], "matmul output leading dimension");

            const CutlassGemmConfig cfg{
                .M = m,
                .N = n,
                .K = k,
                .lda = lda,
                .ldb = ldb,
                .ldd = ldd,
                .bias_stride = 0,
                .split_k_slices = 1,
            };

            pi::tensorlib::KernelLaunchArguments arguments{};

            pi::tensorlib::KernelDataArg cfg_storage{};
            cfg_storage.bytes.resize(sizeof(CutlassGemmConfig));
            std::memcpy(cfg_storage.bytes.data(), &cfg, sizeof(CutlassGemmConfig));

            constexpr float alpha_f32 = 1.0f;
            const float beta_f32 = accumulate_output ? 1.0f : 0.0f;

            arguments.args.emplace_back(std::move(cfg_storage));
            arguments.args.emplace_back(a->dataptr());
            arguments.args.emplace_back(b->dataptr());
            arguments.args.emplace_back(static_cast<void *>(nullptr));
            arguments.args.emplace_back(result->dataptr());

            if (dtype == pi::tensorlib::DataType::FLOAT16 && use_fp16_accumulation)
            {
                const uint16_t alpha_half = pi::tensorlib::utils::Fp16FromFp32(alpha_f32);
                const uint16_t beta_half = pi::tensorlib::utils::Fp16FromFp32(beta_f32);
                arguments.args.emplace_back(alpha_half);
                arguments.args.emplace_back(beta_half);
            }
            else
            {
                arguments.args.emplace_back(alpha_f32);
                arguments.args.emplace_back(beta_f32);
            }

            arguments.args.emplace_back(static_cast<void *>(nullptr));

            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_GEMM"); env != nullptr)
            {
                std::clog << "[Gemm] CUTLASS kernel " << kernel_bin.function_name << " launch params: M=" << m
                          << ", N=" << n << ", K=" << k << ", lda=" << lda << ", ldb=" << ldb << ", ldd=" << ldd
                          << ", block_m=" << kernel_bin.meta.block_size_m
                          << ", block_n=" << kernel_bin.meta.block_size_n
                          << ", block_k=" << kernel_bin.meta.block_size_k << '\n';
            }

            const uint32_t block_threads = kernel_bin.num_warps * CUDA_WARP_SIZE;
            const uint32_t grid_tiles_m = CEIL_DIV(static_cast<uint32_t>(m), kernel_bin.meta.block_size_m);
            const uint32_t grid_tiles_n = CEIL_DIV(static_cast<uint32_t>(n), kernel_bin.meta.block_size_n);
            const auto [grid_dim_x, grid_dim_y] =
                ComputeCutlassGridShape(grid_tiles_m, grid_tiles_n, kernel_bin.meta.swizzle_size);
            arguments.grid_dim_x = grid_dim_x;
            arguments.grid_dim_y = grid_dim_y;
            arguments.grid_dim_z = 1;
            arguments.block_dim_x = block_threads;
            arguments.block_dim_y = 1;
            arguments.block_dim_z = 1;
            arguments.shared_mem_bytes = kernel_bin.shared_mem_bytes;
            arguments.device_ordinal = static_cast<int>(device_ordinal);
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel_bin.data, kernel_bin.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel_bin.data, kernel_bin.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor CreateTritonMatmulComputeKernelDescriptor(
    const pi::tensorlib::DataType dtype, const pi::tensorlib::DataType input_dtype,
    const kernel_bin_t<kernel_meta_gemm_t> &kernel_bin, const std::string &kernel_name, const bool aligned,
    const MatmulTransposeState transpose_state)
{
    const auto dtype_name = std::string(pi::tensorlib::GetDataTypeName(dtype));
    return pi::tensorlib::ComputeKernelDescriptor{
        // Use a unique kernel name per specialization so the cache does not alias different binaries.
        .kernel_name = kernel_name,
        .function_name = kernel_bin.function_name,
        .expected_arg_count = kernel_bin.arg_count,
        .argument_provider = [dtype, input_dtype, dtype_name, &kernel_bin, kernel_name, aligned,
                              transpose_state](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                               const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 2 || outputs.size() != 1)
            {
                throw std::runtime_error("matmul expects two inputs and one output");
            }

            const auto &a = inputs[0];
            const auto &b = inputs[1];
            const auto &result = outputs[0];

            const auto expected_input_dtype = (dtype == pi::tensorlib::DataType::FLOAT32) ? input_dtype : dtype;
            if (dtype == pi::tensorlib::DataType::FLOAT32 && expected_input_dtype != pi::tensorlib::DataType::FLOAT16 &&
                expected_input_dtype != pi::tensorlib::DataType::BFLOAT16)
            {
                throw std::runtime_error("matmul fp32 output expects fp16 or bf16 inputs");
            }
            if (a->dtype() != expected_input_dtype || b->dtype() != expected_input_dtype || result->dtype() != dtype)
            {
                throw std::runtime_error("matmul requires " + dtype_name + " output and matching input dtype");
            }

            if (a->shape().ndims() != 2 || b->shape().ndims() != 2 || result->shape().ndims() != 2)
            {
                throw std::runtime_error("matmul expects 2D input and output tensors");
            }

            if (a->shape()[1] != b->shape()[0])
            {
                throw std::runtime_error("Cannot multiply matrices: incompatible shapes");
            }

            if (result->shape()[0] != a->shape()[0] || result->shape()[1] != b->shape()[1])
            {
                throw std::runtime_error("matmul result tensor has unexpected shape");
            }

            auto is_transposed_contiguous = [](const std::shared_ptr<pi::tensorlib::RealTensor> &tensor) -> bool
            {
                if (tensor->shape().ndims() != 2)
                {
                    return false;
                }
                const auto &shape = tensor->shape();
                const auto &strides = tensor->strides();
                return strides[0] == 1 && strides[1] == shape[0];
            };

            const bool a_row_major = pi::tensorlib::shape_utils::IsRowMajorContiguous(a);
            const bool b_row_major = pi::tensorlib::shape_utils::IsRowMajorContiguous(b);
            if ((!a_row_major && !is_transposed_contiguous(a)) || (!b_row_major && !is_transposed_contiguous(b)) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(result))
            {
                throw std::runtime_error("matmul triton kernels require contiguous row-major tensors");
            }

            if (a->device().device_type != pi::tensorlib::DeviceType::GPU ||
                b->device().device_type != pi::tensorlib::DeviceType::GPU ||
                result->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("matmul currently supports GPU tensors only");
            }

            const auto device_ordinal = a->device().ordinal;
            if (b->device().ordinal != device_ordinal || result->device().ordinal != device_ordinal)
            {
                throw std::runtime_error("matmul requires all tensors on the same GPU");
            }

            void *a_ptr = a->dataptr();
            void *b_ptr = b->dataptr();
            void *bias_ptr = nullptr;
            void *d_ptr = result->dataptr();

            const auto m = static_cast<uint32_t>(a->shape()[0]);
            const auto n = static_cast<uint32_t>(b->shape()[1]);
            const auto k = static_cast<uint32_t>(a->shape()[1]);

            const auto stride_am = static_cast<uint32_t>(a->strides()[0]);
            const auto stride_ak = static_cast<uint32_t>(a->strides()[1]);
            const auto stride_bk = static_cast<uint32_t>(b->strides()[0]);
            const auto stride_bn = static_cast<uint32_t>(b->strides()[1]);
            constexpr uint32_t stride_cm = 0;
            constexpr uint32_t stride_cn = 0;
            const auto stride_dm = static_cast<uint32_t>(result->strides()[0]);
            const auto stride_dn = static_cast<uint32_t>(result->strides()[1]);

            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_GEMM"); env != nullptr)
            {
                std::clog << "[Gemm] Selecting Triton matmul kernel " << kernel_name << " (" << kernel_bin.function_name
                          << ")\n";
            }

            const MatmulArgLayout arg_layout = GetMatmulArgLayout(aligned, transpose_state);
            pi::tensorlib::KernelLaunchArguments arguments{};
            arguments.args.reserve(18);
            arguments.args.emplace_back(a_ptr);
            arguments.args.emplace_back(b_ptr);
            arguments.args.emplace_back(bias_ptr);
            arguments.args.emplace_back(d_ptr);
            arguments.args.emplace_back(m);
            arguments.args.emplace_back(n);
            arguments.args.emplace_back(k);
            if (arg_layout.stride_am_runtime)
            {
                arguments.args.emplace_back(stride_am);
            }
            if (arg_layout.stride_ak_runtime)
            {
                arguments.args.emplace_back(stride_ak);
            }
            if (arg_layout.stride_bk_runtime)
            {
                arguments.args.emplace_back(stride_bk);
            }
            if (arg_layout.stride_bn_runtime)
            {
                arguments.args.emplace_back(stride_bn);
            }
            if (arg_layout.stride_cm_runtime)
            {
                arguments.args.emplace_back(stride_cm);
            }
            if (arg_layout.stride_cn_runtime)
            {
                arguments.args.emplace_back(stride_cn);
            }
            if (arg_layout.stride_dm_runtime)
            {
                arguments.args.emplace_back(stride_dm);
            }
            if (arg_layout.stride_dn_runtime)
            {
                arguments.args.emplace_back(stride_dn);
            }
            arguments.args.emplace_back(static_cast<void *>(nullptr));
            arguments.grid_dim_x = ((m + kernel_bin.meta.block_size_m - 1) / kernel_bin.meta.block_size_m) *
                                   ((n + kernel_bin.meta.block_size_n - 1) / kernel_bin.meta.block_size_n);
            arguments.grid_dim_y = 1;
            arguments.grid_dim_z = 1;
            arguments.block_dim_x = TRITON_WARP_SIZE * kernel_bin.num_warps;
            arguments.block_dim_y = 1;
            arguments.block_dim_z = 1;
            arguments.shared_mem_bytes = kernel_bin.shared_mem_bytes;
            arguments.device_ordinal = static_cast<int>(device_ordinal);
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel_bin.data, kernel_bin.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel_bin.data, kernel_bin.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor CreateTritonMatmulGeluBwdComputeKernelDescriptor(
    const pi::tensorlib::DataType dtype, const kernel_bin_t<kernel_meta_gemm_t> kernel_bin,
    const std::string &kernel_name, const bool aligned, const MatmulTransposeState transpose_state)
{
    const auto dtype_name = std::string(pi::tensorlib::GetDataTypeName(dtype));
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_bin.kernel_name,
        .function_name = kernel_bin.function_name,
        .expected_arg_count = kernel_bin.arg_count,
        .argument_provider = [dtype, dtype_name, kernel_bin, kernel_name, aligned,
                              transpose_state](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                               const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 3 || outputs.size() != 1)
            {
                throw std::runtime_error("matmul_gelu_bwd expects three inputs and one output");
            }

            const auto &a = inputs[0];
            const auto &b = inputs[1];
            const auto &pre = inputs[2];
            const auto &result = outputs[0];

            if (a->dtype() != dtype || b->dtype() != dtype || pre->dtype() != dtype || result->dtype() != dtype)
            {
                throw std::runtime_error("matmul_gelu_bwd requires " + dtype_name + " tensors");
            }

            if (a->shape().ndims() != 2 || b->shape().ndims() != 2 || result->shape().ndims() != 2 ||
                pre->shape().ndims() != 2)
            {
                throw std::runtime_error("matmul_gelu_bwd expects 2D input and output tensors");
            }

            if (a->shape()[1] != b->shape()[0])
            {
                throw std::runtime_error("matmul_gelu_bwd cannot multiply matrices: incompatible shapes");
            }

            if (result->shape()[0] != a->shape()[0] || result->shape()[1] != b->shape()[1])
            {
                throw std::runtime_error("matmul_gelu_bwd result tensor has unexpected shape");
            }

            if (pre->shape()[0] != result->shape()[0] || pre->shape()[1] != result->shape()[1])
            {
                throw std::runtime_error("matmul_gelu_bwd expects pre-activation to match output");
            }

            if (a->device().device_type != pi::tensorlib::DeviceType::GPU ||
                b->device().device_type != pi::tensorlib::DeviceType::GPU ||
                pre->device().device_type != pi::tensorlib::DeviceType::GPU ||
                result->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("matmul_gelu_bwd currently supports GPU tensors only");
            }

            const auto device_ordinal = a->device().ordinal;
            if (b->device().ordinal != device_ordinal || pre->device().ordinal != device_ordinal ||
                result->device().ordinal != device_ordinal)
            {
                throw std::runtime_error("matmul_gelu_bwd requires all tensors on the same GPU");
            }

            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_GEMM"); env != nullptr)
            {
                std::clog << "[Gemm] Selecting Triton matmul_gelu_bwd kernel " << kernel_name << " ("
                          << kernel_bin.function_name << ")\n";
            }

            pi::tensorlib::KernelLaunchArguments arguments{};
            const MatmulArgLayout arg_layout = GetMatmulArgLayout(aligned, transpose_state);
            arguments.args.reserve(18);
            arguments.args.emplace_back(a->dataptr());
            arguments.args.emplace_back(b->dataptr());
            arguments.args.emplace_back(pre->dataptr());
            arguments.args.emplace_back(result->dataptr());
            const auto m = static_cast<uint32_t>(a->shape()[0]);
            const auto n = static_cast<uint32_t>(b->shape()[1]);
            const auto k = static_cast<uint32_t>(a->shape()[1]);
            arguments.args.emplace_back(m);
            arguments.args.emplace_back(n);
            arguments.args.emplace_back(k);
            const auto stride_am = static_cast<uint32_t>(a->strides()[0]);
            const auto stride_ak = static_cast<uint32_t>(a->strides()[1]);
            const auto stride_bk = static_cast<uint32_t>(b->strides()[0]);
            const auto stride_bn = static_cast<uint32_t>(b->strides()[1]);
            const auto stride_cm = static_cast<uint32_t>(pre->strides()[0]);
            const auto stride_cn = static_cast<uint32_t>(pre->strides()[1]);
            const auto stride_dm = static_cast<uint32_t>(result->strides()[0]);
            const auto stride_dn = static_cast<uint32_t>(result->strides()[1]);

            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_GEMM_VALUES"); env != nullptr)
            {
                static bool logged = false;
                if (!logged)
                {
                    logged = true;
                    std::clog << "[matmul_gelu_bwd] m=" << m << " n=" << n << " k=" << k << " stride_am=" << stride_am
                              << " stride_ak=" << stride_ak << " stride_bk=" << stride_bk << " stride_bn=" << stride_bn
                              << " stride_cm=" << stride_cm << " stride_cn=" << stride_cn << " stride_dm=" << stride_dm
                              << " stride_dn=" << stride_dn << " aligned=" << (aligned ? 1 : 0)
                              << " a_ptr=" << a->dataptr() << " b_ptr=" << b->dataptr() << " pre_ptr=" << pre->dataptr()
                              << " out_ptr=" << result->dataptr()
                              << " a_ptr_mod16=" << (reinterpret_cast<uintptr_t>(a->dataptr()) % 16)
                              << " b_ptr_mod16=" << (reinterpret_cast<uintptr_t>(b->dataptr()) % 16)
                              << " pre_ptr_mod16=" << (reinterpret_cast<uintptr_t>(pre->dataptr()) % 16)
                              << " out_ptr_mod16=" << (reinterpret_cast<uintptr_t>(result->dataptr()) % 16) << '\n';
                }
            }

            if (arg_layout.stride_am_runtime)
            {
                arguments.args.emplace_back(stride_am);
            }
            if (arg_layout.stride_ak_runtime)
            {
                arguments.args.emplace_back(stride_ak);
            }
            if (arg_layout.stride_bk_runtime)
            {
                arguments.args.emplace_back(stride_bk);
            }
            if (arg_layout.stride_bn_runtime)
            {
                arguments.args.emplace_back(stride_bn);
            }
            if (arg_layout.stride_cm_runtime)
            {
                arguments.args.emplace_back(stride_cm);
            }
            if (arg_layout.stride_cn_runtime)
            {
                arguments.args.emplace_back(stride_cn);
            }
            if (arg_layout.stride_dm_runtime)
            {
                arguments.args.emplace_back(stride_dm);
            }
            if (arg_layout.stride_dn_runtime)
            {
                arguments.args.emplace_back(stride_dn);
            }
            arguments.args.emplace_back(static_cast<void *>(nullptr));

            arguments.grid_dim_x = ((m + kernel_bin.meta.block_size_m - 1) / kernel_bin.meta.block_size_m) *
                                   ((n + kernel_bin.meta.block_size_n - 1) / kernel_bin.meta.block_size_n);
            arguments.grid_dim_y = 1;
            arguments.grid_dim_z = 1;
            arguments.block_dim_x = TRITON_WARP_SIZE * kernel_bin.num_warps;
            arguments.block_dim_y = 1;
            arguments.block_dim_z = 1;
            arguments.shared_mem_bytes = kernel_bin.shared_mem_bytes;
            arguments.device_ordinal = static_cast<int>(device_ordinal);
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel_bin.data, kernel_bin.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel_bin.data, kernel_bin.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor
CreateMatmulGeluBwdComputeKernelDescriptor(const pi::tensorlib::DataType dtype, const bool use_fp16_accumulation,
                                           const bool allow_cutlass, const bool aligned, const bool accumulate_output,
                                           const MatmulTransposeState transpose_state)
{
    if (allow_cutlass)
    {
        const GemmKernelSet *cutlass_kernels = &kCutlassMatmulGeluBwdKernels;
        switch (transpose_state)
        {
            case MatmulTransposeState::kTransposeA:
                cutlass_kernels = &kCutlassMatmulGeluBwdKernelsTA;
                break;
            case MatmulTransposeState::kTransposeB:
                cutlass_kernels = &kCutlassMatmulGeluBwdKernelsTB;
                break;
            case MatmulTransposeState::kTransposeAB:
                cutlass_kernels = &kCutlassMatmulGeluBwdKernelsTAB;
                break;
            case MatmulTransposeState::kNone:
            default:
                break;
        }
        if (const auto *cutlass_kernel = SelectCutlassGemmKernel(*cutlass_kernels, dtype, use_fp16_accumulation);
            cutlass_kernel != nullptr)
        {
            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_GEMM"); env != nullptr)
            {
                std::clog << "[Gemm] Selecting CUTLASS kernel " << cutlass_kernel->function_name << '\n';
            }
            return CreateCutlassMatmulGeluBwdComputeKernelDescriptor(dtype, *cutlass_kernel, use_fp16_accumulation,
                                                                     accumulate_output, transpose_state);
        }
    }

    kernel_bin_t<kernel_meta_gemm_t> kernel{};
    std::string kernel_name{};
    if (dtype == pi::tensorlib::DataType::BFLOAT16)
    {
        switch (transpose_state)
        {
            case MatmulTransposeState::kTransposeA:
                kernel = aligned ? (accumulate_output ? kmatmul_gelu_bwd_bf16_ta_cacc : kmatmul_gelu_bwd_bf16_ta)
                                 : (accumulate_output ? kmatmul_gelu_bwd_bf16_unaligned_ta_cacc
                                                      : kmatmul_gelu_bwd_bf16_unaligned_ta);
                kernel_name = aligned ? (accumulate_output ? "matmul_gelu_bwd_bf16_ta_cacc" : "matmul_gelu_bwd_bf16_ta")
                                      : (accumulate_output ? "matmul_gelu_bwd_bf16_unaligned_ta_cacc"
                                                           : "matmul_gelu_bwd_bf16_unaligned_ta_cstore");
                break;
            case MatmulTransposeState::kTransposeB:
                kernel = aligned ? (accumulate_output ? kmatmul_gelu_bwd_bf16_tb_cacc : kmatmul_gelu_bwd_bf16_tb)
                                 : (accumulate_output ? kmatmul_gelu_bwd_bf16_unaligned_tb_cacc
                                                      : kmatmul_gelu_bwd_bf16_unaligned_tb);
                kernel_name = aligned ? (accumulate_output ? "matmul_gelu_bwd_bf16_tb_cacc" : "matmul_gelu_bwd_bf16_tb")
                                      : (accumulate_output ? "matmul_gelu_bwd_bf16_unaligned_tb_cacc"
                                                           : "matmul_gelu_bwd_bf16_unaligned_tb_cstore");
                break;
            case MatmulTransposeState::kTransposeAB:
                kernel = aligned ? (accumulate_output ? kmatmul_gelu_bwd_bf16_tab_cacc : kmatmul_gelu_bwd_bf16_tab)
                                 : (accumulate_output ? kmatmul_gelu_bwd_bf16_unaligned_tab_cacc
                                                      : kmatmul_gelu_bwd_bf16_unaligned_tab);
                kernel_name = aligned
                                  ? (accumulate_output ? "matmul_gelu_bwd_bf16_tab_cacc" : "matmul_gelu_bwd_bf16_tab")
                                  : (accumulate_output ? "matmul_gelu_bwd_bf16_unaligned_tab_cacc"
                                                       : "matmul_gelu_bwd_bf16_unaligned_tab_cstore");
                break;
            case MatmulTransposeState::kNone:
            default:
                kernel = aligned ? (accumulate_output ? kmatmul_gelu_bwd_bf16_cacc : kmatmul_gelu_bwd_bf16)
                                 : (accumulate_output ? kmatmul_gelu_bwd_bf16_unaligned_cacc
                                                      : kmatmul_gelu_bwd_bf16_unaligned);
                kernel_name = aligned ? (accumulate_output ? "matmul_gelu_bwd_bf16_cacc" : "matmul_gelu_bwd_bf16")
                                      : (accumulate_output ? "matmul_gelu_bwd_bf16_unaligned_cacc"
                                                           : "matmul_gelu_bwd_bf16_unaligned_cstore");
                break;
        }
    }
    else if (dtype == pi::tensorlib::DataType::FLOAT16)
    {
        if (use_fp16_accumulation)
        {
#if PI_TENSORLIB_ENABLE_CUDA
            switch (transpose_state)
            {
                case MatmulTransposeState::kTransposeA:
                    kernel = aligned ? (accumulate_output ? kmatmul_gelu_bwd_fp16_acc_fp16_ta_cacc
                                                          : kmatmul_gelu_bwd_fp16_acc_fp16_ta)
                                     : (accumulate_output ? kmatmul_gelu_bwd_fp16_acc_fp16_unaligned_ta_cacc
                                                          : kmatmul_gelu_bwd_fp16_acc_fp16_unaligned_ta);
                    kernel_name = aligned ? (accumulate_output ? "matmul_gelu_bwd_fp16_acc_fp16_ta_cacc"
                                                               : "matmul_gelu_bwd_fp16_acc_fp16_ta")
                                          : (accumulate_output ? "matmul_gelu_bwd_fp16_acc_fp16_unaligned_ta_cacc"
                                                               : "matmul_gelu_bwd_fp16_acc_fp16_unaligned_ta_cstore");
                    break;
                case MatmulTransposeState::kTransposeB:
                    kernel = aligned ? (accumulate_output ? kmatmul_gelu_bwd_fp16_acc_fp16_tb_cacc
                                                          : kmatmul_gelu_bwd_fp16_acc_fp16_tb)
                                     : (accumulate_output ? kmatmul_gelu_bwd_fp16_acc_fp16_unaligned_tb_cacc
                                                          : kmatmul_gelu_bwd_fp16_acc_fp16_unaligned_tb);
                    kernel_name = aligned ? (accumulate_output ? "matmul_gelu_bwd_fp16_acc_fp16_tb_cacc"
                                                               : "matmul_gelu_bwd_fp16_acc_fp16_tb")
                                          : (accumulate_output ? "matmul_gelu_bwd_fp16_acc_fp16_unaligned_tb_cacc"
                                                               : "matmul_gelu_bwd_fp16_acc_fp16_unaligned_tb_cstore");
                    break;
                case MatmulTransposeState::kTransposeAB:
                    kernel = aligned ? (accumulate_output ? kmatmul_gelu_bwd_fp16_acc_fp16_tab_cacc
                                                          : kmatmul_gelu_bwd_fp16_acc_fp16_tab)
                                     : (accumulate_output ? kmatmul_gelu_bwd_fp16_acc_fp16_unaligned_tab_cacc
                                                          : kmatmul_gelu_bwd_fp16_acc_fp16_unaligned_tab);
                    kernel_name = aligned ? (accumulate_output ? "matmul_gelu_bwd_fp16_acc_fp16_tab_cacc"
                                                               : "matmul_gelu_bwd_fp16_acc_fp16_tab")
                                          : (accumulate_output ? "matmul_gelu_bwd_fp16_acc_fp16_unaligned_tab_cacc"
                                                               : "matmul_gelu_bwd_fp16_acc_fp16_unaligned_tab_cstore");
                    break;
                case MatmulTransposeState::kNone:
                default:
                    kernel = aligned ? (accumulate_output ? kmatmul_gelu_bwd_fp16_acc_fp16_cacc
                                                          : kmatmul_gelu_bwd_fp16_acc_fp16)
                                     : (accumulate_output ? kmatmul_gelu_bwd_fp16_acc_fp16_unaligned_cacc
                                                          : kmatmul_gelu_bwd_fp16_acc_fp16_unaligned);
                    kernel_name = aligned ? (accumulate_output ? "matmul_gelu_bwd_fp16_acc_fp16_cacc"
                                                               : "matmul_gelu_bwd_fp16_acc_fp16")
                                          : (accumulate_output ? "matmul_gelu_bwd_fp16_acc_fp16_unaligned_cacc"
                                                               : "matmul_gelu_bwd_fp16_acc_fp16_unaligned_cstore");
                    break;
            }
#else
            RequireFp16AccumulationSupported(true);
#endif
        }
        else
        {
            switch (transpose_state)
            {
                case MatmulTransposeState::kTransposeA:
                    kernel = aligned ? (accumulate_output ? kmatmul_gelu_bwd_fp16_ta_cacc : kmatmul_gelu_bwd_fp16_ta)
                                     : (accumulate_output ? kmatmul_gelu_bwd_fp16_unaligned_ta_cacc
                                                          : kmatmul_gelu_bwd_fp16_unaligned_ta);
                    kernel_name = aligned
                                      ? (accumulate_output ? "matmul_gelu_bwd_fp16_ta_cacc" : "matmul_gelu_bwd_fp16_ta")
                                      : (accumulate_output ? "matmul_gelu_bwd_fp16_unaligned_ta_cacc"
                                                           : "matmul_gelu_bwd_fp16_unaligned_ta_cstore");
                    break;
                case MatmulTransposeState::kTransposeB:
                    kernel = aligned ? (accumulate_output ? kmatmul_gelu_bwd_fp16_tb_cacc : kmatmul_gelu_bwd_fp16_tb)
                                     : (accumulate_output ? kmatmul_gelu_bwd_fp16_unaligned_tb_cacc
                                                          : kmatmul_gelu_bwd_fp16_unaligned_tb);
                    kernel_name = aligned
                                      ? (accumulate_output ? "matmul_gelu_bwd_fp16_tb_cacc" : "matmul_gelu_bwd_fp16_tb")
                                      : (accumulate_output ? "matmul_gelu_bwd_fp16_unaligned_tb_cacc"
                                                           : "matmul_gelu_bwd_fp16_unaligned_tb_cstore");
                    break;
                case MatmulTransposeState::kTransposeAB:
                    kernel = aligned ? (accumulate_output ? kmatmul_gelu_bwd_fp16_tab_cacc : kmatmul_gelu_bwd_fp16_tab)
                                     : (accumulate_output ? kmatmul_gelu_bwd_fp16_unaligned_tab_cacc
                                                          : kmatmul_gelu_bwd_fp16_unaligned_tab);
                    kernel_name =
                        aligned ? (accumulate_output ? "matmul_gelu_bwd_fp16_tab_cacc" : "matmul_gelu_bwd_fp16_tab")
                                : (accumulate_output ? "matmul_gelu_bwd_fp16_unaligned_tab_cacc"
                                                     : "matmul_gelu_bwd_fp16_unaligned_tab_cstore");
                    break;
                case MatmulTransposeState::kNone:
                default:
                    kernel = aligned ? (accumulate_output ? kmatmul_gelu_bwd_fp16_cacc : kmatmul_gelu_bwd_fp16)
                                     : (accumulate_output ? kmatmul_gelu_bwd_fp16_unaligned_cacc
                                                          : kmatmul_gelu_bwd_fp16_unaligned);
                    kernel_name = aligned ? (accumulate_output ? "matmul_gelu_bwd_fp16_cacc" : "matmul_gelu_bwd_fp16")
                                          : (accumulate_output ? "matmul_gelu_bwd_fp16_unaligned_cacc"
                                                               : "matmul_gelu_bwd_fp16_unaligned_cstore");
                    break;
            }
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data type for matmul_gelu_bwd kernel");
    }

    return CreateTritonMatmulGeluBwdComputeKernelDescriptor(dtype, kernel, kernel_name, aligned, transpose_state);
}

static pi::tensorlib::ComputeKernelDescriptor
CreateAddMMGeluComputeKernelDescriptor(const pi::tensorlib::DataType dtype, const bool use_fp16_accumulation,
                                       const bool allow_cutlass, const bool accumulate_output, const bool aligned,
                                       const bool write_out_preact)
{
    kernel_bin_t<kernel_meta_gemm_t> kernel{};
    std::string kernel_name{};
    switch (dtype)
    {
        case pi::tensorlib::DataType::BFLOAT16:
        {
            if (aligned)
            {
                if (write_out_preact)
                {
                    kernel = accumulate_output ? kaddmm_gelu_preact_bf16_cacc : kaddmm_gelu_preact_bf16;
                    kernel_name = accumulate_output ? "addmm_gelu_preact_bf16_cacc" : "addmm_gelu_preact_bf16_cstore";
                }
                else
                {
                    kernel = accumulate_output ? kaddmm_gelu_bf16_cacc : kaddmm_gelu_bf16;
                    kernel_name = accumulate_output ? "addmm_gelu_bf16_cacc" : "addmm_gelu_bf16_cstore";
                }
            }
            else
            {
                if (write_out_preact)
                {
                    kernel =
                        accumulate_output ? kaddmm_gelu_preact_bf16_unaligned_cacc : kaddmm_gelu_preact_bf16_unaligned;
                    kernel_name = accumulate_output ? "addmm_gelu_preact_bf16_unaligned_cacc"
                                                    : "addmm_gelu_preact_bf16_unaligned_cstore";
                }
                else
                {
                    kernel = accumulate_output ? kaddmm_gelu_bf16_unaligned_cacc : kaddmm_gelu_bf16_unaligned;
                    kernel_name =
                        accumulate_output ? "addmm_gelu_bf16_unaligned_cacc" : "addmm_gelu_bf16_unaligned_cstore";
                }
            }
            break;
        }
        case pi::tensorlib::DataType::FLOAT16:
        {
            if (use_fp16_accumulation)
            {
                RequireFp16AccumulationSupported(true);
            }
            if (aligned)
            {
#if PI_TENSORLIB_ENABLE_CUDA
                if (use_fp16_accumulation)
                {
                    if (write_out_preact)
                    {
                        kernel = accumulate_output ? kaddmm_gelu_preact_fp16_acc_fp16_cacc
                                                   : kaddmm_gelu_preact_fp16_acc_fp16;
                        kernel_name = accumulate_output ? "addmm_gelu_preact_fp16_acc_fp16_cacc"
                                                        : "addmm_gelu_preact_fp16_acc_fp16_cstore";
                    }
                    else
                    {
                        kernel = accumulate_output ? kaddmm_gelu_fp16_acc_fp16_cacc : kaddmm_gelu_fp16_acc_fp16;
                        kernel_name =
                            accumulate_output ? "addmm_gelu_fp16_acc_fp16_cacc" : "addmm_gelu_fp16_acc_fp16_cstore";
                    }
                }
                else
#endif
                {
                    if (write_out_preact)
                    {
                        kernel = accumulate_output ? kaddmm_gelu_preact_fp16_cacc : kaddmm_gelu_preact_fp16;
                        kernel_name =
                            accumulate_output ? "addmm_gelu_preact_fp16_cacc" : "addmm_gelu_preact_fp16_cstore";
                    }
                    else
                    {
                        kernel = accumulate_output ? kaddmm_gelu_fp16_cacc : kaddmm_gelu_fp16;
                        kernel_name = accumulate_output ? "addmm_gelu_fp16_cacc" : "addmm_gelu_fp16_cstore";
                    }
                }
            }
            else
            {
#if PI_TENSORLIB_ENABLE_CUDA
                if (use_fp16_accumulation)
                {
                    if (write_out_preact)
                    {
                        kernel = accumulate_output ? kaddmm_gelu_preact_fp16_acc_fp16_unaligned_cacc
                                                   : kaddmm_gelu_preact_fp16_acc_fp16_unaligned;
                        kernel_name = accumulate_output ? "addmm_gelu_preact_fp16_acc_fp16_unaligned_cacc"
                                                        : "addmm_gelu_preact_fp16_acc_fp16_unaligned_cstore";
                    }
                    else
                    {
                        kernel = accumulate_output ? kaddmm_gelu_fp16_acc_fp16_unaligned_cacc
                                                   : kaddmm_gelu_fp16_acc_fp16_unaligned;
                        kernel_name = accumulate_output ? "addmm_gelu_fp16_acc_fp16_unaligned_cacc"
                                                        : "addmm_gelu_fp16_acc_fp16_unaligned_cstore";
                    }
                }
                else
#endif
                {
                    if (write_out_preact)
                    {
                        kernel = accumulate_output ? kaddmm_gelu_preact_fp16_unaligned_cacc
                                                   : kaddmm_gelu_preact_fp16_unaligned;
                        kernel_name = accumulate_output ? "addmm_gelu_preact_fp16_unaligned_cacc"
                                                        : "addmm_gelu_preact_fp16_unaligned_cstore";
                    }
                    else
                    {
                        kernel = accumulate_output ? kaddmm_gelu_fp16_unaligned_cacc : kaddmm_gelu_fp16_unaligned;
                        kernel_name =
                            accumulate_output ? "addmm_gelu_fp16_unaligned_cacc" : "addmm_gelu_fp16_unaligned_cstore";
                    }
                }
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type for addmm_gelu kernel");
    }
    const std::string dtype_name = pi::tensorlib::GetDataTypeName(dtype);

    if (allow_cutlass)
    {
        const auto &cutlass_kernels = write_out_preact ? kCutlassAddmmGeluPreactKernels : kCutlassAddmmGeluKernels;
        if (const auto *cutlass_kernel = SelectCutlassGemmKernel(cutlass_kernels, dtype, use_fp16_accumulation);
            cutlass_kernel != nullptr)
        {
            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_GEMM"); env != nullptr)
            {
                std::clog << "[Gemm] Selecting CUTLASS kernel " << cutlass_kernel->function_name << '\n';
            }
            return CreateCutlassGemmComputeKernelDescriptor(dtype, *cutlass_kernel, use_fp16_accumulation,
                                                            accumulate_output, write_out_preact);
        }
    }

    if (const char *env = std::getenv("FBAMTRAIN_DEBUG_GEMM"); env != nullptr)
    {
        std::clog << "[Gemm] Selecting Triton kernel " << kernel.function_name << '\n';
    }

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [kernel, dtype, dtype_name, aligned,
                              write_out_preact](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                                const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            const size_t expected_outputs = write_out_preact ? 2u : 1u;
            if (inputs.size() != 3 || outputs.size() != expected_outputs)
            {
                throw std::runtime_error("addmm_gelu expects three inputs and " + std::to_string(expected_outputs) +
                                         " output(s)");
            }

            const auto &a = inputs[0];
            const auto &b = inputs[1];
            const auto &bias = inputs[2];
            const auto &result = outputs[0];
            const std::shared_ptr<pi::tensorlib::RealTensor> pre_act =
                write_out_preact ? outputs[1] : std::shared_ptr<pi::tensorlib::RealTensor>{};

            if (a->dtype() != dtype || b->dtype() != dtype || bias->dtype() != dtype || result->dtype() != dtype ||
                (write_out_preact && pre_act->dtype() != dtype))
            {
                throw std::runtime_error("addmm_gelu requires " + std::string(dtype_name) + " tensors");
            }
            if (a->device().device_type != pi::tensorlib::DeviceType::GPU ||
                b->device().device_type != pi::tensorlib::DeviceType::GPU ||
                bias->device().device_type != pi::tensorlib::DeviceType::GPU ||
                result->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("addmm_gelu currently supports GPU tensors only");
            }

            const auto device_ordinal = write_out_preact
                                            ? ValidateSameDeviceOrdinal("addmm_gelu", {a, b, bias, result, pre_act})
                                            : ValidateSameDeviceOrdinal("addmm_gelu", {a, b, bias, result});
            if (a->shape()[1] != b->shape()[0])
            {
                throw std::runtime_error("Cannot multiply matrices: incompatible shapes");
            }
            if (write_out_preact)
            {
                if (pre_act->shape()[0] != result->shape()[0] || pre_act->shape()[1] != result->shape()[1])
                {
                    throw std::runtime_error("addmm_gelu pre-activation output shape mismatch");
                }
            }

            void *a_ptr = a->dataptr();
            void *b_ptr = b->dataptr();
            void *c_ptr = bias->dataptr();
            void *d_ptr = result->dataptr();
            void *pre_ptr = write_out_preact ? pre_act->dataptr() : nullptr;

            const auto m = static_cast<uint32_t>(a->shape()[0]);
            const auto n = static_cast<uint32_t>(b->shape()[1]);
            const auto k = static_cast<uint32_t>(a->shape()[1]);

            const auto stride_am = static_cast<uint32_t>(a->strides()[0]);
            const auto stride_bk = static_cast<uint32_t>(b->strides()[0]);
            constexpr uint32_t stride_cm = 0;
            const auto stride_dm = static_cast<uint32_t>(result->strides()[0]);
            const auto stride_em = write_out_preact ? static_cast<uint32_t>(pre_act->strides()[0]) : 0u;
            const auto stride_ak = static_cast<uint32_t>(a->strides()[1]);
            const auto stride_bn = static_cast<uint32_t>(b->strides()[1]);
            constexpr uint32_t stride_cn = 1;
            const auto stride_dn = static_cast<uint32_t>(result->strides()[1]);
            const auto stride_en = write_out_preact ? static_cast<uint32_t>(pre_act->strides()[1]) : 0u;

            const AddmmArgLayout arg_layout = GetAddmmArgLayout(aligned);
            pi::tensorlib::KernelLaunchArguments arguments{};
            arguments.args.reserve(18);
            arguments.args.emplace_back(a_ptr);
            arguments.args.emplace_back(b_ptr);
            arguments.args.emplace_back(c_ptr);
            arguments.args.emplace_back(d_ptr);
            if (write_out_preact)
            {
                arguments.args.emplace_back(pre_ptr);
            }
            arguments.args.emplace_back(m);
            arguments.args.emplace_back(n);
            arguments.args.emplace_back(k);
            if (arg_layout.stride_am_runtime)
            {
                arguments.args.emplace_back(stride_am);
            }
            if (arg_layout.stride_ak_runtime)
            {
                arguments.args.emplace_back(stride_ak);
            }
            if (arg_layout.stride_bk_runtime)
            {
                arguments.args.emplace_back(stride_bk);
            }
            if (arg_layout.stride_bn_runtime)
            {
                arguments.args.emplace_back(stride_bn);
            }
            if (arg_layout.stride_cm_runtime)
            {
                arguments.args.emplace_back(stride_cm);
            }
            if (arg_layout.stride_cn_runtime)
            {
                arguments.args.emplace_back(stride_cn);
            }
            if (arg_layout.stride_dm_runtime)
            {
                arguments.args.emplace_back(stride_dm);
            }
            if (arg_layout.stride_dn_runtime)
            {
                arguments.args.emplace_back(stride_dn);
            }
            if (write_out_preact)
            {
                arguments.args.emplace_back(stride_em);
                if (arg_layout.stride_dn_runtime)
                {
                    arguments.args.emplace_back(stride_en);
                }
            }
            arguments.args.emplace_back(static_cast<void *>(nullptr));
            arguments.grid_dim_x = ((m + kernel.meta.block_size_m - 1) / kernel.meta.block_size_m) *
                                   ((n + kernel.meta.block_size_n - 1) / kernel.meta.block_size_n);
            arguments.grid_dim_y = 1;
            arguments.grid_dim_z = 1;
            arguments.block_dim_x = TRITON_WARP_SIZE * kernel.num_warps;
            arguments.block_dim_y = 1;
            arguments.block_dim_z = 1;
            arguments.shared_mem_bytes = kernel.shared_mem_bytes;
            arguments.device_ordinal = device_ordinal;
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel.data, kernel.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel.data, kernel.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor
CreateAddMMComputeKernelDescriptor(const pi::tensorlib::DataType dtype, const bool use_fp16_accumulation,
                                   const bool allow_cutlass, const bool accumulate_output, const bool aligned)
{
    kernel_bin_t<kernel_meta_gemm_t> kernel{};
    std::string kernel_name{};
    if (dtype == pi::tensorlib::DataType::FLOAT32)
    {
        if (use_fp16_accumulation)
        {
            RequireFp16AccumulationSupported(true);
        }
        if (allow_cutlass)
        {
            if (const auto *cutlass_kernel =
                    SelectCutlassGemmKernel(kCutlassAddmmKernels, dtype, use_fp16_accumulation);
                cutlass_kernel != nullptr)
            {
                if (const char *env = std::getenv("FBAMTRAIN_DEBUG_GEMM"); env != nullptr)
                {
                    std::clog << "[Gemm] Selecting CUTLASS kernel " << cutlass_kernel->function_name << '\n';
                }
                return CreateCutlassGemmComputeKernelDescriptor(dtype, *cutlass_kernel, use_fp16_accumulation,
                                                                accumulate_output);
            }
        }

        if (aligned)
        {
#if PI_TENSORLIB_ENABLE_CUDA
            if (use_fp16_accumulation)
            {
                kernel = accumulate_output ? kaddmm_fp16_acc_out_fp32_cacc : kaddmm_fp16_acc_out_fp32;
                kernel_name = accumulate_output ? "addmm_fp16_acc_out_fp32_cacc" : "addmm_fp16_acc_out_fp32_cstore";
            }
            else
#endif
            {
                kernel = accumulate_output ? kaddmm_fp16_out_fp32_cacc : kaddmm_fp16_out_fp32;
                kernel_name = accumulate_output ? "addmm_fp16_out_fp32_cacc" : "addmm_fp16_out_fp32_cstore";
            }
        }
        else
        {
#if PI_TENSORLIB_ENABLE_CUDA
            if (use_fp16_accumulation)
            {
                kernel =
                    accumulate_output ? kaddmm_fp16_acc_out_fp32_unaligned_cacc : kaddmm_fp16_acc_out_fp32_unaligned;
                kernel_name = accumulate_output ? "addmm_fp16_acc_out_fp32_unaligned_cacc"
                                                : "addmm_fp16_acc_out_fp32_unaligned_cstore";
            }
            else
#endif
            {
                kernel = accumulate_output ? kaddmm_fp16_out_fp32_unaligned_cacc : kaddmm_fp16_out_fp32_unaligned;
                kernel_name =
                    accumulate_output ? "addmm_fp16_out_fp32_unaligned_cacc" : "addmm_fp16_out_fp32_unaligned_cstore";
            }
        }

        if (const char *env = std::getenv("FBAMTRAIN_DEBUG_GEMM"); env != nullptr)
        {
            std::clog << "[Gemm] Selecting Triton kernel " << kernel.function_name << '\n';
        }
        const std::string dtype_name = pi::tensorlib::GetDataTypeName(dtype);
        return pi::tensorlib::ComputeKernelDescriptor{
            .kernel_name = kernel_name,
            .function_name = kernel.function_name,
            .expected_arg_count = kernel.arg_count,
            .argument_provider = [kernel, dtype,
                                  aligned](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                           const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
                -> pi::tensorlib::KernelLaunchArguments
            {
                if (inputs.size() != 3 || outputs.size() != 1)
                {
                    throw std::runtime_error("addmm expects three inputs and one output");
                }

                const auto &a = inputs[0];
                const auto &b = inputs[1];
                const auto &bias = inputs[2];
                const auto &result = outputs[0];

                if (a->dtype() != pi::tensorlib::DataType::FLOAT16 || b->dtype() != pi::tensorlib::DataType::FLOAT16 ||
                    bias->dtype() != pi::tensorlib::DataType::FLOAT32 || result->dtype() != dtype)
                {
                    throw std::runtime_error("addmm requires fp16 inputs and fp32 output");
                }
                const auto device_ordinal = a->device().ordinal;
                if (b->device().ordinal != device_ordinal || bias->device().ordinal != device_ordinal ||
                    result->device().ordinal != device_ordinal)
                {
                    throw std::runtime_error("addmm requires all tensors on the same GPU");
                }

                const auto m64 = a->shape()[0];
                const auto k64 = a->shape()[1];
                const auto k_b64 = b->shape()[0];
                const auto n64 = b->shape()[1];

                if (k64 != k_b64)
                {
                    throw std::runtime_error("Cannot multiply matrices: incompatible shapes");
                }

                if (result->shape()[0] != m64 || result->shape()[1] != n64)
                {
                    throw std::runtime_error("addmm result tensor has unexpected shape");
                }

                if (bias->shape().ndims() != 1 || bias->shape()[0] != n64)
                {
                    throw std::runtime_error("addmm kernels require 1D bias vectors");
                }

                auto to_i32 = [](const uint64_t value, const char *what) -> int32_t
                {
                    if (value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
                    {
                        std::stringstream ss;
                        ss << what << " exceeds supported range";
                        throw std::runtime_error(ss.str());
                    }
                    return static_cast<int32_t>(value);
                };

                const int32_t m = to_i32(m64, "addmm rows");
                const int32_t n = to_i32(n64, "addmm columns");
                const int32_t k = to_i32(k64, "addmm shared dimension");
                if (const char *env = std::getenv("FBAMTRAIN_DEBUG_GEMM_VALUES"); env != nullptr)
                {
                    static bool logged = false;
                    if (!logged)
                    {
                        logged = true;
                        std::clog << "[addmm] m=" << m << " n=" << n << " k=" << k << " lda=" << a->strides()[0]
                                  << " ldb=" << b->strides()[0] << " out_stride=" << result->strides()[0] << '\n';
                        auto b_storage = b->storage();
                        if (b_storage->device().device_type != pi::tensorlib::DeviceType::CPU)
                        {
                            b_storage = b_storage->toCPU();
                        }
                        const auto *b_ptr = static_cast<const uint16_t *>(b_storage->dataptr());
                        std::clog << "[addmm] b[0,0..4]=" << b_ptr[0] << ", " << b_ptr[1] << ", " << b_ptr[2] << ", "
                                  << b_ptr[3] << ", " << b_ptr[4] << '\n';
                        auto a_storage = a->storage();
                        if (a_storage->device().device_type != pi::tensorlib::DeviceType::CPU)
                        {
                            a_storage = a_storage->toCPU();
                        }
                        const auto *a_ptr = static_cast<const uint16_t *>(a_storage->dataptr());
                        std::clog << "[addmm] a[0,0..4]=" << a_ptr[0] << ", " << a_ptr[1] << ", " << a_ptr[2] << ", "
                                  << a_ptr[3] << ", " << a_ptr[4] << " a[0,6..10]=" << a_ptr[6] << ", " << a_ptr[7]
                                  << ", " << a_ptr[8] << ", " << a_ptr[9] << ", " << a_ptr[10] << '\n';
                    }
                }

                const auto stride_am = static_cast<uint32_t>(a->strides()[0]);
                const auto stride_ak = static_cast<uint32_t>(a->strides()[1]);
                const auto stride_bk = static_cast<uint32_t>(b->strides()[0]);
                const auto stride_bn = static_cast<uint32_t>(b->strides()[1]);
                constexpr uint32_t stride_cm = 0;
                constexpr uint32_t stride_cn = 1;
                const auto stride_dm = static_cast<uint32_t>(result->strides()[0]);
                const auto stride_dn = static_cast<uint32_t>(result->strides()[1]);

                const AddmmArgLayout arg_layout = GetAddmmArgLayout(aligned);
                pi::tensorlib::KernelLaunchArguments arguments{};
                arguments.args.reserve(18);
                arguments.args.emplace_back(a->dataptr());
                arguments.args.emplace_back(b->dataptr());
                arguments.args.emplace_back(bias->dataptr());
                arguments.args.emplace_back(result->dataptr());
                arguments.args.emplace_back(m);
                arguments.args.emplace_back(n);
                arguments.args.emplace_back(k);
                if (arg_layout.stride_am_runtime)
                {
                    arguments.args.emplace_back(stride_am);
                }
                if (arg_layout.stride_ak_runtime)
                {
                    arguments.args.emplace_back(stride_ak);
                }
                if (arg_layout.stride_bk_runtime)
                {
                    arguments.args.emplace_back(stride_bk);
                }
                if (arg_layout.stride_bn_runtime)
                {
                    arguments.args.emplace_back(stride_bn);
                }
                if (arg_layout.stride_cm_runtime)
                {
                    arguments.args.emplace_back(stride_cm);
                }
                if (arg_layout.stride_cn_runtime)
                {
                    arguments.args.emplace_back(stride_cn);
                }
                if (arg_layout.stride_dm_runtime)
                {
                    arguments.args.emplace_back(stride_dm);
                }
                if (arg_layout.stride_dn_runtime)
                {
                    arguments.args.emplace_back(stride_dn);
                }
                arguments.args.emplace_back(static_cast<void *>(nullptr));
                // addmm_act uses a 1D launch grid that packs (m, n) tiles into axis 0.
                arguments.grid_dim_x = ((m + kernel.meta.block_size_m - 1) / kernel.meta.block_size_m) *
                                       ((n + kernel.meta.block_size_n - 1) / kernel.meta.block_size_n);
                arguments.grid_dim_y = 1;
                arguments.grid_dim_z = 1;
                arguments.block_dim_x = TRITON_WARP_SIZE * kernel.num_warps;
                arguments.block_dim_y = 1;
                arguments.block_dim_z = 1;
                arguments.shared_mem_bytes = kernel.shared_mem_bytes;
                arguments.device_ordinal = device_ordinal;
                return arguments;
            },
            .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel.data, kernel.size),
            .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel.data, kernel.size),
        };
    }

    switch (dtype)
    {
        case pi::tensorlib::DataType::BFLOAT16:
        {
            if (aligned)
            {
                kernel = accumulate_output ? kaddmm_bf16_cacc : kaddmm_bf16;
                kernel_name = accumulate_output ? "addmm_bf16_cacc" : "addmm_bf16_cstore";
            }
            else
            {
                kernel = accumulate_output ? kaddmm_bf16_unaligned_cacc : kaddmm_bf16_unaligned;
                kernel_name = accumulate_output ? "addmm_bf16_unaligned_cacc" : "addmm_bf16_unaligned_cstore";
            }
            break;
        }
        case pi::tensorlib::DataType::FLOAT16:
        {
            if (use_fp16_accumulation)
            {
                RequireFp16AccumulationSupported(true);
            }
            if (aligned)
            {
#if PI_TENSORLIB_ENABLE_CUDA
                if (use_fp16_accumulation)
                {
                    kernel = accumulate_output ? kaddmm_fp16_acc_fp16_cacc : kaddmm_fp16_acc_fp16;
                    kernel_name = accumulate_output ? "addmm_fp16_acc_fp16_cacc" : "addmm_fp16_acc_fp16_cstore";
                }
                else
#endif
                {
                    kernel = accumulate_output ? kaddmm_fp16_cacc : kaddmm_fp16;
                    kernel_name = accumulate_output ? "addmm_fp16_cacc" : "addmm_fp16_cstore";
                }
            }
            else
            {
#if PI_TENSORLIB_ENABLE_CUDA
                if (use_fp16_accumulation)
                {
                    kernel = accumulate_output ? kaddmm_fp16_acc_fp16_unaligned_cacc : kaddmm_fp16_acc_fp16_unaligned;
                    kernel_name = accumulate_output ? "addmm_fp16_acc_fp16_unaligned_cacc"
                                                    : "addmm_fp16_acc_fp16_unaligned_cstore";
                }
                else
#endif
                {
                    kernel = accumulate_output ? kaddmm_fp16_unaligned_cacc : kaddmm_fp16_unaligned;
                    kernel_name = accumulate_output ? "addmm_fp16_unaligned_cacc" : "addmm_fp16_unaligned_cstore";
                }
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type for addmm kernel");
    }
    const std::string dtype_name = pi::tensorlib::GetDataTypeName(dtype);

    if (allow_cutlass)
    {
        if (const auto *cutlass_kernel = SelectCutlassGemmKernel(kCutlassAddmmKernels, dtype, use_fp16_accumulation);
            cutlass_kernel != nullptr)
        {
            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_GEMM"); env != nullptr)
            {
                std::clog << "[Gemm] Selecting CUTLASS kernel " << cutlass_kernel->function_name << '\n';
            }
            return CreateCutlassGemmComputeKernelDescriptor(dtype, *cutlass_kernel, use_fp16_accumulation,
                                                            accumulate_output);
        }
    }

    if (const char *env = std::getenv("FBAMTRAIN_DEBUG_GEMM"); env != nullptr)
    {
        std::clog << "[Gemm] Selecting Triton kernel " << kernel.function_name << '\n';
    }

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [kernel, dtype, dtype_name,
                              aligned](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                       const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 3 || outputs.size() != 1)
            {
                throw std::runtime_error("addmm expects three inputs and one output");
            }

            const auto &a = inputs[0];
            const auto &b = inputs[1];
            const auto &bias = inputs[2];
            const auto &result = outputs[0];

            if (a->dtype() != dtype || b->dtype() != dtype || bias->dtype() != dtype || result->dtype() != dtype)
            {
                throw std::runtime_error("addmm requires " + std::string(dtype_name) + " tensors");
            }
            if (a->device().device_type != pi::tensorlib::DeviceType::GPU ||
                b->device().device_type != pi::tensorlib::DeviceType::GPU ||
                bias->device().device_type != pi::tensorlib::DeviceType::GPU ||
                result->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("addmm currently supports GPU tensors only");
            }

            const auto device_ordinal = ValidateSameDeviceOrdinal("addmm", {a, b, bias, result});
            if (a->shape()[1] != b->shape()[0])
            {
                throw std::runtime_error("Cannot multiply matrices: incompatible shapes");
            }

            void *a_ptr = a->dataptr();
            void *b_ptr = b->dataptr();
            void *c_ptr = bias->dataptr();
            void *d_ptr = result->dataptr();

            const auto m = static_cast<uint32_t>(a->shape()[0]);
            const auto n = static_cast<uint32_t>(b->shape()[1]);
            const auto k = static_cast<uint32_t>(a->shape()[1]);

            const auto stride_am = static_cast<uint32_t>(a->strides()[0]);
            const auto stride_bk = static_cast<uint32_t>(b->strides()[0]);
            constexpr uint32_t stride_cm = 0;
            const auto stride_dm = static_cast<uint32_t>(result->strides()[0]);
            const auto stride_ak = static_cast<uint32_t>(a->strides()[1]);
            const auto stride_bn = static_cast<uint32_t>(b->strides()[1]);
            constexpr uint32_t stride_cn = 1;
            const auto stride_dn = static_cast<uint32_t>(result->strides()[1]);

            const AddmmArgLayout arg_layout = GetAddmmArgLayout(aligned);
            pi::tensorlib::KernelLaunchArguments arguments{};
            arguments.args.reserve(18);
            arguments.args.emplace_back(a_ptr);
            arguments.args.emplace_back(b_ptr);
            arguments.args.emplace_back(c_ptr);
            arguments.args.emplace_back(d_ptr);
            arguments.args.emplace_back(m);
            arguments.args.emplace_back(n);
            arguments.args.emplace_back(k);
            if (arg_layout.stride_am_runtime)
            {
                arguments.args.emplace_back(stride_am);
            }
            if (arg_layout.stride_ak_runtime)
            {
                arguments.args.emplace_back(stride_ak);
            }
            if (arg_layout.stride_bk_runtime)
            {
                arguments.args.emplace_back(stride_bk);
            }
            if (arg_layout.stride_bn_runtime)
            {
                arguments.args.emplace_back(stride_bn);
            }
            if (arg_layout.stride_cm_runtime)
            {
                arguments.args.emplace_back(stride_cm);
            }
            if (arg_layout.stride_cn_runtime)
            {
                arguments.args.emplace_back(stride_cn);
            }
            if (arg_layout.stride_dm_runtime)
            {
                arguments.args.emplace_back(stride_dm);
            }
            if (arg_layout.stride_dn_runtime)
            {
                arguments.args.emplace_back(stride_dn);
            }
            arguments.args.emplace_back(static_cast<void *>(nullptr));
            arguments.grid_dim_x = ((m + kernel.meta.block_size_m - 1) / kernel.meta.block_size_m) *
                                   ((n + kernel.meta.block_size_n - 1) / kernel.meta.block_size_n);
            arguments.grid_dim_y = 1;
            arguments.grid_dim_z = 1;
            arguments.block_dim_x = TRITON_WARP_SIZE * kernel.num_warps;
            arguments.block_dim_y = 1;
            arguments.block_dim_z = 1;
            arguments.shared_mem_bytes = kernel.shared_mem_bytes;
            arguments.device_ordinal = device_ordinal;
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel.data, kernel.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel.data, kernel.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor
CreateMatmulComputeKernelDescriptor(const pi::tensorlib::DataType dtype, const pi::tensorlib::DataType input_dtype,
                                    const bool use_fp16_accumulation, const bool allow_cutlass,
                                    const bool accumulate_output, const bool aligned,
                                    const MatmulTransposeState transpose_state)
{
    if (use_fp16_accumulation)
    {
        RequireFp16AccumulationSupported(true);
    }
    auto select_triton_matmul_set = [&](const bool aligned_flag) -> const GemmKernelSet *
    {
        switch (transpose_state)
        {
            case MatmulTransposeState::kTransposeA:
                return aligned_flag ? (accumulate_output ? &kTritonMatmulKernelsCACCTA : &kTritonMatmulKernelsTA)
                                    : (accumulate_output ? &kTritonMatmulUnalignedKernelsCACCTA
                                                         : &kTritonMatmulUnalignedKernelsTA);
            case MatmulTransposeState::kTransposeB:
                return aligned_flag ? (accumulate_output ? &kTritonMatmulKernelsCACCTB : &kTritonMatmulKernelsTB)
                                    : (accumulate_output ? &kTritonMatmulUnalignedKernelsCACCTB
                                                         : &kTritonMatmulUnalignedKernelsTB);
            case MatmulTransposeState::kTransposeAB:
                return aligned_flag ? (accumulate_output ? &kTritonMatmulKernelsCACCTAB : &kTritonMatmulKernelsTAB)
                                    : (accumulate_output ? &kTritonMatmulUnalignedKernelsCACCTAB
                                                         : &kTritonMatmulUnalignedKernelsTAB);
            case MatmulTransposeState::kNone:
            default:
                return aligned_flag
                           ? (accumulate_output ? &kTritonMatmulKernelsCacc : &kTritonMatmulKernels)
                           : (accumulate_output ? &kTritonMatmulUnalignedKernelsCacc : &kTritonMatmulUnalignedKernels);
        }
    };
    auto select_cutlass_matmul_set = [&]() -> const GemmKernelSet *
    {
        switch (transpose_state)
        {
            case MatmulTransposeState::kTransposeA:
                return &kCutlassMatmulKernelsTA;
            case MatmulTransposeState::kTransposeB:
                return &kCutlassMatmulKernelsTB;
            case MatmulTransposeState::kTransposeAB:
                return &kCutlassMatmulKernelsTAB;
            case MatmulTransposeState::kNone:
            default:
                return &kCutlassMatmulKernels;
        }
    };
    auto select_cutlass_fp32_out_set = [&]() -> const GemmKernelSet *
    {
        switch (transpose_state)
        {
            case MatmulTransposeState::kTransposeA:
                return &kCutlassMatmulFp32OutKernelsTA;
            case MatmulTransposeState::kTransposeB:
                return &kCutlassMatmulFp32OutKernelsTB;
            case MatmulTransposeState::kTransposeAB:
                return &kCutlassMatmulFp32OutKernelsTAB;
            case MatmulTransposeState::kNone:
            default:
                return &kCutlassMatmulFp32OutKernels;
        }
    };

    if (dtype == pi::tensorlib::DataType::FLOAT32)
    {
        if (use_fp16_accumulation)
        {
            throw std::runtime_error("use_fp16_matmul_acc is not supported when matmul output dtype is FLOAT32");
        }
        if (input_dtype != pi::tensorlib::DataType::FLOAT16 && input_dtype != pi::tensorlib::DataType::BFLOAT16)
        {
            throw std::runtime_error("FP32 output matmul expects FP16 or BF16 inputs");
        }
        auto select_triton_fp32_out_set = [&](const bool aligned_flag) -> const GemmKernelSet *
        {
            switch (transpose_state)
            {
                case MatmulTransposeState::kTransposeA:
                    return aligned_flag ? (accumulate_output ? &kTritonMatmulFp32OutKernelsCACCTA
                                                             : &kTritonMatmulFp32OutKernelsTA)
                                        : (accumulate_output ? &kTritonMatmulFp32OutUnalignedKernelsCACCTA
                                                             : &kTritonMatmulFp32OutUnalignedKernelsTA);
                case MatmulTransposeState::kTransposeB:
                    return aligned_flag ? (accumulate_output ? &kTritonMatmulFp32OutKernelsCACCTB
                                                             : &kTritonMatmulFp32OutKernelsTB)
                                        : (accumulate_output ? &kTritonMatmulFp32OutUnalignedKernelsCACCTB
                                                             : &kTritonMatmulFp32OutUnalignedKernelsTB);
                case MatmulTransposeState::kTransposeAB:
                    return aligned_flag ? (accumulate_output ? &kTritonMatmulFp32OutKernelsCACCTAB
                                                             : &kTritonMatmulFp32OutKernelsTAB)
                                        : (accumulate_output ? &kTritonMatmulFp32OutUnalignedKernelsCACCTAB
                                                             : &kTritonMatmulFp32OutUnalignedKernelsTAB);
                case MatmulTransposeState::kNone:
                default:
                    return aligned_flag
                               ? (accumulate_output ? &kTritonMatmulFp32OutKernelsCacc : &kTritonMatmulFp32OutKernels)
                               : (accumulate_output ? &kTritonMatmulFp32OutUnalignedKernelsCacc
                                                    : &kTritonMatmulFp32OutUnalignedKernels);
            }
        };

        if (allow_cutlass)
        {
            const auto *cutlass_set = select_cutlass_fp32_out_set();
            if (const auto *cutlass_kernel = SelectFp32OutKernel(*cutlass_set, input_dtype, use_fp16_accumulation);
                cutlass_kernel != nullptr)
            {
                if (const char *env = std::getenv("FBAMTRAIN_DEBUG_GEMM"); env != nullptr)
                {
                    std::clog << "[Gemm] Selecting CUTLASS kernel " << cutlass_kernel->function_name << '\n';
                }
                return CreateCutlassMatmulComputeKernelDescriptor(dtype, input_dtype, *cutlass_kernel,
                                                                  use_fp16_accumulation, accumulate_output,
                                                                  transpose_state);
            }
        }

        const auto *triton_kernels = select_triton_fp32_out_set(aligned);
        if (const auto *triton_kernel = SelectFp32OutKernel(*triton_kernels, input_dtype, use_fp16_accumulation);
            triton_kernel != nullptr)
        {
            std::string kernel_name;
            if (input_dtype == pi::tensorlib::DataType::BFLOAT16)
            {
                kernel_name = "matmul_bf16_out_fp32";
            }
            else
            {
                kernel_name = use_fp16_accumulation ? "matmul_fp16_acc_out_fp32" : "matmul_fp16_out_fp32";
            }
            if (!aligned)
            {
                kernel_name += "_unaligned";
            }
            switch (transpose_state)
            {
                case MatmulTransposeState::kTransposeA:
                    kernel_name += "_ta";
                    break;
                case MatmulTransposeState::kTransposeB:
                    kernel_name += "_tb";
                    break;
                case MatmulTransposeState::kTransposeAB:
                    kernel_name += "_tab";
                    break;
                case MatmulTransposeState::kNone:
                default:
                    break;
            }
            kernel_name += accumulate_output ? "_cacc" : "_cstore";
            return CreateTritonMatmulComputeKernelDescriptor(dtype, input_dtype, *triton_kernel, kernel_name, aligned,
                                                             transpose_state);
        }
        throw std::runtime_error("No matmul implementation available for " +
                                 std::string(pi::tensorlib::GetDataTypeName(input_dtype)) + " -> FP32");
    }
    if (allow_cutlass)
    {
        const auto *cutlass_set = select_cutlass_matmul_set();
        if (const auto *cutlass_kernel = SelectCutlassGemmKernel(*cutlass_set, dtype, use_fp16_accumulation);
            cutlass_kernel != nullptr)
        {
            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_GEMM"); env != nullptr)
            {
                std::clog << "[Gemm] Selecting CUTLASS kernel " << cutlass_kernel->function_name << '\n';
            }
            return CreateCutlassMatmulComputeKernelDescriptor(dtype, input_dtype, *cutlass_kernel,
                                                              use_fp16_accumulation, accumulate_output,
                                                              transpose_state);
        }
    }
    const auto *triton_kernels = select_triton_matmul_set(aligned);
    if (const auto *triton_kernel = SelectCutlassGemmKernel(*triton_kernels, dtype, use_fp16_accumulation);
        triton_kernel != nullptr)
    {
        std::string kernel_name;
        switch (dtype)
        {
            case pi::tensorlib::DataType::BFLOAT16:
            {
                kernel_name = aligned ? "matmul_bf16" : "matmul_bf16_unaligned";
                break;
            }
            case pi::tensorlib::DataType::FLOAT16:
            {
                if (aligned)
                {
                    kernel_name = use_fp16_accumulation ? "matmul_fp16_acc_fp16" : "matmul_fp16";
                }
                else
                {
                    kernel_name = use_fp16_accumulation ? "matmul_fp16_acc_fp16_unaligned" : "matmul_fp16_unaligned";
                }
                break;
            }
            default:
                kernel_name = aligned ? "matmul" : "matmul_unaligned";
                break;
        }
        switch (transpose_state)
        {
            case MatmulTransposeState::kTransposeA:
            {
                kernel_name += "_ta";
                break;
            }
            case MatmulTransposeState::kTransposeB:
            {
                kernel_name += "_tb";
                break;
            }
            case MatmulTransposeState::kTransposeAB:
            {
                kernel_name += "_tab";
                break;
            }
            case MatmulTransposeState::kNone:
            default:
                break;
        }
        kernel_name += accumulate_output ? "_cacc" : "_cstore";
        if (const char *env = std::getenv("FBAMTRAIN_DEBUG_GEMM"); env != nullptr)
        {
            std::clog << "[Gemm] Selecting Triton matmul kernel " << kernel_name << " (" << triton_kernel->function_name
                      << ")\n";
        }
        return CreateTritonMatmulComputeKernelDescriptor(dtype, input_dtype, *triton_kernel, kernel_name, aligned,
                                                         transpose_state);
    }
    throw std::runtime_error("No matmul implementation available for dtype " +
                             std::string(pi::tensorlib::GetDataTypeName(dtype)));
}

void MatmulFusePass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    auto &entries = execution_plan.entries;
    if (entries.empty())
    {
        return;
    }

    std::unordered_map<uint64_t, std::vector<size_t>> consumer_indices{};
    consumer_indices.reserve(entries.size() * 2);
    std::unordered_map<uint64_t, std::vector<size_t>> non_create_producers{};
    non_create_producers.reserve(entries.size() * 2);
    for (size_t entry_idx = 0; entry_idx < entries.size(); ++entry_idx)
    {
        const auto &entry = entries[entry_idx];
        for (const auto &input : entry.inputs)
        {
            if (input == nullptr)
            {
                continue;
            }
            consumer_indices[input->id()].push_back(entry_idx);
        }
        if (entry.op_type != pi::tensorlib::OpType::CREATE_TENSOR)
        {
            for (const auto &output : entry.outputs)
            {
                if (output != nullptr)
                {
                    non_create_producers[output->id()].push_back(entry_idx);
                }
            }
        }
    }

    std::vector<uint8_t> removed(entries.size(), 0u);
    const auto is_alive = [&removed](const size_t idx) { return idx < removed.size() && removed[idx] == 0u; };
    const auto get_consumers = [&consumer_indices](const uint64_t tensor_id) -> const std::vector<size_t> *
    {
        if (const auto it = consumer_indices.find(tensor_id); it != consumer_indices.end())
        {
            return &it->second;
        }
        return nullptr;
    };
    const auto find_last_producer = [&non_create_producers, &entries, &is_alive](const uint64_t tensor_id)
        -> std::optional<size_t>
    {
        const auto it = non_create_producers.find(tensor_id);
        if (it == non_create_producers.end())
        {
            return std::nullopt;
        }
        for (auto rit = it->second.rbegin(); rit != it->second.rend(); ++rit)
        {
            const size_t producer_idx = *rit;
            if (!is_alive(producer_idx))
            {
                continue;
            }
            const auto &producer = entries[producer_idx];
            if (producer.op_type == pi::tensorlib::OpType::CREATE_TENSOR)
            {
                continue;
            }
            if (EntryProducesTensor(producer, tensor_id))
            {
                return producer_idx;
            }
        }
        return std::nullopt;
    };
    const auto replace_tensor_inputs =
        [&entries, &consumer_indices, &get_consumers, &is_alive](const uint64_t from_id,
                                                                 const std::shared_ptr<pi::tensorlib::RealTensor>
                                                                     &to_tensor)
    {
        const auto *consumers = get_consumers(from_id);
        if (consumers == nullptr || to_tensor == nullptr)
        {
            return;
        }
        auto &to_consumers = consumer_indices[to_tensor->id()];
        for (const size_t consumer_idx : *consumers)
        {
            if (!is_alive(consumer_idx))
            {
                continue;
            }
            auto &entry = entries[consumer_idx];
            bool rewired = false;
            for (auto &input : entry.inputs)
            {
                if (input != nullptr && input->id() == from_id)
                {
                    input = to_tensor;
                    rewired = true;
                }
            }
            if (rewired)
            {
                to_consumers.push_back(consumer_idx);
            }
        }
    };

    std::vector<std::pair<uint64_t, size_t>> move_create_before_requests{};
    move_create_before_requests.reserve(16);
    std::unordered_set<uint64_t> maybe_unused_create_tensors{};
    maybe_unused_create_tensors.reserve(16);

    for (size_t i = 0; i < entries.size(); ++i)
    {
        if (!is_alive(i) || entries[i].op_type != pi::tensorlib::OpType::MATMUL)
        {
            continue;
        }

        auto &matmul_op = entries[i];
        if (matmul_op.inputs.size() < 2 || matmul_op.outputs.empty())
        {
            continue;
        }
        const auto &matmul_output = matmul_op.outputs[0];
        if (matmul_output == nullptr)
        {
            continue;
        }

        const auto dtype = matmul_output->dtype();
        const bool allow_fp32_out = (dtype == pi::tensorlib::DataType::FLOAT32);
        if (!allow_fp32_out && !IsSupportedHalfType(dtype))
        {
            continue;
        }

        bool requested_fp16_accumulation = false;
        if (const auto attr_it = matmul_op.attributes.find("use_fp16_matmul_acc");
            attr_it != matmul_op.attributes.end() && attr_it->second.type() == typeid(bool))
        {
            requested_fp16_accumulation = std::any_cast<bool>(attr_it->second);
        }
        if (allow_fp32_out && requested_fp16_accumulation)
        {
            throw std::runtime_error("use_fp16_matmul_acc is not supported when matmul output dtype is FLOAT32");
        }
        if (dtype == pi::tensorlib::DataType::FLOAT16)
        {
            RequireFp16AccumulationSupported(requested_fp16_accumulation);
        }
        const bool use_fp16_accumulation =
            (dtype == pi::tensorlib::DataType::FLOAT16) ? requested_fp16_accumulation : false;
        bool accumulate_output = false;
        if (const auto attr_it = matmul_op.attributes.find("accumulate_output");
            attr_it != matmul_op.attributes.end() && attr_it->second.type() == typeid(bool))
        {
            accumulate_output = std::any_cast<bool>(attr_it->second);
        }
        bool write_out_preact = false;
        if (const auto attr_it = matmul_op.attributes.find("write_out_preact");
            attr_it != matmul_op.attributes.end() && attr_it->second.type() == typeid(bool))
        {
            write_out_preact = std::any_cast<bool>(attr_it->second);
        }

        const auto &matmul_input_a = matmul_op.inputs[0];
        const auto &matmul_input_b = matmul_op.inputs[1];
        if (matmul_input_a == nullptr || matmul_input_b == nullptr)
        {
            continue;
        }

        const bool debug_gemm = std::getenv("FBAMTRAIN_DEBUG_GEMM") != nullptr;

        size_t act_bwd_idx = entries.size();
        if (const auto *matmul_consumers = get_consumers(matmul_output->id()); matmul_consumers != nullptr)
        {
            for (const size_t consumer_idx : *matmul_consumers)
            {
                if (!is_alive(consumer_idx))
                {
                    continue;
                }
                const auto &act_candidate = entries[consumer_idx];
                if (act_candidate.op_type != pi::tensorlib::OpType::ACT_FN_BWD || act_candidate.outputs.size() != 1 ||
                    act_candidate.inputs.size() != 2)
                {
                    continue;
                }
                const auto &act_out = act_candidate.outputs[0];
                const auto &upstream = act_candidate.inputs[1];
                if (act_out == nullptr || upstream == nullptr || !EntryConsumesTensor(act_candidate, matmul_output->id()))
                {
                    continue;
                }
                if (upstream->id() != matmul_output->id())
                {
                    continue;
                }
                const auto attr_it = act_candidate.attributes.find("activation_function");
                if (attr_it == act_candidate.attributes.end() ||
                    attr_it->second.type() != typeid(pi::tensorlib::ActivationFunction) ||
                    std::any_cast<pi::tensorlib::ActivationFunction>(attr_it->second) !=
                        pi::tensorlib::ActivationFunction::GELU)
                {
                    continue;
                }
                act_bwd_idx = consumer_idx;
                break;
            }
        }

        bool fused_matmul_gelu_bwd = false;
        if (act_bwd_idx != entries.size() && is_alive(act_bwd_idx))
        {
            const auto &act_op = entries[act_bwd_idx];
            const auto &act_out = act_op.outputs[0];
            const auto &pre_act = act_op.inputs[0];
            const auto &upstream = act_op.inputs[1];
            if (act_out != nullptr && pre_act != nullptr && upstream != nullptr)
            {
                size_t upstream_use_count = 0;
                if (const auto *upstream_consumers = get_consumers(upstream->id()); upstream_consumers != nullptr)
                {
                    for (const size_t consumer_idx : *upstream_consumers)
                    {
                        if (!is_alive(consumer_idx))
                        {
                            continue;
                        }
                        const auto &entry = entries[consumer_idx];
                        if (!EntryConsumesTensor(entry, upstream->id()) ||
                            entry.op_type == pi::tensorlib::OpType::DELETE_TENSOR ||
                            entry.op_type == pi::tensorlib::OpType::FILL_ZEROS)
                        {
                            continue;
                        }
                        ++upstream_use_count;
                    }
                }

                if (upstream_use_count == 1 && pre_act->shape().ndims() == 2 && upstream->shape().ndims() == 2 &&
                    pre_act->shape()[0] == upstream->shape()[0] && pre_act->shape()[1] == upstream->shape()[1] &&
                    matmul_input_a->shape().ndims() == 2 && matmul_input_b->shape().ndims() == 2)
                {
                    const auto pre_act_producer_idx = find_last_producer(pre_act->id());
                    if (!pre_act_producer_idx.has_value() || *pre_act_producer_idx <= i)
                    {
                        const auto &a_tensor = matmul_input_a;
                        const auto &b_tensor = matmul_input_b;
                        const auto &out_tensor = act_out;

                        auto is_transposed_contiguous =
                            [](const std::shared_ptr<pi::tensorlib::RealTensor> &tensor) -> bool
                        {
                            if (tensor->shape().ndims() != 2)
                            {
                                return false;
                            }
                            const auto &shape = tensor->shape();
                            const auto &strides = tensor->strides();
                            return strides[0] == 1 && strides[1] == shape[0];
                        };

                        const bool a_row_major = pi::tensorlib::shape_utils::IsRowMajorContiguous(a_tensor);
                        const bool b_row_major = pi::tensorlib::shape_utils::IsRowMajorContiguous(b_tensor);
                        const bool a_transposed = !a_row_major && is_transposed_contiguous(a_tensor);
                        const bool b_transposed = !b_row_major && is_transposed_contiguous(b_tensor);

                        MatmulTransposeState transpose_state = MatmulTransposeState::kNone;
                        if (a_transposed && b_transposed)
                        {
                            transpose_state = MatmulTransposeState::kTransposeAB;
                        }
                        else if (a_transposed)
                        {
                            transpose_state = MatmulTransposeState::kTransposeA;
                        }
                        else if (b_transposed)
                        {
                            transpose_state = MatmulTransposeState::kTransposeB;
                        }

                        const uint64_t m_dim = a_tensor->shape()[0];
                        const uint64_t k_dim = a_tensor->shape()[1];
                        const uint64_t n_dim = b_tensor->shape()[1];
                        const bool triton_aligned = IsTritonGemmAligned(
                            m_dim, n_dim, k_dim, a_tensor->strides()[0], a_tensor->strides()[1], b_tensor->strides()[0],
                            b_tensor->strides()[1], out_tensor->strides()[0], transpose_state);
                        const bool prefer_cutlass = (GetGemmBackendPreference() == GemmBackendPreference::Cutlass);
                        auto offset_aligned = [](const std::shared_ptr<pi::tensorlib::RealTensor> &tensor) -> bool
                        {
                            constexpr uint64_t alignment_elems = 8; // matches CUTLASS_GEMM_ALIGNMENT_{A,B}
                            return (tensor->storageOffset() % alignment_elems) == 0;
                        };
                        const bool aligned_a = offset_aligned(a_tensor);
                        const bool aligned_b = offset_aligned(b_tensor);
                        const bool aligned_pre = offset_aligned(pre_act);
                        const bool aligned_out = offset_aligned(out_tensor);
                        const bool pre_row_major = pi::tensorlib::shape_utils::IsRowMajorContiguous(pre_act);
                        const bool out_row_major = pi::tensorlib::shape_utils::IsRowMajorContiguous(out_tensor);
                        const bool a_transposed_ok = (transpose_state == MatmulTransposeState::kTransposeA ||
                                                      transpose_state == MatmulTransposeState::kTransposeAB) &&
                                                     is_transposed_contiguous(a_tensor);
                        const bool b_transposed_ok = (transpose_state == MatmulTransposeState::kTransposeB ||
                                                      transpose_state == MatmulTransposeState::kTransposeAB) &&
                                                     is_transposed_contiguous(b_tensor);

                        const bool a_layout_ok = (transpose_state == MatmulTransposeState::kTransposeA ||
                                                  transpose_state == MatmulTransposeState::kTransposeAB)
                                                     ? a_transposed_ok
                                                     : a_row_major;
                        const bool b_layout_ok = (transpose_state == MatmulTransposeState::kTransposeB ||
                                                  transpose_state == MatmulTransposeState::kTransposeAB)
                                                     ? b_transposed_ok
                                                     : b_row_major;

                        const bool row_major_ok = a_layout_ok && b_layout_ok && pre_row_major && out_row_major;
                        const GemmKernelSet *cutlass_kernels = &kCutlassMatmulGeluBwdKernels;
                        switch (transpose_state)
                        {
                            case MatmulTransposeState::kTransposeA:
                                cutlass_kernels = &kCutlassMatmulGeluBwdKernelsTA;
                                break;
                            case MatmulTransposeState::kTransposeB:
                                cutlass_kernels = &kCutlassMatmulGeluBwdKernelsTB;
                                break;
                            case MatmulTransposeState::kTransposeAB:
                                cutlass_kernels = &kCutlassMatmulGeluBwdKernelsTAB;
                                break;
                            case MatmulTransposeState::kNone:
                            default:
                                break;
                        }
                        const auto *cutlass_kernel =
                            SelectCutlassGemmKernel(*cutlass_kernels, dtype, use_fp16_accumulation);
                        bool allow_cutlass_gelu_bwd = false;
                        if (prefer_cutlass && cutlass_kernel != nullptr && row_major_ok && aligned_a && aligned_b &&
                            aligned_pre && aligned_out)
                        {
                            const uint64_t block_m = cutlass_kernel->meta.block_size_m;
                            const uint64_t block_n = cutlass_kernel->meta.block_size_n;
                            const uint64_t block_k = cutlass_kernel->meta.block_size_k;
                            auto stride_aligned = [](uint64_t stride) -> bool { return (stride % 8u) == 0u; };
                            const bool lda_ok = (transpose_state == MatmulTransposeState::kTransposeA ||
                                                 transpose_state == MatmulTransposeState::kTransposeAB)
                                                    ? true
                                                    : stride_aligned(a_tensor->strides()[0]);
                            const bool ldb_ok = (transpose_state == MatmulTransposeState::kTransposeB ||
                                                 transpose_state == MatmulTransposeState::kTransposeAB)
                                                    ? true
                                                    : stride_aligned(b_tensor->strides()[0]);
                            const bool ldd_ok = stride_aligned(out_tensor->strides()[0]);
                            const bool ldp_ok = stride_aligned(pre_act->strides()[0]);
                            if (block_m != 0 && block_n != 0 && block_k != 0 && m_dim != 0 && n_dim != 0 &&
                                k_dim != 0 && lda_ok && ldb_ok && ldd_ok && ldp_ok)
                            {
                                allow_cutlass_gelu_bwd = true;
                            }
                            else if (debug_gemm)
                            {
                                std::clog << "[Gemm] Skipping CUTLASS matmul_gelu_bwd due to alignment requirements "
                                          << "(aligned_a=" << aligned_a << ", aligned_b=" << aligned_b
                                          << ", aligned_pre=" << aligned_pre << ", aligned_out=" << aligned_out
                                          << ", lda=" << a_tensor->strides()[0] << ", ldb=" << b_tensor->strides()[0]
                                          << ", ldd=" << out_tensor->strides()[0] << ", ldp=" << pre_act->strides()[0]
                                          << ")\n";
                            }
                        }

                        const uint64_t act_out_id = act_out->id();
                        replace_tensor_inputs(act_out_id, matmul_output);
                        matmul_op.inputs.push_back(pre_act);
                        matmul_op.op_type = std::nullopt;
                        matmul_op.kernel_descriptor = CreateMatmulGeluBwdComputeKernelDescriptor(
                            dtype, use_fp16_accumulation, allow_cutlass_gelu_bwd, triton_aligned, accumulate_output,
                            transpose_state);

                        if (debug_gemm && matmul_op.kernel_descriptor.has_value())
                        {
                            std::clog << "[Gemm] Fusing matmul+gelu_bwd into "
                                      << matmul_op.kernel_descriptor->kernel_name << " for entry id " << matmul_op.id
                                      << '\n';
                        }

                        removed[act_bwd_idx] = 1u;
                        maybe_unused_create_tensors.insert(act_out_id);
                        fused_matmul_gelu_bwd = true;
                    }
                }
            }
        }

        if (fused_matmul_gelu_bwd)
        {
            continue;
        }

        size_t plus_idx = entries.size();
        size_t bias_input_idx = 1;
        if (const auto *matmul_consumers = get_consumers(matmul_output->id()); matmul_consumers != nullptr)
        {
            for (const size_t consumer_idx : *matmul_consumers)
            {
                if (!is_alive(consumer_idx))
                {
                    continue;
                }
                const auto &plus_candidate = entries[consumer_idx];
                if (plus_candidate.op_type != pi::tensorlib::OpType::PLUS || plus_candidate.inputs.size() != 2)
                {
                    continue;
                }
                const auto &p0 = plus_candidate.inputs[0];
                const auto &p1 = plus_candidate.inputs[1];
                if (p0 != nullptr && p0->id() == matmul_output->id())
                {
                    plus_idx = consumer_idx;
                    bias_input_idx = 1;
                    break;
                }
                if (p1 != nullptr && p1->id() == matmul_output->id())
                {
                    plus_idx = consumer_idx;
                    bias_input_idx = 0;
                    break;
                }
            }
        }
        if (plus_idx == entries.size() || !is_alive(plus_idx))
        {
            continue;
        }

        auto &plus_op = entries[plus_idx];
        const auto &bias_candidate = plus_op.inputs[bias_input_idx];
        if (bias_candidate == nullptr || bias_candidate->shape().ndims() != 1)
        {
            continue;
        }

        if (allow_fp32_out)
        {
            if (matmul_input_a->dtype() != pi::tensorlib::DataType::FLOAT16 ||
                matmul_input_b->dtype() != pi::tensorlib::DataType::FLOAT16)
            {
                continue;
            }
        }
        if (matmul_input_a->shape().ndims() != 2 || matmul_input_b->shape().ndims() != 2 ||
            matmul_output->shape().ndims() != 2)
        {
            continue;
        }

        const uint64_t m_dim = matmul_input_a->shape()[0];
        const uint64_t k_dim = matmul_input_a->shape()[1];
        const uint64_t n_dim = matmul_input_b->shape()[1];
        if (bias_candidate->shape()[0] != n_dim)
        {
            continue;
        }
        auto offset_aligned = [](const std::shared_ptr<pi::tensorlib::RealTensor> &tensor) -> bool
        {
            constexpr uint64_t alignment_elems = 8; // matches CUTLASS_GEMM_ALIGNMENT_{A,B}
            return (tensor->storageOffset() % alignment_elems) == 0;
        };

        const bool aligned_a = offset_aligned(matmul_input_a);
        const bool aligned_b = offset_aligned(matmul_input_b);
        const bool aligned_bias = offset_aligned(bias_candidate);
        const bool aligned_out = offset_aligned(matmul_output);

        auto supports_cutlass = [&](const kernel_bin_t<kernel_meta_gemm_t> *kernel_ptr, const char *label) -> bool
        {
            if (kernel_ptr == nullptr)
            {
                return false;
            }
            const uint64_t block_m = kernel_ptr->meta.block_size_m;
            const uint64_t block_n = kernel_ptr->meta.block_size_n;
            const uint64_t block_k = kernel_ptr->meta.block_size_k;
            if (block_m == 0 || block_n == 0 || block_k == 0)
            {
                return false;
            }
            if (m_dim == 0 || n_dim == 0 || k_dim == 0)
            {
                return false;
            }
            if (!aligned_a || !aligned_b || !aligned_bias || !aligned_out)
            {
                if (debug_gemm)
                {
                    std::clog << "[Gemm] Skipping CUTLASS " << label << " due to tensor offset alignment "
                              << "(aligned_a=" << aligned_a << ", aligned_b=" << aligned_b
                              << ", aligned_bias=" << aligned_bias << ", aligned_out=" << aligned_out << ")\n";
                }
                return false;
            }

            const uint64_t lda = matmul_input_a->strides()[0];
            const uint64_t ldb = matmul_input_b->strides()[0];
            const uint64_t ldd = matmul_output->strides()[0];

            auto stride_aligned = [](uint64_t stride) -> bool { return (stride % 8u) == 0u; };

            if (!stride_aligned(lda) || !stride_aligned(ldb) || !stride_aligned(ldd))
            {
                if (debug_gemm)
                {
                    std::clog << "[Gemm] Skipping CUTLASS " << label << " due to stride alignment "
                              << "(lda=" << lda << ", ldb=" << ldb << ", ldd=" << ldd << ")\n";
                }
                return false;
            }

            return true;
        };

        const bool prefer_cutlass = (GetGemmBackendPreference() == GemmBackendPreference::Cutlass);

        const auto &cutlass_gelu_set = write_out_preact ? kCutlassAddmmGeluPreactKernels : kCutlassAddmmGeluKernels;
        const char *cutlass_gelu_label = write_out_preact ? "addmm_gelu_preact" : "addmm_gelu";
        const bool allow_cutlass_gelu =
            prefer_cutlass &&
            supports_cutlass(SelectCutlassGemmKernel(cutlass_gelu_set, dtype, use_fp16_accumulation), cutlass_gelu_label);
        const bool allow_cutlass_addmm =
            prefer_cutlass &&
            supports_cutlass(SelectCutlassGemmKernel(kCutlassAddmmKernels, dtype, use_fp16_accumulation), "addmm");

        size_t act_idx = entries.size();
        if (!plus_op.outputs.empty() && plus_op.outputs[0] != nullptr)
        {
            if (const auto *plus_consumers = get_consumers(plus_op.outputs[0]->id()); plus_consumers != nullptr)
            {
                for (const size_t consumer_idx : *plus_consumers)
                {
                    if (!is_alive(consumer_idx))
                    {
                        continue;
                    }
                    const auto &act_op = entries[consumer_idx];
                    if (act_op.op_type != pi::tensorlib::OpType::ACT_FN || act_op.inputs.empty() ||
                        act_op.inputs[0] == nullptr || act_op.inputs[0]->id() != plus_op.outputs[0]->id())
                    {
                        continue;
                    }

                    const auto attr_it = act_op.attributes.find("activation_function");
                    if (attr_it != act_op.attributes.end() &&
                        attr_it->second.type() == typeid(pi::tensorlib::ActivationFunction) &&
                        std::any_cast<pi::tensorlib::ActivationFunction>(attr_it->second) ==
                            pi::tensorlib::ActivationFunction::GELU)
                    {
                        act_idx = consumer_idx;
                        break;
                    }
                }
            }
        }

        const pi::tensorlib::ExecutionEntry plus_entry = plus_op;
        std::optional<pi::tensorlib::ExecutionEntry> act_entry{};
        if (act_idx != entries.size() && is_alive(act_idx))
        {
            act_entry = entries[act_idx];
        }

        // Rewire the fused output to the post-plus / post-activation tensor so downstream
        // consumers read the fused result.
        const uint64_t matmul_flops = 2u * m_dim * n_dim * k_dim;
        const uint64_t bias_flops = m_dim * n_dim; // one add per output element

        if (act_entry.has_value() && !act_entry->outputs.empty() && act_entry->outputs[0] != nullptr)
        {
            const uint64_t old_output_id = matmul_output->id();
            if (matmul_output->retained() && !write_out_preact)
            {
                throw std::runtime_error("addmm_gelu fusion requires write_out_preact when the matmul output is retained");
            }
            const uint64_t output_stride = matmul_output->strides()[0];
            const auto pre_act_tensor = matmul_output;
            matmul_op.outputs[0] = act_entry->outputs[0];
            non_create_producers[matmul_op.outputs[0]->id()].push_back(i);
            if (write_out_preact && pre_act_tensor != nullptr)
            {
                matmul_op.outputs.push_back(pre_act_tensor);
            }
            matmul_op.op_type = std::nullopt;
            matmul_op.kernel_descriptor = CreateAddMMGeluComputeKernelDescriptor(
                dtype, use_fp16_accumulation, allow_cutlass_gelu, accumulate_output,
                IsTritonGemmAligned(m_dim, n_dim, k_dim, matmul_input_a->strides()[0], matmul_input_a->strides()[1],
                                    matmul_input_b->strides()[0], matmul_input_b->strides()[1], output_stride,
                                    MatmulTransposeState::kNone),
                write_out_preact);
            matmul_op.inputs.push_back(bias_candidate);
            const uint64_t gelu_flops = EstimateActivationFlops(pi::tensorlib::ActivationFunction::GELU, m_dim * n_dim);
            matmul_op.flop_estimate = matmul_flops + bias_flops + gelu_flops;

            if (debug_gemm)
            {
                if (matmul_op.kernel_descriptor.has_value())
                {
                    std::clog << "[Gemm] Fusing matmul+plus+gelu into " << matmul_op.kernel_descriptor->kernel_name
                              << " for entry id " << matmul_op.id << '\n';
                }
                else
                {
                    std::clog << "[Gemm] Fusing matmul+plus+gelu for entry id " << matmul_op.id << '\n';
                }
            }

            removed[plus_idx] = 1u;
            removed[act_idx] = 1u;
            move_create_before_requests.emplace_back(matmul_op.outputs[0]->id(), i);
            if (!write_out_preact)
            {
                maybe_unused_create_tensors.insert(old_output_id);
            }
        }
        else if (!plus_entry.outputs.empty() && plus_entry.outputs[0] != nullptr)
        {
            matmul_op.outputs[0] = plus_entry.outputs[0];
            non_create_producers[matmul_op.outputs[0]->id()].push_back(i);
            matmul_op.op_type = std::nullopt;
            matmul_op.kernel_descriptor = CreateAddMMComputeKernelDescriptor(
                dtype, use_fp16_accumulation, allow_cutlass_addmm, accumulate_output,
                IsTritonGemmAligned(m_dim, n_dim, k_dim, matmul_input_a->strides()[0], matmul_input_a->strides()[1],
                                    matmul_input_b->strides()[0], matmul_input_b->strides()[1], matmul_output->strides()[0],
                                    MatmulTransposeState::kNone));
            matmul_op.inputs.push_back(bias_candidate);
            matmul_op.flop_estimate = matmul_flops + bias_flops;

            if (debug_gemm)
            {
                std::clog << "[Gemm] Fusing matmul+plus into addmm for entry id " << matmul_op.id << '\n';
            }

            removed[plus_idx] = 1u;
        }
    }

    constexpr size_t kInvalidIndex = std::numeric_limits<size_t>::max();
    std::vector<size_t> old_to_new(entries.size(), kInvalidIndex);
    size_t next_index = 0;
    for (size_t i = 0; i < entries.size(); ++i)
    {
        if (!is_alive(i))
        {
            continue;
        }
        old_to_new[i] = next_index;
        ++next_index;
    }

    if (next_index != entries.size())
    {
        std::vector<pi::tensorlib::ExecutionEntry> compacted_entries{};
        compacted_entries.reserve(next_index);
        for (size_t i = 0; i < entries.size(); ++i)
        {
            if (is_alive(i))
            {
                compacted_entries.push_back(std::move(entries[i]));
            }
        }
        entries = std::move(compacted_entries);
    }

    for (const auto &[tensor_id, old_target_idx] : move_create_before_requests)
    {
        if (old_target_idx >= old_to_new.size())
        {
            continue;
        }
        const size_t target_idx = old_to_new[old_target_idx];
        if (target_idx == kInvalidIndex || target_idx >= entries.size())
        {
            continue;
        }
        MoveCreateTensorBefore(execution_plan, tensor_id, target_idx);
    }

    RemoveCreateTensorsIfUnused(execution_plan, maybe_unused_create_tensors);
}

void MatmulImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &matmul_op : execution_plan.entries)
    {
        if (matmul_op.op_type != pi::tensorlib::OpType::MATMUL)
        {
            continue;
        }

        if (matmul_op.inputs.size() < 2 || matmul_op.outputs.empty())
        {
            continue;
        }

        const auto output_dtype = matmul_op.outputs[0]->dtype();
        const bool allow_fp32_out = (output_dtype == pi::tensorlib::DataType::FLOAT32);
        if (!allow_fp32_out && !IsSupportedHalfType(output_dtype))
        {
            continue;
        }

        bool requested_fp16_accumulation = false;
        if (const auto attr_it = matmul_op.attributes.find("use_fp16_matmul_acc");
            attr_it != matmul_op.attributes.end() && attr_it->second.type() == typeid(bool))
        {
            requested_fp16_accumulation = std::any_cast<bool>(attr_it->second);
        }
        if (allow_fp32_out && requested_fp16_accumulation)
        {
            throw std::runtime_error("use_fp16_matmul_acc is not supported when matmul output dtype is FLOAT32");
        }
        if (output_dtype == pi::tensorlib::DataType::FLOAT16)
        {
            RequireFp16AccumulationSupported(requested_fp16_accumulation);
        }
        const bool use_fp16_accumulation =
            (output_dtype == pi::tensorlib::DataType::FLOAT16) ? requested_fp16_accumulation : false;
        bool accumulate_output = false;
        if (const auto attr_it = matmul_op.attributes.find("accumulate_output");
            attr_it != matmul_op.attributes.end() && attr_it->second.type() == typeid(bool))
        {
            accumulate_output = std::any_cast<bool>(attr_it->second);
        }

        const auto &matmul_input_a = matmul_op.inputs[0];
        const auto &matmul_input_b = matmul_op.inputs[1];
        const auto &matmul_output = matmul_op.outputs[0];
        if (matmul_input_a == nullptr || matmul_input_b == nullptr || matmul_output == nullptr)
        {
            continue;
        }
        const auto input_dtype = matmul_input_a->dtype();
        if (matmul_input_b->dtype() != input_dtype)
        {
            continue;
        }
        if (allow_fp32_out)
        {
            if (input_dtype != pi::tensorlib::DataType::FLOAT16 &&
                input_dtype != pi::tensorlib::DataType::BFLOAT16)
            {
                continue;
            }
        }
        else if (input_dtype != output_dtype)
        {
            continue;
        }
        if (matmul_input_a->shape().ndims() != 2 || matmul_input_b->shape().ndims() != 2 ||
            matmul_output->shape().ndims() != 2)
        {
            continue;
        }

        if (matmul_input_a->shape()[1] != matmul_input_b->shape()[0])
        {
            continue;
        }

        auto is_transposed_contiguous = [](const std::shared_ptr<pi::tensorlib::RealTensor> &tensor) -> bool
        {
            if (tensor->shape().ndims() != 2)
            {
                return false;
            }
            const auto &shape = tensor->shape();
            const auto &strides = tensor->strides();
            return strides[0] == 1 && strides[1] == shape[0];
        };

        const bool a_row_major = pi::tensorlib::shape_utils::IsRowMajorContiguous(matmul_input_a);
        const bool b_row_major = pi::tensorlib::shape_utils::IsRowMajorContiguous(matmul_input_b);
        const bool out_row_major = pi::tensorlib::shape_utils::IsRowMajorContiguous(matmul_output);
        const bool a_transposed = !a_row_major && is_transposed_contiguous(matmul_input_a);
        const bool b_transposed = !b_row_major && is_transposed_contiguous(matmul_input_b);

        if ((!a_row_major && !a_transposed) || (!b_row_major && !b_transposed) || !out_row_major)
        {
            continue;
        }

        const bool debug_gemm = std::getenv("FBAMTRAIN_DEBUG_GEMM") != nullptr;
        const uint64_t flops = 2u * matmul_output->shape()[0] * matmul_output->shape()[1] * matmul_input_a->shape()[1];

        const bool prefer_cutlass = (GetGemmBackendPreference() == GemmBackendPreference::Cutlass);

        const MatmulTransposeState transpose_state = GetMatmulTransposeState(a_transposed, b_transposed);
        const uint64_t input_alignment = (transpose_state == MatmulTransposeState::kNone) ? 8u : 2u;
        const uint64_t output_alignment = (output_dtype == pi::tensorlib::DataType::FLOAT32) ? 4u : 8u;
        auto offset_aligned = [input_alignment](const std::shared_ptr<pi::tensorlib::RealTensor> &tensor) -> bool
        { return (tensor->storageOffset() % input_alignment) == 0; };
        auto output_offset_aligned =
            [output_alignment](const std::shared_ptr<pi::tensorlib::RealTensor> &tensor) -> bool
        { return (tensor->storageOffset() % output_alignment) == 0; };
        const auto *cutlass_kernel = [&]() -> const kernel_bin_t<kernel_meta_gemm_t> *
        {
            if (output_dtype == pi::tensorlib::DataType::FLOAT32)
            {
                switch (transpose_state)
                {
                    case MatmulTransposeState::kTransposeA:
                        return SelectFp32OutKernel(kCutlassMatmulFp32OutKernelsTA, input_dtype,
                                                   use_fp16_accumulation);
                    case MatmulTransposeState::kTransposeB:
                        return SelectFp32OutKernel(kCutlassMatmulFp32OutKernelsTB, input_dtype,
                                                   use_fp16_accumulation);
                    case MatmulTransposeState::kTransposeAB:
                        return SelectFp32OutKernel(kCutlassMatmulFp32OutKernelsTAB, input_dtype,
                                                   use_fp16_accumulation);
                    case MatmulTransposeState::kNone:
                    default:
                        return SelectFp32OutKernel(kCutlassMatmulFp32OutKernels, input_dtype, use_fp16_accumulation);
                }
            }
            switch (transpose_state)
            {
                case MatmulTransposeState::kTransposeA:
                    return SelectCutlassGemmKernel(kCutlassMatmulKernelsTA, output_dtype, use_fp16_accumulation);
                case MatmulTransposeState::kTransposeB:
                    return SelectCutlassGemmKernel(kCutlassMatmulKernelsTB, output_dtype, use_fp16_accumulation);
                case MatmulTransposeState::kTransposeAB:
                    return SelectCutlassGemmKernel(kCutlassMatmulKernelsTAB, output_dtype, use_fp16_accumulation);
                case MatmulTransposeState::kNone:
                default:
                    return SelectCutlassGemmKernel(kCutlassMatmulKernels, output_dtype, use_fp16_accumulation);
            }
        }();
        bool allow_cutlass = prefer_cutlass && cutlass_kernel != nullptr;

        const bool aligned_a = offset_aligned(matmul_input_a);
        const bool aligned_b = offset_aligned(matmul_input_b);
        const bool aligned_out = output_offset_aligned(matmul_output);

        auto stride_aligned_inputs = [input_alignment](const uint64_t stride) -> bool
        { return stride % input_alignment == 0u; };
        auto stride_aligned_output = [output_alignment](const uint64_t stride) -> bool
        { return stride % output_alignment == 0u; };

        const uint64_t lda = (transpose_state == MatmulTransposeState::kTransposeA ||
                              transpose_state == MatmulTransposeState::kTransposeAB)
                                 ? matmul_input_a->strides()[1]
                                 : matmul_input_a->strides()[0];
        const uint64_t ldb = (transpose_state == MatmulTransposeState::kTransposeB ||
                              transpose_state == MatmulTransposeState::kTransposeAB)
                                 ? matmul_input_b->strides()[1]
                                 : matmul_input_b->strides()[0];
        const uint64_t ldd = matmul_output->strides()[0];
        const uint64_t stride_am = matmul_input_a->strides()[0];
        const uint64_t stride_ak = matmul_input_a->strides()[1];
        const uint64_t stride_bk = matmul_input_b->strides()[0];
        const uint64_t stride_bn = matmul_input_b->strides()[1];
        const uint64_t stride_dm = matmul_output->strides()[0];
        const bool triton_aligned =
            IsTritonGemmAligned(matmul_input_a->shape()[0], matmul_input_b->shape()[1], matmul_input_a->shape()[1],
                                stride_am, stride_ak, stride_bk, stride_bn, stride_dm, transpose_state);

        if (allow_cutlass)
        {
            const uint64_t block_m = cutlass_kernel->meta.block_size_m;
            const uint64_t block_n = cutlass_kernel->meta.block_size_n;
            const uint64_t block_k = cutlass_kernel->meta.block_size_k;
            const uint64_t m_dim = matmul_input_a->shape()[0];
            const uint64_t n_dim = matmul_input_b->shape()[1];
            const uint64_t k_dim = matmul_input_a->shape()[1];

            if (block_m == 0 || block_n == 0 || block_k == 0)
            {
                if (debug_gemm)
                {
                    std::clog << "[Gemm] Skipping CUTLASS matmul due to tile size requirements "
                              << "(m=" << m_dim << ", n=" << n_dim << ", k=" << k_dim << ", block_m=" << block_m
                              << ", block_n=" << block_n << ", block_k=" << block_k << ")\n";
                }
                allow_cutlass = false;
            }
        }

        if (allow_cutlass)
        {
            if (!aligned_a || !aligned_b || !aligned_out || !stride_aligned_inputs(lda) ||
                !stride_aligned_inputs(ldb) || !stride_aligned_output(ldd))
            {
                if (debug_gemm)
                {
                    std::clog << "[Gemm] Skipping CUTLASS matmul due to alignment requirements "
                              << "(aligned_a=" << aligned_a << ", aligned_b=" << aligned_b
                              << ", aligned_out=" << aligned_out << ", lda=" << lda << ", ldb=" << ldb
                              << ", ldd=" << ldd << ")\n";
                }
                allow_cutlass = false;
            }
        }

        matmul_op.op_type = std::nullopt;
        matmul_op.kernel_descriptor = CreateMatmulComputeKernelDescriptor(
            output_dtype, input_dtype, use_fp16_accumulation, allow_cutlass, accumulate_output, triton_aligned,
            transpose_state);
        matmul_op.flop_estimate = flops;
    }
}
