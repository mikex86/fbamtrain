#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <kernels/kernel_binaries.h>

namespace fbamtrain::kernel_gen::nv
{
enum class GeneratedConv2dDType
{
    BF16,
    FP16,
};

enum class GeneratedConv2dOperation
{
    Fprop,
    Dgrad,
    Wgrad,
};

struct GeneratedConv2dRequest
{
    GeneratedConv2dOperation operation{};
    GeneratedConv2dDType dtype{};
    int32_t n{};
    int32_t h{};
    int32_t w{};
    int32_t c{};
    int32_t k{};
    int32_t r{};
    int32_t s{};
    int32_t pad_h{};
    int32_t pad_w{};
    int32_t stride_h{};
    int32_t stride_w{};
    int32_t dilation_h{};
    int32_t dilation_w{};
    int32_t groups{1};
    void *ptr_a{};
    void *ptr_b{};
    void *ptr_c{};
    void *ptr_d{};
    float alpha{1.0f};
    float beta{0.0f};
    int device_ordinal{};
};

struct GeneratedConv2dLaunchPlan
{
    std::vector<uint8_t> params;
    uint32_t grid_dim_x{1};
    uint32_t grid_dim_y{1};
    uint32_t grid_dim_z{1};
    uint32_t block_dim_x{1};
    uint32_t block_dim_y{1};
    uint32_t block_dim_z{1};
    uint32_t shared_mem_bytes{0};
    uint32_t cluster_dim_x{1};
    uint32_t cluster_dim_y{1};
    uint32_t cluster_dim_z{1};
};

bool PrepareGeneratedConv2dLaunch(const kernel_meta_conv2d_cutlass_t &meta,
                                  const GeneratedConv2dRequest &request,
                                  GeneratedConv2dLaunchPlan &plan,
                                  std::string &error);
} // namespace fbamtrain::kernel_gen::nv
