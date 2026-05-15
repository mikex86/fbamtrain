#include "passes.h"

#include "shape_utils.h"
#include "passes/conv2d_bin_map.h"
#include "passes/pass_utils.h"

#include <kernels/kernel_binaries.h>
#include <nv/generated_conv2d_launch.h>
#include <utils.h>

#include <any>
#include <array>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

enum class CutlassConv2dImplementation
{
    Cutlass2,
    Cutlass3,
};

struct Conv2dKernelKey
{
    uint32_t in_channels;
    uint32_t out_channels;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t dilation_h;
    uint32_t dilation_w;

    bool operator==(const Conv2dKernelKey &) const = default;
};

struct Conv2dPaddedKernelKey
{
    Conv2dKernelKey kernel;
    std::array<uint32_t, 2> padding;

    bool operator==(const Conv2dPaddedKernelKey &) const = default;
};

struct Conv2dBinLookupKey
{
    Conv2dPaddedKernelKey conv;
    uint32_t batch;
    uint32_t input_h;
    uint32_t input_w;

    bool operator==(const Conv2dBinLookupKey &) const = default;
};

struct Conv2dKernelEntry
{
    Conv2dKernelKey key;
    const kernel_bin_t<kernel_meta_conv2d_t> *bf16_kernel;
    const kernel_bin_t<kernel_meta_conv2d_t> *fp16_kernel;
};


static constexpr std::array<Conv2dKernelEntry, 7> kConv2dKernels{{
    Conv2dKernelEntry{Conv2dKernelKey{3, 8, 3, 3, 1, 1, 1, 1}, &kconv2d_bf16_ic3_oc8_k3,
                      &kconv2d_fp16_ic3_oc8_k3},
    Conv2dKernelEntry{Conv2dKernelKey{3, 8, 3, 3, 1, 1, 2, 2}, &kconv2d_bf16_ic3_oc8_k3_dil2,
                      &kconv2d_fp16_ic3_oc8_k3_dil2},
    Conv2dKernelEntry{Conv2dKernelKey{32, 64, 3, 3, 1, 1, 1, 1}, &kconv2d_bf16_ic32_oc64_k3,
                      &kconv2d_fp16_ic32_oc64_k3},
    Conv2dKernelEntry{Conv2dKernelKey{768, 768, 3, 3, 1, 1, 2, 2}, &kconv2d_bf16_ic768_oc768_k3_dil2,
                      &kconv2d_fp16_ic768_oc768_k3_dil2},
    Conv2dKernelEntry{Conv2dKernelKey{768, 1536, 3, 3, 1, 1, 1, 1}, &kconv2d_bf16_ic768_oc1536_k3,
                      &kconv2d_fp16_ic768_oc1536_k3},
    Conv2dKernelEntry{Conv2dKernelKey{1024, 1024, 3, 3, 1, 1, 2, 2}, &kconv2d_bf16_ic1024_oc1024_k3_dil2,
                      &kconv2d_fp16_ic1024_oc1024_k3_dil2},
    Conv2dKernelEntry{Conv2dKernelKey{1024, 2048, 3, 3, 1, 1, 1, 1}, &kconv2d_bf16_ic1024_oc2048_k3,
                      &kconv2d_fp16_ic1024_oc2048_k3},
}};

static constexpr std::array<Conv2dKernelEntry, 7> kConv2dDgradKernels{{
    Conv2dKernelEntry{Conv2dKernelKey{3, 8, 3, 3, 1, 1, 1, 1}, &kconv2d_dgrad_bf16_ic3_oc8_k3,
                      &kconv2d_dgrad_fp16_ic3_oc8_k3},
    Conv2dKernelEntry{Conv2dKernelKey{3, 8, 3, 3, 1, 1, 2, 2}, &kconv2d_dgrad_bf16_ic3_oc8_k3_dil2,
                      &kconv2d_dgrad_fp16_ic3_oc8_k3_dil2},
    Conv2dKernelEntry{Conv2dKernelKey{32, 64, 3, 3, 1, 1, 1, 1}, &kconv2d_dgrad_bf16_ic32_oc64_k3,
                      &kconv2d_dgrad_fp16_ic32_oc64_k3},
    Conv2dKernelEntry{Conv2dKernelKey{768, 768, 3, 3, 1, 1, 2, 2}, &kconv2d_dgrad_bf16_ic768_oc768_k3_dil2,
                      &kconv2d_dgrad_fp16_ic768_oc768_k3_dil2},
    Conv2dKernelEntry{Conv2dKernelKey{768, 1536, 3, 3, 1, 1, 1, 1}, &kconv2d_dgrad_bf16_ic768_oc1536_k3,
                      &kconv2d_dgrad_fp16_ic768_oc1536_k3},
    Conv2dKernelEntry{Conv2dKernelKey{1024, 1024, 3, 3, 1, 1, 2, 2}, &kconv2d_dgrad_bf16_ic1024_oc1024_k3_dil2,
                      &kconv2d_dgrad_fp16_ic1024_oc1024_k3_dil2},
    Conv2dKernelEntry{Conv2dKernelKey{1024, 2048, 3, 3, 1, 1, 1, 1}, &kconv2d_dgrad_bf16_ic1024_oc2048_k3,
                      &kconv2d_dgrad_fp16_ic1024_oc2048_k3},
}};

static constexpr std::array<Conv2dKernelEntry, 7> kConv2dWgradKernels{{
    Conv2dKernelEntry{Conv2dKernelKey{3, 8, 3, 3, 1, 1, 1, 1}, &kconv2d_wgrad_bf16_ic3_oc8_k3,
                      &kconv2d_wgrad_fp16_ic3_oc8_k3},
    Conv2dKernelEntry{Conv2dKernelKey{3, 8, 3, 3, 1, 1, 2, 2}, &kconv2d_wgrad_bf16_ic3_oc8_k3_dil2,
                      &kconv2d_wgrad_fp16_ic3_oc8_k3_dil2},
    Conv2dKernelEntry{Conv2dKernelKey{32, 64, 3, 3, 1, 1, 1, 1}, &kconv2d_wgrad_bf16_ic32_oc64_k3,
                      &kconv2d_wgrad_fp16_ic32_oc64_k3},
    Conv2dKernelEntry{Conv2dKernelKey{768, 768, 3, 3, 1, 1, 2, 2}, &kconv2d_wgrad_bf16_ic768_oc768_k3_dil2,
                      &kconv2d_wgrad_fp16_ic768_oc768_k3_dil2},
    Conv2dKernelEntry{Conv2dKernelKey{768, 1536, 3, 3, 1, 1, 1, 1}, &kconv2d_wgrad_bf16_ic768_oc1536_k3,
                      &kconv2d_wgrad_fp16_ic768_oc1536_k3},
    Conv2dKernelEntry{Conv2dKernelKey{1024, 1024, 3, 3, 1, 1, 2, 2}, &kconv2d_wgrad_bf16_ic1024_oc1024_k3_dil2,
                      &kconv2d_wgrad_fp16_ic1024_oc1024_k3_dil2},
    Conv2dKernelEntry{Conv2dKernelKey{1024, 2048, 3, 3, 1, 1, 1, 1}, &kconv2d_wgrad_bf16_ic1024_oc2048_k3,
                      &kconv2d_wgrad_fp16_ic1024_oc2048_k3},
}};

enum class Conv2dKernelBackend
{
    Triton,
    Cutlass,
};

enum class Conv2dBackendPreference
{
    Cutlass,
    Triton,
};

enum class CutlassConv2dImplementationPreference
{
    Auto,
    Cutlass2,
    Cutlass3,
};

static Conv2dBackendPreference GetConv2dBackendPreference()
{
    const auto env = GetEnvValue("FBAMTRAIN_PREFER_CONV2D_BACKEND");
#if !PI_TENSORLIB_ENABLE_CUDA
    (void)env;
    return Conv2dBackendPreference::Triton;
#else
    if (!env.has_value())
    {
        return Conv2dBackendPreference::Cutlass;
    }
    if (*env == "cutlass")
    {
        return Conv2dBackendPreference::Cutlass;
    }
    if (*env == "triton")
    {
        return Conv2dBackendPreference::Triton;
    }
    throw std::runtime_error(
        "FBAMTRAIN_PREFER_CONV2D_BACKEND must be one of: cutlass, triton");
#endif
}

static CutlassConv2dImplementationPreference GetConv2dWgradCutlassImplementationPreference()
{
    const auto env = GetEnvValue("FBAMTRAIN_CONV2D_WGRAD_CUTLASS_IMPL");
    if (!env.has_value() || *env == "auto")
    {
        return CutlassConv2dImplementationPreference::Auto;
    }
    if (*env == "cutlass2")
    {
        return CutlassConv2dImplementationPreference::Cutlass2;
    }
    if (*env == "cutlass3")
    {
        return CutlassConv2dImplementationPreference::Cutlass3;
    }
    throw std::runtime_error(
        "FBAMTRAIN_CONV2D_WGRAD_CUTLASS_IMPL must be one of: auto, cutlass2, cutlass3");
}

struct SelectedConv2dKernel
{
    Conv2dKernelBackend backend{Conv2dKernelBackend::Triton};
    CutlassConv2dImplementation cutlass_implementation{CutlassConv2dImplementation::Cutlass2};
    const kernel_bin_t<kernel_meta_conv2d_t> *triton_kernel{nullptr};
    const kernel_bin_t<kernel_meta_conv2d_cutlass_t> *cutlass_kernel{nullptr};
};

struct Conv2dAttributes
{
    std::array<uint32_t, 2> stride{};
    std::array<uint32_t, 2> padding{};
    std::array<uint32_t, 2> dilation{};
    bool accumulate_output{false};
};

static Conv2dAttributes ParseConv2dAttributes(const pi::tensorlib::ExecutionEntry &entry)
{
    const auto stride_it = entry.attributes.find("stride");
    const auto padding_it = entry.attributes.find("padding");
    const auto dilation_it = entry.attributes.find("dilation");
    const auto accumulate_it = entry.attributes.find("accumulate_output");
    if (stride_it == entry.attributes.end() || padding_it == entry.attributes.end())
    {
        throw std::runtime_error("conv2d operation missing stride or padding attributes");
    }

    const auto stride = std::any_cast<std::array<uint32_t, 2>>(stride_it->second);
    const auto padding = std::any_cast<std::array<uint32_t, 2>>(padding_it->second);
    const auto dilation = dilation_it != entry.attributes.end()
                              ? std::any_cast<std::array<uint32_t, 2>>(dilation_it->second)
                              : std::array<uint32_t, 2>{1, 1};
    const bool accumulate_output =
        accumulate_it != entry.attributes.end() ? std::any_cast<bool>(accumulate_it->second) : false;
    return Conv2dAttributes{stride, padding, dilation, accumulate_output};
}

static void HashCombine(std::size_t &seed, const uint32_t value)
{
    seed ^= std::hash<uint32_t>{}(value) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
}

struct Conv2dKernelKeyHash
{
    std::size_t operator()(const Conv2dKernelKey &key) const
    {
        std::size_t seed = 0;
        HashCombine(seed, key.in_channels);
        HashCombine(seed, key.out_channels);
        HashCombine(seed, key.kernel_h);
        HashCombine(seed, key.kernel_w);
        HashCombine(seed, key.stride_h);
        HashCombine(seed, key.stride_w);
        HashCombine(seed, key.dilation_h);
        HashCombine(seed, key.dilation_w);
        return seed;
    }
};

struct Conv2dPaddedKernelKeyHash
{
    std::size_t operator()(const Conv2dPaddedKernelKey &key) const
    {
        std::size_t seed = Conv2dKernelKeyHash{}(key.kernel);
        HashCombine(seed, key.padding[0]);
        HashCombine(seed, key.padding[1]);
        return seed;
    }
};

struct Conv2dBinLookupKeyHash
{
    std::size_t operator()(const Conv2dBinLookupKey &key) const
    {
        std::size_t seed = Conv2dPaddedKernelKeyHash{}(key.conv);
        HashCombine(seed, key.batch);
        HashCombine(seed, key.input_h);
        HashCombine(seed, key.input_w);
        return seed;
    }
};

static Conv2dKernelKey MakeConv2dKernelKey(const uint32_t in_channels, const uint32_t out_channels,
                                           const uint32_t kernel_h, const uint32_t kernel_w,
                                           const uint32_t stride_h, const uint32_t stride_w,
                                           const uint32_t dilation_h, const uint32_t dilation_w)
{
    return Conv2dKernelKey{in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, dilation_h,
                           dilation_w};
}

static Conv2dKernelKey MakeConv2dKernelKey(const Conv2dBinKernelKey &key)
{
    return MakeConv2dKernelKey(key.in_channels, key.out_channels, key.kernel_h, key.kernel_w, key.stride_h,
                               key.stride_w, key.dilation_h, key.dilation_w);
}

static Conv2dPaddedKernelKey MakeConv2dPaddedKernelKey(const Conv2dKernelKey &key,
                                                       const std::array<uint32_t, 2> &padding)
{
    return Conv2dPaddedKernelKey{key, padding};
}

static Conv2dBinLookupKey MakeConv2dBinLookupKey(const Conv2dKernelKey &key,
                                                 const std::array<uint32_t, 2> &padding,
                                                 const uint32_t batch, const uint32_t input_h,
                                                 const uint32_t input_w)
{
    return Conv2dBinLookupKey{MakeConv2dPaddedKernelKey(key, padding), batch, input_h, input_w};
}

using CutlassConv2dBinKernelMap =
    std::unordered_map<Conv2dBinLookupKey, const CutlassConv2dBinKernelEntry *, Conv2dBinLookupKeyHash>;
using CutlassConv2dBwdBinKernelMap =
    std::unordered_map<Conv2dBinLookupKey, const CutlassConv2dBwdBinKernelEntry *, Conv2dBinLookupKeyHash>;

struct CutlassConv2dKernelEntry
{
    Conv2dKernelKey key;
    std::array<uint32_t, 2> padding;
    const kernel_bin_t<kernel_meta_conv2d_cutlass_t> *bf16_kernel;
    const kernel_bin_t<kernel_meta_conv2d_cutlass_t> *fp16_kernel;
    const kernel_bin_t<kernel_meta_conv2d_cutlass_t> *fp16_acc_fp16_kernel;
};

#if PI_TENSORLIB_ENABLE_CUDA
static constexpr std::array<CutlassConv2dKernelEntry, 5> kCutlassConv2dKernelsNv{{
    CutlassConv2dKernelEntry{
        Conv2dKernelKey{32, 64, 3, 3, 1, 1, 1, 1}, {1, 1}, &kconv2d_cutlass_bf16_stride1_pad1_dil1_ic32_oc64,
        &kconv2d_cutlass_fp16_stride1_pad1_dil1_ic32_oc64, &kconv2d_cutlass_fp16_acc_fp16_stride1_pad1_dil1_ic32_oc64},
    CutlassConv2dKernelEntry{
        Conv2dKernelKey{768, 1536, 3, 3, 1, 1, 1, 1}, {1, 1}, &kconv2d_cutlass_bf16_stride1_pad1_dil1_ic768_oc1536,
        &kconv2d_cutlass_fp16_stride1_pad1_dil1_ic768_oc1536,
        &kconv2d_cutlass_fp16_acc_fp16_stride1_pad1_dil1_ic768_oc1536},
    CutlassConv2dKernelEntry{
        Conv2dKernelKey{1024, 2048, 3, 3, 1, 1, 1, 1}, {1, 1}, &kconv2d_cutlass_bf16_stride1_pad1_dil1_ic1024_oc2048,
        &kconv2d_cutlass_fp16_stride1_pad1_dil1_ic1024_oc2048,
        &kconv2d_cutlass_fp16_acc_fp16_stride1_pad1_dil1_ic1024_oc2048},
    CutlassConv2dKernelEntry{
        Conv2dKernelKey{768, 768, 3, 3, 1, 1, 2, 2}, {2, 2}, &kconv2d_cutlass_bf16_stride1_pad2_dil2_ic768_oc768,
        &kconv2d_cutlass_fp16_stride1_pad2_dil2_ic768_oc768,
        &kconv2d_cutlass_fp16_acc_fp16_stride1_pad2_dil2_ic768_oc768},
    CutlassConv2dKernelEntry{
        Conv2dKernelKey{1024, 1024, 3, 3, 1, 1, 2, 2}, {2, 2}, &kconv2d_cutlass_bf16_stride1_pad2_dil2_ic1024_oc1024,
        &kconv2d_cutlass_fp16_stride1_pad2_dil2_ic1024_oc1024,
        &kconv2d_cutlass_fp16_acc_fp16_stride1_pad2_dil2_ic1024_oc1024},
}};
#else
static constexpr std::array<CutlassConv2dKernelEntry, 0> kCutlassConv2dKernelsNv{};
#endif

struct CutlassConv2dBwdKernelEntry
{
    Conv2dKernelKey key;
    std::array<uint32_t, 2> padding;
    const kernel_bin_t<kernel_meta_conv2d_cutlass_t> *bf16_kernel;
    const kernel_bin_t<kernel_meta_conv2d_cutlass_t> *fp16_kernel;
};

#if PI_TENSORLIB_ENABLE_CUDA
static constexpr std::array<CutlassConv2dBwdKernelEntry, 5> kCutlassConv2dDgradKernelsNv{{
    CutlassConv2dBwdKernelEntry{
        Conv2dKernelKey{32, 64, 3, 3, 1, 1, 1, 1}, {1, 1}, &kconv2d_dgrad_cutlass_bf16_stride1_pad1_dil1_ic32_oc64,
        &kconv2d_dgrad_cutlass_fp16_stride1_pad1_dil1_ic32_oc64},
    CutlassConv2dBwdKernelEntry{
        Conv2dKernelKey{768, 1536, 3, 3, 1, 1, 1, 1}, {1, 1},
        &kconv2d_dgrad_cutlass_bf16_stride1_pad1_dil1_ic768_oc1536,
        &kconv2d_dgrad_cutlass_fp16_stride1_pad1_dil1_ic768_oc1536},
    CutlassConv2dBwdKernelEntry{
        Conv2dKernelKey{1024, 2048, 3, 3, 1, 1, 1, 1}, {1, 1},
        &kconv2d_dgrad_cutlass_bf16_stride1_pad1_dil1_ic1024_oc2048,
        &kconv2d_dgrad_cutlass_fp16_stride1_pad1_dil1_ic1024_oc2048},
    CutlassConv2dBwdKernelEntry{
        Conv2dKernelKey{768, 768, 3, 3, 1, 1, 2, 2}, {2, 2},
        &kconv2d_dgrad_cutlass_bf16_stride1_pad2_dil2_ic768_oc768,
        &kconv2d_dgrad_cutlass_fp16_stride1_pad2_dil2_ic768_oc768},
    CutlassConv2dBwdKernelEntry{
        Conv2dKernelKey{1024, 1024, 3, 3, 1, 1, 2, 2}, {2, 2},
        &kconv2d_dgrad_cutlass_bf16_stride1_pad2_dil2_ic1024_oc1024,
        &kconv2d_dgrad_cutlass_fp16_stride1_pad2_dil2_ic1024_oc1024},
}};
#else
static constexpr std::array<CutlassConv2dBwdKernelEntry, 0> kCutlassConv2dDgradKernelsNv{};
#endif

#if PI_TENSORLIB_ENABLE_CUDA
static constexpr std::array<CutlassConv2dBwdKernelEntry, 5> kCutlassConv2dWgradKernelsNv{{
    CutlassConv2dBwdKernelEntry{
        Conv2dKernelKey{32, 64, 3, 3, 1, 1, 1, 1}, {1, 1}, &kconv2d_wgrad_cutlass_bf16_stride1_pad1_dil1_ic32_oc64,
        &kconv2d_wgrad_cutlass_fp16_stride1_pad1_dil1_ic32_oc64},
    CutlassConv2dBwdKernelEntry{
        Conv2dKernelKey{768, 1536, 3, 3, 1, 1, 1, 1}, {1, 1},
        &kconv2d_wgrad_cutlass_bf16_stride1_pad1_dil1_ic768_oc1536,
        &kconv2d_wgrad_cutlass_fp16_stride1_pad1_dil1_ic768_oc1536},
    CutlassConv2dBwdKernelEntry{
        Conv2dKernelKey{1024, 2048, 3, 3, 1, 1, 1, 1}, {1, 1},
        &kconv2d_wgrad_cutlass_bf16_stride1_pad1_dil1_ic1024_oc2048,
        &kconv2d_wgrad_cutlass_fp16_stride1_pad1_dil1_ic1024_oc2048},
    CutlassConv2dBwdKernelEntry{
        Conv2dKernelKey{768, 768, 3, 3, 1, 1, 2, 2}, {2, 2},
        &kconv2d_wgrad_cutlass_bf16_stride1_pad2_dil2_ic768_oc768,
        &kconv2d_wgrad_cutlass_fp16_stride1_pad2_dil2_ic768_oc768},
    CutlassConv2dBwdKernelEntry{
        Conv2dKernelKey{1024, 1024, 3, 3, 1, 1, 2, 2}, {2, 2},
        &kconv2d_wgrad_cutlass_bf16_stride1_pad2_dil2_ic1024_oc1024,
        &kconv2d_wgrad_cutlass_fp16_stride1_pad2_dil2_ic1024_oc1024},
}};
#else
static constexpr std::array<CutlassConv2dBwdKernelEntry, 0> kCutlassConv2dWgradKernelsNv{};
#endif

static CutlassConv2dImplementation ToCutlassConv2dImplementation(
    const CutlassConv2dBinImplementation implementation)
{
    switch (implementation)
    {
        case CutlassConv2dBinImplementation::Cutlass2:
            return CutlassConv2dImplementation::Cutlass2;
        case CutlassConv2dBinImplementation::Cutlass3:
            return CutlassConv2dImplementation::Cutlass3;
    }
    return CutlassConv2dImplementation::Cutlass2;
}

static const char *ToString(const CutlassConv2dImplementation implementation)
{
    switch (implementation)
    {
        case CutlassConv2dImplementation::Cutlass2:
            return "CUTLASS2";
        case CutlassConv2dImplementation::Cutlass3:
            return "CUTLASS3";
    }
    return "unknown CUTLASS";
}

static bool AllowsCutlassImplementation(const CutlassConv2dImplementationPreference preference,
                                        const CutlassConv2dImplementation implementation)
{
    switch (preference)
    {
        case CutlassConv2dImplementationPreference::Auto:
            return true;
        case CutlassConv2dImplementationPreference::Cutlass2:
            return implementation == CutlassConv2dImplementation::Cutlass2;
        case CutlassConv2dImplementationPreference::Cutlass3:
            return implementation == CutlassConv2dImplementation::Cutlass3;
    }
    return false;
}

template <typename Entry, std::size_t N>
static std::unordered_map<Conv2dKernelKey, const Entry *, Conv2dKernelKeyHash> BuildConv2dKernelMap(
    const std::array<Entry, N> &entries)
{
    std::unordered_map<Conv2dKernelKey, const Entry *, Conv2dKernelKeyHash> map;
    map.reserve(entries.size());
    for (const auto &entry : entries)
    {
        map.emplace(entry.key, &entry);
    }
    return map;
}

template <typename Entry, std::size_t N>
static std::unordered_map<Conv2dPaddedKernelKey, const Entry *, Conv2dPaddedKernelKeyHash>
BuildCutlassConv2dKernelMap(const std::array<Entry, N> &entries)
{
    std::unordered_map<Conv2dPaddedKernelKey, const Entry *, Conv2dPaddedKernelKeyHash> map;
    map.reserve(entries.size());
    for (const auto &entry : entries)
    {
        map.emplace(MakeConv2dPaddedKernelKey(entry.key, entry.padding), &entry);
    }
    return map;
}

template <typename Entry, std::size_t N>
static std::unordered_map<Conv2dBinLookupKey, const Entry *, Conv2dBinLookupKeyHash> BuildConv2dBinKernelMap(
    const std::array<Entry, N> &entries)
{
    std::unordered_map<Conv2dBinLookupKey, const Entry *, Conv2dBinLookupKeyHash> map;
    map.reserve(entries.size());
    for (const auto &entry : entries)
    {
        map.emplace(MakeConv2dBinLookupKey(MakeConv2dKernelKey(entry.key), entry.padding, entry.bin.batch,
                                           entry.bin.height, entry.bin.width),
                    &entry);
    }
    return map;
}

template <const auto &Entries>
static const auto &Conv2dKernelMap()
{
    static const auto map = BuildConv2dKernelMap(Entries);
    return map;
}

template <const auto &Entries>
static const auto &CutlassConv2dKernelMap()
{
    static const auto map = BuildCutlassConv2dKernelMap(Entries);
    return map;
}

template <const auto &Entries>
static const auto &Conv2dBinKernelMapFor()
{
    static const auto map = BuildConv2dBinKernelMap(Entries);
    return map;
}

static bool DebugConv2d()
{
    return std::getenv("FBAMTRAIN_DEBUG_CONV2D") != nullptr;
}

static const kernel_bin_t<kernel_meta_conv2d_t> *SelectTritonKernel(
    const pi::tensorlib::DataType dtype, const Conv2dKernelEntry *entry)
{
    if (entry == nullptr)
    {
        return nullptr;
    }
    return dtype == pi::tensorlib::DataType::BFLOAT16 ? entry->bf16_kernel : entry->fp16_kernel;
}

static const kernel_bin_t<kernel_meta_conv2d_cutlass_t> *SelectCutlassKernel(
    const pi::tensorlib::DataType dtype, const CutlassConv2dBwdKernelEntry *entry)
{
    if (entry == nullptr)
    {
        return nullptr;
    }
    return dtype == pi::tensorlib::DataType::BFLOAT16 ? entry->bf16_kernel : entry->fp16_kernel;
}

static const kernel_bin_t<kernel_meta_conv2d_cutlass_t> *SelectCutlassKernel(
    const pi::tensorlib::DataType dtype, const CutlassConv2dBwdBinKernelEntry *entry)
{
    if (entry == nullptr)
    {
        return nullptr;
    }
    return dtype == pi::tensorlib::DataType::BFLOAT16 ? entry->bf16_kernel : entry->fp16_kernel;
}

static const kernel_bin_t<kernel_meta_conv2d_cutlass_t> *SelectCutlassKernel(
    const pi::tensorlib::DataType dtype, const CutlassConv2dBinKernelEntry *entry,
    const bool use_fp16_accumulation)
{
    if (entry == nullptr)
    {
        return nullptr;
    }
    if (dtype == pi::tensorlib::DataType::BFLOAT16)
    {
        return entry->bf16_kernel;
    }
    return (use_fp16_accumulation && entry->fp16_acc_fp16_kernel != nullptr) ? entry->fp16_acc_fp16_kernel
                                                                            : entry->fp16_kernel;
}

static const kernel_bin_t<kernel_meta_conv2d_cutlass_t> *SelectCutlassKernel(
    const pi::tensorlib::DataType dtype, const CutlassConv2dKernelEntry *entry,
    const bool use_fp16_accumulation)
{
    if (entry == nullptr)
    {
        return nullptr;
    }
    if (dtype == pi::tensorlib::DataType::BFLOAT16)
    {
        return entry->bf16_kernel;
    }
    return (use_fp16_accumulation && entry->fp16_acc_fp16_kernel != nullptr) ? entry->fp16_acc_fp16_kernel
                                                                            : entry->fp16_kernel;
}

static void LogBinLookup(const char *op_name, const bool hit, const uint32_t in_channels,
                         const uint32_t out_channels, const uint32_t kernel_h, const uint32_t kernel_w,
                         const uint32_t stride_h, const uint32_t stride_w, const uint32_t dilation_h,
                         const uint32_t dilation_w, const std::array<uint32_t, 2> &padding,
                         const uint32_t batch, const uint32_t input_h, const uint32_t input_w,
                         const char *kernel_name)
{
    if (!DebugConv2d())
    {
        return;
    }
    std::clog << "[Conv2d] " << op_name << " bin " << (hit ? "hit" : "miss") << ": query="
              << "(ic=" << in_channels << ", oc=" << out_channels << ", kh=" << kernel_h << ", kw=" << kernel_w
              << ", stride=" << stride_h << "," << stride_w << ", dilation=" << dilation_h << ","
              << dilation_w << ", pad=" << padding[0] << "," << padding[1] << ", b=" << batch
              << ", h=" << input_h << ", w=" << input_w << ")";
    if (hit)
    {
        std::clog << " kernel=" << (kernel_name != nullptr ? kernel_name : "<null>");
    }
    std::clog << '\n';
}

static SelectedConv2dKernel FinishSelection(
    const char *op_name, const bool prefer_cutlass,
    const kernel_bin_t<kernel_meta_conv2d_cutlass_t> *cutlass_kernel,
    const kernel_bin_t<kernel_meta_conv2d_t> *triton_kernel,
    const CutlassConv2dImplementation cutlass_implementation = CutlassConv2dImplementation::Cutlass2)
{
    const auto pick_cutlass = [&](const char *prefix)
    {
        if (DebugConv2d())
        {
            std::clog << "[Conv2d] " << prefix << ' ' << ToString(cutlass_implementation) << ' ' << op_name
                      << " kernel " << cutlass_kernel->function_name << '\n';
        }
        return SelectedConv2dKernel{
            .backend = Conv2dKernelBackend::Cutlass,
            .cutlass_implementation = cutlass_implementation,
            .triton_kernel = nullptr,
            .cutlass_kernel = cutlass_kernel,
        };
    };
    const auto pick_triton = [&](const char *prefix)
    {
        if (DebugConv2d())
        {
            std::clog << "[Conv2d] " << prefix << " Triton " << op_name << " kernel "
                      << triton_kernel->function_name << '\n';
        }
        return SelectedConv2dKernel{
            .backend = Conv2dKernelBackend::Triton,
            .triton_kernel = triton_kernel,
            .cutlass_kernel = nullptr,
        };
    };

    if (prefer_cutlass && cutlass_kernel != nullptr)
    {
        return pick_cutlass("Selecting");
    }
    if (!prefer_cutlass && triton_kernel != nullptr)
    {
        return pick_triton("Selecting");
    }
    if (cutlass_kernel != nullptr)
    {
        return pick_cutlass("Falling back to");
    }
    if (triton_kernel != nullptr)
    {
        return pick_triton("Falling back to");
    }
    throw std::runtime_error("Unsupported conv2d " + std::string(op_name) + " configuration");
}

static const Conv2dKernelEntry *LookupTritonConv2dKernel(
    const std::unordered_map<Conv2dKernelKey, const Conv2dKernelEntry *, Conv2dKernelKeyHash> &map,
    const Conv2dKernelKey &query)
{
    const auto match = map.find(query);
    return match != map.end() ? match->second : nullptr;
}

template <typename Entry>
static const Entry *LookupCutlassConv2dBinKernel(
    const std::unordered_map<Conv2dBinLookupKey, const Entry *, Conv2dBinLookupKeyHash> &map,
    const Conv2dKernelKey &query, const std::array<uint32_t, 2> &padding, const uint32_t batch,
    const uint32_t input_h, const uint32_t input_w)
{
    const auto match = map.find(MakeConv2dBinLookupKey(query, padding, batch, input_h, input_w));
    return match != map.end() ? match->second : nullptr;
}

static SelectedConv2dKernel SelectConv2dKernel(const uint32_t in_channels, const uint32_t out_channels,
                                               const uint32_t kernel_h, const uint32_t kernel_w,
                                               const uint32_t stride_h, const uint32_t stride_w,
                                               const uint32_t dilation_h, const uint32_t dilation_w,
                                               const std::array<uint32_t, 2> &padding, const uint32_t batch,
                                               const uint32_t input_h, const uint32_t input_w,
                                               const pi::tensorlib::DataType dtype, const bool has_bias,
                                               const bool use_fp16_accumulation)
{
    const bool prefer_cutlass = GetConv2dBackendPreference() == Conv2dBackendPreference::Cutlass;
    const auto query = MakeConv2dKernelKey(in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
                                           dilation_h, dilation_w);
    const auto triton_kernel =
        SelectTritonKernel(dtype, LookupTritonConv2dKernel(Conv2dKernelMap<kConv2dKernels>(), query));
    const kernel_bin_t<kernel_meta_conv2d_cutlass_t> *cutlass_kernel = nullptr;
    auto cutlass_implementation = CutlassConv2dImplementation::Cutlass2;

    if (prefer_cutlass && !has_bias)
    {
        const auto *bin_match =
            LookupCutlassConv2dBinKernel(Conv2dBinKernelMapFor<kCutlassConv2dBinKernels>(), query, padding,
                                         batch, input_h, input_w);
        if (bin_match != nullptr)
        {
            cutlass_implementation = ToCutlassConv2dImplementation(bin_match->implementation);
            cutlass_kernel = SelectCutlassKernel(dtype, bin_match, use_fp16_accumulation);
        }
        LogBinLookup("fwd", cutlass_kernel != nullptr, in_channels, out_channels, kernel_h, kernel_w, stride_h,
                     stride_w, dilation_h, dilation_w, padding, batch, input_h, input_w,
                     cutlass_kernel != nullptr ? cutlass_kernel->function_name : nullptr);
        if (cutlass_kernel == nullptr)
        {
            const auto &cutlass_map = CutlassConv2dKernelMap<kCutlassConv2dKernelsNv>();
            const auto match = cutlass_map.find(MakeConv2dPaddedKernelKey(query, padding));
            cutlass_kernel = SelectCutlassKernel(dtype, match != cutlass_map.end() ? match->second : nullptr,
                                                 use_fp16_accumulation);
            cutlass_implementation = CutlassConv2dImplementation::Cutlass2;
        }
    }
    return FinishSelection("fwd", prefer_cutlass, cutlass_kernel, triton_kernel, cutlass_implementation);
}

template <typename BinMapFn, typename MapFn>
static SelectedConv2dKernel SelectConv2dBwdKernel(const char *op_name, BinMapFn get_bin_map, MapFn get_cutlass_map,
                                                  const std::unordered_map<Conv2dKernelKey,
                                                                           const Conv2dKernelEntry *,
                                                                           Conv2dKernelKeyHash> &triton_map,
                                                  const uint32_t in_channels, const uint32_t out_channels,
                                                  const uint32_t kernel_h, const uint32_t kernel_w,
                                                  const uint32_t stride_h, const uint32_t stride_w,
                                                  const uint32_t dilation_h, const uint32_t dilation_w,
                                                  const std::array<uint32_t, 2> &padding,
                                                  const uint32_t batch, const uint32_t input_h,
                                                  const uint32_t input_w, const pi::tensorlib::DataType dtype,
                                                  const CutlassConv2dImplementationPreference implementation_preference =
                                                      CutlassConv2dImplementationPreference::Auto)
{
    const bool prefer_cutlass = GetConv2dBackendPreference() == Conv2dBackendPreference::Cutlass;
    const auto query = MakeConv2dKernelKey(in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
                                           dilation_h, dilation_w);
    const auto triton_kernel = SelectTritonKernel(dtype, LookupTritonConv2dKernel(triton_map, query));
    const kernel_bin_t<kernel_meta_conv2d_cutlass_t> *cutlass_kernel = nullptr;
    auto cutlass_implementation = CutlassConv2dImplementation::Cutlass2;

    if (prefer_cutlass)
    {
        const auto *bin_match =
            LookupCutlassConv2dBinKernel(get_bin_map(), query, padding, batch, input_h, input_w);
        if (bin_match != nullptr)
        {
            const auto bin_implementation = ToCutlassConv2dImplementation(bin_match->implementation);
            if (AllowsCutlassImplementation(implementation_preference, bin_implementation))
            {
                cutlass_kernel = SelectCutlassKernel(dtype, bin_match);
                cutlass_implementation = bin_implementation;
            }
        }
        LogBinLookup(op_name, cutlass_kernel != nullptr, in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
                     dilation_h, dilation_w, padding, batch, input_h, input_w,
                     cutlass_kernel != nullptr ? cutlass_kernel->function_name : nullptr);
        if (cutlass_kernel == nullptr &&
            implementation_preference != CutlassConv2dImplementationPreference::Cutlass3)
        {
            const auto &cutlass_map = get_cutlass_map();
            const auto match = cutlass_map.find(MakeConv2dPaddedKernelKey(query, padding));
            cutlass_kernel = SelectCutlassKernel(dtype, match != cutlass_map.end() ? match->second : nullptr);
            cutlass_implementation = CutlassConv2dImplementation::Cutlass2;
        }
        if (cutlass_kernel == nullptr &&
            implementation_preference != CutlassConv2dImplementationPreference::Auto)
        {
            throw std::runtime_error("Requested " + std::string(ToString(
                                                implementation_preference ==
                                                        CutlassConv2dImplementationPreference::Cutlass3
                                                    ? CutlassConv2dImplementation::Cutlass3
                                                    : CutlassConv2dImplementation::Cutlass2)) +
                                     " conv2d " + op_name + " kernel is not available for this shape");
        }
    }

    return FinishSelection(op_name, prefer_cutlass, cutlass_kernel, triton_kernel, cutlass_implementation);
}

static SelectedConv2dKernel SelectConv2dDgradKernel(const uint32_t in_channels, const uint32_t out_channels,
                                                    const uint32_t kernel_h, const uint32_t kernel_w,
                                                    const uint32_t stride_h, const uint32_t stride_w,
                                                    const uint32_t dilation_h, const uint32_t dilation_w,
                                                    const std::array<uint32_t, 2> &padding,
                                                    const uint32_t batch, const uint32_t input_h,
                                                    const uint32_t input_w,
                                                    const pi::tensorlib::DataType dtype)
{
    return SelectConv2dBwdKernel("dgrad", Conv2dBinKernelMapFor<kCutlassConv2dDgradBinKernels>,
                                 CutlassConv2dKernelMap<kCutlassConv2dDgradKernelsNv>,
                                 Conv2dKernelMap<kConv2dDgradKernels>(), in_channels, out_channels, kernel_h,
                                 kernel_w, stride_h, stride_w, dilation_h, dilation_w, padding, batch, input_h,
                                 input_w, dtype);
}

static SelectedConv2dKernel SelectConv2dWgradKernel(const uint32_t in_channels, const uint32_t out_channels,
                                                    const uint32_t kernel_h, const uint32_t kernel_w,
                                                    const uint32_t stride_h, const uint32_t stride_w,
                                                    const uint32_t dilation_h, const uint32_t dilation_w,
                                                    const std::array<uint32_t, 2> &padding,
                                                    const uint32_t batch, const uint32_t input_h,
                                                    const uint32_t input_w,
                                                    const pi::tensorlib::DataType dtype)
{
    return SelectConv2dBwdKernel("wgrad", Conv2dBinKernelMapFor<kCutlassConv2dWgradBinKernels>,
                                 CutlassConv2dKernelMap<kCutlassConv2dWgradKernelsNv>,
                                 Conv2dKernelMap<kConv2dWgradKernels>(), in_channels, out_channels, kernel_h,
                                 kernel_w, stride_h, stride_w, dilation_h, dilation_w, padding, batch, input_h,
                                 input_w, dtype, GetConv2dWgradCutlassImplementationPreference());
}

pi::tensorlib::ComputeKernelDescriptor CreateTritonConv2dComputeKernelDescriptor(
    const pi::tensorlib::DataType dtype, const std::array<uint32_t, 2> &stride,
    const std::array<uint32_t, 2> &padding, const std::array<uint32_t, 2> &dilation,
    const kernel_bin_t<kernel_meta_conv2d_t> &kernel_bin)
{
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_bin.kernel_name,
        .function_name = kernel_bin.function_name,
        .expected_arg_count = kernel_bin.arg_count,
        .argument_provider = [stride, padding, dilation, kernel_ptr = &kernel_bin, dtype](
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() < 2 || inputs.size() > 3)
            {
                throw std::runtime_error("conv2d expects two or three inputs (input, weight, [bias])");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("conv2d expects a single output tensor");
            }

            const auto &input = inputs[0];
            const auto &weight = inputs[1];
            const auto &output = outputs[0];
            const std::shared_ptr<pi::tensorlib::RealTensor> bias = inputs.size() == 3 ? inputs[2] : nullptr;

            if (input->dtype() != dtype || weight->dtype() != dtype || output->dtype() != dtype ||
                (bias && bias->dtype() != dtype))
            {
                throw std::runtime_error("conv2d currently supports BFLOAT16 or FLOAT16 tensors");
            }

            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                weight->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU ||
                (bias && bias->device().device_type != pi::tensorlib::DeviceType::GPU))
            {
                throw std::runtime_error("conv2d requires GPU tensors");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("conv2d", {input, weight, output, bias});

            if (input->shape().ndims() != 4 || weight->shape().ndims() != 4 || output->shape().ndims() != 4)
            {
                throw std::runtime_error("conv2d expects 4D input, weight, and output tensors");
            }
            if (bias && (bias->shape().ndims() != 1))
            {
                throw std::runtime_error("conv2d bias must be 1D");
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(weight))
            {
                throw std::runtime_error("conv2d weight must be contiguous");
            }

            const auto batch64 = input->shape()[0];
            const auto in_h64 = input->shape()[1];
            const auto in_w64 = input->shape()[2];
            const auto in_channels64 = input->shape()[3];

            const auto out_channels64 = weight->shape()[0];
            const auto kernel_h64 = weight->shape()[1];
            const auto kernel_w64 = weight->shape()[2];
            const auto kernel_in_channels64 = weight->shape()[3];

            if (kernel_in_channels64 != in_channels64)
            {
                throw std::runtime_error("conv2d weight in_channels must match input channels");
            }

            const auto output_h64 = output->shape()[1];
            const auto output_w64 = output->shape()[2];
            const auto out_channels_out64 = output->shape()[3];
            if (out_channels_out64 != out_channels64)
            {
                throw std::runtime_error("conv2d output channels must match weight out_channels");
            }
            if (bias && bias->shape()[0] != out_channels64)
            {
                throw std::runtime_error("conv2d bias must have size equal to out_channels");
            }

            auto to_u32 = [](const uint64_t value, const char *what) -> uint32_t
            {
                if (value > std::numeric_limits<uint32_t>::max())
                {
                    std::stringstream ss;
                    ss << what << " exceeds supported range";
                    throw std::runtime_error(ss.str());
                }
                return static_cast<uint32_t>(value);
            };

            const auto batch = to_u32(batch64, "conv2d batch size");
            const auto in_channels = to_u32(in_channels64, "conv2d input channels");
            const auto in_h = to_u32(in_h64, "conv2d input height");
            const auto in_w = to_u32(in_w64, "conv2d input width");

            const auto out_channels = to_u32(out_channels64, "conv2d output channels");
            const auto kernel_h = to_u32(kernel_h64, "conv2d kernel height");
            const auto kernel_w = to_u32(kernel_w64, "conv2d kernel width");

            const auto out_h = to_u32(output_h64, "conv2d output height");
            const auto out_w = to_u32(output_w64, "conv2d output width");

            const auto dilation_h = to_u32(dilation[0], "conv2d dilation height");
            const auto dilation_w = to_u32(dilation[1], "conv2d dilation width");

            const uint64_t effective_kernel_h = 1 + static_cast<uint64_t>(dilation_h) * (kernel_h64 - 1);
            const uint64_t effective_kernel_w = 1 + static_cast<uint64_t>(dilation_w) * (kernel_w64 - 1);

            const int64_t numerator_h = static_cast<int64_t>(in_h64 + 2 * padding[0]) -
                                        static_cast<int64_t>(effective_kernel_h);
            const int64_t numerator_w = static_cast<int64_t>(in_w64 + 2 * padding[1]) -
                                        static_cast<int64_t>(effective_kernel_w);
            if (numerator_h < 0 || numerator_w < 0)
            {
                throw std::runtime_error("conv2d padding/stride/dilation/kernel configuration results in negative "
                                         "output size");
            }
            if (numerator_h % static_cast<int64_t>(stride[0]) != 0 ||
                numerator_w % static_cast<int64_t>(stride[1]) != 0)
            {
                throw std::runtime_error("conv2d configuration must produce integer output spatial dimensions");
            }
            const auto expected_out_h =
                static_cast<uint64_t>(numerator_h / static_cast<int64_t>(stride[0]) + 1);
            const auto expected_out_w =
                static_cast<uint64_t>(numerator_w / static_cast<int64_t>(stride[1]) + 1);
            if (expected_out_h != output_h64 || expected_out_w != output_w64)
            {
                throw std::runtime_error("conv2d output dimensions mismatch expected shape from stride/padding/dilation");
            }

            if (in_channels != kernel_ptr->meta.in_channels || kernel_h != kernel_ptr->meta.kernel_h ||
                kernel_w != kernel_ptr->meta.kernel_w)
            {
                throw std::runtime_error("conv2d configuration not covered by compiled kernel set");
            }

            const auto &input_strides = input->strides();
            const auto &weight_strides = weight->strides();
            const auto &output_strides = output->strides();

            const auto in_stride_n = to_u32(input_strides[0], "conv2d input stride N");
            const auto in_stride_h = to_u32(input_strides[1], "conv2d input stride H");
            const auto in_stride_w = to_u32(input_strides[2], "conv2d input stride W");
            const auto in_stride_c = to_u32(input_strides[3], "conv2d input stride C");

            const auto weight_stride_oc = to_u32(weight_strides[0], "conv2d weight stride OC");
            const auto weight_stride_kh = to_u32(weight_strides[1], "conv2d weight stride KH");
            const auto weight_stride_kw = to_u32(weight_strides[2], "conv2d weight stride KW");
            const auto weight_stride_ic = to_u32(weight_strides[3], "conv2d weight stride IC");

            const auto out_stride_n = to_u32(output_strides[0], "conv2d output stride N");
            const auto out_stride_h = to_u32(output_strides[1], "conv2d output stride H");
            const auto out_stride_w = to_u32(output_strides[2], "conv2d output stride W");
            const auto out_stride_c = to_u32(output_strides[3], "conv2d output stride C");

            const auto grid_dim_y =
                to_u32(batch64 * output_h64, "conv2d grid dim Y (batch * output height)");

            void *input_ptr = input->dataptr();
            void *weight_ptr = weight->dataptr();
            void *output_ptr = output->dataptr();
            void *bias_ptr = bias ? bias->dataptr() : output_ptr;

            const uint32_t has_bias = bias ? 1U : 0U;
            const uint32_t block_dim_x = TRITON_WARP_SIZE * kernel_ptr->num_warps;

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {input_ptr,
                         weight_ptr,
                         bias_ptr,
                         output_ptr,
                         batch,
                         in_channels,
                         in_h,
                         in_w,
                         out_channels,
                         kernel_h,
                        kernel_w,
                        stride[0],
                        stride[1],
                        dilation_h,
                        dilation_w,
                        padding[0],
                        padding[1],
                        out_h,
                        out_w,
                         in_stride_n,
                         in_stride_h,
                         in_stride_w,
                         in_stride_c,
                         weight_stride_kh,
                         weight_stride_kw,
                         weight_stride_ic,
                         weight_stride_oc,
                         out_stride_n,
                         out_stride_h,
                         out_stride_w,
                         out_stride_c,
                         has_bias,
                         static_cast<void *>(nullptr)},
                .grid_dim_x = CEIL_DIV(out_channels, kernel_ptr->meta.block_oc),
                .grid_dim_y = grid_dim_y,
                .grid_dim_z = CEIL_DIV(out_w, kernel_ptr->meta.block_pixels),
                .block_dim_x = block_dim_x,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kernel_ptr->shared_mem_bytes,
                .device_ordinal = static_cast<int>(device_ordinal)};
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel_bin.data, kernel_bin.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel_bin.data, kernel_bin.size),
    };
}

pi::tensorlib::ComputeKernelDescriptor CreateTritonConv2dDgradComputeKernelDescriptor(
    const pi::tensorlib::DataType dtype, const std::array<uint32_t, 2> &stride,
    const std::array<uint32_t, 2> &padding, const std::array<uint32_t, 2> &dilation,
    const kernel_bin_t<kernel_meta_conv2d_t> &kernel_bin)
{
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_bin.kernel_name,
        .function_name = kernel_bin.function_name,
        .expected_arg_count = kernel_bin.arg_count,
        .argument_provider = [stride, padding, dilation, kernel_ptr = &kernel_bin, dtype](
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 2)
            {
                throw std::runtime_error("conv2d dgrad expects two inputs (upstream, weight)");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("conv2d dgrad expects a single output tensor (grad_input)");
            }

            const auto &upstream = inputs[0];
            const auto &weight = inputs[1];
            const auto &grad_input = outputs[0];

            if (upstream->dtype() != dtype || weight->dtype() != dtype || grad_input->dtype() != dtype)
            {
                throw std::runtime_error("conv2d dgrad currently supports BFLOAT16 or FLOAT16 tensors");
            }

            if (upstream->device().device_type != pi::tensorlib::DeviceType::GPU ||
                weight->device().device_type != pi::tensorlib::DeviceType::GPU ||
                grad_input->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("conv2d dgrad requires GPU tensors");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("conv2d_dgrad", {upstream, weight, grad_input});

            if (upstream->shape().ndims() != 4 || weight->shape().ndims() != 4 || grad_input->shape().ndims() != 4)
            {
                throw std::runtime_error("conv2d dgrad expects 4D upstream, weight, and grad_input tensors");
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(upstream) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(weight) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(grad_input))
            {
                throw std::runtime_error("conv2d dgrad expects contiguous upstream, weight, and grad_input tensors");
            }

            const auto batch64 = grad_input->shape()[0];
            const auto in_h64 = grad_input->shape()[1];
            const auto in_w64 = grad_input->shape()[2];
            const auto in_channels64 = grad_input->shape()[3];

            const auto out_channels64 = weight->shape()[0];
            const auto kernel_h64 = weight->shape()[1];
            const auto kernel_w64 = weight->shape()[2];
            const auto kernel_in_channels64 = weight->shape()[3];

            if (kernel_in_channels64 != in_channels64)
            {
                throw std::runtime_error("conv2d dgrad weight in_channels must match grad_input channels");
            }

            const uint64_t effective_kernel_h =
                1 + static_cast<uint64_t>(dilation[0]) * (kernel_h64 - 1);
            const uint64_t effective_kernel_w =
                1 + static_cast<uint64_t>(dilation[1]) * (kernel_w64 - 1);

            const int64_t numerator_h =
                static_cast<int64_t>(in_h64 + 2 * padding[0]) - static_cast<int64_t>(effective_kernel_h);
            const int64_t numerator_w =
                static_cast<int64_t>(in_w64 + 2 * padding[1]) - static_cast<int64_t>(effective_kernel_w);
            if (numerator_h < 0 || numerator_w < 0)
            {
                throw std::runtime_error("conv2d dgrad configuration results in negative output size");
            }
            if (numerator_h % static_cast<int64_t>(stride[0]) != 0 ||
                numerator_w % static_cast<int64_t>(stride[1]) != 0)
            {
                throw std::runtime_error("conv2d dgrad configuration must produce integer output spatial dimensions");
            }

            const auto out_h64 =
                static_cast<uint64_t>(numerator_h / static_cast<int64_t>(stride[0]) + 1);
            const auto out_w64 =
                static_cast<uint64_t>(numerator_w / static_cast<int64_t>(stride[1]) + 1);

            if (upstream->shape()[0] != batch64 || upstream->shape()[1] != out_h64 ||
                upstream->shape()[2] != out_w64 || upstream->shape()[3] != out_channels64)
            {
                throw std::runtime_error("conv2d dgrad upstream shape mismatch");
            }

            if (in_channels64 != kernel_ptr->meta.in_channels || kernel_h64 != kernel_ptr->meta.kernel_h ||
                kernel_w64 != kernel_ptr->meta.kernel_w)
            {
                throw std::runtime_error("conv2d dgrad configuration not covered by compiled kernel set");
            }

            auto to_u32 = [](const uint64_t value, const char *what) -> uint32_t
            {
                if (value > std::numeric_limits<uint32_t>::max())
                {
                    std::stringstream ss;
                    ss << what << " exceeds supported range";
                    throw std::runtime_error(ss.str());
                }
                return static_cast<uint32_t>(value);
            };

            const auto batch = to_u32(batch64, "conv2d dgrad batch size");
            const auto in_channels = to_u32(in_channels64, "conv2d dgrad input channels");
            const auto in_h = to_u32(in_h64, "conv2d dgrad input height");
            const auto in_w = to_u32(in_w64, "conv2d dgrad input width");

            const auto out_channels = to_u32(out_channels64, "conv2d dgrad output channels");
            const auto kernel_h = to_u32(kernel_h64, "conv2d dgrad kernel height");
            const auto kernel_w = to_u32(kernel_w64, "conv2d dgrad kernel width");
            const auto out_h = to_u32(out_h64, "conv2d dgrad output height");
            const auto out_w = to_u32(out_w64, "conv2d dgrad output width");

            const auto &dout_strides = upstream->strides();
            const auto &weight_strides = weight->strides();
            const auto &dinput_strides = grad_input->strides();

            const auto dout_stride_n = to_u32(dout_strides[0], "conv2d dgrad upstream stride N");
            const auto dout_stride_h = to_u32(dout_strides[1], "conv2d dgrad upstream stride H");
            const auto dout_stride_w = to_u32(dout_strides[2], "conv2d dgrad upstream stride W");
            const auto dout_stride_c = to_u32(dout_strides[3], "conv2d dgrad upstream stride C");

            const auto weight_stride_oc = to_u32(weight_strides[0], "conv2d dgrad weight stride OC");
            const auto weight_stride_kh = to_u32(weight_strides[1], "conv2d dgrad weight stride KH");
            const auto weight_stride_kw = to_u32(weight_strides[2], "conv2d dgrad weight stride KW");
            const auto weight_stride_ic = to_u32(weight_strides[3], "conv2d dgrad weight stride IC");

            const auto dinput_stride_n = to_u32(dinput_strides[0], "conv2d dgrad output stride N");
            const auto dinput_stride_h = to_u32(dinput_strides[1], "conv2d dgrad output stride H");
            const auto dinput_stride_w = to_u32(dinput_strides[2], "conv2d dgrad output stride W");
            const auto dinput_stride_c = to_u32(dinput_strides[3], "conv2d dgrad output stride C");

            const uint64_t grid_x64 = CEIL_DIV(in_channels64, kernel_ptr->meta.block_k);

            const uint32_t block_dim_x = TRITON_WARP_SIZE * kernel_ptr->num_warps;
            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {upstream->dataptr(),
                         weight->dataptr(),
                         grad_input->dataptr(),
                         batch,
                         in_channels,
                         in_h,
                         in_w,
                         out_channels,
                         kernel_h,
                         kernel_w,
                         stride[0],
                         stride[1],
                         dilation[0],
                         dilation[1],
                         padding[0],
                         padding[1],
                         out_h,
                         out_w,
                         dout_stride_n,
                         dout_stride_h,
                         dout_stride_w,
                         dout_stride_c,
                         weight_stride_oc,
                         weight_stride_kh,
                         weight_stride_kw,
                         weight_stride_ic,
                         dinput_stride_n,
                         dinput_stride_h,
                         dinput_stride_w,
                         dinput_stride_c,
                         static_cast<void *>(nullptr)},
                .grid_dim_x = to_u32(grid_x64, "conv2d dgrad grid dim X"),
                .grid_dim_y = to_u32(batch64 * in_h64, "conv2d dgrad grid dim Y"),
                .grid_dim_z = to_u32(CEIL_DIV(in_w64, kernel_ptr->meta.block_pixels), "conv2d dgrad grid dim Z"),
                .block_dim_x = block_dim_x,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kernel_ptr->shared_mem_bytes,
                .device_ordinal = static_cast<int>(device_ordinal)};
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel_bin.data, kernel_bin.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel_bin.data, kernel_bin.size),
    };
}

pi::tensorlib::ComputeKernelDescriptor CreateTritonConv2dWgradComputeKernelDescriptor(
    const pi::tensorlib::DataType dtype, const std::array<uint32_t, 2> &stride,
    const std::array<uint32_t, 2> &padding, const std::array<uint32_t, 2> &dilation,
    const bool accumulate_output, const kernel_bin_t<kernel_meta_conv2d_t> &kernel_bin)
{
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_bin.kernel_name,
        .function_name = kernel_bin.function_name,
        .expected_arg_count = kernel_bin.arg_count,
        .argument_provider = [stride, padding, dilation, accumulate_output, kernel_ptr = &kernel_bin, dtype](
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 2)
            {
                throw std::runtime_error("conv2d wgrad expects two inputs (input, upstream)");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("conv2d wgrad expects a single output tensor (grad_weight)");
            }

            const auto &input = inputs[0];
            const auto &upstream = inputs[1];
            const auto &grad_weight = outputs[0];

            if (input->dtype() != dtype || upstream->dtype() != dtype || grad_weight->dtype() != dtype)
            {
                throw std::runtime_error("conv2d wgrad currently supports BFLOAT16 or FLOAT16 tensors");
            }

            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                upstream->device().device_type != pi::tensorlib::DeviceType::GPU ||
                grad_weight->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("conv2d wgrad requires GPU tensors");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("conv2d_wgrad", {input, upstream, grad_weight});

            if (input->shape().ndims() != 4 || upstream->shape().ndims() != 4 || grad_weight->shape().ndims() != 4)
            {
                throw std::runtime_error("conv2d wgrad expects 4D input, upstream, and grad_weight tensors");
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(input) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(upstream) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(grad_weight))
            {
                throw std::runtime_error("conv2d wgrad expects contiguous input, upstream, and grad_weight tensors");
            }

            const auto batch64 = input->shape()[0];
            const auto in_h64 = input->shape()[1];
            const auto in_w64 = input->shape()[2];
            const auto in_channels64 = input->shape()[3];

            const auto out_channels64 = grad_weight->shape()[0];
            const auto kernel_h64 = grad_weight->shape()[1];
            const auto kernel_w64 = grad_weight->shape()[2];
            const auto kernel_in_channels64 = grad_weight->shape()[3];

            if (kernel_in_channels64 != in_channels64)
            {
                throw std::runtime_error("conv2d wgrad grad_weight in_channels must match input channels");
            }

            const uint64_t effective_kernel_h =
                1 + static_cast<uint64_t>(dilation[0]) * (kernel_h64 - 1);
            const uint64_t effective_kernel_w =
                1 + static_cast<uint64_t>(dilation[1]) * (kernel_w64 - 1);

            const int64_t numerator_h =
                static_cast<int64_t>(in_h64 + 2 * padding[0]) - static_cast<int64_t>(effective_kernel_h);
            const int64_t numerator_w =
                static_cast<int64_t>(in_w64 + 2 * padding[1]) - static_cast<int64_t>(effective_kernel_w);
            if (numerator_h < 0 || numerator_w < 0)
            {
                throw std::runtime_error("conv2d wgrad configuration results in negative output size");
            }
            if (numerator_h % static_cast<int64_t>(stride[0]) != 0 ||
                numerator_w % static_cast<int64_t>(stride[1]) != 0)
            {
                throw std::runtime_error("conv2d wgrad configuration must produce integer output spatial dimensions");
            }

            const auto out_h64 =
                static_cast<uint64_t>(numerator_h / static_cast<int64_t>(stride[0]) + 1);
            const auto out_w64 =
                static_cast<uint64_t>(numerator_w / static_cast<int64_t>(stride[1]) + 1);

            if (upstream->shape()[0] != batch64 || upstream->shape()[1] != out_h64 ||
                upstream->shape()[2] != out_w64 || upstream->shape()[3] != out_channels64)
            {
                throw std::runtime_error("conv2d wgrad upstream shape mismatch");
            }

            if (in_channels64 != kernel_ptr->meta.in_channels || kernel_h64 != kernel_ptr->meta.kernel_h ||
                kernel_w64 != kernel_ptr->meta.kernel_w)
            {
                throw std::runtime_error("conv2d wgrad configuration not covered by compiled kernel set");
            }

            auto to_u32 = [](const uint64_t value, const char *what) -> uint32_t
            {
                if (value > std::numeric_limits<uint32_t>::max())
                {
                    std::stringstream ss;
                    ss << what << " exceeds supported range";
                    throw std::runtime_error(ss.str());
                }
                return static_cast<uint32_t>(value);
            };

            const auto batch = to_u32(batch64, "conv2d wgrad batch size");
            const auto in_channels = to_u32(in_channels64, "conv2d wgrad input channels");
            const auto in_h = to_u32(in_h64, "conv2d wgrad input height");
            const auto in_w = to_u32(in_w64, "conv2d wgrad input width");

            const auto out_channels = to_u32(out_channels64, "conv2d wgrad output channels");
            const auto kernel_h = to_u32(kernel_h64, "conv2d wgrad kernel height");
            const auto kernel_w = to_u32(kernel_w64, "conv2d wgrad kernel width");
            const auto out_h = to_u32(out_h64, "conv2d wgrad output height");
            const auto out_w = to_u32(out_w64, "conv2d wgrad output width");

            const auto &input_strides = input->strides();
            const auto &dout_strides = upstream->strides();
            const auto &dweight_strides = grad_weight->strides();

            const auto input_stride_n = to_u32(input_strides[0], "conv2d wgrad input stride N");
            const auto input_stride_h = to_u32(input_strides[1], "conv2d wgrad input stride H");
            const auto input_stride_w = to_u32(input_strides[2], "conv2d wgrad input stride W");
            const auto input_stride_c = to_u32(input_strides[3], "conv2d wgrad input stride C");

            const auto dout_stride_n = to_u32(dout_strides[0], "conv2d wgrad upstream stride N");
            const auto dout_stride_h = to_u32(dout_strides[1], "conv2d wgrad upstream stride H");
            const auto dout_stride_w = to_u32(dout_strides[2], "conv2d wgrad upstream stride W");
            const auto dout_stride_c = to_u32(dout_strides[3], "conv2d wgrad upstream stride C");

            const auto dweight_stride_oc = to_u32(dweight_strides[0], "conv2d wgrad weight stride OC");
            const auto dweight_stride_kh = to_u32(dweight_strides[1], "conv2d wgrad weight stride KH");
            const auto dweight_stride_kw = to_u32(dweight_strides[2], "conv2d wgrad weight stride KW");
            const auto dweight_stride_ic = to_u32(dweight_strides[3], "conv2d wgrad weight stride IC");
            const auto accumulate_flag = static_cast<uint32_t>(accumulate_output ? 1u : 0u);

            const uint64_t grid_x64 = CEIL_DIV(out_channels64, kernel_ptr->meta.block_oc);
            const uint64_t grid_y64 = CEIL_DIV(in_channels64, kernel_ptr->meta.block_k);
            const uint64_t grid_z64 = kernel_h64 * kernel_w64;

            const uint32_t block_dim_x = TRITON_WARP_SIZE * kernel_ptr->num_warps;
            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {input->dataptr(),
                         upstream->dataptr(),
                         grad_weight->dataptr(),
                         batch,
                         in_channels,
                         in_h,
                         in_w,
                         out_channels,
                         kernel_h,
                         kernel_w,
                         stride[0],
                         stride[1],
                         dilation[0],
                         dilation[1],
                         padding[0],
                         padding[1],
                         out_h,
                         out_w,
                         input_stride_n,
                         input_stride_h,
                         input_stride_w,
                         input_stride_c,
                         dout_stride_n,
                         dout_stride_h,
                         dout_stride_w,
                         dout_stride_c,
                         dweight_stride_oc,
                         dweight_stride_kh,
                         dweight_stride_kw,
                         dweight_stride_ic,
                         accumulate_flag,
                         static_cast<void *>(nullptr)},
                .grid_dim_x = to_u32(grid_x64, "conv2d wgrad grid dim X"),
                .grid_dim_y = to_u32(grid_y64, "conv2d wgrad grid dim Y"),
                .grid_dim_z = to_u32(grid_z64, "conv2d wgrad grid dim Z"),
                .block_dim_x = block_dim_x,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kernel_ptr->shared_mem_bytes,
                .device_ordinal = static_cast<int>(device_ordinal)};
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel_bin.data, kernel_bin.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel_bin.data, kernel_bin.size),
    };
}

struct CutlassConv2dConfig
{
    int32_t N;
    int32_t H;
    int32_t W;
    int32_t C;
    int32_t K;
    int32_t R;
    int32_t S;
    int32_t pad_h;
    int32_t pad_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t dilation_h;
    int32_t dilation_w;
    int32_t groups;
    int32_t split_k_slices;
};

static fbamtrain::kernel_gen::nv::GeneratedConv2dDType ToGeneratedConv2dDType(
    const pi::tensorlib::DataType dtype)
{
    switch (dtype)
    {
        case pi::tensorlib::DataType::BFLOAT16:
            return fbamtrain::kernel_gen::nv::GeneratedConv2dDType::BF16;
        case pi::tensorlib::DataType::FLOAT16:
            return fbamtrain::kernel_gen::nv::GeneratedConv2dDType::FP16;
        default:
            throw std::runtime_error("CUTLASS3 conv2d launch currently supports BFLOAT16 or FLOAT16 tensors");
    }
}

static pi::tensorlib::KernelLaunchArguments PrepareGeneratedConv2dLaunchArguments(
    const kernel_meta_conv2d_cutlass_t &meta, const char *function_name, const char *op_name,
    const fbamtrain::kernel_gen::nv::GeneratedConv2dRequest &request)
{
    fbamtrain::kernel_gen::nv::GeneratedConv2dLaunchPlan plan{};
    std::string prepare_error;
    if (!fbamtrain::kernel_gen::nv::PrepareGeneratedConv2dLaunch(meta, request, plan, prepare_error))
    {
        throw std::runtime_error("failed to prepare generated conv2d " + std::string(op_name) + " launch for " +
                                 std::string(function_name) + ": " + prepare_error);
    }

    pi::tensorlib::KernelDataArg params_storage{};
    params_storage.bytes = std::move(plan.params);

    pi::tensorlib::KernelLaunchArguments arguments{};
    arguments.args.emplace_back(std::move(params_storage));
    arguments.grid_dim_x = plan.grid_dim_x;
    arguments.grid_dim_y = plan.grid_dim_y;
    arguments.grid_dim_z = plan.grid_dim_z;
    arguments.block_dim_x = plan.block_dim_x;
    arguments.block_dim_y = plan.block_dim_y;
    arguments.block_dim_z = plan.block_dim_z;
    arguments.shared_mem_bytes = plan.shared_mem_bytes;
    arguments.cluster_dim_x = plan.cluster_dim_x;
    arguments.cluster_dim_y = plan.cluster_dim_y;
    arguments.cluster_dim_z = plan.cluster_dim_z;
    arguments.device_ordinal = request.device_ordinal;

    if (DebugConv2d())
    {
        std::clog << "[Conv2d] prepared CUTLASS3 raw-param " << op_name << " launch for "
                  << function_name << " grid=(" << arguments.grid_dim_x << ',' << arguments.grid_dim_y
                  << ',' << arguments.grid_dim_z << ") block=(" << arguments.block_dim_x << ','
                  << arguments.block_dim_y << ',' << arguments.block_dim_z << ") cluster=("
                  << arguments.cluster_dim_x << ',' << arguments.cluster_dim_y << ','
                  << arguments.cluster_dim_z << ")\n";
    }

    return arguments;
}

pi::tensorlib::ComputeKernelDescriptor CreateCutlassConv2dComputeKernelDescriptor(
    const pi::tensorlib::DataType dtype, const std::array<uint32_t, 2> &stride,
    const std::array<uint32_t, 2> &padding, const std::array<uint32_t, 2> &dilation,
    const kernel_bin_t<kernel_meta_conv2d_cutlass_t> &kernel_bin, const bool use_fp16_accumulation,
    const CutlassConv2dImplementation cutlass_implementation)
{
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_bin.kernel_name,
        .function_name = kernel_bin.function_name,
        .expected_arg_count = kernel_bin.arg_count,
        .argument_provider = [stride, padding, dilation, kernel_ptr = &kernel_bin, dtype, use_fp16_accumulation,
                              cutlass_implementation](
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() < 2 || inputs.size() > 3)
            {
                throw std::runtime_error("conv2d expects two or three inputs (input, weight, [bias])");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("conv2d expects a single output tensor");
            }

            const auto &input = inputs[0];
            const auto &weight = inputs[1];
            const auto &output = outputs[0];
            const std::shared_ptr<pi::tensorlib::RealTensor> bias = inputs.size() == 3 ? inputs[2] : nullptr;

            if (bias)
            {
                throw std::runtime_error("conv2d cutlass kernels currently require no bias");
            }

            if (input->dtype() != dtype || weight->dtype() != dtype || output->dtype() != dtype)
            {
                throw std::runtime_error("conv2d currently supports BFLOAT16 or FLOAT16 tensors");
            }

            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                weight->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("conv2d requires GPU tensors");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("conv2d", {input, weight, output});

            if (input->shape().ndims() != 4 || weight->shape().ndims() != 4 || output->shape().ndims() != 4)
            {
                throw std::runtime_error("conv2d expects 4D input, weight, and output tensors");
            }

            const auto batch64 = input->shape()[0];
            const auto in_h64 = input->shape()[1];
            const auto in_w64 = input->shape()[2];
            const auto in_channels64 = input->shape()[3];

            const auto out_channels64 = weight->shape()[0];
            const auto kernel_h64 = weight->shape()[1];
            const auto kernel_w64 = weight->shape()[2];
            const auto kernel_in_channels64 = weight->shape()[3];

            if (kernel_in_channels64 != in_channels64)
            {
                throw std::runtime_error("conv2d weight in_channels must match input channels for Cutlass kernels");
            }
            if (kernel_h64 == 0 || kernel_w64 == 0)
            {
                throw std::runtime_error("conv2d kernel dimensions must be greater than zero");
            }

            if (output->shape()[0] != batch64)
            {
                throw std::runtime_error("conv2d output batch dimension must match input for Cutlass kernels");
            }
            if (output->shape()[3] != out_channels64)
            {
                throw std::runtime_error("conv2d output channel dimension must match weight out_channels");
            }

            const auto output_h64 = output->shape()[1];
            const auto output_w64 = output->shape()[2];

            const uint64_t effective_kernel_h =
                1 + static_cast<uint64_t>(dilation[0]) * (kernel_h64 - 1);
            const uint64_t effective_kernel_w =
                1 + static_cast<uint64_t>(dilation[1]) * (kernel_w64 - 1);

            const int64_t numerator_h =
                static_cast<int64_t>(in_h64 + 2 * padding[0]) - static_cast<int64_t>(effective_kernel_h);
            const int64_t numerator_w =
                static_cast<int64_t>(in_w64 + 2 * padding[1]) - static_cast<int64_t>(effective_kernel_w);
            if (numerator_h < 0 || numerator_w < 0)
            {
                throw std::runtime_error("conv2d configuration results in negative output spatial size");
            }
            if (numerator_h % static_cast<int64_t>(stride[0]) != 0 ||
                numerator_w % static_cast<int64_t>(stride[1]) != 0)
            {
                throw std::runtime_error("conv2d configuration must produce integer output spatial dimensions");
            }

            const auto expected_out_h64 =
                static_cast<uint64_t>(numerator_h / static_cast<int64_t>(stride[0]) + 1);
            const auto expected_out_w64 =
                static_cast<uint64_t>(numerator_w / static_cast<int64_t>(stride[1]) + 1);

            if (output_h64 != expected_out_h64 || output_w64 != expected_out_w64)
            {
                throw std::runtime_error("conv2d output dimensions mismatch expected Cutlass kernel shape");
            }

            auto to_u32 = [](const uint64_t value, const char *what) -> uint32_t
            {
                if (value > std::numeric_limits<uint32_t>::max())
                {
                    std::stringstream ss;
                    ss << what << " exceeds supported range";
                    throw std::runtime_error(ss.str());
                }
                return static_cast<uint32_t>(value);
            };

            const auto batch = to_u32(batch64, "conv2d batch size");
            const auto in_channels = to_u32(in_channels64, "conv2d input channels");
            const auto in_h = to_u32(in_h64, "conv2d input height");
            const auto in_w = to_u32(in_w64, "conv2d input width");

            const auto out_channels = to_u32(out_channels64, "conv2d output channels");
            const auto kernel_h = to_u32(kernel_h64, "conv2d kernel height");
            const auto kernel_w = to_u32(kernel_w64, "conv2d kernel width");

            const auto out_h = to_u32(output_h64, "conv2d output height");
            const auto out_w = to_u32(output_w64, "conv2d output width");

            if (kernel_ptr->meta.in_channels != 0 && in_channels != kernel_ptr->meta.in_channels)
            {
                throw std::runtime_error("conv2d configuration not covered by compiled Cutlass kernel set");
            }
            if (kernel_h != kernel_ptr->meta.kernel_h || kernel_w != kernel_ptr->meta.kernel_w)
            {
                throw std::runtime_error("conv2d configuration not covered by compiled Cutlass kernel set");
            }

            const uint64_t total_pixels64 = batch64 * output_h64 * output_w64;
            const auto total_pixels = to_u32(total_pixels64, "conv2d total output elements (batch*out_h*out_w)");

            void *input_ptr = input->dataptr();
            void *weight_ptr = weight->dataptr();
            void *output_ptr = output->dataptr();

            const uint32_t block_threads = kernel_ptr->num_warps * CUDA_WARP_SIZE;
            const uint32_t grid_x = CEIL_DIV(total_pixels, kernel_ptr->meta.block_pixels);
            const uint32_t grid_y = CEIL_DIV(out_channels, kernel_ptr->meta.block_oc);

            CutlassConv2dConfig cfg{
                .N = static_cast<int32_t>(batch),
                .H = static_cast<int32_t>(in_h),
                .W = static_cast<int32_t>(in_w),
                .C = static_cast<int32_t>(in_channels),
                .K = static_cast<int32_t>(out_channels),
                .R = static_cast<int32_t>(kernel_h),
                .S = static_cast<int32_t>(kernel_w),
                .pad_h = static_cast<int32_t>(padding[0]),
                .pad_w = static_cast<int32_t>(padding[1]),
                .stride_h = static_cast<int32_t>(stride[0]),
                .stride_w = static_cast<int32_t>(stride[1]),
                .dilation_h = static_cast<int32_t>(dilation[0]),
                .dilation_w = static_cast<int32_t>(dilation[1]),
                .groups = 1,
                .split_k_slices = 1,
            };

            pi::tensorlib::KernelLaunchArguments arguments{};
            pi::tensorlib::KernelDataArg cfg_storage{};
            cfg_storage.bytes.resize(sizeof(CutlassConv2dConfig));
            std::memcpy(cfg_storage.bytes.data(), &cfg, sizeof(CutlassConv2dConfig));

            const float alpha = 1.0f;
            const float beta = 0.0f;

            if (cutlass_implementation == CutlassConv2dImplementation::Cutlass3)
            {
                const fbamtrain::kernel_gen::nv::GeneratedConv2dRequest request{
                    .operation = fbamtrain::kernel_gen::nv::GeneratedConv2dOperation::Fprop,
                    .dtype = ToGeneratedConv2dDType(dtype),
                    .n = static_cast<int32_t>(batch),
                    .h = static_cast<int32_t>(in_h),
                    .w = static_cast<int32_t>(in_w),
                    .c = static_cast<int32_t>(in_channels),
                    .k = static_cast<int32_t>(out_channels),
                    .r = static_cast<int32_t>(kernel_h),
                    .s = static_cast<int32_t>(kernel_w),
                    .pad_h = static_cast<int32_t>(padding[0]),
                    .pad_w = static_cast<int32_t>(padding[1]),
                    .stride_h = static_cast<int32_t>(stride[0]),
                    .stride_w = static_cast<int32_t>(stride[1]),
                    .dilation_h = static_cast<int32_t>(dilation[0]),
                    .dilation_w = static_cast<int32_t>(dilation[1]),
                    .groups = 1,
                    .ptr_a = input_ptr,
                    .ptr_b = weight_ptr,
                    .ptr_c = output_ptr,
                    .ptr_d = output_ptr,
                    .alpha = alpha,
                    .beta = beta,
                    .device_ordinal = static_cast<int>(device_ordinal),
                };
                return PrepareGeneratedConv2dLaunchArguments(kernel_ptr->meta, kernel_ptr->function_name, "fwd",
                                                             request);
            }

            arguments.args.emplace_back(std::move(cfg_storage));
            arguments.args.emplace_back(input_ptr);
            arguments.args.emplace_back(weight_ptr);
            arguments.args.emplace_back(output_ptr);
            if (use_fp16_accumulation && dtype == pi::tensorlib::DataType::FLOAT16)
            {
                const uint16_t alpha_half = pi::tensorlib::utils::Fp16FromFp32(alpha);
                const uint16_t beta_half = pi::tensorlib::utils::Fp16FromFp32(beta);

                arguments.args.emplace_back(alpha_half);
                arguments.args.emplace_back(beta_half);
            }
            else
            {
                arguments.args.emplace_back(alpha);
                arguments.args.emplace_back(beta);
            }
            arguments.args.emplace_back(static_cast<void *>(nullptr));

            arguments.grid_dim_x = grid_x;
            arguments.grid_dim_y = grid_y;
            arguments.grid_dim_z = 1;
            arguments.block_dim_x = block_threads;
            arguments.block_dim_y = 1;
            arguments.block_dim_z = 1;
            arguments.shared_mem_bytes = kernel_ptr->shared_mem_bytes;
            arguments.device_ordinal = static_cast<int>(device_ordinal);
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel_bin.data, kernel_bin.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel_bin.data, kernel_bin.size),
    };
}

pi::tensorlib::ComputeKernelDescriptor CreateCutlassConv2dDgradComputeKernelDescriptor(
    const pi::tensorlib::DataType dtype, const std::array<uint32_t, 2> &stride,
    const std::array<uint32_t, 2> &padding, const std::array<uint32_t, 2> &dilation,
    const kernel_bin_t<kernel_meta_conv2d_cutlass_t> &kernel_bin,
    const CutlassConv2dImplementation cutlass_implementation)
{
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_bin.kernel_name,
        .function_name = kernel_bin.function_name,
        .expected_arg_count = kernel_bin.arg_count,
        .argument_provider = [stride, padding, dilation, kernel_ptr = &kernel_bin, dtype,
                              cutlass_implementation](
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 2)
            {
                throw std::runtime_error("conv2d dgrad expects two inputs (upstream, weight)");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("conv2d dgrad expects a single output tensor");
            }

            const auto &upstream = inputs[0];
            const auto &weight = inputs[1];
            const auto &grad_input = outputs[0];

            if (upstream->dtype() != dtype || weight->dtype() != dtype || grad_input->dtype() != dtype)
            {
                throw std::runtime_error("conv2d dgrad currently supports BFLOAT16 or FLOAT16 tensors");
            }

            if (upstream->device().device_type != pi::tensorlib::DeviceType::GPU ||
                weight->device().device_type != pi::tensorlib::DeviceType::GPU ||
                grad_input->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("conv2d dgrad requires GPU tensors");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("conv2d_dgrad", {upstream, weight, grad_input});

            if (upstream->shape().ndims() != 4 || weight->shape().ndims() != 4 || grad_input->shape().ndims() != 4)
            {
                throw std::runtime_error("conv2d dgrad expects 4D upstream, weight, and grad_input tensors");
            }

            const auto batch64 = grad_input->shape()[0];
            const auto in_h64 = grad_input->shape()[1];
            const auto in_w64 = grad_input->shape()[2];
            const auto in_channels64 = grad_input->shape()[3];

            const auto out_channels64 = weight->shape()[0];
            const auto kernel_h64 = weight->shape()[1];
            const auto kernel_w64 = weight->shape()[2];
            const auto kernel_in_channels64 = weight->shape()[3];

            if (kernel_in_channels64 != in_channels64)
            {
                throw std::runtime_error("conv2d dgrad weight in_channels must match grad_input channels");
            }
            if (kernel_h64 == 0 || kernel_w64 == 0)
            {
                throw std::runtime_error("conv2d dgrad kernel dimensions must be greater than zero");
            }

            const uint64_t effective_kernel_h =
                1 + static_cast<uint64_t>(dilation[0]) * (kernel_h64 - 1);
            const uint64_t effective_kernel_w =
                1 + static_cast<uint64_t>(dilation[1]) * (kernel_w64 - 1);

            const int64_t numerator_h =
                static_cast<int64_t>(in_h64 + 2 * padding[0]) - static_cast<int64_t>(effective_kernel_h);
            const int64_t numerator_w =
                static_cast<int64_t>(in_w64 + 2 * padding[1]) - static_cast<int64_t>(effective_kernel_w);
            if (numerator_h < 0 || numerator_w < 0)
            {
                throw std::runtime_error("conv2d dgrad configuration results in negative output spatial size");
            }
            if (numerator_h % static_cast<int64_t>(stride[0]) != 0 ||
                numerator_w % static_cast<int64_t>(stride[1]) != 0)
            {
                throw std::runtime_error("conv2d dgrad configuration must produce integer output spatial dimensions");
            }

            const auto out_h64 =
                static_cast<uint64_t>(numerator_h / static_cast<int64_t>(stride[0]) + 1);
            const auto out_w64 =
                static_cast<uint64_t>(numerator_w / static_cast<int64_t>(stride[1]) + 1);

            if (upstream->shape()[0] != batch64 || upstream->shape()[1] != out_h64 ||
                upstream->shape()[2] != out_w64 || upstream->shape()[3] != out_channels64)
            {
                throw std::runtime_error("conv2d dgrad upstream shape mismatch");
            }

            if (kernel_ptr->meta.in_channels != 0 && in_channels64 != kernel_ptr->meta.in_channels)
            {
                throw std::runtime_error("conv2d dgrad configuration not covered by compiled Cutlass kernel set");
            }
            if (kernel_h64 != kernel_ptr->meta.kernel_h || kernel_w64 != kernel_ptr->meta.kernel_w)
            {
                throw std::runtime_error("conv2d dgrad configuration not covered by compiled Cutlass kernel set");
            }

            auto to_u32 = [](const uint64_t value, const char *what) -> uint32_t
            {
                if (value > std::numeric_limits<uint32_t>::max())
                {
                    std::stringstream ss;
                    ss << what << " exceeds supported range";
                    throw std::runtime_error(ss.str());
                }
                return static_cast<uint32_t>(value);
            };

            const auto batch = to_u32(batch64, "conv2d dgrad batch size");
            const auto in_channels = to_u32(in_channels64, "conv2d dgrad input channels");
            const auto in_h = to_u32(in_h64, "conv2d dgrad input height");
            const auto in_w = to_u32(in_w64, "conv2d dgrad input width");

            const auto out_channels = to_u32(out_channels64, "conv2d dgrad output channels");
            const auto kernel_h = to_u32(kernel_h64, "conv2d dgrad kernel height");
            const auto kernel_w = to_u32(kernel_w64, "conv2d dgrad kernel width");

            const auto out_h = to_u32(out_h64, "conv2d dgrad output height");
            const auto out_w = to_u32(out_w64, "conv2d dgrad output width");

            const uint64_t total_pixels64 = batch64 * in_h64 * in_w64;
            const auto total_pixels = to_u32(total_pixels64, "conv2d dgrad total input elements");

            CutlassConv2dConfig cfg{
                .N = static_cast<int32_t>(batch),
                .H = static_cast<int32_t>(in_h),
                .W = static_cast<int32_t>(in_w),
                .C = static_cast<int32_t>(in_channels),
                .K = static_cast<int32_t>(out_channels),
                .R = static_cast<int32_t>(kernel_h),
                .S = static_cast<int32_t>(kernel_w),
                .pad_h = static_cast<int32_t>(padding[0]),
                .pad_w = static_cast<int32_t>(padding[1]),
                .stride_h = static_cast<int32_t>(stride[0]),
                .stride_w = static_cast<int32_t>(stride[1]),
                .dilation_h = static_cast<int32_t>(dilation[0]),
                .dilation_w = static_cast<int32_t>(dilation[1]),
                .groups = 1,
                .split_k_slices = 1,
            };

            pi::tensorlib::KernelLaunchArguments arguments{};
            pi::tensorlib::KernelDataArg cfg_storage{};
            cfg_storage.bytes.resize(sizeof(CutlassConv2dConfig));
            std::memcpy(cfg_storage.bytes.data(), &cfg, sizeof(CutlassConv2dConfig));

            const float alpha = 1.0f;
            const float beta = 0.0f;

            if (cutlass_implementation == CutlassConv2dImplementation::Cutlass3)
            {
                const fbamtrain::kernel_gen::nv::GeneratedConv2dRequest request{
                    .operation = fbamtrain::kernel_gen::nv::GeneratedConv2dOperation::Dgrad,
                    .dtype = ToGeneratedConv2dDType(dtype),
                    .n = static_cast<int32_t>(batch),
                    .h = static_cast<int32_t>(in_h),
                    .w = static_cast<int32_t>(in_w),
                    .c = static_cast<int32_t>(in_channels),
                    .k = static_cast<int32_t>(out_channels),
                    .r = static_cast<int32_t>(kernel_h),
                    .s = static_cast<int32_t>(kernel_w),
                    .pad_h = static_cast<int32_t>(padding[0]),
                    .pad_w = static_cast<int32_t>(padding[1]),
                    .stride_h = static_cast<int32_t>(stride[0]),
                    .stride_w = static_cast<int32_t>(stride[1]),
                    .dilation_h = static_cast<int32_t>(dilation[0]),
                    .dilation_w = static_cast<int32_t>(dilation[1]),
                    .groups = 1,
                    .ptr_a = upstream->dataptr(),
                    .ptr_b = weight->dataptr(),
                    .ptr_c = grad_input->dataptr(),
                    .ptr_d = grad_input->dataptr(),
                    .alpha = alpha,
                    .beta = beta,
                    .device_ordinal = static_cast<int>(device_ordinal),
                };
                return PrepareGeneratedConv2dLaunchArguments(kernel_ptr->meta, kernel_ptr->function_name, "dgrad",
                                                             request);
            }

            arguments.args.emplace_back(std::move(cfg_storage));
            arguments.args.emplace_back(upstream->dataptr());
            arguments.args.emplace_back(weight->dataptr());
            arguments.args.emplace_back(grad_input->dataptr());
            arguments.args.emplace_back(alpha);
            arguments.args.emplace_back(beta);
            arguments.args.emplace_back(static_cast<void *>(nullptr));

            const uint32_t block_threads = kernel_ptr->num_warps * CUDA_WARP_SIZE;
            const uint32_t grid_x = CEIL_DIV(total_pixels, kernel_ptr->meta.block_pixels);
            const uint32_t grid_y = CEIL_DIV(in_channels, kernel_ptr->meta.block_oc);

            arguments.grid_dim_x = grid_x;
            arguments.grid_dim_y = grid_y;
            arguments.grid_dim_z = 1;
            arguments.block_dim_x = block_threads;
            arguments.block_dim_y = 1;
            arguments.block_dim_z = 1;
            arguments.shared_mem_bytes = kernel_ptr->shared_mem_bytes;
            arguments.device_ordinal = static_cast<int>(device_ordinal);
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel_bin.data, kernel_bin.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel_bin.data, kernel_bin.size),
    };
}

pi::tensorlib::ComputeKernelDescriptor CreateCutlassConv2dWgradComputeKernelDescriptor(
    const pi::tensorlib::DataType dtype, const std::array<uint32_t, 2> &stride,
    const std::array<uint32_t, 2> &padding, const std::array<uint32_t, 2> &dilation,
    const bool accumulate_output, const kernel_bin_t<kernel_meta_conv2d_cutlass_t> &kernel_bin,
    const CutlassConv2dImplementation cutlass_implementation)
{
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_bin.kernel_name,
        .function_name = kernel_bin.function_name,
        .expected_arg_count = kernel_bin.arg_count,
        .argument_provider = [stride, padding, dilation, accumulate_output, cutlass_implementation,
                              kernel_ptr = &kernel_bin, dtype](
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 2)
            {
                throw std::runtime_error("conv2d wgrad expects two inputs (input, upstream)");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("conv2d wgrad expects a single output tensor");
            }

            const auto &input = inputs[0];
            const auto &upstream = inputs[1];
            const auto &grad_weight = outputs[0];

            if (input->dtype() != dtype || upstream->dtype() != dtype || grad_weight->dtype() != dtype)
            {
                throw std::runtime_error("conv2d wgrad currently supports BFLOAT16 or FLOAT16 tensors");
            }

            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                upstream->device().device_type != pi::tensorlib::DeviceType::GPU ||
                grad_weight->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("conv2d wgrad requires GPU tensors");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("conv2d_wgrad", {input, upstream, grad_weight});

            if (input->shape().ndims() != 4 || upstream->shape().ndims() != 4 || grad_weight->shape().ndims() != 4)
            {
                throw std::runtime_error("conv2d wgrad expects 4D input, upstream, and grad_weight tensors");
            }

            const auto batch64 = input->shape()[0];
            const auto in_h64 = input->shape()[1];
            const auto in_w64 = input->shape()[2];
            const auto in_channels64 = input->shape()[3];

            const auto out_channels64 = grad_weight->shape()[0];
            const auto kernel_h64 = grad_weight->shape()[1];
            const auto kernel_w64 = grad_weight->shape()[2];
            const auto kernel_in_channels64 = grad_weight->shape()[3];

            if (kernel_in_channels64 != in_channels64)
            {
                throw std::runtime_error("conv2d wgrad grad_weight in_channels must match input channels");
            }
            if (kernel_h64 == 0 || kernel_w64 == 0)
            {
                throw std::runtime_error("conv2d wgrad kernel dimensions must be greater than zero");
            }

            const uint64_t effective_kernel_h =
                1 + static_cast<uint64_t>(dilation[0]) * (kernel_h64 - 1);
            const uint64_t effective_kernel_w =
                1 + static_cast<uint64_t>(dilation[1]) * (kernel_w64 - 1);

            const int64_t numerator_h =
                static_cast<int64_t>(in_h64 + 2 * padding[0]) - static_cast<int64_t>(effective_kernel_h);
            const int64_t numerator_w =
                static_cast<int64_t>(in_w64 + 2 * padding[1]) - static_cast<int64_t>(effective_kernel_w);
            if (numerator_h < 0 || numerator_w < 0)
            {
                throw std::runtime_error("conv2d wgrad configuration results in negative output spatial size");
            }
            if (numerator_h % static_cast<int64_t>(stride[0]) != 0 ||
                numerator_w % static_cast<int64_t>(stride[1]) != 0)
            {
                throw std::runtime_error("conv2d wgrad configuration must produce integer output spatial dimensions");
            }

            const auto out_h64 =
                static_cast<uint64_t>(numerator_h / static_cast<int64_t>(stride[0]) + 1);
            const auto out_w64 =
                static_cast<uint64_t>(numerator_w / static_cast<int64_t>(stride[1]) + 1);

            if (upstream->shape()[0] != batch64 || upstream->shape()[1] != out_h64 ||
                upstream->shape()[2] != out_w64 || upstream->shape()[3] != out_channels64)
            {
                throw std::runtime_error("conv2d wgrad upstream shape mismatch");
            }

            if (kernel_ptr->meta.in_channels != 0 && in_channels64 != kernel_ptr->meta.in_channels)
            {
                throw std::runtime_error("conv2d wgrad configuration not covered by compiled Cutlass kernel set");
            }
            if (kernel_h64 != kernel_ptr->meta.kernel_h || kernel_w64 != kernel_ptr->meta.kernel_w)
            {
                throw std::runtime_error("conv2d wgrad configuration not covered by compiled Cutlass kernel set");
            }

            auto to_u32 = [](const uint64_t value, const char *what) -> uint32_t
            {
                if (value > std::numeric_limits<uint32_t>::max())
                {
                    std::stringstream ss;
                    ss << what << " exceeds supported range";
                    throw std::runtime_error(ss.str());
                }
                return static_cast<uint32_t>(value);
            };

            const auto batch = to_u32(batch64, "conv2d wgrad batch size");
            const auto in_channels = to_u32(in_channels64, "conv2d wgrad input channels");
            const auto in_h = to_u32(in_h64, "conv2d wgrad input height");
            const auto in_w = to_u32(in_w64, "conv2d wgrad input width");

            const auto out_channels = to_u32(out_channels64, "conv2d wgrad output channels");
            const auto kernel_h = to_u32(kernel_h64, "conv2d wgrad kernel height");
            const auto kernel_w = to_u32(kernel_w64, "conv2d wgrad kernel width");

            const auto out_h = to_u32(out_h64, "conv2d wgrad output height");
            const auto out_w = to_u32(out_w64, "conv2d wgrad output width");

            const uint64_t gemm_m64 = out_channels64;
            const uint64_t gemm_n64 = kernel_h64 * kernel_w64 * in_channels64;
            const auto gemm_m = to_u32(gemm_m64, "conv2d wgrad gemm M dimension");
            const auto gemm_n = to_u32(gemm_n64, "conv2d wgrad gemm N dimension");

            const float alpha = 1.0f;
            const float beta = accumulate_output ? 1.0f : 0.0f;

            if (cutlass_implementation == CutlassConv2dImplementation::Cutlass2)
            {
                CutlassConv2dConfig cfg{
                    .N = static_cast<int32_t>(batch),
                    .H = static_cast<int32_t>(in_h),
                    .W = static_cast<int32_t>(in_w),
                    .C = static_cast<int32_t>(in_channels),
                    .K = static_cast<int32_t>(out_channels),
                    .R = static_cast<int32_t>(kernel_h),
                    .S = static_cast<int32_t>(kernel_w),
                    .pad_h = static_cast<int32_t>(padding[0]),
                    .pad_w = static_cast<int32_t>(padding[1]),
                    .stride_h = static_cast<int32_t>(stride[0]),
                    .stride_w = static_cast<int32_t>(stride[1]),
                    .dilation_h = static_cast<int32_t>(dilation[0]),
                    .dilation_w = static_cast<int32_t>(dilation[1]),
                    .groups = 1,
                    .split_k_slices = 1,
                };

                pi::tensorlib::KernelDataArg cfg_storage{};
                cfg_storage.bytes.resize(sizeof(CutlassConv2dConfig));
                std::memcpy(cfg_storage.bytes.data(), &cfg, sizeof(CutlassConv2dConfig));

                pi::tensorlib::KernelLaunchArguments arguments{};
                arguments.args.emplace_back(std::move(cfg_storage));
                arguments.args.emplace_back(upstream->dataptr());
                arguments.args.emplace_back(input->dataptr());
                arguments.args.emplace_back(grad_weight->dataptr());
                arguments.args.emplace_back(alpha);
                arguments.args.emplace_back(beta);
                arguments.args.emplace_back(static_cast<void *>(nullptr));

                const uint32_t block_threads = kernel_ptr->num_warps * CUDA_WARP_SIZE;
                const uint32_t grid_x = CEIL_DIV(gemm_m, kernel_ptr->meta.block_pixels);
                const uint32_t grid_y = CEIL_DIV(gemm_n, kernel_ptr->meta.block_oc);

                arguments.grid_dim_x = grid_x;
                arguments.grid_dim_y = grid_y;
                arguments.grid_dim_z = 1;
                arguments.block_dim_x = block_threads;
                arguments.block_dim_y = 1;
                arguments.block_dim_z = 1;
                arguments.shared_mem_bytes = kernel_ptr->shared_mem_bytes;
                arguments.device_ordinal = static_cast<int>(device_ordinal);
                return arguments;
            }
            else if (cutlass_implementation == CutlassConv2dImplementation::Cutlass3)
            {
                const fbamtrain::kernel_gen::nv::GeneratedConv2dRequest request{
                    .operation = fbamtrain::kernel_gen::nv::GeneratedConv2dOperation::Wgrad,
                    .dtype = ToGeneratedConv2dDType(dtype),
                    .n = static_cast<int32_t>(batch),
                    .h = static_cast<int32_t>(in_h),
                    .w = static_cast<int32_t>(in_w),
                    .c = static_cast<int32_t>(in_channels),
                    .k = static_cast<int32_t>(out_channels),
                    .r = static_cast<int32_t>(kernel_h),
                    .s = static_cast<int32_t>(kernel_w),
                    .pad_h = static_cast<int32_t>(padding[0]),
                    .pad_w = static_cast<int32_t>(padding[1]),
                    .stride_h = static_cast<int32_t>(stride[0]),
                    .stride_w = static_cast<int32_t>(stride[1]),
                    .dilation_h = static_cast<int32_t>(dilation[0]),
                    .dilation_w = static_cast<int32_t>(dilation[1]),
                    .groups = 1,
                    .ptr_a = upstream->dataptr(),
                    .ptr_b = input->dataptr(),
                    .ptr_c = grad_weight->dataptr(),
                    .ptr_d = grad_weight->dataptr(),
                    .alpha = alpha,
                    .beta = beta,
                    .device_ordinal = static_cast<int>(device_ordinal),
                };
                return PrepareGeneratedConv2dLaunchArguments(kernel_ptr->meta, kernel_ptr->function_name, "wgrad",
                                                             request);
            }

            throw std::runtime_error("unsupported CUTLASS conv2d wgrad implementation");
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel_bin.data, kernel_bin.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel_bin.data, kernel_bin.size),
    };
}

void Conv2dImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &entry : execution_plan.entries)
    {
        if (entry.op_type == pi::tensorlib::OpType::CONV2D)
        {
            const auto attrs = ParseConv2dAttributes(entry);
            const auto &stride = attrs.stride;
            const auto &padding = attrs.padding;
            const auto &dilation = attrs.dilation;

            const auto &input_tensor = entry.inputs[0];
            const auto &weight_tensor = entry.inputs[1];
            const auto &output_tensor = entry.outputs[0];
            const auto dtype = input_tensor->dtype();
            if (!IsSupportedHalfType(dtype) || weight_tensor->dtype() != dtype || output_tensor->dtype() != dtype)
            {
                throw std::runtime_error("conv2d requires BFLOAT16 or FLOAT16 tensors");
            }

            const auto in_channels = static_cast<uint32_t>(input_tensor->shape()[3]);
            if (static_cast<uint32_t>(weight_tensor->shape()[3]) != in_channels)
            {
                throw std::runtime_error("conv2d weight tensor channels mismatch expected HWIO layout");
            }

            const auto out_channels = static_cast<uint32_t>(weight_tensor->shape()[0]);
            const auto kernel_h = static_cast<uint32_t>(weight_tensor->shape()[1]);
            const auto kernel_w = static_cast<uint32_t>(weight_tensor->shape()[2]);
            const auto batch = static_cast<uint32_t>(input_tensor->shape()[0]);
            const auto input_h = static_cast<uint32_t>(input_tensor->shape()[1]);
            const auto input_w = static_cast<uint32_t>(input_tensor->shape()[2]);

            const bool has_bias = entry.inputs.size() == 3;
            if (has_bias && entry.inputs[2]->dtype() != dtype)
            {
                throw std::runtime_error("conv2d bias dtype must match input dtype");
            }
            bool requested_fp16_accumulation = false;
            if (const auto attr = entry.attributes.find("use_fp16_conv_acc");
                attr != entry.attributes.end() && attr->second.type() == typeid(bool))
            {
                requested_fp16_accumulation = std::any_cast<bool>(attr->second);
            }
            if (dtype == pi::tensorlib::DataType::FLOAT16)
            {
                RequireFp16AccumulationSupported(requested_fp16_accumulation);
            }
            const bool use_fp16_accumulation =
                (dtype == pi::tensorlib::DataType::FLOAT16) ? requested_fp16_accumulation : false;
            const auto selected =
                SelectConv2dKernel(in_channels, out_channels, kernel_h, kernel_w, stride[0], stride[1], dilation[0],
                                   dilation[1], padding, batch, input_h, input_w, dtype, has_bias,
                                   use_fp16_accumulation);

            entry.op_type = std::nullopt;
            if (selected.backend == Conv2dKernelBackend::Cutlass)
            {
                if (selected.cutlass_kernel == nullptr)
                {
                    throw std::runtime_error("conv2d cutlass kernel metadata missing");
                }
                entry.kernel_descriptor =
                    CreateCutlassConv2dComputeKernelDescriptor(
                        dtype, stride, padding, dilation, *selected.cutlass_kernel, use_fp16_accumulation,
                        selected.cutlass_implementation);
            }
            else
            {
                if (selected.triton_kernel == nullptr)
                {
                    throw std::runtime_error("conv2d triton kernel metadata missing");
                }
                entry.kernel_descriptor =
                    CreateTritonConv2dComputeKernelDescriptor(dtype, stride, padding, dilation, *selected.triton_kernel);
            }
            const uint64_t out_h = output_tensor->shape()[1];
            const uint64_t out_w = output_tensor->shape()[2];
            const uint64_t out_c = output_tensor->shape()[3];
            uint64_t flops =
                2u * out_h * out_w * out_c * static_cast<uint64_t>(kernel_h) * static_cast<uint64_t>(kernel_w) *
                static_cast<uint64_t>(in_channels);
            if (has_bias)
            {
                flops += out_h * out_w * out_c; // bias add
            }
            entry.flop_estimate = flops;
        }
        else if (entry.op_type == pi::tensorlib::OpType::CONV2D_DGRAD)
        {
            const auto attrs = ParseConv2dAttributes(entry);
            const auto &stride = attrs.stride;
            const auto &padding = attrs.padding;
            const auto &dilation = attrs.dilation;

            if (entry.inputs.size() != 2 || entry.outputs.size() != 1)
            {
                throw std::runtime_error("conv2d dgrad expects two inputs and one output");
            }

            const auto &upstream = entry.inputs[0];
            const auto &weight = entry.inputs[1];
            const auto &grad_input = entry.outputs[0];

            const auto dtype = upstream->dtype();
            if (!IsSupportedHalfType(dtype) || weight->dtype() != dtype || grad_input->dtype() != dtype)
            {
                throw std::runtime_error("conv2d dgrad requires BFLOAT16 or FLOAT16 tensors");
            }

            const auto in_channels = static_cast<uint32_t>(grad_input->shape()[3]);
            if (static_cast<uint32_t>(weight->shape()[3]) != in_channels)
            {
                throw std::runtime_error("conv2d dgrad weight tensor channels mismatch expected HWIO layout");
            }

            const auto out_channels = static_cast<uint32_t>(weight->shape()[0]);
            if (static_cast<uint32_t>(upstream->shape()[3]) != out_channels)
            {
                throw std::runtime_error("conv2d dgrad upstream channels must match weight out_channels");
            }

            const auto kernel_h = static_cast<uint32_t>(weight->shape()[1]);
            const auto kernel_w = static_cast<uint32_t>(weight->shape()[2]);
            const auto batch_u32 = static_cast<uint32_t>(grad_input->shape()[0]);
            const auto input_h_u32 = static_cast<uint32_t>(grad_input->shape()[1]);
            const auto input_w_u32 = static_cast<uint32_t>(grad_input->shape()[2]);

            const auto selected =
                SelectConv2dDgradKernel(in_channels, out_channels, kernel_h, kernel_w, stride[0], stride[1],
                                        dilation[0], dilation[1], padding, batch_u32, input_h_u32, input_w_u32, dtype);

            entry.op_type = std::nullopt;
            if (selected.backend == Conv2dKernelBackend::Cutlass)
            {
                if (selected.cutlass_kernel == nullptr)
                {
                    throw std::runtime_error("conv2d dgrad cutlass kernel metadata missing");
                }
                entry.kernel_descriptor =
                    CreateCutlassConv2dDgradComputeKernelDescriptor(dtype, stride, padding, dilation,
                                                                    *selected.cutlass_kernel,
                                                                    selected.cutlass_implementation);
            }
            else
            {
                if (selected.triton_kernel == nullptr)
                {
                    throw std::runtime_error("conv2d dgrad triton kernel metadata missing");
                }
                entry.kernel_descriptor =
                    CreateTritonConv2dDgradComputeKernelDescriptor(dtype, stride, padding, dilation,
                                                                   *selected.triton_kernel);
            }

            const uint64_t batch = upstream->shape()[0];
            const uint64_t out_h = upstream->shape()[1];
            const uint64_t out_w = upstream->shape()[2];
            uint64_t flops =
                2u * batch * out_h * out_w * static_cast<uint64_t>(out_channels) *
                static_cast<uint64_t>(kernel_h) * static_cast<uint64_t>(kernel_w) *
                static_cast<uint64_t>(in_channels);
            entry.flop_estimate = flops;
        }
        else if (entry.op_type == pi::tensorlib::OpType::CONV2D_WGRAD)
        {
            const auto attrs = ParseConv2dAttributes(entry);
            const auto &stride = attrs.stride;
            const auto &padding = attrs.padding;
            const auto &dilation = attrs.dilation;
            const bool accumulate_output = attrs.accumulate_output;

            if (entry.inputs.size() != 2 || entry.outputs.size() != 1)
            {
                throw std::runtime_error("conv2d wgrad expects two inputs and one output");
            }

            const auto &input = entry.inputs[0];
            const auto &upstream = entry.inputs[1];
            const auto &grad_weight = entry.outputs[0];

            const auto dtype = input->dtype();
            if (!IsSupportedHalfType(dtype) || upstream->dtype() != dtype || grad_weight->dtype() != dtype)
            {
                throw std::runtime_error("conv2d wgrad requires BFLOAT16 or FLOAT16 tensors");
            }

            const auto in_channels = static_cast<uint32_t>(input->shape()[3]);
            if (static_cast<uint32_t>(grad_weight->shape()[3]) != in_channels)
            {
                throw std::runtime_error("conv2d wgrad weight tensor channels mismatch expected HWIO layout");
            }

            const auto out_channels = static_cast<uint32_t>(grad_weight->shape()[0]);
            if (static_cast<uint32_t>(upstream->shape()[3]) != out_channels)
            {
                throw std::runtime_error("conv2d wgrad upstream channels must match weight out_channels");
            }

            const auto kernel_h = static_cast<uint32_t>(grad_weight->shape()[1]);
            const auto kernel_w = static_cast<uint32_t>(grad_weight->shape()[2]);
            const auto batch_u32 = static_cast<uint32_t>(input->shape()[0]);
            const auto input_h_u32 = static_cast<uint32_t>(input->shape()[1]);
            const auto input_w_u32 = static_cast<uint32_t>(input->shape()[2]);

            const auto selected =
                SelectConv2dWgradKernel(in_channels, out_channels, kernel_h, kernel_w, stride[0], stride[1],
                                        dilation[0], dilation[1], padding, batch_u32, input_h_u32, input_w_u32, dtype);

            entry.op_type = std::nullopt;
            if (selected.backend == Conv2dKernelBackend::Cutlass)
            {
                if (selected.cutlass_kernel == nullptr)
                {
                    throw std::runtime_error("conv2d wgrad cutlass kernel metadata missing");
                }
                entry.kernel_descriptor =
                    CreateCutlassConv2dWgradComputeKernelDescriptor(dtype, stride, padding, dilation,
                                                                    accumulate_output,
                                                                    *selected.cutlass_kernel,
                                                                    selected.cutlass_implementation);
            }
            else
            {
                if (selected.triton_kernel == nullptr)
                {
                    throw std::runtime_error("conv2d wgrad triton kernel metadata missing");
                }
                entry.kernel_descriptor =
                    CreateTritonConv2dWgradComputeKernelDescriptor(dtype, stride, padding, dilation,
                                                                   accumulate_output,
                                                                   *selected.triton_kernel);
            }

            const uint64_t batch = upstream->shape()[0];
            const uint64_t out_h = upstream->shape()[1];
            const uint64_t out_w = upstream->shape()[2];
            uint64_t flops =
                2u * batch * out_h * out_w * static_cast<uint64_t>(out_channels) *
                static_cast<uint64_t>(kernel_h) * static_cast<uint64_t>(kernel_w) *
                static_cast<uint64_t>(in_channels);
            entry.flop_estimate = flops;
        }
    }
}
