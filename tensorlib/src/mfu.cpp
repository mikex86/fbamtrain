#include "mfu.h"

#include <algorithm>
#include <cctype>
#include <string>

#if PI_TENSORLIB_ENABLE_CUDA
#include <cuda_runtime.h>
#elif PI_TENSORLIB_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

namespace pi::tensorlib
{
namespace
{
struct GpuEntry
{
    const char *name;
    const PromisedPerf *perf;
    float cores_override;
    float clock_override;
};

constexpr PromisedPerf VOLTA{"VOLTA", 125.0f, -1.0f, 125.0f, -1.0f, 640.0f, 1530.0f};
constexpr PromisedPerf AMPERE_DC{"AMPERE_DC", 156.0f, 312.0f, 312.0f, 312.0f, 432.0f, 1410.0f};
constexpr PromisedPerf AMPERE_CONSUMER{"AMPERE_CONSUMER", 40.0f, 80.0f, 80.0f, 160.0f, 336.0f, 1860.0f};
constexpr PromisedPerf HOPPER{"HOPPER", 378.0f, 756.0f, 756.0f, 756.0f, 456.0f, 1620.0f};
constexpr PromisedPerf ADA{"ADA", 82.6f, 165.2f, 165.2f, 330.3f, 512.0f, 2520.0f};
constexpr PromisedPerf RDNA3_RX_7800_XT{"RDNA3_RX7800XT", -1.0f, -1.0f, 74.6f, 74.6f, 1.0f, 1.0f};

constexpr GpuEntry GPU_DB[] = {
    {"Tesla V100-SXM2-16GB", &VOLTA, 640.0f, 1530.0f},
    {"Tesla V100-PCIE-32GB", &VOLTA, 640.0f, 1530.0f},
    {"NVIDIA A100-PCIE-40GB", &AMPERE_DC, 432.0f, 1410.0f},
    {"NVIDIA A100-PCIE-80GB", &AMPERE_DC, 432.0f, 1410.0f},
    {"NVIDIA A100-SXM4-40GB", &AMPERE_DC, 432.0f, 1410.0f},
    {"NVIDIA A100-SXM4-80GB", &AMPERE_DC, 432.0f, 1410.0f},
    {"NVIDIA RTX A2000", &AMPERE_CONSUMER, 104.0f, 1200.0f},
    {"NVIDIA RTX A4000", &AMPERE_CONSUMER, 192.0f, 1560.0f},
    {"NVIDIA RTX A4500", &AMPERE_CONSUMER, 224.0f, 1650.0f},
    {"NVIDIA RTX A5000", &AMPERE_CONSUMER, 256.0f, 1695.0f},
    {"NVIDIA RTX A5500", &AMPERE_CONSUMER, 320.0f, 1770.0f},
    {"NVIDIA RTX A6000", &AMPERE_CONSUMER, 336.0f, 1800.0f},
    {"NVIDIA GeForce RTX 3090 Ti", &AMPERE_CONSUMER, 336.0f, 1860.0f},
    {"NVIDIA GeForce RTX 3090", &AMPERE_CONSUMER, 328.0f, 1695.0f},
    {"NVIDIA GeForce RTX 3080 Ti", &AMPERE_CONSUMER, 320.0f, 1665.0f},
    {"NVIDIA GeForce RTX 3080", &AMPERE_CONSUMER, 272.0f, 1710.0f},
    {"NVIDIA GeForce RTX 3070 Ti", &AMPERE_CONSUMER, 192.0f, 1770.0f},
    {"NVIDIA GeForce RTX 3070", &AMPERE_CONSUMER, 184.0f, 1725.0f},
    {"NVIDIA GeForce RTX 3060 Ti", &AMPERE_CONSUMER, 152.0f, 1665.0f},
    {"NVIDIA GeForce RTX 3060", &AMPERE_CONSUMER, 112.0f, 1777.0f},
    {"NVIDIA RTX A2000 ADA", &ADA, 88.0f, 2130.0f},
    {"NVIDIA RTX A4000 ADA", &ADA, 192.0f, 2175.0f},
    {"NVIDIA RTX A4500 ADA", &ADA, 224.0f, 2580.0f},
    {"NVIDIA RTX A5000 ADA", &ADA, 400.0f, 2550.0f},
    {"NVIDIA RTX A5880 ADA", &ADA, 440.0f, 2460.0f},
    {"NVIDIA RTX A6000 ADA", &ADA, 568.0f, 2505.0f},
    {"NVIDIA GeForce RTX 4090", &ADA, 512.0f, 2520.0f},
    {"NVIDIA GeForce RTX 4080 SUPER", &ADA, 320.0f, 2550.0f},
    {"NVIDIA GeForce RTX 4080", &ADA, 304.0f, 2505.0f},
    {"NVIDIA GeForce RTX 4070 Ti SUPER", &ADA, 264.0f, 2610.0f},
    {"NVIDIA GeForce RTX 4070 Ti", &ADA, 240.0f, 2610.0f},
    {"NVIDIA GeForce RTX 4070 SUPER", &ADA, 224.0f, 2475.0f},
    {"NVIDIA GeForce RTX 4070", &ADA, 184.0f, 2475.0f},
    {"NVIDIA GeForce RTX 4060 Ti", &ADA, 136.0f, 2535.0f},
    {"NVIDIA GeForce RTX 4060", &ADA, 96.0f, 2460.0f},
    {"NVIDIA H100 PCIe", &HOPPER, 456.0f, 1620.0f},
    {"NVIDIA H100 80GB HBM3", &HOPPER, 528.0f, 1830.0f},
    {"AMD Radeon RX 7800 XT", &RDNA3_RX_7800_XT, 1.0f, 1.0f},
};

} // namespace

std::optional<float> GetPromisedTFlops(const std::string &device_name, const PrecisionMode precision)
{
    const GpuEntry *match = nullptr;
    for (const auto &entry : GPU_DB)
    {
        if (device_name == entry.name)
        {
            match = &entry;
            break;
        }
    }
    if (match == nullptr)
    {
        return std::nullopt;
    }

    const PromisedPerf *perf = match->perf;
    float baseline = -1.0f;
    switch (precision)
    {
    case PrecisionMode::FP32:
        baseline = perf->tf_32;
        break;
    case PrecisionMode::FP16:
        baseline = perf->fp16_32;
        break;
    case PrecisionMode::FP16_ACC16:
        baseline = perf->fp16_16 >= 0.0f ? perf->fp16_16 : perf->fp16_32;
        break;
    case PrecisionMode::BF16:
        baseline = perf->bf16_32;
        break;
    }

    if (baseline < 0.0f)
    {
        return std::nullopt;
    }

    const float cores_scale = match->cores_override / perf->cores;
    const float clock_scale = match->clock_override / perf->clock_mhz;
    return baseline * cores_scale * clock_scale;
}

std::optional<float> GetPromisedTFlops(const int device_ordinal, const PrecisionMode precision)
{
#if PI_TENSORLIB_ENABLE_CUDA
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_ordinal) != cudaSuccess)
    {
        return std::nullopt;
    }
    return GetPromisedTFlops(prop.name, precision);
#elif PI_TENSORLIB_ENABLE_HIP
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, device_ordinal) != hipSuccess)
    {
        return std::nullopt;
    }
    return GetPromisedTFlops(prop.name, precision);
#else
    (void)device_ordinal;
    (void)precision;
    return std::nullopt;
#endif
}

} // namespace pi::tensorlib
