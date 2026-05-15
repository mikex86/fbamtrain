#pragma once

#include <kernels/kernel_binaries.h>

#include <activation.h>
#include <launch_utils.h>
#include <tensorlib.h>

#include <cstdlib>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

inline bool IsSupportedHalfType(const pi::tensorlib::DataType dtype)
{
    return dtype == pi::tensorlib::DataType::BFLOAT16 || dtype == pi::tensorlib::DataType::FLOAT16;
}

inline bool IsSupportedAddType(const pi::tensorlib::DataType dtype)
{
    return dtype == pi::tensorlib::DataType::BFLOAT16 || dtype == pi::tensorlib::DataType::FLOAT16 ||
           dtype == pi::tensorlib::DataType::FLOAT32;
}

inline bool IsEnvFlagEnabled(const char *name)
{
    const char *env = std::getenv(name);
    return env != nullptr && env[0] != '\0';
}

inline void RequireFp16AccumulationSupported(const bool use_fp16_accumulation)
{
#if PI_TENSORLIB_ENABLE_HIP && !PI_TENSORLIB_ENABLE_CUDA
    if (use_fp16_accumulation)
    {
        throw std::runtime_error("FP16 accumulation is not supported on AMD because it is not necessary.");
    }
#else
    (void)use_fp16_accumulation;
#endif
}

inline std::optional<std::string_view> GetEnvValue(const char *name)
{
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0')
    {
        return std::nullopt;
    }
    return std::string_view(env);
}

inline std::string_view KernelSuffixForHalf(const pi::tensorlib::DataType dtype)
{
    switch (dtype)
    {
        case pi::tensorlib::DataType::BFLOAT16:
            return "bf16";
        case pi::tensorlib::DataType::FLOAT16:
            return "fp16";
        default:
            throw std::runtime_error("Unsupported half precision data type");
    }
}

inline std::string_view KernelSuffixForAdd(const pi::tensorlib::DataType dtype)
{
    switch (dtype)
    {
        case pi::tensorlib::DataType::BFLOAT16:
            return "bf16";
        case pi::tensorlib::DataType::FLOAT16:
            return "fp16";
        case pi::tensorlib::DataType::FLOAT32:
            return "fp32";
        default:
            throw std::runtime_error("Unsupported add data type");
    }
}

template <typename Meta>
[[nodiscard]] kernel_bin_t<Meta> SelectKernelForHalf(const kernel_bin_t<Meta> &bf16_kernel,
                                              const kernel_bin_t<Meta> &fp16_kernel,
                                              const pi::tensorlib::DataType dtype)
{
    switch (dtype)
    {
        case pi::tensorlib::DataType::BFLOAT16:
            return bf16_kernel;
        case pi::tensorlib::DataType::FLOAT16:
            return fp16_kernel;
        default:
            throw std::runtime_error("Unsupported half precision data type");
    }
}

template <typename Meta>
[[nodiscard]] kernel_bin_t<Meta> SelectKernelForAdd(const kernel_bin_t<Meta> &bf16_kernel,
                                                    const kernel_bin_t<Meta> &fp16_kernel,
                                                    const kernel_bin_t<Meta> &fp32_kernel,
                                                    const pi::tensorlib::DataType dtype)
{
    switch (dtype)
    {
        case pi::tensorlib::DataType::BFLOAT16:
            return bf16_kernel;
        case pi::tensorlib::DataType::FLOAT16:
            return fp16_kernel;
        case pi::tensorlib::DataType::FLOAT32:
            return fp32_kernel;
        default:
            throw std::runtime_error("Unsupported add data type");
    }
}

inline uint64_t EstimateActivationFlops(const pi::tensorlib::ActivationFunction activation, const uint64_t elements)
{
    if (elements == 0)
    {
        return 0;
    }

    constexpr uint64_t kReluFlopsPerElem = 1; // compare
    constexpr uint64_t kGeluFlopsPerElem = 6; // polynomial + tanh approximation in kernel

    switch (activation)
    {
        case pi::tensorlib::ActivationFunction::RELU:
            return kReluFlopsPerElem * elements;
        case pi::tensorlib::ActivationFunction::GELU:
            return kGeluFlopsPerElem * elements;
        default:
            throw std::runtime_error("Unsupported activation in EstimateActivationFlops");
    }
}
