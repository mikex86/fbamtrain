#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace pi::tensorlib
{
    enum class PrecisionMode
    {
        FP32,
        FP16,
        FP16_ACC16,
        BF16
    };

    struct PromisedPerf
    {
        std::string name;
        float tf_32;    /// tensor-core FP32 equivalent
        float bf16_32;  /// bf16 accumulate FP32
        float fp16_32;  /// fp16 accumulate FP32
        float fp16_16;  /// fp16 accumulate FP16
        float cores;
        float clock_mhz;
    };

    [[nodiscard]] std::optional<float> GetPromisedTFlops(const std::string &device_name, PrecisionMode precision);

    [[nodiscard]] std::optional<float> GetPromisedTFlops(int device_ordinal, PrecisionMode precision);

} // namespace pi::tensorlib
