#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace pi::tensorlib::utils
{
    inline float Fp32FromBf16(const uint16_t bf16)
    {
        const uint32_t fp32 = static_cast<uint32_t>(bf16) << 16;
        return std::bit_cast<float>(fp32);
    }

    inline uint16_t Bf16FromFp32(const float value)
    {
        const uint32_t bits = std::bit_cast<uint32_t>(value);
        const uint32_t rounding_bias = ((bits >> 16) & 1u) + 0x7FFFu;
        return static_cast<uint16_t>((bits + rounding_bias) >> 16);
    }

    inline float Fp32FromFp16(const uint16_t value)
    {
        const bool is_negative = (value & 0x8000u) != 0;
        const uint32_t exponent = (value >> 10) & 0x1Fu;
        const uint32_t mantissa = value & 0x03FFu;

        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                return is_negative ? -0.0f : 0.0f;
            }
            const float subnormal = std::ldexp(static_cast<float>(mantissa), -24);
            return is_negative ? -subnormal : subnormal;
        }

        if (exponent == 0x1F)
        {
            if (mantissa == 0)
            {
                return is_negative ? -std::numeric_limits<float>::infinity()
                                   : std::numeric_limits<float>::infinity();
            }
            return std::numeric_limits<float>::quiet_NaN();
        }

        const float significand = 1.0f + static_cast<float>(mantissa) / 1024.0f;
        const float normal = std::ldexp(significand, static_cast<int>(exponent) - 15);
        return is_negative ? -normal : normal;
    }

    inline uint16_t Fp16FromFp32(const float value)
    {
        if (std::isnan(value))
        {
            return 0x7E00u;
        }
        if (std::isinf(value))
        {
            return static_cast<uint16_t>(std::signbit(value) ? 0xFC00u : 0x7C00u);
        }

        const uint16_t sign_bit = std::signbit(value) ? static_cast<uint16_t>(0x8000u) : static_cast<uint16_t>(0x0u);
        const double abs_value = std::fabs(static_cast<double>(value));

        if (abs_value == 0.0)
        {
            return sign_bit;
        }

        if (abs_value >= 65504.0)
        {
            return static_cast<uint16_t>(sign_bit | 0x7C00u);
        }

        if (abs_value < std::ldexp(1.0, -14))
        {
            const double scaled = std::ldexp(abs_value, 24);
            auto mantissa = static_cast<uint32_t>(std::nearbyint(scaled));
            if (mantissa == 0)
            {
                return sign_bit;
            }
            if (mantissa > 0x3FFu)
            {
                mantissa = 0x3FFu;
            }
            return static_cast<uint16_t>(sign_bit | static_cast<uint16_t>(mantissa));
        }

        int exponent{};
        double significand = std::frexp(abs_value, &exponent); // abs_value = significand * 2^exponent, significand in [0.5, 1)
        significand *= 2.0;
        exponent -= 1;

        int half_exponent = exponent + 15;
        const double mantissa_double = (significand - 1.0) * 1024.0;
        auto mantissa = static_cast<uint32_t>(std::nearbyint(mantissa_double));

        if (mantissa == 1024u)
        {
            mantissa = 0;
            ++half_exponent;
        }

        if (half_exponent >= 0x1F)
        {
            return static_cast<uint16_t>(sign_bit | 0x7C00u);
        }

        return static_cast<uint16_t>(sign_bit | (static_cast<uint16_t>(half_exponent) << 10) |
                                     static_cast<uint16_t>(mantissa & 0x3FFu));
    }
} // namespace pi::tensorlib::utils
