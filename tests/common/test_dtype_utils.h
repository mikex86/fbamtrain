#pragma once

#include <array>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>

#include <tensorlib.h>

namespace test_utils
{
    inline constexpr std::array<std::pair<pi::tensorlib::DataType, std::string_view>, 2> kAllTestDtypes{{
        {pi::tensorlib::DataType::BFLOAT16, "bf16"},
        {pi::tensorlib::DataType::FLOAT16, "fp16"},
    }};

    constexpr std::string_view GetDtypeSuffix(const pi::tensorlib::DataType dtype)
    {
        switch (dtype)
        {
            case pi::tensorlib::DataType::BFLOAT16:
                return "bf16";
            case pi::tensorlib::DataType::FLOAT16:
                return "fp16";
            default:
                return "unknown";
        }
    }

    inline std::string MakeKey(const std::string_view base, const pi::tensorlib::DataType dtype)
    {
        const auto suffix = GetDtypeSuffix(dtype);
        std::string key;
        key.reserve(base.size() + 1 + suffix.size());
        key.append(base);
        key.push_back('_');
        key.append(suffix);
        return key;
    }

    inline float SelectTolerance(const pi::tensorlib::DataType dtype, const float bf16_tol, const float fp16_tol)
    {
        return dtype == pi::tensorlib::DataType::FLOAT16 ? fp16_tol : bf16_tol;
    }

    inline pi::tensorlib::DataType GetTestDtype()
    {
        if (const char *env = std::getenv("TEST_DTYPE"))
        {
            const std::string_view value{env};
            if (value == "fp16")
            {
                return pi::tensorlib::DataType::FLOAT16;
            }
            if (value == "bf16")
            {
                return pi::tensorlib::DataType::BFLOAT16;
            }
        }
        return pi::tensorlib::DataType::BFLOAT16;
    }

    inline std::string ReferenceFileName(const pi::tensorlib::DataType dtype)
    {
        std::string file{"reference_"};
        file.append(GetDtypeSuffix(dtype));
        file.append(".safetensors");
        return file;
    }
} // namespace test_utils
