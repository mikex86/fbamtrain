#pragma once

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace fbamtrain::formatutils
{
    inline std::string FormatBytes(uint64_t bytes)
    {
        static constexpr const char *suffixes[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB"};
        static constexpr size_t kSuffixCount = sizeof(suffixes) / sizeof(suffixes[0]);
        double value = static_cast<double>(bytes);
        size_t suffix_idx = 0;
        while (value >= 1024.0 && suffix_idx + 1 < kSuffixCount)
        {
            value /= 1024.0;
            ++suffix_idx;
        }
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(value >= 10.0 ? 1 : 2) << value << ' ' << suffixes[suffix_idx];
        return oss.str();
    }
} // namespace fbamtrain::formatutils
