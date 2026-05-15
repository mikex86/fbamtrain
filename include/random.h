#pragma once

#include <cstdint>

namespace fbamtrain
{
    class Random
    {
        uint64_t state;
        static constexpr uint64_t GOLDEN = 0x9E3779B97F4A7C15ULL;

      public:
        explicit Random(uint64_t seed);

        void seed(uint64_t seed);

        uint64_t randu64();

        int64_t randint(int64_t low, int64_t high);

        [[nodiscard]] uint64_t getState() const { return state; }

        void setState(const uint64_t value) { state = value; }
    };
} // namespace fbamtrain
