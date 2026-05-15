#include "random.h"

#include <stdexcept>

fbamtrain::Random::Random(const uint64_t seed) : state(seed) {}

void fbamtrain::Random::seed(const uint64_t seed) { state = seed; }

uint64_t fbamtrain::Random::randu64()
{
    uint64_t z = (state += GOLDEN);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

int64_t fbamtrain::Random::randint(const int64_t low, const int64_t high)
{
    if (high <= low)
        throw std::invalid_argument("high must be > low");
    const auto range = static_cast<uint64_t>(high - low);
    const uint64_t r = randu64();
    const uint64_t v = r % range;
    return low + static_cast<int64_t>(v);
}