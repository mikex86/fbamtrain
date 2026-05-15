#pragma once

#include <cstdint>

namespace at {

struct PhiloxCudaState {
    uint64_t seed;
    uint64_t offset;
};

} // namespace at
