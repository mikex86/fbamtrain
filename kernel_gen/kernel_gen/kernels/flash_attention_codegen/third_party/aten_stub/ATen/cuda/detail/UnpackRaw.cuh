#pragma once

#include <cstdint>
#include <utility>

#include "ATen/cuda/CUDAGeneratorImpl.h"

namespace at::cuda::philox {

__host__ __device__ inline std::pair<uint64_t, uint64_t> unpack(const at::PhiloxCudaState &state) {
    return {state.seed, state.offset};
}

} // namespace at::cuda::philox
