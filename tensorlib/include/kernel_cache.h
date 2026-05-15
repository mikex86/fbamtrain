#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <cstdint>

#if PI_TENSORLIB_ENABLE_CUDA
#include <cuda.h>
#elif PI_TENSORLIB_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

namespace pi::tensorlib
{
#if PI_TENSORLIB_ENABLE_CUDA
    typedef CUmodule kernel_t;
    typedef CUfunction kernel_func_t;
#elif PI_TENSORLIB_ENABLE_HIP
    typedef hipModule_t kernel_t;
    typedef hipFunction_t kernel_func_t;
#endif

    class KernelCache
    {
        std::unordered_map<int, std::unordered_map<std::string, kernel_t>> loaded_kernels_;
        
        std::unordered_map<int, std::unordered_map<std::string, uint64_t>> kernel_hashes_;

      public:
        KernelCache() = default;

        void loadKernel(const std::string &kernel_name, const void *module_data, size_t module_size_bytes, int device_ordinal);

        [[nodiscard]] std::optional<kernel_t> getKernel(const std::string &kernel_name, int device_ordinal);

        [[nodiscard]] std::optional<kernel_func_t> getKernelFunction(const std::string &kernel_name,
                                                                     const std::string &func_name, int device_ordinal);

        static KernelCache &getInstance();
    };
}; // namespace pi::tensorlib
