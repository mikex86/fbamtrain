#pragma once

#include "allocator.h"
#include <any>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace pi::tensorlib
{
    enum class KernelBackend
    {
        CUDA,
        HIP
    };

    /// Forward declaration of RealTensor
    class RealTensor;

    class CudaKernel final
    {
        void *module{nullptr};
        void *function{nullptr};

      public:
        CudaKernel(const std::string &function_name, void *module);

        CudaKernel(const CudaKernel &) = delete;
        CudaKernel(CudaKernel &&) = delete;
        CudaKernel &operator=(const CudaKernel &) = delete;

        ~CudaKernel();
    };

    struct CudaKernelDescriptor
    {
        void *module_data;
        size_t module_size;
    };

    struct HipKernelDescriptor
    {
        void *module_data;
        size_t module_size;
    };

    struct KernelDataArg
    {
        std::vector<uint8_t> bytes;
    };

    struct KernelLaunchArguments
    {
        std::vector<std::any> args;
        uint32_t grid_dim_x{1};
        uint32_t grid_dim_y{1};
        uint32_t grid_dim_z{1};
        uint32_t block_dim_x{1};
        uint32_t block_dim_y{1};
        uint32_t block_dim_z{1};
        uint32_t shared_mem_bytes{0};
        uint32_t cluster_dim_x{1};
        uint32_t cluster_dim_y{1};
        uint32_t cluster_dim_z{1};
        int device_ordinal{0};
    };

    typedef std::function<KernelLaunchArguments(const std::vector<std::shared_ptr<RealTensor>> &inputs,
                                                const std::vector<std::shared_ptr<RealTensor>> &outputs)>
        KernelArgumentProvider;

    struct ComputeKernelDescriptor
    {
        /// Debug identifier for the kernel
        std::string kernel_name;

        /// The name of the module function to launch
        std::string function_name;

        /// Kernel backend to use
#if PI_TENSORLIB_ENABLE_HIP && !PI_TENSORLIB_ENABLE_CUDA
        KernelBackend backend{KernelBackend::HIP};
#else
        KernelBackend backend{KernelBackend::CUDA};
#endif

        /// Expected number of kernel arguments (PTX param count). UINT32_MAX = not specified, will result in error!
        uint32_t expected_arg_count{UINT32_MAX};

        /// Argument provider to use for this kernel
        KernelArgumentProvider argument_provider;

        /// Allocations owned by the kernel (and ONLY the kernel) that need to be freed right after kernel execution
        std::vector<std::shared_ptr<RealTensor>> owned_allocations{};

        /// Populated if backend == KernelBackend::CUDA
        std::optional<CudaKernelDescriptor> cuda_descriptor;

        /// Populated if backend == KernelBackend::HIP
        std::optional<HipKernelDescriptor> hip_descriptor;
    };

    [[nodiscard]] inline std::optional<CudaKernelDescriptor> MakeCudaKernelDescriptor(void *module_data,
                                                                                      const size_t module_size)
    {
#if PI_TENSORLIB_ENABLE_CUDA
        return CudaKernelDescriptor{.module_data = module_data, .module_size = module_size};
#else
        (void)module_data;
        (void)module_size;
        return std::nullopt;
#endif
    }

    [[nodiscard]] inline std::optional<HipKernelDescriptor> MakeHipKernelDescriptor(void *module_data,
                                                                                    const size_t module_size)
    {
#if PI_TENSORLIB_ENABLE_HIP
        return HipKernelDescriptor{.module_data = module_data, .module_size = module_size};
#else
        (void)module_data;
        (void)module_size;
        return std::nullopt;
#endif
    }
}; // namespace pi::tensorlib
