#include "kernel_cache.h"

#include "ctx_management.h"

#include <functional>
#include <string_view>

#if PI_TENSORLIB_ENABLE_CUDA
#include <cuda.h>
#elif PI_TENSORLIB_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include <stdexcept>

namespace
{
    uint64_t HashKernelBytes(const void *data, size_t size_bytes)
    {
        if (data == nullptr || size_bytes == 0)
        {
            return 0;
        }
        const auto view = std::string_view(static_cast<const char *>(data), size_bytes);
        return std::hash<std::string_view>{}(view);
    }
} // namespace

void pi::tensorlib::KernelCache::loadKernel(const std::string &kernel_name, const void *module_data,
                                            const size_t module_size_bytes, const int device_ordinal)
{
    const uint64_t new_hash = HashKernelBytes(module_data, module_size_bytes);
    auto &device_hashes = kernel_hashes_[device_ordinal];
    if (const auto it = device_hashes.find(kernel_name); it != device_hashes.end())
    {
        if (it->second != new_hash)
        {
            throw std::runtime_error("Kernel name collision: '" + kernel_name +
                                     "' already loaded with different binary contents");
        }
        // Already loaded identical binary; no-op.
        return;
    }

#if PI_TENSORLIB_ENABLE_CUDA
    internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);
    CUmodule module;
    if (const CUresult res = cuModuleLoadData(&module, module_data); res != CUDA_SUCCESS)
    {
        const char *err_str;
        cuGetErrorString(res, &err_str);
        throw std::runtime_error(std::string("CUDA Error: ") + err_str);
    }
    loaded_kernels_[device_ordinal].emplace(kernel_name, module);
#elif PI_TENSORLIB_ENABLE_HIP
    hipModule_t module;
    hipError_t err = hipModuleLoadData(&module, module_data);
    if (err != hipSuccess)
    {
        throw std::runtime_error(std::string("HIP Error: ") + hipGetErrorString(err));
    }
    loaded_kernels_[device_ordinal].emplace(kernel_name, module);
#else
    throw std::runtime_error("No backend enabled for kernel loading");
#endif
    device_hashes.emplace(kernel_name, new_hash);
}

std::optional<pi::tensorlib::kernel_t> pi::tensorlib::KernelCache::getKernel(const std::string &kernel_name, const int device_ordinal)
{
    const auto &kernels = loaded_kernels_[device_ordinal];
    return kernels.contains(kernel_name) ? std::make_optional(kernels.at(kernel_name)) : std::nullopt;
}

std::optional<pi::tensorlib::kernel_func_t>
pi::tensorlib::KernelCache::getKernelFunction(const std::string &kernel_name, const std::string &func_name, const int device_ordinal)
{
    const std::optional<kernel_t> kernel = getKernel(kernel_name, device_ordinal);
    if (!kernel.has_value())
    {
        return std::nullopt;
    }
#if PI_TENSORLIB_ENABLE_CUDA
    if (CUfunction func; cuModuleGetFunction(&func, *kernel, func_name.c_str()) == CUDA_SUCCESS)
    {
        return func;
    }
    return std::nullopt;
#elif PI_TENSORLIB_ENABLE_HIP
    if (hipFunction_t func; hipModuleGetFunction(&func, *kernel, func_name.c_str()) == hipSuccess)
    {
        return func;
    }
    return std::nullopt;
#else
    return std::nullopt;
#endif
}

pi::tensorlib::KernelCache &pi::tensorlib::KernelCache::getInstance()
{
    static KernelCache instance{};
    return instance;
}
