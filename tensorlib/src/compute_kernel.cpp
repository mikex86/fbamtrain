#include "compute_kernel.h"

#include <stdexcept>

#if PI_TENSORLIB_ENABLE_CUDA
#include <cuda.h>
#endif

pi::tensorlib::CudaKernel::CudaKernel(const std::string &function_name, void *module) : module(module)
{
#if PI_TENSORLIB_ENABLE_CUDA
    CUfunction cu_func{};
    if (cuModuleGetFunction(&cu_func, static_cast<CUmodule>(module), function_name.c_str()) != CUDA_SUCCESS)
    {
        throw std::invalid_argument("No function named \"" + function_name + "\" found in module!");
    }
    function = cu_func;
#else
    throw std::runtime_error("Cuda support is not enabled!");
#endif
}

pi::tensorlib::CudaKernel::~CudaKernel()
{
#if PI_TENSORLIB_ENABLE_CUDA
    const auto cu_mod = static_cast<CUmodule>(module);
    if (module != nullptr)
    {
        cuModuleUnload(cu_mod);
    }
#else
    throw std::runtime_error("Cuda support is not enabled!");
#endif
}