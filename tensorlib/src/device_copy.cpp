#include "device_copy.h"

#include <kernels/kernel_binaries.h>

#include "ctx_management.h"
#include "execution_backend.h"
#include "gpu_stream.h"
#include "launch_utils.h"
#include "shape_utils.h"
#include "tensorlib.h"

#include <iostream>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#if PI_TENSORLIB_ENABLE_CUDA
#include <cuda.h>
#elif PI_TENSORLIB_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

enum class TransferType
{
    HOST_TO_HOST,     // cpu to cpu
    HOST_TO_DEVICE,   // cpu to gpu
    DEVICE_TO_HOST,   // gpu to cpu
    DEVICE_TO_DEVICE, // gpu to gpu
};

enum class MemoryType
{
    HOST,
    DEVICE,
};

struct StridedCopyConfig
{
    MemoryType src_type{};
    MemoryType dst_type{};
    int src_device_ordinal{-1};
    int dst_device_ordinal{-1};
};

namespace
{
    constexpr size_t kCopyKernelElementBytes = 2;

    uint32_t NarrowToUint32(const size_t value, const char *name)
    {
        if (value > std::numeric_limits<uint32_t>::max())
        {
            throw std::runtime_error(std::string(name) + " exceeds uint32_t range for device copy kernel.");
        }
        return static_cast<uint32_t>(value);
    }

    pi::tensorlib::KernelLaunchArguments BuildDeviceCopyKernelArgs(void *dst_ptr, const void *src_ptr,
                                                                   size_t src_stride_bytes, size_t dst_stride_bytes,
                                                                   size_t width_bytes, size_t height,
                                                                   const int device_ordinal)
    {
        if (width_bytes == 0 || height == 0)
        {
            return pi::tensorlib::KernelLaunchArguments{.device_ordinal = device_ordinal};
        }

        if ((width_bytes % kCopyKernelElementBytes != 0) || (src_stride_bytes % kCopyKernelElementBytes != 0) ||
            (dst_stride_bytes % kCopyKernelElementBytes != 0))
        {
            throw std::runtime_error("device_copy_strided_2d requires strides divisible by two bytes.");
        }

        const uint32_t width_elems = NarrowToUint32(width_bytes / kCopyKernelElementBytes, "width_bytes");
        const uint32_t src_stride_elems =
            NarrowToUint32(src_stride_bytes / kCopyKernelElementBytes, "src_stride_bytes");
        const uint32_t dst_stride_elems =
            NarrowToUint32(dst_stride_bytes / kCopyKernelElementBytes, "dst_stride_bytes");
        const uint32_t rows = NarrowToUint32(height, "height");

        pi::tensorlib::KernelLaunchArguments args{
            .args = {dst_ptr, const_cast<void *>(src_ptr), src_stride_elems, dst_stride_elems, width_elems, rows,
                     static_cast<void *>(nullptr)},
            .grid_dim_x = CEIL_DIV(width_elems, kdevice_copy_strided_2d.meta.block_size_x),
            .grid_dim_y = CEIL_DIV(rows, kdevice_copy_strided_2d.meta.block_size_y),
            .grid_dim_z = 1,
            .block_dim_x = TRITON_WARP_SIZE * kdevice_copy_strided_2d.num_warps,
            .block_dim_y = 1,
            .block_dim_z = 1,
            .shared_mem_bytes = kdevice_copy_strided_2d.shared_mem_bytes,
            .device_ordinal = device_ordinal};
        return args;
    }

    pi::tensorlib::ComputeKernelDescriptor CreateDeviceCopyKernelDescriptor()
    {
        return pi::tensorlib::ComputeKernelDescriptor{
            .kernel_name = "device_copy_strided_2d",
            .function_name = "device_copy_strided_2d",
            .expected_arg_count = kdevice_copy_strided_2d.arg_count,
            .argument_provider =
                [](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                   const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
                -> pi::tensorlib::KernelLaunchArguments
            {
                if (inputs.size() != 1 || outputs.size() != 1)
                {
                    throw std::runtime_error("device_copy_strided_2d expects a single input and output tensor.");
                }
                const auto &src_tensor = inputs[0];
                const auto &dst_tensor = outputs[0];
                if (src_tensor->shape().ndims() != 2 || dst_tensor->shape().ndims() != 2)
                {
                    throw std::runtime_error("device_copy_strided_2d expects rank-2 tensors.");
                }
                if (src_tensor->dtype() != dst_tensor->dtype())
                {
                    throw std::runtime_error("device_copy_strided_2d expects matching input/output dtypes.");
                }
                const auto element_size = pi::tensorlib::GetDataTypeSize(src_tensor->dtype());
                const size_t width_bytes = src_tensor->shape()[1] * element_size;
                const size_t height = src_tensor->shape()[0];
                const size_t src_stride_bytes = src_tensor->strides()[0] * element_size;
                const size_t dst_stride_bytes = dst_tensor->strides()[0] * element_size;
                const auto device_ordinal =
                    ValidateSameDeviceOrdinal("device_copy_strided_2d", {src_tensor, dst_tensor});
                return BuildDeviceCopyKernelArgs(dst_tensor->dataptr(), src_tensor->dataptr(), src_stride_bytes,
                                                 dst_stride_bytes, width_bytes, height, device_ordinal);
            },
            .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kdevice_copy_strided_2d.data, kdevice_copy_strided_2d.size),
            .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kdevice_copy_strided_2d.data, kdevice_copy_strided_2d.size)};
    }

    const pi::tensorlib::ComputeKernelDescriptor &DeviceCopyKernelDescriptor()
    {
        static pi::tensorlib::ComputeKernelDescriptor descriptor = CreateDeviceCopyKernelDescriptor();
        return descriptor;
    }

    void LaunchDeviceCopyKernel(void *dst_ptr, const void *src_ptr, const size_t src_stride_bytes, const size_t dst_stride_bytes,
                                const size_t width_bytes, const size_t height, const pi::tensorlib::gpustream::GpuStream stream,
                                const int device_ordinal)
    {
        if (width_bytes == 0 || height == 0)
        {
            return;
        }
        const pi::tensorlib::KernelLaunchArguments args =
            BuildDeviceCopyKernelArgs(dst_ptr, src_ptr, src_stride_bytes, dst_stride_bytes, width_bytes, height,
                                      device_ordinal);
        pi::tensorlib::ExecutionBackend::LaunchKernel(DeviceCopyKernelDescriptor(), args, stream);
    }

} // namespace

static TransferType GetTransferType(const pi::tensorlib::DeviceType src_device_type,
                                    const pi::tensorlib::DeviceType dst_device_type)
{
    switch (src_device_type)
    {
        case pi::tensorlib::DeviceType::CPU:
            switch (dst_device_type)
            {
                case pi::tensorlib::DeviceType::CPU:
                    return TransferType::HOST_TO_HOST;
                case pi::tensorlib::DeviceType::GPU:
                    return TransferType::HOST_TO_DEVICE;
                default:
                    throw std::runtime_error("Unsupported destination device type for transfer.");
            }
        case pi::tensorlib::DeviceType::GPU:
            switch (dst_device_type)
            {
                case pi::tensorlib::DeviceType::CPU:
                    return TransferType::DEVICE_TO_HOST;
                case pi::tensorlib::DeviceType::GPU:
                    return TransferType::DEVICE_TO_DEVICE;
                default:
                    throw std::runtime_error("Unsupported destination device type for transfer.");
            }
        default:
            throw std::runtime_error("Unsupported source device type for transfer.");
    }
}

static void ValidateStridedCopyConfig(const StridedCopyConfig &config)
{
    if (config.src_type == MemoryType::DEVICE && config.dst_type == MemoryType::DEVICE &&
        config.src_device_ordinal != config.dst_device_ordinal)
    {
        throw std::runtime_error(
            "Strided DEVICE_COPY across different devices is not supported in the slow path (non-contiguous tensors).");
    }
}

static int ActiveDeviceOrdinal(const StridedCopyConfig &config)
{
    if (config.dst_type == MemoryType::DEVICE)
    {
        return config.dst_device_ordinal;
    }
        return config.src_device_ordinal;
}

// Memcopy implementations
static void HostToHostCopy(void *dst, const void *src, const size_t size_in_bytes)
{
    std::memcpy(dst, src, size_in_bytes);
}

static void DeviceToHostCopy(void *dst, const void *src, size_t size_in_bytes, const int src_device_ordinal,
                             pi::tensorlib::gpustream::GpuStream stream)
{
    pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(src_device_ordinal);
#if PI_TENSORLIB_ENABLE_CUDA
    const auto cuda_stream = static_cast<CUstream>(stream);
    cuMemcpyDtoHAsync_v2(dst, reinterpret_cast<CUdeviceptr>(src), size_in_bytes, cuda_stream);
#elif PI_TENSORLIB_ENABLE_HIP
    const auto hip_stream = static_cast<hipStream_t>(stream);
    hipMemcpyDtoHAsync(dst, reinterpret_cast<hipDeviceptr_t>(const_cast<void *>(src)), size_in_bytes, hip_stream);
#else
    throw std::runtime_error("Device to Host copy not supported: No GPU backend enabled.");
#endif
}

static void HostToDeviceCopy(void *dst, const void *src, size_t size_in_bytes, const int dst_device_ordinal,
                             pi::tensorlib::gpustream::GpuStream stream)
{
    pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(dst_device_ordinal);
#if PI_TENSORLIB_ENABLE_CUDA
    const auto cuda_stream = static_cast<CUstream>(stream);
    cuMemcpyHtoDAsync_v2(reinterpret_cast<CUdeviceptr>(dst), src, size_in_bytes, cuda_stream);
#elif PI_TENSORLIB_ENABLE_HIP
    const auto hip_stream = static_cast<hipStream_t>(stream);
    hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst), const_cast<void *>(src), size_in_bytes, hip_stream);
#else
    throw std::runtime_error("Host to Device copy not supported: No GPU backend enabled.");
#endif
}

#if PI_TENSORLIB_ENABLE_CUDA
static CUmemorytype ToCudaMemoryType(const MemoryType type)
{
    return type == MemoryType::DEVICE ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
}
#elif PI_TENSORLIB_ENABLE_HIP
static hipMemcpyKind ToHipMemcpyKind(const StridedCopyConfig &config)
{
    switch (config.src_type)
    {
        case MemoryType::HOST:
            switch (config.dst_type)
            {
                case MemoryType::HOST:
                    return hipMemcpyHostToHost;
                case MemoryType::DEVICE:
                    return hipMemcpyHostToDevice;
            }
            break;
        case MemoryType::DEVICE:
            switch (config.dst_type)
            {
                case MemoryType::HOST:
                    return hipMemcpyDeviceToHost;
                case MemoryType::DEVICE:
                    return hipMemcpyDeviceToDevice;
            }
            break;
    }
    throw std::runtime_error("Unsupported hipMemcpy kind for strided copy.");
}
#endif

static void DeviceToDeviceCopyFast(void *dst, const void *src, const size_t size_in_bytes, const int src_device_ordinal,
                                   const int dst_device_ordinal, pi::tensorlib::gpustream::GpuStream stream)
{
#if PI_TENSORLIB_ENABLE_CUDA
    const auto src_device_ctx = pi::tensorlib::internal::ctxmgmt::GpuGetDeviceCtx(src_device_ordinal);
    const auto dst_device_ctx = pi::tensorlib::internal::ctxmgmt::GpuGetDeviceCtx(dst_device_ordinal);
    const auto cuda_stream = static_cast<CUstream>(stream);
    if (src_device_ordinal != dst_device_ordinal)
    {
        cuMemcpyPeerAsync(reinterpret_cast<CUdeviceptr>(dst), static_cast<CUcontext>(dst_device_ctx),
                          reinterpret_cast<CUdeviceptr>(src), static_cast<CUcontext>(src_device_ctx), size_in_bytes,
                          cuda_stream);
    }
    else
    {
        cuMemcpyDtoDAsync_v2(reinterpret_cast<CUdeviceptr>(dst), reinterpret_cast<CUdeviceptr>(src), size_in_bytes,
                             cuda_stream);
    }
#elif PI_TENSORLIB_ENABLE_HIP
    const auto hip_stream = static_cast<hipStream_t>(stream);
    if (src_device_ordinal != dst_device_ordinal)
    {
        hipMemcpyPeerAsync(dst, dst_device_ordinal, src, src_device_ordinal, size_in_bytes, hip_stream);
    }
    else
    {
        hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst),
                           reinterpret_cast<hipDeviceptr_t>(const_cast<void *>(src)), size_in_bytes, hip_stream);
    }
#else
    throw std::runtime_error("Device to Device copy not supported: No GPU backend enabled.");
#endif
}

static void StridedCopy2d(void *dst, const void *src, const pi::tensorlib::Shape &src_shape,
                          const pi::tensorlib::Strides &src_strides, const pi::tensorlib::Strides &dst_strides,
                          const size_t element_size, const StridedCopyConfig &config,
                          pi::tensorlib::gpustream::GpuStream stream)
{
    // Handle zero-stride broadcast (e.g., views created via broadcast).
    if (src_strides[0] == 0 || dst_strides[0] == 0)
    {
        const size_t width_bytes = src_shape[1] * element_size;
        const size_t height = src_shape[0];
        if (height == 0)
        {
            return;
        }
        const bool src_zero_stride = src_strides[0] == 0;
        const bool dst_zero_stride = dst_strides[0] == 0;
        const size_t src_pitch = src_strides[0] * element_size;
        const size_t dst_pitch = dst_strides[0] * element_size;

        switch (config.src_type)
        {
            case MemoryType::HOST:
                switch (config.dst_type)
                {
                    case MemoryType::HOST:
                    {
                        const size_t src_offset = (!src_zero_stride && dst_zero_stride) ? (height - 1) * src_pitch : 0;
                        const auto *src_row = static_cast<const uint8_t *>(src) + src_offset;
                        HostToHostCopy(dst, src_row, width_bytes);
                        if (!dst_zero_stride && height > 1)
                        {
                            auto *dst_bytes = static_cast<uint8_t *>(dst);
                            for (size_t row = 1; row < height; ++row)
                            {
                                const auto *row_src = src_zero_stride
                                                          ? src_row
                                                          : static_cast<const uint8_t *>(src) + row * src_pitch;
                                auto *dst_row = dst_bytes + row * dst_pitch;
                                HostToHostCopy(dst_row, row_src, width_bytes);
                            }
                        }
                        break;
                    }
                    case MemoryType::DEVICE:
                    {
                        const size_t src_offset = (!src_zero_stride && dst_zero_stride) ? (height - 1) * src_pitch : 0;
                        const auto *src_row = static_cast<const uint8_t *>(src) + src_offset;
                        HostToDeviceCopy(dst, src_row, width_bytes, config.dst_device_ordinal, stream);
                        if (!dst_zero_stride && src_zero_stride && height > 1)
                        {
                            auto *dst_bytes = static_cast<uint8_t *>(dst);
                            auto *dst_row1 = dst_bytes + dst_pitch;
                            const size_t remaining_rows = height - 1;
                            LaunchDeviceCopyKernel(dst_row1, dst_bytes, /*src_stride_bytes=*/0, dst_pitch, width_bytes,
                                                   remaining_rows, stream, config.dst_device_ordinal);
                        }
                        break;
                    }
                }
                break;
            case MemoryType::DEVICE:
                switch (config.dst_type)
                {
                    case MemoryType::HOST:
                    {
                        const size_t src_offset = (!src_zero_stride && dst_zero_stride) ? (height - 1) * src_pitch : 0;
                        const auto *src_row = static_cast<const uint8_t *>(src) + src_offset;
                        DeviceToHostCopy(dst, src_row, width_bytes, config.src_device_ordinal, stream);
                        if (!dst_zero_stride && src_zero_stride && height > 1)
                        {
                            auto *dst_bytes = static_cast<uint8_t *>(dst);
                            for (size_t row = 1; row < height; ++row)
                            {
                                auto *dst_row = dst_bytes + row * dst_pitch;
                                HostToHostCopy(dst_row, dst_bytes, width_bytes);
                            }
                        }
                        break;
                    }
                    case MemoryType::DEVICE:
                    {
                        ValidateStridedCopyConfig(config);
                        const int device_ordinal = ActiveDeviceOrdinal(config);
                        if (device_ordinal >= 0)
                        {
                            pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);
                        }
                        const size_t src_offset = (!src_zero_stride && dst_zero_stride) ? (height - 1) * src_pitch : 0;
                        const auto *src_row = static_cast<const uint8_t *>(src) + src_offset;
                        auto *dst_row = static_cast<uint8_t *>(dst);
                        const size_t rows_to_copy = dst_zero_stride ? 1 : height;
                        LaunchDeviceCopyKernel(dst_row, src_row, src_pitch, dst_pitch, width_bytes, rows_to_copy,
                                               stream, device_ordinal);
                        break;
                    }
                }
                break;
        }
        return;
    }

    // Only support contiguous inner dimension so the memcpy2D pitch copy applies.
    if (src_strides[1] != 1 || dst_strides[1] != 1)
    {
        throw std::runtime_error("DEVICE_COPY slow path expects innermost stride of 1.");
    }

    ValidateStridedCopyConfig(config);
    if (const int device_ordinal = ActiveDeviceOrdinal(config); device_ordinal >= 0)
    {
        pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);
    }

    const size_t width_bytes = src_shape[1] * element_size;
    const size_t height = src_shape[0];

#if PI_TENSORLIB_ENABLE_CUDA
    const auto cuda_stream = static_cast<CUstream>(stream);
    const auto src_pitch = src_strides[0] * element_size;
    const auto dst_pitch = dst_strides[0] * element_size;
    CUDA_MEMCPY2D copy_params{};
    copy_params.srcXInBytes = 0;
    copy_params.srcY = 0;
    copy_params.srcMemoryType = ToCudaMemoryType(config.src_type);
    copy_params.srcHost = const_cast<void *>(src);
    copy_params.srcDevice = reinterpret_cast<CUdeviceptr>(src);
    copy_params.srcPitch = src_pitch;
    copy_params.dstXInBytes = 0;
    copy_params.dstY = 0;
    copy_params.dstMemoryType = ToCudaMemoryType(config.dst_type);
    copy_params.dstHost = dst;
    copy_params.dstDevice = reinterpret_cast<CUdeviceptr>(dst);
    copy_params.dstPitch = dst_pitch;
    copy_params.WidthInBytes = width_bytes;
    copy_params.Height = height;
    const CUresult res = cuMemcpy2DAsync_v2(&copy_params, cuda_stream);
    if (res != CUDA_SUCCESS)
    {
        const char *err_msg{};
        cuGetErrorString(res, &err_msg);
        const std::string msg = err_msg ? err_msg : "unknown";
        throw std::runtime_error("DEVICE_COPY slow path (cudaMemcpy2DAsync) failed: " + msg);
    }
#elif PI_TENSORLIB_ENABLE_HIP
    const auto hip_stream = static_cast<hipStream_t>(stream);
    const auto src_pitch = static_cast<size_t>(src_strides[0] * element_size);
    const auto dst_pitch = static_cast<size_t>(dst_strides[0] * element_size);
    const hipMemcpyKind kind = ToHipMemcpyKind(config);
    const hipError_t res =
        hipMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width_bytes, height, kind, hip_stream);
    if (res != hipSuccess)
    {
        throw std::runtime_error(std::string("DEVICE_COPY slow path (hipMemcpy2DAsync) failed: ")
                                 + hipGetErrorString(res));
    }
#else
    throw std::runtime_error("Strided copy not supported: No GPU backend enabled.");
#endif
}

static void StridedCopy3d(void *dst, const void *src, const pi::tensorlib::Shape &src_shape,
                          const pi::tensorlib::Strides &src_strides, const pi::tensorlib::Strides &dst_strides,
                          const size_t element_size, const StridedCopyConfig &config,
                          pi::tensorlib::gpustream::GpuStream stream)
{
    // Require contiguous innermost dimension so we can reuse the 2D slow copy per slice.
    if (src_strides[2] != 1 || dst_strides[2] != 1)
    {
        throw std::runtime_error("DEVICE_COPY slow path expects innermost stride of 1 for rank-3 tensors.");
    }

    ValidateStridedCopyConfig(config);
    const int device_ordinal = ActiveDeviceOrdinal(config);
    if (device_ordinal >= 0)
    {
        pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);
    }

    const bool can_use_3d =
        (src_strides[0] == src_strides[1] * src_shape[1]) && (dst_strides[0] == dst_strides[1] * src_shape[1]);
    if (can_use_3d)
    {

        const size_t width_bytes = src_shape[2] * element_size;
        const size_t height = src_shape[1];
        const size_t depth = src_shape[0];

#if PI_TENSORLIB_ENABLE_CUDA
        const auto cuda_stream = static_cast<CUstream>(stream);
        CUDA_MEMCPY3D copy_params{};
        copy_params.srcXInBytes = 0;
        copy_params.srcY = 0;
        copy_params.srcZ = 0;
        copy_params.srcMemoryType = ToCudaMemoryType(config.src_type);
        copy_params.srcHost = const_cast<void *>(src);
        copy_params.srcDevice = reinterpret_cast<CUdeviceptr>(src);
        copy_params.srcPitch = src_strides[1] * element_size;
        copy_params.srcHeight = height;
        copy_params.dstXInBytes = 0;
        copy_params.dstY = 0;
        copy_params.dstZ = 0;
        copy_params.dstMemoryType = ToCudaMemoryType(config.dst_type);
        copy_params.dstHost = dst;
        copy_params.dstDevice = reinterpret_cast<CUdeviceptr>(dst);
        copy_params.dstPitch = dst_strides[1] * element_size;
        copy_params.dstHeight = height;
        copy_params.WidthInBytes = width_bytes;
        copy_params.Height = height;
        copy_params.Depth = depth;
        const CUresult res = cuMemcpy3DAsync(&copy_params, cuda_stream);
        if (res != CUDA_SUCCESS)
        {
            const char *err_msg{};
            cuGetErrorString(res, &err_msg);
            const std::string msg = err_msg ? err_msg : "unknown";
            throw std::runtime_error("DEVICE_COPY slow path (cudaMemcpy3DAsync) failed: " + msg);
        }
#elif PI_TENSORLIB_ENABLE_HIP
        const auto hip_stream = static_cast<hipStream_t>(stream);
        hipMemcpy3DParms parms{};
        parms.srcPos = make_hipPos(0, 0, 0);
        parms.dstPos = make_hipPos(0, 0, 0);
        parms.extent = make_hipExtent(width_bytes, height, depth);
        parms.kind = ToHipMemcpyKind(config);
        parms.srcPtr = make_hipPitchedPtr(const_cast<void *>(src), src_strides[1] * element_size, width_bytes, height);
        parms.dstPtr = make_hipPitchedPtr(dst, dst_strides[1] * element_size, width_bytes, height);
        const hipError_t res = hipMemcpy3DAsync(&parms, hip_stream);
        if (res != hipSuccess)
        {
            throw std::runtime_error(std::string("DEVICE_COPY slow path (hipMemcpy3DAsync) failed: ")
                                     + hipGetErrorString(res));
        }
#else
        throw std::runtime_error("Strided copy not supported: No GPU backend enabled.");
#endif
    }
    else
    {
        // Fallback: copy one 2D plane at a time using the 2D slow path.
        const uint64_t depth = src_shape[0];
        const uint64_t inner_rows = src_shape[1];
        const uint64_t inner_cols = src_shape[2];
        for (uint64_t d = 0; d < depth; ++d)
        {
            const auto src_offset_bytes = d * src_strides[0] * element_size;
            const auto dst_offset_bytes = d * dst_strides[0] * element_size;
            const auto *src_slice = static_cast<const uint8_t *>(src) + src_offset_bytes;
            auto *dst_slice = static_cast<uint8_t *>(dst) + dst_offset_bytes;
            pi::tensorlib::Shape slice_shape({inner_rows, inner_cols});
            pi::tensorlib::Strides slice_src_strides({src_strides[1], src_strides[2]});
            pi::tensorlib::Strides slice_dst_strides({dst_strides[1], dst_strides[2]});
            StridedCopy2d(dst_slice, src_slice, slice_shape, slice_src_strides, slice_dst_strides, element_size, config,
                          stream);
        }
    }
}

static void StridedCopySlow(void *dst, const void *src, const pi::tensorlib::Shape &src_shape,
                            const pi::tensorlib::Shape &dst_shape, const pi::tensorlib::Strides &src_strides,
                            const pi::tensorlib::Strides &dst_strides, const size_t element_size,
                            const StridedCopyConfig &config, pi::tensorlib::gpustream::GpuStream stream)
{
    if (src_shape.ndims() != dst_shape.ndims())
    {
        throw std::runtime_error("DEVICE_COPY slow path requires matching ranks.");
    }
    const auto rank = src_shape.ndims();
    switch (rank)
    {
        case 2:
            StridedCopy2d(dst, src, src_shape, src_strides, dst_strides, element_size, config, stream);
            break;
        case 3:
            StridedCopy3d(dst, src, src_shape, src_strides, dst_strides, element_size, config, stream);
            break;
        default:
            throw std::runtime_error("DEVICE_COPY slow path supports rank-2 and rank-3 tensors only (for now).");
    }
}

static void EnsureContiguous(const std::shared_ptr<pi::tensorlib::RealTensor> &tensor, const std::string &tensor_name)
{
    if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(tensor))
    {
        throw std::runtime_error(tensor_name + " tensor must be row-major contiguous for DEVICE_COPY operation.");
    }
}

// Perform device copy between input and output tensors
void pi::tensorlib::internal::device_copy::PerformDeviceCopy(const std::shared_ptr<RealTensor> &input_tensor,
                                                             const std::shared_ptr<RealTensor> &output_tensor,
                                                             gpustream::GpuStream stream)
{
    if (input_tensor->shape() != output_tensor->shape())
    {
        throw std::runtime_error("Input and output tensors must have the same shape for DEVICE_COPY operation.");
    }
    if (input_tensor->dtype() != output_tensor->dtype())
    {
        if (std::getenv("FBAMTRAIN_DEBUG_DEVICE_COPY") != nullptr)
        {
            std::cerr << "DeviceCopy dtype mismatch: input dtype=" << GetDataTypeName(input_tensor->dtype())
                      << " output dtype=" << GetDataTypeName(output_tensor->dtype())
                      << " input shape rank=" << input_tensor->shape().ndims()
                      << " output shape rank=" << output_tensor->shape().ndims() << std::endl;
        }
        throw std::runtime_error("Input and output tensors must have the same data type for DEVICE_COPY operation.");
    }
    const size_t element_size = GetDataTypeSize(input_tensor->dtype());
    const size_t size_in_bytes = input_tensor->shape().numel() * element_size;
    const bool input_contiguous = shape_utils::IsRowMajorContiguous(input_tensor);
    const bool output_contiguous = shape_utils::IsRowMajorContiguous(output_tensor);

    const auto &[input_device_type, input_device_ordinal] = input_tensor->device();
    const auto &[output_device_type, output_device_ordinal] = output_tensor->device();

    switch (GetTransferType(input_device_type, output_device_type))
    {
        case TransferType::HOST_TO_HOST:
        {
            EnsureContiguous(input_tensor, "input_tensor");
            EnsureContiguous(output_tensor, "output_tensor");
            HostToHostCopy(output_tensor->dataptr(), input_tensor->dataptr(), size_in_bytes);
            break;
        }
        case TransferType::DEVICE_TO_HOST:
        {
            if (input_contiguous && output_contiguous)
            {
                EnsureContiguous(output_tensor, "output_tensor");
                EnsureContiguous(input_tensor, "input_tensor");
                DeviceToHostCopy(output_tensor->dataptr(), input_tensor->dataptr(), size_in_bytes, input_device_ordinal,
                                 stream);
            }
            else
            {
                const StridedCopyConfig config{.src_type = MemoryType::DEVICE,
                                         .dst_type = MemoryType::HOST,
                                         .src_device_ordinal = input_device_ordinal,
                                         .dst_device_ordinal = output_device_ordinal};
                StridedCopySlow(output_tensor->dataptr(), input_tensor->dataptr(), input_tensor->shape(),
                                output_tensor->shape(), input_tensor->strides(), output_tensor->strides(),
                                element_size, config, stream);
            }
            break;
        }
        case TransferType::DEVICE_TO_DEVICE:
        {
            if (input_contiguous && output_contiguous)
            {
                // if both tensors are contiguous, we can use the fast path
                DeviceToDeviceCopyFast(output_tensor->dataptr(), input_tensor->dataptr(), size_in_bytes,
                                       input_device_ordinal, output_device_ordinal, stream);
            }
            else
            {
                // fallback to slow path respecting strides
                const StridedCopyConfig config{.src_type = MemoryType::DEVICE,
                                         .dst_type = MemoryType::DEVICE,
                                         .src_device_ordinal = input_device_ordinal,
                                         .dst_device_ordinal = output_device_ordinal};
                StridedCopySlow(output_tensor->dataptr(), input_tensor->dataptr(), input_tensor->shape(),
                                output_tensor->shape(), input_tensor->strides(), output_tensor->strides(),
                                element_size, config, stream);
            }
            break;
        }
        case TransferType::HOST_TO_DEVICE:
        {
            if (input_contiguous && output_contiguous)
            {
                EnsureContiguous(input_tensor, "input_tensor");
                EnsureContiguous(output_tensor, "output_tensor");
                HostToDeviceCopy(output_tensor->dataptr(), input_tensor->dataptr(), size_in_bytes, output_device_ordinal,
                                 stream);
            }
            else
            {
                const StridedCopyConfig config{.src_type = MemoryType::HOST,
                                         .dst_type = MemoryType::DEVICE,
                                         .src_device_ordinal = input_device_ordinal,
                                         .dst_device_ordinal = output_device_ordinal};
                StridedCopySlow(output_tensor->dataptr(), input_tensor->dataptr(), input_tensor->shape(),
                                output_tensor->shape(), input_tensor->strides(), output_tensor->strides(),
                                element_size, config, stream);
            }
            break;
        }
        default:
        {
            throw std::runtime_error("Unsupported transfer type.");
        }
    }

    auto update_last_stream = [stream](const std::shared_ptr<RealTensor> &tensor)
    {
        if (!tensor || tensor->device().device_type != DeviceType::GPU || stream == nullptr)
        {
            return;
        }
        const auto stream_bundle = pi::tensorlib::internal::ctxmgmt::GetStreamBundle(tensor->device().ordinal);
        if (!stream_bundle)
        {
            return;
        }
        const auto stream_id_opt = stream_bundle->getStreamId(stream);
        if (!stream_id_opt.has_value())
        {
            return;
        }
        tensor->storage()->setLastStreamId(*stream_id_opt);
    };
    update_last_stream(input_tensor);
    update_last_stream(output_tensor);
}
