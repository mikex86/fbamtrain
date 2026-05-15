#include "passes.h"

#include "passes/pass_utils.h"
#include <kernels/kernel_binaries.h>

#include <any>
#include <array>
#include <limits>
#include <sstream>
#include <string>

static pi::tensorlib::ComputeKernelDescriptor
CreateAvgPool1dComputeKernelDescriptor(const uint32_t kernel_size, const uint32_t stride, const uint32_t pool_dim,
                                       const pi::tensorlib::DataType dtype)
{
    const auto kernel = SelectKernelForHalf(kavg_pool1d_bf16, kavg_pool1d_fp16, dtype);
    const std::string kernel_name = std::string("avg_pool1d_") + std::string(KernelSuffixForHalf(dtype));
    const std::string dtype_name = pi::tensorlib::GetDataTypeName(dtype);

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [kernel_size, stride, pool_dim, kernel, dtype,
                              dtype_name](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                          const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1)
            {
                throw std::runtime_error("avg_pool1d expects a single input tensor");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("avg_pool1d expects a single output tensor");
            }

            const auto &input = inputs[0];
            const auto &output = outputs[0];

            if (input->dtype() != dtype || output->dtype() != dtype)
            {
                throw std::runtime_error("avg_pool1d currently supports " + dtype_name + " tensors only");
            }

            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("avg_pool1d requires GPU tensors");
            }

            if (input->shape().ndims() != 3 || output->shape().ndims() != 3)
            {
                throw std::runtime_error("avg_pool1d expects 3D tensors shaped (N, C, L)");
            }

            if (pool_dim >= 3)
            {
                throw std::runtime_error("avg_pool1d pool_dim must be less than 3");
            }

            if (kernel_size == 0 || stride == 0)
            {
                throw std::runtime_error("avg_pool1d kernel_size and stride must be > 0");
            }
            if (kernel_size > kernel.meta.max_kernel_size)
            {
                throw std::runtime_error("avg_pool1d kernel_size exceeds compiled maximum");
            }

            const auto batches64 = input->shape()[0];
            const auto channels64 = input->shape()[1];
            const auto length_in64 = input->shape()[2];
            const auto output_length_dim64 = output->shape()[pool_dim];

            if (pool_dim != 0 && output->shape()[0] != batches64)
            {
                throw std::runtime_error("avg_pool1d output batch dimension must match input");
            }
            if (pool_dim != 1 && output->shape()[1] != channels64)
            {
                throw std::runtime_error("avg_pool1d output channel dimension must match input");
            }
            if (pool_dim != 2 && output->shape()[2] != length_in64)
            {
                throw std::runtime_error("avg_pool1d output trailing dimension must match input");
            }

            const auto pool_length_in64 = input->shape()[pool_dim];
            if (pool_length_in64 < kernel_size)
            {
                throw std::runtime_error("avg_pool1d input length must be >= kernel_size");
            }

            const auto expected_length_out = 1 + (pool_length_in64 - kernel_size) / stride;
            if (expected_length_out != output_length_dim64)
            {
                throw std::runtime_error("avg_pool1d output length does not match kernel configuration");
            }

            const auto &input_strides = input->strides();
            const auto &output_strides = output->strides();
            if (input_strides.strides().size() != 3 || output_strides.strides().size() != 3)
            {
                throw std::runtime_error("avg_pool1d expects 3D stride information");
            }
            auto to_u32 = [](const uint64_t value, const char *what) -> uint32_t
            {
                if (value > std::numeric_limits<uint32_t>::max())
                {
                    std::stringstream ss;
                    ss << what << " exceeds supported range";
                    throw std::runtime_error(ss.str());
                }
                return static_cast<uint32_t>(value);
            };

            const auto batches = to_u32(batches64, "avg_pool1d batch size");
            const auto channels = to_u32(channels64, "avg_pool1d channel size");
            const auto length_in = to_u32(length_in64, "avg_pool1d trailing input length");
            const auto pool_length_in = to_u32(pool_length_in64, "avg_pool1d pooling input length");
            const auto pool_length_out = to_u32(output_length_dim64, "avg_pool1d pooling output length");

            const auto stride_in_n = to_u32(input_strides[0], "avg_pool1d input stride dim0");
            const auto stride_in_c = to_u32(input_strides[1], "avg_pool1d input stride dim1");
            const auto stride_in_l = to_u32(input_strides[2], "avg_pool1d input stride dim2");

            const auto stride_out_n = to_u32(output_strides[0], "avg_pool1d output stride dim0");
            const auto stride_out_c = to_u32(output_strides[1], "avg_pool1d output stride dim1");
            const auto stride_out_l = to_u32(output_strides[2], "avg_pool1d output stride dim2");

            const uint32_t pool_stride_in = pool_dim == 0 ? stride_in_n : (pool_dim == 1 ? stride_in_c : stride_in_l);
            const uint32_t pool_stride_out =
                pool_dim == 0 ? stride_out_n : (pool_dim == 1 ? stride_out_c : stride_out_l);

            const std::array<uint32_t, 3> dims{batches, channels, length_in};
            const std::array<uint32_t, 3> in_strides{stride_in_n, stride_in_c, stride_in_l};
            const std::array<uint32_t, 3> out_strides{stride_out_n, stride_out_c, stride_out_l};

            std::array<uint32_t, 2> outer_sizes{};
            std::array<uint32_t, 2> outer_in_strides{};
            std::array<uint32_t, 2> outer_out_strides{};
            {
                uint32_t outer_index = 0;
                for (uint32_t dim = 0; dim < 3; ++dim)
                {
                    if (dim == pool_dim)
                    {
                        continue;
                    }
                    outer_sizes[outer_index] = dims[dim];
                    outer_in_strides[outer_index] = in_strides[dim];
                    outer_out_strides[outer_index] = out_strides[dim];
                    ++outer_index;
                }
            }

            const auto kernel_size_u32 = to_u32(kernel_size, "avg_pool1d kernel_size");
            const auto stride_u32 = to_u32(stride, "avg_pool1d stride");
            const auto outer_size0 = outer_sizes[0];
            const auto outer_size1 = outer_sizes[1];

            const auto device_ordinal = ValidateSameDeviceOrdinal("avg_pool1d", {input, output});

            void *input_ptr = input->dataptr();
            void *output_ptr = output->dataptr();

            const float inv_kernel_size = 1.0f / static_cast<float>(kernel_size_u32);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {input_ptr, output_ptr, outer_in_strides[0], outer_in_strides[1], pool_stride_in,
                         outer_out_strides[0], outer_out_strides[1], pool_stride_out, outer_size0, outer_size1,
                         pool_length_in, pool_length_out, kernel_size_u32, stride_u32, inv_kernel_size,
                         static_cast<void *>(nullptr)},
                .grid_dim_x = ((pool_length_out + kernel.meta.block_size - 1) / kernel.meta.block_size),
                .grid_dim_y = outer_size0 * outer_size1,
                .grid_dim_z = 1,
                .block_dim_x = TRITON_WARP_SIZE * kernel.num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kernel.shared_mem_bytes,
                .device_ordinal = device_ordinal};
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel.data, kernel.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel.data, kernel.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor
CreateAvgPool2dComputeKernelDescriptor(const pi::tensorlib::DataType dtype, const std::array<uint32_t, 2> &kernel_size,
                                       const std::array<uint32_t, 2> &stride, const std::array<uint32_t, 2> &padding,
                                       bool channels_last)
{
    const bool is_nhwc_2x2_s2 =
        channels_last && kernel_size[0] == 2 && kernel_size[1] == 2 && stride[0] == 2 && stride[1] == 2 &&
        padding[0] == 0 && padding[1] == 0;
#if PI_TENSORLIB_ENABLE_CUDA
    const bool use_nhwc_2x2_s2 = is_nhwc_2x2_s2;
    const auto kernel = use_nhwc_2x2_s2
                            ? SelectKernelForHalf(kavg_pool2d_nhwc_2x2_s2_bf16, kavg_pool2d_nhwc_2x2_s2_fp16, dtype)
                            : SelectKernelForHalf(kavg_pool2d_bf16, kavg_pool2d_fp16, dtype);
    const std::string kernel_name =
        use_nhwc_2x2_s2 ? std::string("avg_pool2d_nhwc_2x2_s2_") + std::string(KernelSuffixForHalf(dtype))
                        : std::string("avg_pool2d_") + std::string(KernelSuffixForHalf(dtype));
#else
    const bool use_nhwc_2x2_s2 = false;
    const auto kernel = SelectKernelForHalf(kavg_pool2d_bf16, kavg_pool2d_fp16, dtype);
    const std::string kernel_name = std::string("avg_pool2d_") + std::string(KernelSuffixForHalf(dtype));
#endif

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [kernel, kernel_size, stride, padding, channels_last, use_nhwc_2x2_s2,
                              dtype](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                     const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1)
            {
                throw std::runtime_error("avg_pool2d expects a single input tensor");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("avg_pool2d expects a single output tensor");
            }

            const auto &input = inputs[0];
            const auto &output = outputs[0];

            if (input->dtype() != dtype || output->dtype() != dtype)
            {
                throw std::runtime_error("avg_pool2d currently supports BFLOAT16 or FLOAT16 tensors");
            }
            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("avg_pool2d requires GPU tensors");
            }

            if (input->shape().ndims() != 4 || output->shape().ndims() != 4)
            {
                throw std::runtime_error("avg_pool2d expects 4D tensors shaped (N, C, H, W) or (N, H, W, C)");
            }

            if (kernel_size[0] == 0 || kernel_size[1] == 0)
            {
                throw std::runtime_error("avg_pool2d kernel dimensions must be > 0");
            }
            if (stride[0] == 0 || stride[1] == 0)
            {
                throw std::runtime_error("avg_pool2d stride dimensions must be > 0");
            }
            if (kernel_size[0] > kernel.meta.max_kernel_h || kernel_size[1] > kernel.meta.max_kernel_w)
            {
                throw std::runtime_error("avg_pool2d kernel dimensions exceed compiled maximum");
            }

            const size_t batch_dim = 0;
            const size_t channel_dim = channels_last ? 3 : 1;
            const size_t height_dim = channels_last ? 1 : 2;
            const size_t width_dim = channels_last ? 2 : 3;

            const auto batches64 = input->shape()[batch_dim];
            const auto channels64 = input->shape()[channel_dim];
            const auto height_in64 = input->shape()[height_dim];
            const auto width_in64 = input->shape()[width_dim];

            if (output->shape()[batch_dim] != batches64)
            {
                throw std::runtime_error("avg_pool2d output batch dimension must match input");
            }
            if (output->shape()[channel_dim] != channels64)
            {
                throw std::runtime_error("avg_pool2d output channel dimension must match input");
            }

            const auto pad_h = static_cast<uint64_t>(padding[0]);
            const auto pad_w = static_cast<uint64_t>(padding[1]);
            const auto kernel_h = static_cast<uint64_t>(kernel_size[0]);
            const auto kernel_w = static_cast<uint64_t>(kernel_size[1]);
            const auto stride_h = static_cast<uint64_t>(stride[0]);
            const auto stride_w = static_cast<uint64_t>(stride[1]);

            if (height_in64 + 2 * pad_h < kernel_h || width_in64 + 2 * pad_w < kernel_w)
            {
                throw std::runtime_error("avg_pool2d kernel does not fit within padded input dimensions");
            }

            const auto expected_height_out64 = 1 + (height_in64 + 2 * pad_h - kernel_h) / stride_h;
            const auto expected_width_out64 = 1 + (width_in64 + 2 * pad_w - kernel_w) / stride_w;

            if (expected_height_out64 == 0 || expected_width_out64 == 0)
            {
                throw std::runtime_error("avg_pool2d expected output dimensions must be positive");
            }
            if (output->shape()[height_dim] != expected_height_out64)
            {
                throw std::runtime_error("avg_pool2d output height does not match kernel configuration");
            }
            if (output->shape()[width_dim] != expected_width_out64)
            {
                throw std::runtime_error("avg_pool2d output width does not match kernel configuration");
            }

            const auto &input_strides = input->strides();
            const auto &output_strides = output->strides();
            if (input_strides.strides().size() != 4 || output_strides.strides().size() != 4)
            {
                throw std::runtime_error("avg_pool2d expects 4D stride information");
            }

            auto to_u32 = [](const uint64_t value, const char *what) -> uint32_t
            {
                if (value > std::numeric_limits<uint32_t>::max())
                {
                    std::stringstream ss;
                    ss << what << " exceeds supported range";
                    throw std::runtime_error(ss.str());
                }
                return static_cast<uint32_t>(value);
            };

            const auto batches = to_u32(batches64, "avg_pool2d batch size");
            const auto channels = to_u32(channels64, "avg_pool2d channel size");
            const auto height_in = to_u32(height_in64, "avg_pool2d input height");
            const auto width_in = to_u32(width_in64, "avg_pool2d input width");
            const auto height_out = to_u32(expected_height_out64, "avg_pool2d output height");
            const auto width_out = to_u32(expected_width_out64, "avg_pool2d output width");

            const auto stride_in_n = to_u32(input_strides[batch_dim], "avg_pool2d input stride batch dim");
            const auto stride_in_c = to_u32(input_strides[channel_dim], "avg_pool2d input stride channel dim");
            const auto stride_in_h = to_u32(input_strides[height_dim], "avg_pool2d input stride height dim");
            const auto stride_in_w = to_u32(input_strides[width_dim], "avg_pool2d input stride width dim");

            const auto stride_out_n = to_u32(output_strides[batch_dim], "avg_pool2d output stride batch dim");
            const auto stride_out_c = to_u32(output_strides[channel_dim], "avg_pool2d output stride channel dim");
            const auto stride_out_h = to_u32(output_strides[height_dim], "avg_pool2d output stride height dim");
            const auto stride_out_w = to_u32(output_strides[width_dim], "avg_pool2d output stride width dim");

            if (use_nhwc_2x2_s2 && (stride_in_c != 1 || stride_out_c != 1))
            {
                throw std::runtime_error("avg_pool2d NHWC 2x2 stride-2 kernel requires contiguous channel dimension");
            }

            const auto kernel_h_u32 = to_u32(kernel_h, "avg_pool2d kernel height");
            const auto kernel_w_u32 = to_u32(kernel_w, "avg_pool2d kernel width");
            const auto stride_h_u32 = to_u32(stride_h, "avg_pool2d stride height");
            const auto stride_w_u32 = to_u32(stride_w, "avg_pool2d stride width");
            const auto padding_h_u32 = to_u32(pad_h, "avg_pool2d padding height");
            const auto padding_w_u32 = to_u32(pad_w, "avg_pool2d padding width");

            const uint64_t outer_product64 = static_cast<uint64_t>(batches) * static_cast<uint64_t>(channels);
            const auto outer_product = to_u32(outer_product64, "avg_pool2d outer dimension product");

            const auto device_ordinal = ValidateSameDeviceOrdinal("avg_pool2d", {input, output});

            void *input_ptr = input->dataptr();
            void *output_ptr = output->dataptr();

            const float inv_kernel_size =
                1.0f / static_cast<float>(static_cast<uint64_t>(kernel_h_u32) * static_cast<uint64_t>(kernel_w_u32));

            uint64_t grid_x = 0;
            if (use_nhwc_2x2_s2)
            {
                const uint64_t total_hw64 = static_cast<uint64_t>(height_out) * static_cast<uint64_t>(width_out);
                const auto total_hw = to_u32(total_hw64, "avg_pool2d output spatial product");
                const uint32_t blocks_hw = (total_hw + kernel.meta.block_size_h - 1) / kernel.meta.block_size_h;
                const uint32_t blocks_c = (channels + kernel.meta.block_size_w - 1) / kernel.meta.block_size_w;
                grid_x =
                    static_cast<uint64_t>(batches) * static_cast<uint64_t>(blocks_hw) * static_cast<uint64_t>(blocks_c);
            }
            else
            {
                const uint32_t blocks_w = (width_out + kernel.meta.block_size_w - 1) / kernel.meta.block_size_w;
                const uint32_t blocks_h = (height_out + kernel.meta.block_size_h - 1) / kernel.meta.block_size_h;
                grid_x = static_cast<uint64_t>(blocks_w) * static_cast<uint64_t>(blocks_h) *
                         static_cast<uint64_t>(outer_product);
            }
            if (grid_x > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("avg_pool2d grid dimension overflow");
            }

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {input_ptr,     output_ptr,    stride_in_n,     stride_in_c,
                         stride_in_h,   stride_in_w,   stride_out_n,    stride_out_c,
                         stride_out_h,  stride_out_w,  batches,         channels,
                         height_in,     width_in,      height_out,      width_out,
                         kernel_h_u32,  kernel_w_u32,  stride_h_u32,    stride_w_u32,
                         padding_h_u32, padding_w_u32, inv_kernel_size, static_cast<void *>(nullptr)},
                .grid_dim_x = static_cast<uint32_t>(grid_x),
                .grid_dim_y = 1,
                .grid_dim_z = 1,
                .block_dim_x = TRITON_WARP_SIZE * kernel.num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kernel.shared_mem_bytes,
                .device_ordinal = device_ordinal};
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel.data, kernel.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel.data, kernel.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor
CreateAvgPool2dBwdComputeKernelDescriptor(const pi::tensorlib::DataType dtype, const std::array<uint32_t, 2> &kernel_size,
                                          const std::array<uint32_t, 2> &stride, const std::array<uint32_t, 2> &padding,
                                          bool channels_last, const bool accumulate)
{
    const bool is_nhwc_2x2_s2_noaccum =
        !accumulate && channels_last && kernel_size[0] == 2 && kernel_size[1] == 2 && stride[0] == 2 &&
        stride[1] == 2 && padding[0] == 0 && padding[1] == 0;
#if PI_TENSORLIB_ENABLE_CUDA
    const bool use_nhwc_2x2_s2_noaccum = is_nhwc_2x2_s2_noaccum;
    const auto kernel =
        use_nhwc_2x2_s2_noaccum
            ? SelectKernelForHalf(kavg_pool2d_bwd_noaccum_nhwc_2x2_s2_bf16,
                                  kavg_pool2d_bwd_noaccum_nhwc_2x2_s2_fp16, dtype)
            : (accumulate ? SelectKernelForHalf(kavg_pool2d_bwd_bf16, kavg_pool2d_bwd_fp16, dtype)
                          : SelectKernelForHalf(kavg_pool2d_bwd_noaccum_bf16, kavg_pool2d_bwd_noaccum_fp16, dtype));
#else
    const bool use_nhwc_2x2_s2_noaccum = false;
    const auto kernel =
        accumulate ? SelectKernelForHalf(kavg_pool2d_bwd_bf16, kavg_pool2d_bwd_fp16, dtype)
                   : SelectKernelForHalf(kavg_pool2d_bwd_noaccum_bf16, kavg_pool2d_bwd_noaccum_fp16, dtype);
#endif
    const std::string kernel_name =
        use_nhwc_2x2_s2_noaccum
            ? std::string("avg_pool2d_bwd_noaccum_nhwc_2x2_s2_") + std::string(KernelSuffixForHalf(dtype))
            : (accumulate ? std::string("avg_pool2d_bwd_") + std::string(KernelSuffixForHalf(dtype))
                          : std::string("avg_pool2d_bwd_noaccum_") + std::string(KernelSuffixForHalf(dtype)));

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [kernel, kernel_size, stride, padding, channels_last, use_nhwc_2x2_s2_noaccum,
                              dtype, accumulate](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                     const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1)
            {
                throw std::runtime_error("avg_pool2d_bwd expects a single upstream tensor");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("avg_pool2d_bwd expects a single grad_input tensor");
            }

            const auto &upstream = inputs[0];
            const auto &grad_input = outputs[0];

            if (upstream->dtype() != dtype || grad_input->dtype() != dtype)
            {
                throw std::runtime_error("avg_pool2d_bwd currently supports BFLOAT16 or FLOAT16 tensors");
            }
            if (upstream->device().device_type != pi::tensorlib::DeviceType::GPU ||
                grad_input->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("avg_pool2d_bwd requires GPU tensors");
            }

            if (upstream->shape().ndims() != 4 || grad_input->shape().ndims() != 4)
            {
                throw std::runtime_error("avg_pool2d_bwd expects 4D tensors shaped (N, C, H, W) or (N, H, W, C)");
            }

            if (kernel_size[0] == 0 || kernel_size[1] == 0)
            {
                throw std::runtime_error("avg_pool2d_bwd kernel dimensions must be > 0");
            }
            if (stride[0] == 0 || stride[1] == 0)
            {
                throw std::runtime_error("avg_pool2d_bwd stride dimensions must be > 0");
            }
            if (!accumulate && (stride[0] != kernel_size[0] || stride[1] != kernel_size[1]))
            {
                throw std::runtime_error("avg_pool2d_bwd non-accumulating kernel requires stride == kernel size");
            }
            if (kernel_size[0] > kernel.meta.max_kernel_h || kernel_size[1] > kernel.meta.max_kernel_w)
            {
                throw std::runtime_error("avg_pool2d_bwd kernel dimensions exceed compiled maximum");
            }

            const size_t batch_dim = 0;
            const size_t channel_dim = channels_last ? 3 : 1;
            const size_t height_dim = channels_last ? 1 : 2;
            const size_t width_dim = channels_last ? 2 : 3;

            const auto batches64 = grad_input->shape()[batch_dim];
            const auto channels64 = grad_input->shape()[channel_dim];
            const auto height_in64 = grad_input->shape()[height_dim];
            const auto width_in64 = grad_input->shape()[width_dim];

            if (upstream->shape()[batch_dim] != batches64)
            {
                throw std::runtime_error("avg_pool2d_bwd upstream batch dimension must match input");
            }
            if (upstream->shape()[channel_dim] != channels64)
            {
                throw std::runtime_error("avg_pool2d_bwd upstream channel dimension must match input");
            }

            const auto pad_h = static_cast<uint64_t>(padding[0]);
            const auto pad_w = static_cast<uint64_t>(padding[1]);
            const auto kernel_h = static_cast<uint64_t>(kernel_size[0]);
            const auto kernel_w = static_cast<uint64_t>(kernel_size[1]);
            const auto stride_h = static_cast<uint64_t>(stride[0]);
            const auto stride_w = static_cast<uint64_t>(stride[1]);

            if (height_in64 + 2 * pad_h < kernel_h || width_in64 + 2 * pad_w < kernel_w)
            {
                throw std::runtime_error("avg_pool2d_bwd kernel does not fit within padded input dimensions");
            }
            if (!accumulate)
            {
                if ((height_in64 + 2 * pad_h) % stride_h != 0 || (width_in64 + 2 * pad_w) % stride_w != 0)
                {
                    throw std::runtime_error("avg_pool2d_bwd non-accumulating kernel requires full coverage");
                }
            }

            const auto expected_height_out64 = 1 + (height_in64 + 2 * pad_h - kernel_h) / stride_h;
            const auto expected_width_out64 = 1 + (width_in64 + 2 * pad_w - kernel_w) / stride_w;

            if (expected_height_out64 == 0 || expected_width_out64 == 0)
            {
                throw std::runtime_error("avg_pool2d_bwd expected output dimensions must be positive");
            }
            if (upstream->shape()[height_dim] != expected_height_out64)
            {
                throw std::runtime_error("avg_pool2d_bwd upstream height does not match kernel configuration");
            }
            if (upstream->shape()[width_dim] != expected_width_out64)
            {
                throw std::runtime_error("avg_pool2d_bwd upstream width does not match kernel configuration");
            }

            const auto &upstream_strides = upstream->strides();
            const auto &grad_strides = grad_input->strides();
            if (upstream_strides.strides().size() != 4 || grad_strides.strides().size() != 4)
            {
                throw std::runtime_error("avg_pool2d_bwd expects 4D stride information");
            }

            auto to_u32 = [](const uint64_t value, const char *what) -> uint32_t
            {
                if (value > std::numeric_limits<uint32_t>::max())
                {
                    std::stringstream ss;
                    ss << what << " exceeds supported range";
                    throw std::runtime_error(ss.str());
                }
                return static_cast<uint32_t>(value);
            };

            const auto batches = to_u32(batches64, "avg_pool2d_bwd batch size");
            const auto channels = to_u32(channels64, "avg_pool2d_bwd channel size");
            const auto height_in = to_u32(height_in64, "avg_pool2d_bwd input height");
            const auto width_in = to_u32(width_in64, "avg_pool2d_bwd input width");
            const auto height_out = to_u32(expected_height_out64, "avg_pool2d_bwd output height");
            const auto width_out = to_u32(expected_width_out64, "avg_pool2d_bwd output width");

            const auto stride_up_n = to_u32(upstream_strides[batch_dim], "avg_pool2d_bwd upstream stride batch dim");
            const auto stride_up_c = to_u32(upstream_strides[channel_dim], "avg_pool2d_bwd upstream stride channel dim");
            const auto stride_up_h = to_u32(upstream_strides[height_dim], "avg_pool2d_bwd upstream stride height dim");
            const auto stride_up_w = to_u32(upstream_strides[width_dim], "avg_pool2d_bwd upstream stride width dim");

            const auto stride_grad_n = to_u32(grad_strides[batch_dim], "avg_pool2d_bwd grad stride batch dim");
            const auto stride_grad_c = to_u32(grad_strides[channel_dim], "avg_pool2d_bwd grad stride channel dim");
            const auto stride_grad_h = to_u32(grad_strides[height_dim], "avg_pool2d_bwd grad stride height dim");
            const auto stride_grad_w = to_u32(grad_strides[width_dim], "avg_pool2d_bwd grad stride width dim");

            if (use_nhwc_2x2_s2_noaccum && (stride_up_c != 1 || stride_grad_c != 1))
            {
                throw std::runtime_error(
                    "avg_pool2d_bwd NHWC 2x2 stride-2 kernel requires contiguous channel dimension");
            }

            const auto kernel_h_u32 = to_u32(kernel_h, "avg_pool2d_bwd kernel height");
            const auto kernel_w_u32 = to_u32(kernel_w, "avg_pool2d_bwd kernel width");
            const auto stride_h_u32 = to_u32(stride_h, "avg_pool2d_bwd stride height");
            const auto stride_w_u32 = to_u32(stride_w, "avg_pool2d_bwd stride width");
            const auto padding_h_u32 = to_u32(pad_h, "avg_pool2d_bwd padding height");
            const auto padding_w_u32 = to_u32(pad_w, "avg_pool2d_bwd padding width");

            const uint64_t outer_product64 = static_cast<uint64_t>(batches) * static_cast<uint64_t>(channels);
            const auto outer_product = to_u32(outer_product64, "avg_pool2d_bwd outer dimension product");

            const auto device_ordinal = ValidateSameDeviceOrdinal("avg_pool2d_bwd", {upstream, grad_input});

            const float inv_kernel_size =
                1.0f / static_cast<float>(static_cast<uint64_t>(kernel_h_u32) * static_cast<uint64_t>(kernel_w_u32));

            uint64_t grid_x = 0;
            if (use_nhwc_2x2_s2_noaccum)
            {
                const uint64_t total_hw64 = static_cast<uint64_t>(height_out) * static_cast<uint64_t>(width_out);
                const auto total_hw = to_u32(total_hw64, "avg_pool2d_bwd output spatial product");
                const uint32_t blocks_hw = (total_hw + kernel.meta.block_size_h - 1) / kernel.meta.block_size_h;
                const uint32_t blocks_c = (channels + kernel.meta.block_size_w - 1) / kernel.meta.block_size_w;
                grid_x =
                    static_cast<uint64_t>(batches) * static_cast<uint64_t>(blocks_hw) * static_cast<uint64_t>(blocks_c);
            }
            else
            {
                const uint32_t blocks_w = (width_out + kernel.meta.block_size_w - 1) / kernel.meta.block_size_w;
                const uint32_t blocks_h = (height_out + kernel.meta.block_size_h - 1) / kernel.meta.block_size_h;
                grid_x = static_cast<uint64_t>(blocks_w) * static_cast<uint64_t>(blocks_h) *
                         static_cast<uint64_t>(outer_product);
            }
            if (grid_x > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("avg_pool2d_bwd grid dimension overflow");
            }

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {upstream->dataptr(),
                         grad_input->dataptr(),
                         stride_up_n,
                         stride_up_c,
                         stride_up_h,
                         stride_up_w,
                         stride_grad_n,
                         stride_grad_c,
                         stride_grad_h,
                         stride_grad_w,
                         batches,
                         channels,
                         height_in,
                         width_in,
                         height_out,
                         width_out,
                         kernel_h_u32,
                         kernel_w_u32,
                         stride_h_u32,
                         stride_w_u32,
                         padding_h_u32,
                         padding_w_u32,
                         inv_kernel_size,
                         static_cast<void *>(nullptr)},
                .grid_dim_x = static_cast<uint32_t>(grid_x),
                .grid_dim_y = 1,
                .grid_dim_z = 1,
                .block_dim_x = TRITON_WARP_SIZE * kernel.num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kernel.shared_mem_bytes,
                .device_ordinal = device_ordinal};
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel.data, kernel.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel.data, kernel.size),
    };
}

void AvgPool1dImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &op : execution_plan.entries)
    {
        if (op.op_type == pi::tensorlib::OpType::AVG_POOL1D)
        {
            const auto kernel_size = std::any_cast<uint32_t>(op.attributes.at("kernel_size"));
            const auto stride = std::any_cast<uint32_t>(op.attributes.at("stride"));
            const auto pool_dim = std::any_cast<uint32_t>(op.attributes.at("pool_dim"));
            const auto &input = op.inputs.at(0);
            const auto &output = op.outputs.at(0);
            const auto dtype = input->dtype();
            if (!IsSupportedHalfType(dtype) || output->dtype() != dtype)
            {
                throw std::runtime_error("avg_pool1d requires BFLOAT16 or FLOAT16 tensors");
            }
            op.op_type = std::nullopt;
            op.kernel_descriptor = CreateAvgPool1dComputeKernelDescriptor(kernel_size, stride, pool_dim, dtype);
            const uint64_t out_elements = op.outputs.at(0)->shape().numel();
            const uint64_t kernel_flops = static_cast<uint64_t>(kernel_size) + 1; // sum + scale
            op.flop_estimate = out_elements * kernel_flops;
        }
    }
}

void AvgPool2dImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &op : execution_plan.entries)
    {
        if (op.op_type != pi::tensorlib::OpType::AVG_POOL2D)
        {
            continue;
        }

        const auto kernel_it = op.attributes.find("kernel_size");
        const auto stride_it = op.attributes.find("stride");
        const auto padding_it = op.attributes.find("padding");
        if (kernel_it == op.attributes.end() || stride_it == op.attributes.end() || padding_it == op.attributes.end())
        {
            throw std::runtime_error("avg_pool2d operation missing kernel_size, stride, or padding attributes");
        }

        const auto kernel_size = std::any_cast<std::array<uint32_t, 2>>(kernel_it->second);
        const auto stride = std::any_cast<std::array<uint32_t, 2>>(stride_it->second);
        const auto padding = std::any_cast<std::array<uint32_t, 2>>(padding_it->second);
        const auto channels_last_it = op.attributes.find("channels_last");
        const bool channels_last =
            channels_last_it != op.attributes.end() ? std::any_cast<bool>(channels_last_it->second) : false;

        if (op.inputs.empty() || op.outputs.empty())
        {
            throw std::runtime_error("avg_pool2d operation requires input and output tensors");
        }
        const auto &input = op.inputs[0];
        const auto &output = op.outputs[0];
        const auto dtype = input->dtype();
        if (!IsSupportedHalfType(dtype) || output->dtype() != dtype)
        {
            throw std::runtime_error("avg_pool2d requires BFLOAT16 or FLOAT16 tensors");
        }

        op.op_type = std::nullopt;
        op.kernel_descriptor =
            CreateAvgPool2dComputeKernelDescriptor(dtype, kernel_size, stride, padding, channels_last);
        const uint64_t out_elements = op.outputs.at(0)->shape().numel();
        const uint64_t kernel_elems = static_cast<uint64_t>(kernel_size[0]) * static_cast<uint64_t>(kernel_size[1]);
        op.flop_estimate = out_elements * (kernel_elems + 1); // sum pool + apply scale
    }
}

void AvgPool2dBwdImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &op : execution_plan.entries)
    {
        if (op.op_type != pi::tensorlib::OpType::AVG_POOL2D_BWD)
        {
            continue;
        }

        const auto kernel_it = op.attributes.find("kernel_size");
        const auto stride_it = op.attributes.find("stride");
        const auto padding_it = op.attributes.find("padding");
        if (kernel_it == op.attributes.end() || stride_it == op.attributes.end() || padding_it == op.attributes.end())
        {
            throw std::runtime_error("avg_pool2d_bwd operation missing kernel_size, stride, or padding attributes");
        }

        const auto kernel_size = std::any_cast<std::array<uint32_t, 2>>(kernel_it->second);
        const auto stride = std::any_cast<std::array<uint32_t, 2>>(stride_it->second);
        const auto padding = std::any_cast<std::array<uint32_t, 2>>(padding_it->second);
        const auto channels_last_it = op.attributes.find("channels_last");
        const bool channels_last =
            channels_last_it != op.attributes.end() ? std::any_cast<bool>(channels_last_it->second) : false;
        const auto accumulate_it = op.attributes.find("accumulate");
        const bool accumulate =
            accumulate_it != op.attributes.end() ? std::any_cast<bool>(accumulate_it->second) : true;

        if (op.inputs.empty() || op.outputs.empty())
        {
            throw std::runtime_error("avg_pool2d_bwd operation requires input and output tensors");
        }
        const auto &upstream = op.inputs[0];
        const auto &grad_input = op.outputs[0];
        const auto dtype = upstream->dtype();
        if (!IsSupportedHalfType(dtype) || grad_input->dtype() != dtype)
        {
            throw std::runtime_error("avg_pool2d_bwd requires BFLOAT16 or FLOAT16 tensors");
        }

        op.op_type = std::nullopt;
        op.kernel_descriptor =
            CreateAvgPool2dBwdComputeKernelDescriptor(dtype, kernel_size, stride, padding, channels_last, accumulate);
        const uint64_t out_elements = op.inputs.at(0)->shape().numel();
        const uint64_t kernel_elems = static_cast<uint64_t>(kernel_size[0]) * static_cast<uint64_t>(kernel_size[1]);
        op.flop_estimate = out_elements * (kernel_elems + 1); // scale + scatter adds
    }
}
