#include "functional.h"
#include "op_graph.h"
#include "shape_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

void pi::tensorlib::DeviceCopy(OpGraph &graph, const TraceTensor &input, const TraceTensor &output,
                               const GpuStreamDescriptor &stream_desc)
{
    graph.recordOperation(OperationEntry{.type = OpType::DEVICE_COPY,
                                         .inputs = {input},
                                         .outputs = {output},
                                         .attributes = {},
                                         .gpu_stream_desc = stream_desc});
}

void pi::tensorlib::FillZeros(OpGraph &graph, TraceTensor &tensor, const GpuStreamDescriptor &compute_stream_descriptor)
{
    graph.recordOperation(OperationEntry{.type = OpType::FILL_ZEROS,
                                         .inputs = {tensor},
                                         .outputs = {tensor},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});
}

void pi::tensorlib::FillUniform(OpGraph &graph, TraceTensor &tensor, float min, float max, uint32_t seed,
                                const GpuStreamDescriptor &compute_stream_descriptor)
{
    if (const auto dtype = tensor.dtype();
        dtype != DataType::FLOAT32 && dtype != DataType::BFLOAT16 && dtype != DataType::FLOAT16)
    {
        throw std::invalid_argument("FillUniform only supports FLOAT32, BFLOAT16, and FLOAT16");
    }
    std::unordered_map<std::string, std::any> attributes{};
    attributes.reserve(4);

    attributes.emplace("min_value", min);
    attributes.emplace("max_value", max);
    attributes.emplace("seed", seed);

    graph.recordOperation(OperationEntry{.type = OpType::FILL_UNIFORM,
                                         .inputs = {tensor},
                                         .outputs = {tensor},
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});
}

void pi::tensorlib::FillConstant(OpGraph &graph, TraceTensor &tensor, float value,
                                 const GpuStreamDescriptor &compute_stream_descriptor)
{
    std::unordered_map<std::string, std::any> attributes{};
    attributes.reserve(2);

    attributes.emplace("value", value);

    graph.recordOperation(OperationEntry{.type = OpType::FILL_CONSTANT,
                                         .inputs = {tensor},
                                         .outputs = {tensor},
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});
}

void pi::tensorlib::FillNormal(OpGraph &graph, TraceTensor &tensor, const float mean, const float std, uint32_t seed,
                               const GpuStreamDescriptor &compute_stream_descriptor)
{
    std::unordered_map<std::string, std::any> attributes{};
    attributes.reserve(3);

    attributes.emplace("mean", mean);
    attributes.emplace("std", std);
    attributes.emplace("seed", seed);

    graph.recordOperation(OperationEntry{.type = OpType::FILL_NORMAL,
                                         .inputs = {tensor},
                                         .outputs = {tensor},
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});
}

void pi::tensorlib::KaimingUniformInit(OpGraph &graph, TraceTensor &tensor, const uint32_t features,
                                       const uint32_t seed, const GpuStreamDescriptor &compute_stream_descriptor)
{
    const float k = 1.0f / static_cast<float>(features);
    const float bound = std::sqrt(k);
    FillUniform(graph, tensor, -bound, bound, seed, compute_stream_descriptor);
}

void pi::tensorlib::OptimizerSgd(OpGraph &graph, TraceTensor &param, const TraceTensor &grad, TraceTensor &velocity,
                                 const float learning_rate, const float momentum, const float weight_decay,
                                 const bool nesterov, const GpuStreamDescriptor &compute_stream_descriptor)
{
    std::unordered_map<std::string, std::any> attributes{};
    attributes.reserve(5);
    attributes.emplace("learning_rate", learning_rate);
    attributes.emplace("momentum", momentum);
    attributes.emplace("weight_decay", weight_decay);
    attributes.emplace("nesterov", nesterov ? 1 : 0);

    graph.recordOperation(OperationEntry{.type = OpType::OPTIMIZER_SGD,
                                         .inputs = {param, grad, velocity},
                                         .outputs = {param, velocity},
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});
}

void pi::tensorlib::OptimizerAdamw(OpGraph &graph, TraceTensor &param, const TraceTensor &grad, TraceTensor &m,
                                   TraceTensor &v, const TraceTensor &bias_correction1,
                                   const TraceTensor &bias_correction2, const float learning_rate, const float beta1,
                                   const float beta2, const float eps, const float weight_decay,
                                   const GpuStreamDescriptor &compute_stream_descriptor)
{
    std::unordered_map<std::string, std::any> attributes{};
    attributes.reserve(6);
    attributes.emplace("learning_rate", learning_rate);
    attributes.emplace("beta1", beta1);
    attributes.emplace("beta2", beta2);
    attributes.emplace("eps", eps);
    attributes.emplace("weight_decay", weight_decay);

    graph.recordOperation(OperationEntry{.type = OpType::OPTIMIZER_ADAMW,
                                         .inputs = {param, grad, m, v, bias_correction1, bias_correction2},
                                         .outputs = {param, m, v},
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});
}

pi::tensorlib::TraceTensor
pi::tensorlib::ScaledDotProductAttentionFwd(OpGraph &graph, const TraceTensor &query, const TraceTensor &key,
                                            const TraceTensor &value, float softmax_scale, bool causal,
                                            const bool use_fp16_flash_attn_acc, std::optional<TraceTensor> scratch_out,
                                            const GpuStreamDescriptor &compute_stream_descriptor)
{
    // assert query, key, value are on the same device
    if (query.device() != key.device() || query.device() != value.device())
    {
        throw std::runtime_error("query, key, value must be on the same device");
    }

    // assert query, key, value have the same dtype
    if (query.dtype() != key.dtype() || query.dtype() != value.dtype())
    {
        throw std::invalid_argument("query, key, value must have the same dtype");
    }

    // assert query, key, value have 4 dimensions
    if (query.shape().ndims() != 4 || key.shape().ndims() != 4 || value.shape().ndims() != 4)
    {
        throw std::invalid_argument("query, key, value must have 4 dimensions");
    }

    // assert query, key, value have the same batch size
    if (query.shape()[0] != key.shape()[0] || query.shape()[0] != value.shape()[0])
    {
        throw std::invalid_argument("query, key, value must have the same batch size");
    }

    // assert query, key, value have the same number of heads
    if (query.shape()[2] != key.shape()[2] || query.shape()[2] != value.shape()[2])
    {
        throw std::invalid_argument("query, key, value must have the same head dimension");
    }

    // assert query, key, value have the same sequence length
    if (query.shape()[1] != key.shape()[1] || query.shape()[1] != value.shape()[1])
    {
        throw std::invalid_argument("query, key, value must have the same sequence length");
    }

    // assert query, key, value have the same head size
    if (query.shape()[3] != key.shape()[3] || query.shape()[3] != value.shape()[3])
    {
        throw std::invalid_argument("query, key, value must have the same head size");
    }

    const auto b = query.shape()[0];
    const auto t = query.shape()[1];
    const auto h = query.shape()[2];
    const auto hs = query.shape()[3];

    const auto dtype = query.dtype();
    const auto device = query.device();

    std::unordered_map<std::string, std::any> attributes{};
    attributes.reserve(3);

    attributes.emplace("softmax_scale", softmax_scale);
    attributes.emplace("causal", causal);
    if (use_fp16_flash_attn_acc && dtype == DataType::FLOAT16)
    {
        attributes.emplace("use_fp16_flash_attn_acc", true);
    }

    TraceTensor output = graph.createTensor({b, t, h, hs}, dtype, device, compute_stream_descriptor, false);
    if (scratch_out.has_value())
    {
        TraceTensor scratch = *scratch_out;
        if (scratch.device() != device)
        {
            throw std::invalid_argument("attention scratch tensor must be on the same device as inputs");
        }
        if (scratch.dtype() != DataType::FLOAT32)
        {
            throw std::invalid_argument("attention scratch tensor must be FLOAT32");
        }
        if (scratch.shape().ndims() != 3 || scratch.shape()[0] != b || scratch.shape()[1] != h ||
            scratch.shape()[2] != t)
        {
            throw std::invalid_argument("attention scratch tensor must have shape (B, H, T)");
        }
        if (!graph.hasTensor(scratch.id()))
        {
            graph.createTensor(scratch);
        }
        graph.recordOperation(OperationEntry{.type = OpType::MHA_ATTN_FWD,
                                             .inputs = {query, key, value},
                                             .outputs = {output, scratch},
                                             .attributes = attributes,
                                             .gpu_stream_desc = compute_stream_descriptor});
    }
    else
    {
        graph.recordOperation(OperationEntry{.type = OpType::MHA_ATTN_FWD,
                                             .inputs = {query, key, value},
                                             .outputs = {output},
                                             .attributes = attributes,
                                             .gpu_stream_desc = compute_stream_descriptor});
    }
    return output;
}

void pi::tensorlib::ScaledDotProductAttentionBwdInto(OpGraph &graph, const TraceTensor &query, const TraceTensor &key,
                                                     const TraceTensor &value, const TraceTensor &output,
                                                     const TraceTensor &scratch, const TraceTensor &upstream,
                                                     const TraceTensor &grad_q, const TraceTensor &grad_k,
                                                     const TraceTensor &grad_v, float softmax_scale, bool causal,
                                                     const GpuStreamDescriptor &compute_stream_descriptor)
{
    if (query.device() != key.device() || query.device() != value.device() || query.device() != output.device() ||
        query.device() != upstream.device() || query.device() != scratch.device())
    {
        throw std::runtime_error("attention tensors must be on the same device");
    }
    if (query.dtype() != key.dtype() || query.dtype() != value.dtype() || query.dtype() != output.dtype() ||
        query.dtype() != upstream.dtype())
    {
        throw std::invalid_argument("attention tensors must have matching dtype");
    }
    if (query.shape().ndims() != 4 || key.shape().ndims() != 4 || value.shape().ndims() != 4 ||
        output.shape().ndims() != 4 || upstream.shape().ndims() != 4)
    {
        throw std::invalid_argument("attention tensors must have 4 dimensions");
    }
    if (query.shape()[0] != key.shape()[0] || query.shape()[0] != value.shape()[0] ||
        query.shape()[0] != output.shape()[0] || query.shape()[0] != upstream.shape()[0])
    {
        throw std::invalid_argument("attention tensors must have the same batch size");
    }
    if (query.shape()[2] != key.shape()[2] || query.shape()[2] != value.shape()[2])
    {
        throw std::invalid_argument("attention tensors must have the same number of heads");
    }
    if (query.shape()[1] != key.shape()[1] || query.shape()[1] != value.shape()[1])
    {
        throw std::invalid_argument("attention tensors must have the same sequence length");
    }
    if (query.shape()[3] != key.shape()[3] || query.shape()[3] != value.shape()[3])
    {
        throw std::invalid_argument("attention tensors must have the same head size");
    }
    if (output.shape()[1] != query.shape()[1] || output.shape()[2] != query.shape()[2] ||
        output.shape()[3] != query.shape()[3])
    {
        throw std::invalid_argument("attention output must have shape (B, T, H, HS)");
    }
    if (upstream.shape() != output.shape())
    {
        throw std::invalid_argument("attention upstream must match output shape");
    }
    if (grad_q.device() != query.device() || grad_k.device() != query.device() || grad_v.device() != query.device())
    {
        throw std::invalid_argument("attention gradients must be on the same device as inputs");
    }
    if (grad_q.dtype() != query.dtype() || grad_k.dtype() != query.dtype() || grad_v.dtype() != query.dtype())
    {
        throw std::invalid_argument("attention gradients must match input dtype");
    }
    if (grad_q.shape() != query.shape() || grad_k.shape() != query.shape() || grad_v.shape() != query.shape())
    {
        throw std::invalid_argument("attention gradients must have the same shape as inputs");
    }
    if (scratch.dtype() != DataType::FLOAT32)
    {
        throw std::invalid_argument("attention scratch tensor must be FLOAT32");
    }
    if (scratch.shape().ndims() != 3 || scratch.shape()[0] != query.shape()[0] ||
        scratch.shape()[1] != query.shape()[2] || scratch.shape()[2] != query.shape()[1])
    {
        throw std::invalid_argument("attention scratch tensor must have shape (B, H, T)");
    }
    if (causal)
    {
        throw std::invalid_argument("causal attention backward is not implemented");
    }

    const auto b = query.shape()[0];
    const auto t = query.shape()[1];
    const auto h = query.shape()[2];

    TraceTensor delta =
        graph.createTensor({b, h, t}, DataType::FLOAT32, output.device(), compute_stream_descriptor, false);
    graph.recordOperation(OperationEntry{.type = OpType::MHA_ATTN_BWD_PRE,
                                         .inputs = {output, upstream},
                                         .outputs = {delta},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});

    std::unordered_map<std::string, std::any> attributes{};
    attributes.emplace("softmax_scale", softmax_scale);
    attributes.emplace("causal", causal);

    graph.recordOperation(OperationEntry{.type = OpType::MHA_ATTN_BWD,
                                         .inputs = {query, key, value, output, upstream, scratch, delta},
                                         .outputs = {grad_q, grad_k, grad_v},
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});

    graph.deleteTensor(delta);
}

pi::tensorlib::ScaledDotProductAttentionBwdResult pi::tensorlib::ScaledDotProductAttentionBwd(
    OpGraph &graph, const TraceTensor &query, const TraceTensor &key, const TraceTensor &value,
    const TraceTensor &output, const TraceTensor &scratch, const TraceTensor &upstream, const float softmax_scale,
    const bool causal, const GpuStreamDescriptor &compute_stream_descriptor)
{
    TraceTensor grad_q =
        graph.createTensor(output.shape().dims(), query.dtype(), query.device(), compute_stream_descriptor, false);
    TraceTensor grad_k =
        graph.createTensor(output.shape().dims(), key.dtype(), key.device(), compute_stream_descriptor, false);
    TraceTensor grad_v =
        graph.createTensor(output.shape().dims(), value.dtype(), value.device(), compute_stream_descriptor, false);
    ScaledDotProductAttentionBwdInto(graph, query, key, value, output, scratch, upstream, grad_q, grad_k, grad_v,
                                     softmax_scale, causal, compute_stream_descriptor);
    return {grad_q, grad_k, grad_v};
}
pi::tensorlib::TraceTensor pi::tensorlib::LayerNormFwd(OpGraph &graph, const TraceTensor &input,
                                                       const TraceTensor &weight, const TraceTensor &bias, float eps,
                                                       const GpuStreamDescriptor &compute_stream_descriptor)
{
    // assert input, weight, bias are on the same device
    if (input.device() != weight.device() || input.device() != bias.device())
    {
        throw std::runtime_error("input, weight, bias must be on the same device");
    }

    // assert input, weight, bias have the same dtype
    if (input.dtype() != weight.dtype() || input.dtype() != bias.dtype())
    {
        throw std::invalid_argument("input, weight, bias must have the same dtype");
    }

    // assert input has at least 2 dimensions
    if (input.shape().ndims() < 2)
    {
        throw std::invalid_argument("input must have at least 2 dimensions");
    }

    // assert weight, bias have 1 dimension
    if (weight.shape().ndims() != 1 || bias.shape().ndims() != 1)
    {
        throw std::invalid_argument("weight, bias must have 1 dimension");
    }

    // assert weight, bias have the same size as the last dimension of input
    if (weight.shape()[0] != input.shape().dims().back() || bias.shape()[0] != input.shape().dims().back())
    {
        throw std::invalid_argument("weight, bias must have the same size as the last dimension of input");
    }
    std::unordered_map<std::string, std::any> attributes{};
    attributes.reserve(1);
    attributes.emplace("eps", eps);

    TraceTensor output =
        graph.createTensor(input.shape().dims(), input.dtype(), input.device(), compute_stream_descriptor, false);
    graph.recordOperation(OperationEntry{.type = OpType::LAYER_NORM_FWD,
                                         .inputs = {input, weight, bias},
                                         .outputs = {output},
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});
    return output;
}

pi::tensorlib::TraceTensor pi::tensorlib::RmsNormFwd(OpGraph &graph, const TraceTensor &input,
                                                     const TraceTensor &weight, const float eps,
                                                     const GpuStreamDescriptor &compute_stream_descriptor)
{
    if (input.device() != weight.device())
    {
        throw std::runtime_error("input and weight must be on the same device");
    }

    if (input.dtype() != weight.dtype())
    {
        throw std::invalid_argument("input and weight must have the same dtype");
    }

    if (input.shape().ndims() < 1)
    {
        throw std::invalid_argument("input must have at least 1 dimension");
    }

    if (weight.shape().ndims() != 1)
    {
        throw std::invalid_argument("weight must have 1 dimension");
    }

    const auto hidden_size = input.shape().dims().back();
    if (weight.shape()[0] != hidden_size)
    {
        throw std::invalid_argument("weight must match the last dimension of input");
    }

    TraceTensor output =
        graph.createTensor(input.shape().dims(), input.dtype(), input.device(), compute_stream_descriptor, false);
    std::unordered_map<std::string, std::any> attributes{};
    attributes.emplace("eps", eps);

    graph.recordOperation(OperationEntry{.type = OpType::RMS_NORM_FWD,
                                         .inputs = {input, weight},
                                         .outputs = {output},
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});
    return output;
}

void pi::tensorlib::RmsNormFwdInplace(OpGraph &graph, const TraceTensor &input, const TraceTensor &weight, float eps,
                                      const GpuStreamDescriptor &compute_stream_descriptor)
{
    if (input.device() != weight.device())
    {
        throw std::runtime_error("input and weight must be on the same device");
    }

    if (input.dtype() != weight.dtype())
    {
        throw std::invalid_argument("input and weight must have the same dtype");
    }

    if (input.shape().ndims() < 1)
    {
        throw std::invalid_argument("input must have at least 1 dimension");
    }

    if (weight.shape().ndims() != 1)
    {
        throw std::invalid_argument("weight must have 1 dimension");
    }

    if (const auto hidden_size = input.shape().dims().back(); weight.shape()[0] != hidden_size)
    {
        throw std::invalid_argument("weight must match the last dimension of input");
    }

    std::unordered_map<std::string, std::any> attributes{};
    attributes.emplace("eps", eps);

    graph.recordOperation(OperationEntry{.type = OpType::RMS_NORM_FWD,
                                         .inputs = {input, weight},
                                         .outputs = {input},
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});
}

void pi::tensorlib::RmsNormBwd(OpGraph &graph, const TraceTensor &input, const TraceTensor &weight,
                               const TraceTensor &upstream, const TraceTensor &grad_input, const TraceTensor &x_hat,
                               const float eps, const GpuStreamDescriptor &compute_stream_descriptor)
{
    if (input.device() != weight.device() || input.device() != upstream.device() ||
        input.device() != grad_input.device() || input.device() != x_hat.device())
    {
        throw std::runtime_error("RmsNormBwd requires all tensors to be on the same device");
    }

    if (input.dtype() != weight.dtype() || input.dtype() != upstream.dtype() || input.dtype() != grad_input.dtype() ||
        input.dtype() != x_hat.dtype())
    {
        throw std::invalid_argument("RmsNormBwd requires all tensors to have the same dtype");
    }

    if (input.shape().ndims() < 1)
    {
        throw std::invalid_argument("RmsNormBwd input must have at least 1 dimension");
    }
    if (weight.shape().ndims() != 1)
    {
        throw std::invalid_argument("RmsNormBwd weight must have 1 dimension");
    }

    const auto hidden_size = input.shape().dims().back();
    if (weight.shape()[0] != hidden_size)
    {
        throw std::invalid_argument("RmsNormBwd weight must match the last dimension of input");
    }
    if (upstream.shape() != input.shape() || grad_input.shape() != input.shape() || x_hat.shape() != input.shape())
    {
        throw std::invalid_argument("RmsNormBwd input, upstream, grad_input, and x_hat must match shapes");
    }

    std::unordered_map<std::string, std::any> attributes{};
    attributes.emplace("eps", eps);

    graph.recordOperation(OperationEntry{.type = OpType::RMS_NORM_BWD,
                                         .inputs = {input, weight, upstream},
                                         .outputs = {grad_input, x_hat},
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});
}

pi::tensorlib::TraceTensor pi::tensorlib::AvgPool1d(OpGraph &graph, const TraceTensor &input,
                                                    const uint32_t kernel_size, const uint32_t stride,
                                                    const GpuStreamDescriptor &compute_stream_descriptor,
                                                    int64_t pool_dim)
{
    if (kernel_size == 0)
    {
        throw std::invalid_argument("AvgPool1d kernel_size must be > 0");
    }
    if (stride == 0)
    {
        throw std::invalid_argument("AvgPool1d stride must be > 0");
    }

    if (input.shape().ndims() != 3)
    {
        throw std::invalid_argument("AvgPool1d input must be 3D (N, C, L)");
    }

    const auto ndims = static_cast<int64_t>(input.shape().ndims());
    if (pool_dim < 0)
    {
        pool_dim += ndims;
    }
    if (pool_dim < 0 || pool_dim >= ndims)
    {
        throw std::invalid_argument("AvgPool1d pool_dim is out of range");
    }

    const auto pool_length_in = input.shape()[pool_dim];
    if (pool_length_in < kernel_size)
    {
        throw std::invalid_argument("AvgPool1d requires pooling dimension >= kernel_size");
    }

    const auto batches = input.shape()[0];
    const auto channels = input.shape()[1];
    const auto depth = input.shape()[2];

    const auto pool_length_out = 1 + (pool_length_in - kernel_size) / stride;
    if (pool_length_out == 0)
    {
        throw std::invalid_argument("AvgPool1d produced zero-length output");
    }

    std::vector<uint64_t> output_shape = {batches, channels, depth};
    output_shape[static_cast<size_t>(pool_dim)] = pool_length_out;
    TraceTensor output =
        graph.createTensor(output_shape, input.dtype(), input.device(), compute_stream_descriptor, false);

    std::unordered_map<std::string, std::any> attributes{};
    attributes.emplace("kernel_size", kernel_size);
    attributes.emplace("stride", stride);
    attributes.emplace("pool_dim", static_cast<uint32_t>(pool_dim));

    graph.recordOperation(OperationEntry{.type = OpType::AVG_POOL1D,
                                         .inputs = {input},
                                         .outputs = {output},
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});

    return output;
}

pi::tensorlib::TraceTensor pi::tensorlib::AvgPool2d(OpGraph &graph, const TraceTensor &input,
                                                    const std::array<uint32_t, 2> &kernel_size,
                                                    const std::array<uint32_t, 2> &stride,
                                                    const GpuStreamDescriptor &compute_stream_descriptor,
                                                    const std::array<uint32_t, 2> &padding, bool channels_last)
{
    if (kernel_size[0] == 0 || kernel_size[1] == 0)
    {
        throw std::invalid_argument("AvgPool2d kernel dimensions must be > 0");
    }
    if (stride[0] == 0 || stride[1] == 0)
    {
        throw std::invalid_argument("AvgPool2d strides must be > 0");
    }

    if (input.shape().ndims() != 4)
    {
        throw std::invalid_argument("AvgPool2d input must be 4D (N, C, H, W) or (N, H, W, C)");
    }

    const size_t batch_dim = 0;
    const size_t channel_dim = channels_last ? 3 : 1;
    const size_t height_dim = channels_last ? 1 : 2;
    const size_t width_dim = channels_last ? 2 : 3;

    const auto batches = input.shape()[batch_dim];
    const auto channels = input.shape()[channel_dim];
    const auto height_in = input.shape()[height_dim];
    const auto width_in = input.shape()[width_dim];

    const uint32_t pad_h = padding[0];
    const uint32_t pad_w = padding[1];

    const auto kernel_h = kernel_size[0];
    const auto kernel_w = kernel_size[1];
    const auto stride_h = stride[0];
    const auto stride_w = stride[1];

    if (height_in + 2 * pad_h < kernel_h || width_in + 2 * pad_w < kernel_w)
    {
        throw std::invalid_argument("AvgPool2d requires pooling region to fit within padded input dimensions");
    }

    const auto height_out = 1 + (height_in + 2 * pad_h - kernel_h) / stride_h;
    const auto width_out = 1 + (width_in + 2 * pad_w - kernel_w) / stride_w;
    if (height_out == 0 || width_out == 0)
    {
        throw std::invalid_argument("AvgPool2d produced zero-sized output");
    }

    std::vector<uint64_t> output_shape = input.shape().dims();
    output_shape[height_dim] = height_out;
    output_shape[width_dim] = width_out;
    TraceTensor output =
        graph.createTensor(output_shape, input.dtype(), input.device(), compute_stream_descriptor, false);

    std::unordered_map<std::string, std::any> attributes{};
    attributes.emplace("kernel_size", kernel_size);
    attributes.emplace("stride", stride);
    attributes.emplace("padding", padding);
    attributes.emplace("channels_last", channels_last);

    graph.recordOperation(OperationEntry{.type = OpType::AVG_POOL2D,
                                         .inputs = {input},
                                         .outputs = {output},
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});
    return output;
}

pi::tensorlib::TraceTensor pi::tensorlib::Conv2d(OpGraph &graph, const TraceTensor &input, const TraceTensor &weight,
                                                 const TraceTensor *bias, const std::array<uint32_t, 2> &stride,
                                                 const std::array<uint32_t, 2> &padding,
                                                 const GpuStreamDescriptor &compute_stream_descriptor,
                                                 const std::array<uint32_t, 2> &dilation, const bool use_fp16_conv_acc)
{
    const auto dtype = input.dtype();
    if (dtype != weight.dtype() || (dtype != DataType::BFLOAT16 && dtype != DataType::FLOAT16))
    {
        throw std::invalid_argument("Conv2d currently supports BFLOAT16 or FLOAT16 input and weight tensors");
    }
    if (bias != nullptr && bias->dtype() != dtype)
    {
        throw std::invalid_argument("Conv2d bias must match input dtype (BFLOAT16 or FLOAT16) when provided");
    }

    if (input.device() != weight.device())
    {
        throw std::invalid_argument("Conv2d input and weight tensors must be on the same device");
    }
    if (bias != nullptr && bias->device() != input.device())
    {
        throw std::invalid_argument("Conv2d bias tensor must be on the same device as input");
    }

    if (input.shape().ndims() != 4)
    {
        throw std::invalid_argument("Conv2d input must be 4D (N, H, W, C)");
    }
    if (weight.shape().ndims() != 4)
    {
        throw std::invalid_argument("Conv2d weight must be 4D (K_h, K_w, C_in, C_out)");
    }

    const uint32_t stride_h = stride[0];
    const uint32_t stride_w = stride[1];
    if (stride_h == 0 || stride_w == 0)
    {
        throw std::invalid_argument("Conv2d stride values must be greater than zero");
    }

    const uint32_t pad_h = padding[0];
    const uint32_t pad_w = padding[1];

    const uint32_t dilation_h = dilation[0];
    const uint32_t dilation_w = dilation[1];
    if (dilation_h == 0 || dilation_w == 0)
    {
        throw std::invalid_argument("Conv2d dilation values must be greater than zero");
    }

    const uint64_t batch = input.shape()[0];
    const uint64_t in_h = input.shape()[1];
    const uint64_t in_w = input.shape()[2];
    const uint64_t in_channels = input.shape()[3];

    const uint64_t out_channels = weight.shape()[0];
    const uint64_t kernel_h = weight.shape()[1];
    const uint64_t kernel_w = weight.shape()[2];
    const uint64_t kernel_in_channels = weight.shape()[3];

    if (kernel_in_channels != in_channels)
    {
        throw std::invalid_argument("Conv2d weight in_channels must match input in_channels");
    }
    if (kernel_h == 0 || kernel_w == 0)
    {
        throw std::invalid_argument("Conv2d kernel dimensions must be greater than zero");
    }

    const uint64_t effective_kernel_h = 1 + static_cast<uint64_t>(dilation_h) * (kernel_h - 1);
    const uint64_t effective_kernel_w = 1 + static_cast<uint64_t>(dilation_w) * (kernel_w - 1);

    const int64_t out_h = static_cast<int64_t>(in_h + 2 * pad_h) - static_cast<int64_t>(effective_kernel_h);
    const int64_t out_w = static_cast<int64_t>(in_w + 2 * pad_w) - static_cast<int64_t>(effective_kernel_w);
    if (out_h < 0 || out_w < 0)
    {
        throw std::invalid_argument(
            "Conv2d padding/stride/dilation/kernel configuration results in negative output size");
    }
    if (((out_h) % static_cast<int64_t>(stride_h) != 0) || ((out_w) % static_cast<int64_t>(stride_w) != 0))
    {
        throw std::invalid_argument("Conv2d configuration must produce integer output spatial dimensions");
    }

    const auto output_h = static_cast<uint64_t>(out_h / static_cast<int64_t>(stride_h) + 1);
    const auto output_w = static_cast<uint64_t>(out_w / static_cast<int64_t>(stride_w) + 1);
    if (output_h == 0 || output_w == 0)
    {
        throw std::invalid_argument("Conv2d produced zero-sized output");
    }

    if (bias != nullptr)
    {
        if (bias->shape().ndims() != 1 || bias->shape()[0] != out_channels)
        {
            throw std::invalid_argument("Conv2d bias tensor must be 1D with size equal to out_channels");
        }
    }

    TraceTensor output = graph.createTensor({batch, output_h, output_w, out_channels}, input.dtype(), input.device(),
                                            compute_stream_descriptor, false);

    std::unordered_map<std::string, std::any> attributes{};
    attributes.emplace("stride", stride);
    attributes.emplace("padding", padding);
    attributes.emplace("dilation", dilation);

    if (use_fp16_conv_acc && input.dtype() == DataType::FLOAT16)
    {
        attributes.emplace("use_fp16_conv_acc", true);
    }

    std::vector inputs{input, weight};
    if (bias != nullptr)
    {
        inputs.push_back(*bias);
    }

    graph.recordOperation(OperationEntry{.type = OpType::CONV2D,
                                         .inputs = inputs,
                                         .outputs = {output},
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});

    return output;
}

void pi::tensorlib::Conv2dDgradInto(OpGraph &graph, const TraceTensor &upstream, const TraceTensor &weight,
                                    const TraceTensor &grad_input, const std::array<uint32_t, 2> &stride,
                                    const std::array<uint32_t, 2> &padding, const std::array<uint32_t, 2> &dilation,
                                    const GpuStreamDescriptor &compute_stream_descriptor)
{
    const auto dtype = upstream.dtype();
    if (dtype != weight.dtype() || dtype != grad_input.dtype() ||
        (dtype != DataType::BFLOAT16 && dtype != DataType::FLOAT16))
    {
        throw std::invalid_argument("Conv2dDgrad expects BFLOAT16 or FLOAT16 tensors with matching dtypes");
    }

    if (upstream.device() != weight.device() || upstream.device() != grad_input.device())
    {
        throw std::invalid_argument("Conv2dDgrad tensors must be on the same device");
    }

    if (upstream.shape().ndims() != 4 || weight.shape().ndims() != 4 || grad_input.shape().ndims() != 4)
    {
        throw std::invalid_argument("Conv2dDgrad expects upstream, weight, and grad_input to be 4D tensors");
    }

    const uint32_t stride_h = stride[0];
    const uint32_t stride_w = stride[1];
    if (stride_h == 0 || stride_w == 0)
    {
        throw std::invalid_argument("Conv2dDgrad stride values must be greater than zero");
    }

    const uint32_t pad_h = padding[0];
    const uint32_t pad_w = padding[1];

    const uint32_t dilation_h = dilation[0];
    const uint32_t dilation_w = dilation[1];
    if (dilation_h == 0 || dilation_w == 0)
    {
        throw std::invalid_argument("Conv2dDgrad dilation values must be greater than zero");
    }

    const uint64_t batch = grad_input.shape()[0];
    const uint64_t in_h = grad_input.shape()[1];
    const uint64_t in_w = grad_input.shape()[2];
    const uint64_t in_channels = grad_input.shape()[3];

    const uint64_t out_channels = weight.shape()[0];
    const uint64_t kernel_h = weight.shape()[1];
    const uint64_t kernel_w = weight.shape()[2];
    const uint64_t kernel_in_channels = weight.shape()[3];

    if (kernel_in_channels != in_channels)
    {
        throw std::invalid_argument("Conv2dDgrad weight in_channels must match grad_input channels");
    }
    if (kernel_h == 0 || kernel_w == 0)
    {
        throw std::invalid_argument("Conv2dDgrad kernel dimensions must be greater than zero");
    }

    const uint64_t effective_kernel_h = 1 + static_cast<uint64_t>(dilation_h) * (kernel_h - 1);
    const uint64_t effective_kernel_w = 1 + static_cast<uint64_t>(dilation_w) * (kernel_w - 1);

    const int64_t out_h_num = static_cast<int64_t>(in_h + 2 * pad_h) - static_cast<int64_t>(effective_kernel_h);
    const int64_t out_w_num = static_cast<int64_t>(in_w + 2 * pad_w) - static_cast<int64_t>(effective_kernel_w);
    if (out_h_num < 0 || out_w_num < 0)
    {
        throw std::invalid_argument("Conv2dDgrad configuration results in negative output size");
    }
    if ((out_h_num % static_cast<int64_t>(stride_h) != 0) || (out_w_num % static_cast<int64_t>(stride_w) != 0))
    {
        throw std::invalid_argument("Conv2dDgrad configuration must produce integer output spatial dimensions");
    }

    const uint64_t out_h = static_cast<uint64_t>(out_h_num / static_cast<int64_t>(stride_h) + 1);
    const uint64_t out_w = static_cast<uint64_t>(out_w_num / static_cast<int64_t>(stride_w) + 1);

    if (upstream.shape()[0] != batch || upstream.shape()[1] != out_h || upstream.shape()[2] != out_w ||
        upstream.shape()[3] != out_channels)
    {
        throw std::invalid_argument("Conv2dDgrad upstream shape must be (B, out_h, out_w, out_channels)");
    }

    std::unordered_map<std::string, std::any> attributes{};
    attributes.emplace("stride", stride);
    attributes.emplace("padding", padding);
    attributes.emplace("dilation", dilation);

    graph.recordOperation(OperationEntry{.type = OpType::CONV2D_DGRAD,
                                         .inputs = {upstream, weight},
                                         .outputs = {grad_input},
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});
}

void pi::tensorlib::Conv2dWgradInto(OpGraph &graph, const TraceTensor &input, const TraceTensor &upstream,
                                    const TraceTensor &grad_weight, const std::array<uint32_t, 2> &stride,
                                    const std::array<uint32_t, 2> &padding, const std::array<uint32_t, 2> &dilation,
                                    const GpuStreamDescriptor &compute_stream_descriptor, const bool accumulate_output)
{
    const auto dtype = input.dtype();
    if (dtype != upstream.dtype() || dtype != grad_weight.dtype() ||
        (dtype != DataType::BFLOAT16 && dtype != DataType::FLOAT16))
    {
        throw std::invalid_argument("Conv2dWgrad expects BFLOAT16 or FLOAT16 tensors with matching dtypes");
    }

    if (input.device() != upstream.device() || input.device() != grad_weight.device())
    {
        throw std::invalid_argument("Conv2dWgrad tensors must be on the same device");
    }

    if (input.shape().ndims() != 4 || upstream.shape().ndims() != 4 || grad_weight.shape().ndims() != 4)
    {
        throw std::invalid_argument("Conv2dWgrad expects input, upstream, and grad_weight to be 4D tensors");
    }

    const uint32_t stride_h = stride[0];
    const uint32_t stride_w = stride[1];
    if (stride_h == 0 || stride_w == 0)
    {
        throw std::invalid_argument("Conv2dWgrad stride values must be greater than zero");
    }

    const uint32_t pad_h = padding[0];
    const uint32_t pad_w = padding[1];

    const uint32_t dilation_h = dilation[0];
    const uint32_t dilation_w = dilation[1];
    if (dilation_h == 0 || dilation_w == 0)
    {
        throw std::invalid_argument("Conv2dWgrad dilation values must be greater than zero");
    }

    const uint64_t batch = input.shape()[0];
    const uint64_t in_h = input.shape()[1];
    const uint64_t in_w = input.shape()[2];
    const uint64_t in_channels = input.shape()[3];

    const uint64_t out_channels = grad_weight.shape()[0];
    const uint64_t kernel_h = grad_weight.shape()[1];
    const uint64_t kernel_w = grad_weight.shape()[2];
    const uint64_t kernel_in_channels = grad_weight.shape()[3];

    if (kernel_in_channels != in_channels)
    {
        throw std::invalid_argument("Conv2dWgrad grad_weight in_channels must match input channels");
    }
    if (kernel_h == 0 || kernel_w == 0)
    {
        throw std::invalid_argument("Conv2dWgrad kernel dimensions must be greater than zero");
    }

    const uint64_t effective_kernel_h = 1 + static_cast<uint64_t>(dilation_h) * (kernel_h - 1);
    const uint64_t effective_kernel_w = 1 + static_cast<uint64_t>(dilation_w) * (kernel_w - 1);

    const int64_t out_h_num = static_cast<int64_t>(in_h + 2 * pad_h) - static_cast<int64_t>(effective_kernel_h);
    const int64_t out_w_num = static_cast<int64_t>(in_w + 2 * pad_w) - static_cast<int64_t>(effective_kernel_w);
    if (out_h_num < 0 || out_w_num < 0)
    {
        throw std::invalid_argument("Conv2dWgrad configuration results in negative output size");
    }
    if ((out_h_num % static_cast<int64_t>(stride_h) != 0) || (out_w_num % static_cast<int64_t>(stride_w) != 0))
    {
        throw std::invalid_argument("Conv2dWgrad configuration must produce integer output spatial dimensions");
    }

    const uint64_t out_h = static_cast<uint64_t>(out_h_num / static_cast<int64_t>(stride_h) + 1);
    const uint64_t out_w = static_cast<uint64_t>(out_w_num / static_cast<int64_t>(stride_w) + 1);

    if (upstream.shape()[0] != batch || upstream.shape()[1] != out_h || upstream.shape()[2] != out_w ||
        upstream.shape()[3] != out_channels)
    {
        throw std::invalid_argument("Conv2dWgrad upstream shape must be (B, out_h, out_w, out_channels)");
    }

    std::unordered_map<std::string, std::any> attributes{};
    attributes.emplace("stride", stride);
    attributes.emplace("padding", padding);
    attributes.emplace("dilation", dilation);
    if (accumulate_output)
    {
        attributes.emplace("accumulate_output", true);
    }

    graph.recordOperation(OperationEntry{.type = OpType::CONV2D_WGRAD,
                                         .inputs = {input, upstream},
                                         .outputs = {grad_weight},
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});
}

pi::tensorlib::TraceTensor pi::tensorlib::Mean(OpGraph &graph, const TraceTensor &input, int64_t dim, bool keepdim,
                                               const GpuStreamDescriptor &compute_stream_descriptor)
{
    const auto dtype = input.dtype();
    if (dtype != DataType::BFLOAT16 && dtype != DataType::FLOAT16)
    {
        throw std::invalid_argument("Mean currently supports BFLOAT16 or FLOAT16 tensors only");
    }
    const auto ndim = static_cast<int64_t>(input.shape().ndims());
    if (ndim == 0)
    {
        throw std::invalid_argument("Mean operation does not support scalar inputs");
    }

    if (dim < 0)
    {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim)
    {
        throw std::out_of_range("Mean reduction dimension is out of range");
    }

    if (const auto reduction_size = input.shape()[dim]; reduction_size == 0)
    {
        throw std::invalid_argument("Mean reduction dimension must be non-zero");
    }

    std::vector<uint64_t> output_dims;
    output_dims.reserve(static_cast<size_t>(ndim));
    if (keepdim)
    {
        for (size_t d = 0; d < input.shape().ndims(); ++d)
        {
            output_dims.push_back(d == static_cast<size_t>(dim) ? 1 : input.shape()[static_cast<int64_t>(d)]);
        }
    }
    else
    {
        for (size_t d = 0; d < input.shape().ndims(); ++d)
        {
            if (static_cast<int64_t>(d) == dim)
            {
                continue;
            }
            output_dims.push_back(input.shape()[static_cast<int64_t>(d)]);
        }
    }

    TraceTensor output = graph.createTensor(output_dims, dtype, input.device(), compute_stream_descriptor, false);

    std::unordered_map<std::string, std::any> attributes{};
    attributes.emplace("dim", static_cast<int64_t>(dim));
    attributes.emplace("keepdim", keepdim);

    graph.recordOperation(OperationEntry{.type = OpType::MEAN,
                                         .inputs = {input},
                                         .outputs = {output},
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});
    return output;
}

pi::tensorlib::TraceTensor pi::tensorlib::ReduceSum(OpGraph &graph, const TraceTensor &input, int64_t dim, bool keepdim,
                                                    const DataType output_dtype,
                                                    const GpuStreamDescriptor &compute_stream_descriptor)
{
    const auto dtype = input.dtype();
    if (dtype != DataType::BFLOAT16 && dtype != DataType::FLOAT16 && dtype != DataType::FLOAT32)
    {
        throw std::invalid_argument("ReduceSum currently supports BFLOAT16, FLOAT16, or FLOAT32 tensors only");
    }
    const auto ndim = static_cast<int64_t>(input.shape().ndims());
    if (ndim == 0)
    {
        throw std::invalid_argument("ReduceSum operation does not support scalar inputs");
    }

    if (dim < 0)
    {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim)
    {
        throw std::out_of_range("ReduceSum reduction dimension is out of range");
    }

    if (const auto reduction_size = input.shape()[dim]; reduction_size == 0)
    {
        throw std::invalid_argument("ReduceSum reduction dimension must be non-zero");
    }

    std::vector<uint64_t> output_dims;
    output_dims.reserve(static_cast<size_t>(ndim));
    if (keepdim)
    {
        for (size_t d = 0; d < input.shape().ndims(); ++d)
        {
            output_dims.push_back(d == static_cast<size_t>(dim) ? 1 : input.shape()[static_cast<int64_t>(d)]);
        }
    }
    else
    {
        for (size_t d = 0; d < input.shape().ndims(); ++d)
        {
            if (static_cast<int64_t>(d) == dim)
            {
                continue;
            }
            output_dims.push_back(input.shape()[static_cast<int64_t>(d)]);
        }
    }

    TraceTensor output =
        graph.createTensor(output_dims, DataType::FLOAT32, input.device(), compute_stream_descriptor, false);

    std::unordered_map<std::string, std::any> attributes{};
    attributes.emplace("dim", static_cast<int64_t>(dim));
    attributes.emplace("keepdim", keepdim);

    graph.recordOperation(OperationEntry{.type = OpType::REDUCE_SUM,
                                         .inputs = {input},
                                         .outputs = {output},
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});

    if (output_dtype != DataType::FLOAT32)
    {
        TraceTensor casted_output =
            graph.createTensor(output.shape().dims(), output_dtype, output.device(), compute_stream_descriptor, false);
        graph.recordOperation(OperationEntry{.type = OpType::CAST,
                                             .inputs = {output},
                                             .outputs = {casted_output},
                                             .attributes = {},
                                             .gpu_stream_desc = compute_stream_descriptor});
        return casted_output;
    }

    return output;
}

namespace
{
    // TODO: THIS IS UGLY, REFACTOR SOON
    std::vector<uint64_t> BroadcastShape(const pi::tensorlib::TraceTensor &lhs, const pi::tensorlib::TraceTensor &rhs,
                                         const char *op_name)
    {
        if (!pi::tensorlib::shape_utils::IsBroadcastable(lhs.shape(), rhs.shape()))
        {
            throw std::runtime_error(std::string(op_name) + " requires broadcastable shapes");
        }

        const auto &lhs_dims = lhs.shape().dims();
        const auto &rhs_dims = rhs.shape().dims();
        const size_t out_rank = std::max(lhs_dims.size(), rhs_dims.size());
        std::vector<uint64_t> out_dims(out_rank, 1);
        for (size_t i = 0; i < out_rank; ++i)
        {
            const auto lhs_idx = (i < out_rank - lhs_dims.size()) ? 1 : lhs_dims[i - (out_rank - lhs_dims.size())];
            const auto rhs_idx = (i < out_rank - rhs_dims.size()) ? 1 : rhs_dims[i - (out_rank - rhs_dims.size())];
            if (lhs_idx != rhs_idx && lhs_idx != 1 && rhs_idx != 1)
            {
                throw std::runtime_error(std::string(op_name) + " requires broadcastable shapes");
            }
            out_dims[i] = lhs_idx > rhs_idx ? lhs_idx : rhs_idx;
        }
        return out_dims;
    }
} // namespace

pi::tensorlib::TraceTensor pi::tensorlib::Div(OpGraph &graph, const TraceTensor &lhs, const TraceTensor &rhs,
                                              const GpuStreamDescriptor &compute_stream_descriptor)
{
    const auto broadcast_shape = BroadcastShape(lhs, rhs, "Div");
    TraceTensor output =
        graph.createTensor(broadcast_shape, lhs.dtype(), lhs.device(), compute_stream_descriptor, false);
    Div(graph, lhs, rhs, output, compute_stream_descriptor);
    return output;
}

void pi::tensorlib::Div(OpGraph &graph, const TraceTensor &lhs, const TraceTensor &rhs, const TraceTensor &output,
                        const GpuStreamDescriptor &compute_stream_descriptor)
{
    if (lhs.device() != rhs.device() || lhs.device() != output.device())
    {
        throw std::runtime_error("Div requires all tensors to be on the same device");
    }
    if (lhs.dtype() != rhs.dtype() || lhs.dtype() != output.dtype())
    {
        throw std::runtime_error("Div requires all tensors to have the same dtype");
    }

    const auto broadcast_shape = BroadcastShape(lhs, rhs, "Div");
    if (output.shape().dims() != broadcast_shape)
    {
        throw std::runtime_error("Div output shape must match broadcasted input shape");
    }

    graph.recordOperation(OperationEntry{.type = OpType::DIV,
                                         .inputs = {lhs, rhs},
                                         .outputs = {output},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});
}

pi::tensorlib::TraceTensor pi::tensorlib::Sqrt(OpGraph &graph, const TraceTensor &input,
                                               const GpuStreamDescriptor &compute_stream_descriptor)
{
    TraceTensor output =
        graph.createTensor(input.shape().dims(), input.dtype(), input.device(), compute_stream_descriptor, false);
    Sqrt(graph, input, output, compute_stream_descriptor);
    return output;
}

void pi::tensorlib::Sqrt(OpGraph &graph, const TraceTensor &input, const TraceTensor &output,
                         const GpuStreamDescriptor &compute_stream_descriptor)
{
    if (input.device() != output.device())
    {
        throw std::runtime_error("Sqrt requires input and output on the same device");
    }
    if (input.dtype() != output.dtype())
    {
        throw std::runtime_error("Sqrt requires input and output to have the same dtype");
    }
    if (input.shape().dims() != output.shape().dims())
    {
        throw std::runtime_error("Sqrt output shape must match input shape");
    }

    graph.recordOperation(OperationEntry{.type = OpType::SQRT,
                                         .inputs = {input},
                                         .outputs = {output},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});
}

namespace
{
    std::string ReductionToString(const pi::tensorlib::Reduction reduction)
    {
        switch (reduction)
        {
            case pi::tensorlib::Reduction::MEAN:
                return "mean";
            case pi::tensorlib::Reduction::ADD:
                return "add";
            default:
                throw std::invalid_argument("Unsupported reduction for CrossEntropyOnTargets");
        }
    }
} // namespace

void pi::tensorlib::CrossEntropyOnTargets(OpGraph &graph, const TraceTensor &logits, const TraceTensor &targets,
                                          const TraceTensor &output, const Reduction reduction,
                                          const GpuStreamDescriptor &compute_stream_descriptor,
                                          const bool reduce_over_rows)
{
    if (logits.device().device_type != DeviceType::GPU || targets.device().device_type != DeviceType::GPU ||
        output.device().device_type != DeviceType::GPU)
    {
        throw std::runtime_error("CrossEntropyOnTargets supports GPU tensors only");
    }
    if (logits.device() != targets.device() || logits.device() != output.device())
    {
        throw std::runtime_error("CrossEntropyOnTargets requires logits, targets, and output on the same device");
    }
    if (logits.dtype() != DataType::BFLOAT16 && logits.dtype() != DataType::FLOAT16)
    {
        throw std::runtime_error("CrossEntropyOnTargets supports BF16/FP16 logits only");
    }
    if (targets.dtype() != DataType::UINT32)
    {
        throw std::runtime_error("CrossEntropyOnTargets requires UINT32 targets");
    }

    if (logits.shape().ndims() < 1)
    {
        throw std::runtime_error("CrossEntropyOnTargets expects logits with at least one dimension");
    }
    const auto vocab = logits.shape()[logits.shape().ndims() - 1];
    if (vocab == 0)
    {
        throw std::runtime_error("CrossEntropyOnTargets requires non-empty class dimension");
    }
    const auto logits_elements = logits.shape().numel();
    if (logits_elements % vocab != 0)
    {
        throw std::runtime_error("CrossEntropyOnTargets expects logits.numel divisible by vocab");
    }

    const auto rows = logits_elements / vocab;
    if (targets.shape().numel() != rows)
    {
        throw std::runtime_error("CrossEntropyOnTargets expects targets to match logits without class dimension");
    }

    const DataType row_output_dtype = (reduction == Reduction::MEAN) ? logits.dtype() : DataType::FLOAT32;

    if (reduce_over_rows)
    {
        if (output.shape().numel() != 1)
        {
            throw std::runtime_error("CrossEntropyOnTargets reduce_over_rows expects scalar output");
        }
        if (output.dtype() != DataType::FLOAT32)
        {
            throw std::runtime_error("CrossEntropyOnTargets reduce_over_rows expects FLOAT32 output");
        }
    }
    else
    {
        switch (reduction)
        {
            case Reduction::MEAN:
            {
                if (output.dtype() != logits.dtype())
                {
                    throw std::runtime_error(
                        "CrossEntropyOnTargets mean reduction expects output dtype to match logits");
                }
                if (output.shape().numel() != rows)
                {
                    throw std::runtime_error("CrossEntropyOnTargets mean reduction output size mismatch");
                }
                break;
            }
            case Reduction::ADD:
            {
                if (output.dtype() != DataType::FLOAT32)
                {
                    throw std::runtime_error("CrossEntropyOnTargets add reduction expects FLOAT32 output");
                }
                if (output.shape().numel() != rows)
                {
                    throw std::runtime_error("CrossEntropyOnTargets add reduction expects per-row output");
                }
                break;
            }
            default:
                throw std::invalid_argument("Unsupported reduction for CrossEntropyOnTargets");
        }
    }

    const bool need_row_output = !reduce_over_rows;
    TraceTensor row_output = output;
    if (!need_row_output)
    {
        row_output = graph.createTensor({rows}, row_output_dtype, logits.device(), compute_stream_descriptor, false);
    }

    std::unordered_map<std::string, std::any> attributes{};
    attributes.emplace("reduction", ReductionToString(reduction));

    graph.recordOperation(OperationEntry{.type = OpType::CROSS_ENTROPY_ON_TARGETS,
                                         .inputs = {logits, targets},
                                         .outputs = {row_output},
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});

    if (reduce_over_rows)
    {
        std::vector<TraceTensor> temporaries{};
        temporaries.reserve(4);

        TraceTensor reduced = row_output;
        if (reduction == Reduction::MEAN && row_output.dtype() != DataType::FLOAT32)
        {
            TraceTensor casted = row_output.to(graph, DataType::FLOAT32, compute_stream_descriptor);
            temporaries.push_back(row_output);
            reduced = casted;
        }
        while (reduced.shape().numel() > 1)
        {
            constexpr int64_t block_size = 128;
            TraceTensor next = ReduceSumPartial(graph, reduced, compute_stream_descriptor, block_size);
            temporaries.push_back(reduced);
            reduced = next;
        }
        if (reduction == Reduction::ADD)
        {
            DeviceCopy(graph, reduced, output, compute_stream_descriptor);
        }
        else // Reduction::MEAN
        {
            TraceTensor denom =
                graph.createTensor({1}, DataType::FLOAT32, logits.device(), compute_stream_descriptor, false);
            FillConstant(graph, denom, static_cast<float>(rows), compute_stream_descriptor);
            Div(graph, reduced, denom, output, compute_stream_descriptor);
            graph.deleteTensor(denom);
        }

        for (auto &tmp : temporaries)
        {
            graph.deleteTensor(tmp);
        }
        if (reduced.id() != output.id())
        {
            graph.deleteTensor(reduced);
        }
    }
}

pi::tensorlib::TraceTensor
pi::tensorlib::CrossEntropyOnTargetsBackward(OpGraph &graph, const TraceTensor &logits, const TraceTensor &targets,
                                             const TraceTensor &upstream, Reduction reduction,
                                             const GpuStreamDescriptor &compute_stream_descriptor,
                                             const bool reduce_over_rows, const bool is_useful)
{
    if (logits.device().device_type != DeviceType::GPU || targets.device() != logits.device() ||
        upstream.device() != logits.device())
    {
        throw std::runtime_error("CrossEntropyOnTargetsBackward supports GPU tensors with matching devices only");
    }
    if (logits.dtype() != DataType::FLOAT16 && logits.dtype() != DataType::BFLOAT16)
    {
        throw std::runtime_error("CrossEntropyOnTargetsBackward supports BF16/FP16 logits only");
    }
    if (targets.dtype() != DataType::UINT32)
    {
        throw std::runtime_error("CrossEntropyOnTargetsBackward requires UINT32 targets");
    }
    if (upstream.dtype() != DataType::FLOAT32)
    {
        throw std::runtime_error("CrossEntropyOnTargetsBackward requires FLOAT32 upstream gradient");
    }
    const auto vocab = logits.shape()[logits.shape().ndims() - 1];
    const auto rows = logits.shape().numel() / vocab;
    if (reduce_over_rows)
    {
        if (upstream.shape().numel() != 1)
        {
            throw std::runtime_error("CrossEntropyOnTargetsBackward reduce_over_rows expects scalar upstream");
        }
    }
    else
    {
        if (upstream.shape().numel() != rows)
        {
            throw std::runtime_error("CrossEntropyOnTargetsBackward expects upstream to have one element per row");
        }
    }

    TraceTensor upstream_for_kernel = upstream;
    if (reduce_over_rows)
    {
        TraceTensor broadcast =
            graph.createTensor({rows, 1}, DataType::FLOAT32, logits.device(), compute_stream_descriptor, false);
        FillZeros(graph, broadcast, compute_stream_descriptor);
        InplaceAdd(graph, broadcast, upstream, compute_stream_descriptor);
        if (reduction == Reduction::MEAN)
        {
            TraceTensor denom =
                graph.createTensor({rows, 1}, DataType::FLOAT32, logits.device(), compute_stream_descriptor, false);
            FillConstant(graph, denom, static_cast<float>(rows), compute_stream_descriptor);
            InplaceDiv(graph, broadcast, denom, compute_stream_descriptor);
            graph.deleteTensor(denom);
        }
        upstream_for_kernel = broadcast;
    }

    TraceTensor grad =
        graph.createTensor(logits.shape().dims(), logits.dtype(), logits.device(), compute_stream_descriptor, false);
    std::unordered_map<std::string, std::any> attributes{};
    attributes.emplace("reduction", reduction == Reduction::MEAN ? std::string("mean") : std::string("add"));
    attributes.emplace("reduce_over_rows", reduce_over_rows);
    graph.recordOperation(OperationEntry{.type = OpType::CROSS_ENTROPY_ON_TARGETS_BWD,
                                         .inputs = {logits, targets, upstream_for_kernel},
                                         .outputs = {grad},
                                         .is_useful = is_useful,
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});
    return grad;
}

pi::tensorlib::TraceTensor pi::tensorlib::CrossEntropyOnTargets(OpGraph &graph, const TraceTensor &logits,
                                                                const TraceTensor &targets, const Reduction reduction,
                                                                const GpuStreamDescriptor &compute_stream_descriptor,
                                                                const bool reduce_over_rows)
{
    const auto output_dims = reduce_over_rows ? std::vector<uint64_t>{1} : targets.shape().dims();
    DataType output_dtype = (reduction == Reduction::MEAN) ? logits.dtype() : DataType::FLOAT32;
    if (reduce_over_rows)
    {
        output_dtype = DataType::FLOAT32;
    }
    TraceTensor output =
        graph.createTensor(output_dims, output_dtype, logits.device(), compute_stream_descriptor, false);
    CrossEntropyOnTargets(graph, logits, targets, output, reduction, compute_stream_descriptor, reduce_over_rows);
    return output;
}

pi::tensorlib::TraceTensor pi::tensorlib::ReduceSumPartial(OpGraph &graph, const TraceTensor &input,
                                                           const GpuStreamDescriptor &compute_stream_descriptor,
                                                           const int64_t block_size)
{
    if (block_size <= 0)
    {
        throw std::runtime_error("ReduceSumPartial requires positive block_size");
    }
    if (input.device().device_type != DeviceType::GPU)
    {
        throw std::runtime_error("ReduceSumPartial supports GPU tensors only");
    }

    const auto elements = input.shape().numel();
    const uint64_t output_elems =
        (elements + static_cast<uint64_t>(block_size) - 1) / static_cast<uint64_t>(block_size);

    TraceTensor output =
        graph.createTensor({output_elems}, DataType::FLOAT32, input.device(), compute_stream_descriptor, false);

    std::unordered_map<std::string, std::any> attributes{};
    attributes.emplace("block_size", block_size);

    graph.recordOperation(OperationEntry{.type = OpType::REDUCE_SUM_PARTIAL,
                                         .inputs = {input},
                                         .outputs = {output},
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});
    return output;
}

void pi::tensorlib::InplaceAdd(OpGraph &graph, const TraceTensor &dst, const TraceTensor &rhs,
                               const GpuStreamDescriptor &compute_stream_descriptor)
{
    if (dst.device() != rhs.device())
    {
        throw std::runtime_error("InplaceAdd requires inputs on the same device");
    }
    if (dst.dtype() != rhs.dtype())
    {
        throw std::runtime_error("InplaceAdd requires inputs with matching dtypes");
    }
    if (!shape_utils::IsBroadcastable(dst.shape(), rhs.shape()))
    {
        throw std::runtime_error("InplaceAdd requires broadcastable shapes");
    }
    graph.recordOperation(OperationEntry{.type = OpType::PLUS,
                                         .inputs = {dst, rhs},
                                         .outputs = {dst},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});
}

void pi::tensorlib::InplaceMul(OpGraph &graph, const TraceTensor &dst, const TraceTensor &rhs,
                               const GpuStreamDescriptor &compute_stream_descriptor)
{
    if (dst.device() != rhs.device())
    {
        throw std::runtime_error("InplaceMul requires inputs on the same device");
    }
    if (dst.dtype() != rhs.dtype())
    {
        throw std::runtime_error("InplaceMul requires inputs with matching dtypes");
    }
    if (!shape_utils::IsBroadcastable(dst.shape(), rhs.shape()))
    {
        throw std::runtime_error("InplaceMul requires broadcastable shapes");
    }
    graph.recordOperation(OperationEntry{.type = OpType::MUL,
                                         .inputs = {dst, rhs},
                                         .outputs = {dst},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});
}

void pi::tensorlib::InplaceDiv(OpGraph &graph, const TraceTensor &dst, const TraceTensor &rhs,
                               const GpuStreamDescriptor &compute_stream_descriptor)
{
    if (dst.device() != rhs.device())
    {
        throw std::runtime_error("InplaceDiv requires inputs on the same device");
    }
    if (dst.dtype() != rhs.dtype())
    {
        throw std::runtime_error("InplaceDiv requires inputs with matching dtypes");
    }
    if (!shape_utils::IsBroadcastable(dst.shape(), rhs.shape()))
    {
        throw std::runtime_error("InplaceDiv requires broadcastable shapes");
    }
    graph.recordOperation(OperationEntry{.type = OpType::DIV,
                                         .inputs = {dst, rhs},
                                         .outputs = {dst},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});
}

namespace
{
    void ValidateActivationOperand(const pi::tensorlib::TraceTensor &tensor)
    {
        if (tensor.dtype() != pi::tensorlib::DataType::BFLOAT16 && tensor.dtype() != pi::tensorlib::DataType::FLOAT16)
        {
            throw std::invalid_argument("Activation operations currently support only BFLOAT16 or FLOAT16 tensors");
        }
        if (tensor.device().device_type != pi::tensorlib::DeviceType::GPU)
        {
            throw std::invalid_argument("Activation operations require GPU tensors");
        }
    }

    std::unordered_map<std::string, std::any>
    BuildActivationAttributes(const pi::tensorlib::ActivationFunction activation_fn)
    {
        std::unordered_map<std::string, std::any> attributes{};
        attributes.emplace("activation_function", activation_fn);
        return attributes;
    }
} // namespace

pi::tensorlib::TraceTensor pi::tensorlib::Gelu(OpGraph &graph, const TraceTensor &input,
                                               const GpuStreamDescriptor &compute_stream_descriptor)
{
    ValidateActivationOperand(input);

    TraceTensor output =
        graph.createTensor(input.shape().dims(), input.dtype(), input.device(), compute_stream_descriptor, false);
    graph.recordOperation(OperationEntry{.type = OpType::ACT_FN,
                                         .inputs = {input},
                                         .outputs = {output},
                                         .attributes = BuildActivationAttributes(GELU),
                                         .gpu_stream_desc = compute_stream_descriptor});
    return output;
}

pi::tensorlib::TraceTensor pi::tensorlib::GeluBackward(OpGraph &graph, const TraceTensor &input,
                                                       const TraceTensor &upstream,
                                                       const GpuStreamDescriptor &compute_stream_descriptor)
{
    ValidateActivationOperand(input);
    ValidateActivationOperand(upstream);

    if (input.dtype() != upstream.dtype())
    {
        throw std::invalid_argument("GeluBackward requires matching input/upstream dtypes");
    }
    if (input.device() != upstream.device())
    {
        throw std::invalid_argument("GeluBackward requires input/upstream on the same device");
    }
    if (input.shape().ndims() != upstream.shape().ndims())
    {
        throw std::invalid_argument("GeluBackward requires input/upstream ranks to match");
    }
    for (size_t dim = 0; dim < input.shape().ndims(); ++dim)
    {
        if (input.shape()[dim] != upstream.shape()[dim])
        {
            throw std::invalid_argument("GeluBackward requires input/upstream shapes to match");
        }
    }

    TraceTensor output =
        graph.createTensor(input.shape().dims(), input.dtype(), input.device(), compute_stream_descriptor, false);
    auto attributes = BuildActivationAttributes(GELU);
    graph.recordOperation(OperationEntry{.type = OpType::ACT_FN_BWD,
                                         .inputs = {input, upstream},
                                         .outputs = {output},
                                         .attributes = attributes,
                                         .gpu_stream_desc = compute_stream_descriptor});
    return output;
}

void pi::tensorlib::GeluInplace(OpGraph &graph, const TraceTensor &tensor,
                                const GpuStreamDescriptor &compute_stream_descriptor)
{
    ValidateActivationOperand(tensor);

    graph.recordOperation(OperationEntry{.type = OpType::ACT_FN,
                                         .inputs = {tensor},
                                         .outputs = {tensor},
                                         .attributes = BuildActivationAttributes(GELU),
                                         .gpu_stream_desc = compute_stream_descriptor});
}

pi::tensorlib::TraceTensor pi::tensorlib::Relu(OpGraph &graph, const TraceTensor &input,
                                               const GpuStreamDescriptor &compute_stream_descriptor)
{
    ValidateActivationOperand(input);

    TraceTensor output =
        graph.createTensor(input.shape().dims(), input.dtype(), input.device(), compute_stream_descriptor, false);
    graph.recordOperation(OperationEntry{.type = OpType::ACT_FN,
                                         .inputs = {input},
                                         .outputs = {output},
                                         .attributes = BuildActivationAttributes(RELU),
                                         .gpu_stream_desc = compute_stream_descriptor});
    return output;
}

void pi::tensorlib::ReluInplace(OpGraph &graph, const TraceTensor &tensor,
                                const GpuStreamDescriptor &compute_stream_descriptor)
{
    ValidateActivationOperand(tensor);

    graph.recordOperation(OperationEntry{.type = OpType::ACT_FN,
                                         .inputs = {tensor},
                                         .outputs = {tensor},
                                         .attributes = BuildActivationAttributes(RELU),
                                         .gpu_stream_desc = compute_stream_descriptor});
}

pi::tensorlib::LstmCellResult pi::tensorlib::LstmCellFwd(OpGraph &graph, const TraceTensor &gates,
                                                         const TraceTensor &c_prev,
                                                         const GpuStreamDescriptor &compute_stream_descriptor)
{
    if (gates.device().device_type != DeviceType::GPU || c_prev.device().device_type != DeviceType::GPU)
    {
        throw std::runtime_error("LSTM_CELL_FWD currently supports GPU tensors only");
    }
    if (gates.dtype() != DataType::FLOAT16 || c_prev.dtype() != DataType::FLOAT16)
    {
        throw std::runtime_error("LSTM_CELL_FWD expects FLOAT16 inputs");
    }
    if (gates.shape().ndims() != 2)
    {
        throw std::runtime_error("LSTM_CELL_FWD expects gates shaped (B, 4H)");
    }
    if (c_prev.shape().ndims() != 2)
    {
        throw std::runtime_error("LSTM_CELL_FWD expects c_prev shaped (B, H)");
    }
    const auto batch = gates.shape()[0];
    const auto gate_dim = gates.shape()[1];
    if (gate_dim % 4 != 0)
    {
        throw std::runtime_error("LSTM_CELL_FWD gate dimension must be divisible by 4");
    }
    const auto hidden = gate_dim / 4;
    if (c_prev.shape()[0] != batch || c_prev.shape()[1] != hidden)
    {
        throw std::runtime_error("LSTM_CELL_FWD c_prev shape mismatch");
    }

    TraceTensor h_out =
        graph.createTensor({batch, hidden}, DataType::FLOAT16, gates.device(), compute_stream_descriptor, false);
    TraceTensor c_out =
        graph.createTensor({batch, hidden}, DataType::FLOAT16, gates.device(), compute_stream_descriptor, false);

    graph.recordOperation(OperationEntry{.type = OpType::LSTM_CELL_FWD,
                                         .inputs = {gates, c_prev},
                                         .outputs = {h_out, c_out},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});

    return LstmCellResult{.h = h_out, .c = c_out};
}

void pi::tensorlib::LstmCellFwdInplace(OpGraph &graph, const TraceTensor &gates, const TraceTensor &c_prev,
                                       const TraceTensor &h_out, const TraceTensor &c_out,
                                       const GpuStreamDescriptor &compute_stream_descriptor)
{
    if (gates.device().device_type != DeviceType::GPU || c_prev.device().device_type != DeviceType::GPU ||
        h_out.device().device_type != DeviceType::GPU || c_out.device().device_type != DeviceType::GPU)
    {
        throw std::runtime_error("LSTM_CELL_FWD inplace supports GPU tensors only");
    }
    const bool fp16_case = gates.dtype() == DataType::FLOAT16 && c_prev.dtype() == DataType::FLOAT16 &&
                           h_out.dtype() == DataType::FLOAT16 && c_out.dtype() == DataType::FLOAT16;
    const bool fp32_state_fp16_case = gates.dtype() == DataType::FLOAT32 && c_prev.dtype() == DataType::FLOAT32 &&
                                      h_out.dtype() == DataType::FLOAT16 && c_out.dtype() == DataType::FLOAT32;
    const bool fp32_state_bf16_case = gates.dtype() == DataType::FLOAT32 && c_prev.dtype() == DataType::FLOAT32 &&
                                      h_out.dtype() == DataType::BFLOAT16 && c_out.dtype() == DataType::FLOAT32;
    if (!fp16_case && !fp32_state_fp16_case && !fp32_state_bf16_case)
    {
        throw std::runtime_error(
            "LSTM_CELL_FWD inplace expects FLOAT16 inputs/outputs or FLOAT32 gates/c_prev/c_out with FLOAT16/BFLOAT16 hidden");
    }
    if (gates.shape().ndims() != 2)
    {
        throw std::runtime_error("LSTM_CELL_FWD inplace expects gates shaped (B, 4H)");
    }
    if (c_prev.shape().ndims() != 2 || h_out.shape().ndims() != 2 || c_out.shape().ndims() != 2)
    {
        throw std::runtime_error("LSTM_CELL_FWD inplace expects 2D tensors for states");
    }
    const auto batch = gates.shape()[0];
    const auto gate_dim = gates.shape()[1];
    if (gate_dim % 4 != 0)
    {
        throw std::runtime_error("LSTM_CELL_FWD inplace gate dimension must be divisible by 4");
    }
    const auto hidden = gate_dim / 4;
    if (c_prev.shape()[0] != batch || c_prev.shape()[1] != hidden || h_out.shape()[0] != batch ||
        h_out.shape()[1] != hidden || c_out.shape()[0] != batch || c_out.shape()[1] != hidden)
    {
        throw std::runtime_error("LSTM_CELL_FWD inplace state shape mismatch");
    }

    graph.recordOperation(OperationEntry{.type = OpType::LSTM_CELL_FWD,
                                         .inputs = {gates, c_prev},
                                         .outputs = {h_out, c_out},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});
}

void pi::tensorlib::LstmCellFwdInplace(OpGraph &graph, const TraceTensor &gates, const TraceTensor &c_prev,
                                       const TraceTensor &h_out, const TraceTensor &c_out, const TraceTensor &y_out,
                                       const GpuStreamDescriptor &compute_stream_descriptor)
{
    if (gates.device().device_type != DeviceType::GPU || c_prev.device().device_type != DeviceType::GPU ||
        h_out.device().device_type != DeviceType::GPU || c_out.device().device_type != DeviceType::GPU ||
        y_out.device().device_type != DeviceType::GPU)
    {
        throw std::runtime_error("LSTM_CELL_FWD inplace supports GPU tensors only");
    }
    const bool fp16_case = gates.dtype() == DataType::FLOAT16 && c_prev.dtype() == DataType::FLOAT16 &&
                           h_out.dtype() == DataType::FLOAT16 && c_out.dtype() == DataType::FLOAT16 &&
                           y_out.dtype() == DataType::FLOAT16;
    const bool fp32_state_fp16_case = gates.dtype() == DataType::FLOAT32 && c_prev.dtype() == DataType::FLOAT32 &&
                                      h_out.dtype() == DataType::FLOAT16 && c_out.dtype() == DataType::FLOAT32 &&
                                      y_out.dtype() == DataType::FLOAT16;
    const bool fp32_state_bf16_case = gates.dtype() == DataType::FLOAT32 && c_prev.dtype() == DataType::FLOAT32 &&
                                      h_out.dtype() == DataType::BFLOAT16 && c_out.dtype() == DataType::FLOAT32 &&
                                      y_out.dtype() == DataType::BFLOAT16;
    if (!fp16_case && !fp32_state_fp16_case && !fp32_state_bf16_case)
    {
        throw std::runtime_error(
            "LSTM_CELL_FWD inplace expects FLOAT16 tensors or mixed FP32 state with FLOAT16/BFLOAT16 hidden outputs");
    }
    if (gates.shape().ndims() != 2)
    {
        throw std::runtime_error("LSTM_CELL_FWD inplace expects gates shaped (B, 4H)");
    }
    if (c_prev.shape().ndims() != 2 || h_out.shape().ndims() != 2 || c_out.shape().ndims() != 2 ||
        y_out.shape().ndims() != 2)
    {
        throw std::runtime_error("LSTM_CELL_FWD inplace expects 2D tensors for states");
    }
    const auto batch = gates.shape()[0];
    const auto gate_dim = gates.shape()[1];
    if (gate_dim % 4 != 0)
    {
        throw std::runtime_error("LSTM_CELL_FWD inplace gate dimension must be divisible by 4");
    }
    const auto hidden = gate_dim / 4;
    if (c_prev.shape()[0] != batch || c_prev.shape()[1] != hidden || h_out.shape()[0] != batch ||
        h_out.shape()[1] != hidden || c_out.shape()[0] != batch || c_out.shape()[1] != hidden ||
        y_out.shape()[0] != batch || y_out.shape()[1] != hidden)
    {
        throw std::runtime_error("LSTM_CELL_FWD inplace state shape mismatch");
    }

    graph.recordOperation(OperationEntry{.type = OpType::LSTM_CELL_FWD,
                                         .inputs = {gates, c_prev},
                                         .outputs = {h_out, c_out, y_out},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});
}

void pi::tensorlib::AwaitAsyncTransfers(OpGraph &graph, const TransferType transfer_side, const Device &gpu_device,
                                        const GpuStreamDescriptor &waiting_stream_desc)
{
    const auto event = graph.createGpuEvent(gpu_device);
    switch (transfer_side)
    {
        case TransferType::D2H:
        {
            graph.recordGpuEvent(event, GpuStreamDescriptors::D2H);
            graph.awaitGpuEvent(event, waiting_stream_desc);
            graph.deleteGpuEvent(event);
            break;
        }
        case TransferType::H2D:
        {
            graph.recordGpuEvent(event, GpuStreamDescriptors::H2D);
            graph.awaitGpuEvent(event, waiting_stream_desc);
            graph.deleteGpuEvent(event);
            break;
        }
    }
}

void pi::tensorlib::AwaitComputeForTransfer(OpGraph &graph, const TransferType transfer_side, const Device &device,
                                            const GpuStreamDescriptor &waiting_stream_desc)
{
    const auto event = graph.createGpuEvent(device);
    switch (transfer_side)
    {
        case TransferType::D2H:
        {
            graph.recordGpuEvent(event, waiting_stream_desc);
            graph.awaitGpuEvent(event, GpuStreamDescriptors::D2H);
            graph.deleteGpuEvent(event);
            break;
        }
        case TransferType::H2D:
        {
            graph.recordGpuEvent(event, waiting_stream_desc);
            graph.awaitGpuEvent(event, GpuStreamDescriptors::H2D);
            graph.deleteGpuEvent(event);
            break;
        }
    }
}

void pi::tensorlib::AwaitComputeStream(OpGraph &graph, const Device &device,
                                       const GpuStreamDescriptor &compute_stream_descriptor)
{
    const auto event = graph.createGpuEvent(device);
    graph.recordGpuEvent(event, compute_stream_descriptor);
    graph.awaitGpuEvent(event, GpuStreamDescriptors::Main);
    graph.deleteGpuEvent(event);
}

void pi::tensorlib::AwaitMainStream(OpGraph &graph, const Device &device,
                                    const GpuStreamDescriptor &compute_stream_descriptor)
{
    const auto event = graph.createGpuEvent(device);
    graph.recordGpuEvent(event, GpuStreamDescriptors::Main);
    graph.awaitGpuEvent(event, compute_stream_descriptor);
    graph.deleteGpuEvent(event);
}
