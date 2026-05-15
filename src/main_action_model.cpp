#include "main_action_model.h"

#include "functional.h"

#include <utility>

fbamtrain::ActionModelModule::ActionModelModule(FbamModelConfiguration config, const uint32_t batch_size,
                                                const size_t vocab_size, pi::tensorlib::OpGraph &graph,
                                                const pi::tensorlib::Device device, const pi::tensorlib::DataType dtype,
                                                const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor,
                                                const pi::tensorlib::GpuStreamDescriptor &head_stream_descriptor)
    : Module("action_model"), config_(std::move(config)), rng_seed_(config_.model_init_seed),
      compute_stream_descriptor_(compute_stream_descriptor),
      head_stream_descriptor_(head_stream_descriptor),
      lstm_(name_ + ".lstm", config_.n_embed, config_.n_embed, config_.recompute_interval,
            config_.streaming_chunk_size, device, graph, rng_seed_, dtype, compute_stream_descriptor_),
      head_(name_ + ".head", config_.n_embed, vocab_size, device, dtype, pi::tensorlib::ActivationFunction::NONE, graph,
            rng_seed_, head_stream_descriptor_, /*has_bias=*/false, /*use_fp16_matmul_acc=*/false)
{
    (void)batch_size; // kept for potential future use
}

pi::tensorlib::LstmForwardStreamingResult
fbamtrain::ActionModelModule::buildForward(pi::tensorlib::OpGraph &graph,
                                           const std::initializer_list<pi::tensorlib::TraceTensor> inputs,
                                           bool save_input_for_backward)
{
    const auto &x = inputs.begin()[0];
    const auto &h0 = inputs.begin()[1];
    const auto &c0 = inputs.begin()[2];
    auto lstm_out = lstm_.buildForward(graph, {x, h0, c0}, save_input_for_backward);
    return lstm_out; // returns allocation. search "ALLOC_100" for deallocation-site
}

std::vector<pi::tensorlib::ParameterEntry> fbamtrain::ActionModelModule::parameters() const
{
    const auto lstm_params = lstm_.parameters();
    const auto head_params = head_.parameters();
    std::vector<pi::tensorlib::ParameterEntry> params;
    params.reserve(lstm_params.size() + head_params.size());
    for (const auto &p : lstm_params)
    {
        params.push_back(p);
    }
    for (const auto &p : head_params)
    {
        params.push_back(p);
    }
    return params;
}

void fbamtrain::ActionModelModule::buildBackward(
    pi::tensorlib::OpGraph &graph, const ActionModelBackwardInput &backward_input,
    const std::unordered_map<std::string, pi::tensorlib::TraceTensor> &parameter_gradients,
    const std::unordered_map<std::string, pi::tensorlib::TraceTensor> &operand_gradients)
{
    const auto &head_params = head_.parameters();
    if (head_params.empty())
    {
        throw std::runtime_error("Action model head has no parameters");
    }
    const auto head_weight = head_params[0].tensor;
    const auto head_weight_dtype = head_weight.dtype();
    
    std::optional<pi::tensorlib::TraceTensor> head_weight_grad{};
    if (const auto it = parameter_gradients.find(head_params[0].name); it != parameter_gradients.end())
    {
        head_weight_grad = it->second;
    }

    const auto &output_host = backward_input.output_host;
    const auto &action_targets_host = backward_input.action_targets_host;

    if (output_host.shape().ndims() != 3)
    {
        throw std::runtime_error("Action model output must be a 3D tensor (T, B, H)");
    }
    if (action_targets_host.shape().ndims() != 2)
    {
        throw std::runtime_error("Action model targets must be a 2D tensor (T, B)");
    }
    if (output_host.shape()[0] != action_targets_host.shape()[0] ||
        output_host.shape()[1] != action_targets_host.shape()[1])
    {
        throw std::runtime_error("Action model output and targets must agree on (T, B)");
    }

    const uint64_t seq_len = output_host.shape()[0];
    const uint64_t batch_size = output_host.shape()[1];
    const uint64_t n_embed = output_host.shape()[2];

    if (n_embed != static_cast<uint64_t>(config_.n_embed))
    {
        throw std::runtime_error("Action model output embedding size mismatch");
    }

    const uint64_t vocab_size = head_weight.shape()[1];
    const uint64_t streaming_chunk = lstm_.streamingChunkSize();
    const auto ce_stream_desc = backward_input.cross_entropy_stream_desc;

    const float loss_scale = backward_input.loss_scale;
    if (loss_scale <= 0.0f)
    {
        throw std::runtime_error("Action model loss scale must be positive.");
    }

    const auto output_dtype = output_host.dtype();
    pi::tensorlib::TraceTensor y_chunk_gpu =
        graph.createTensor({streaming_chunk, batch_size, n_embed}, output_dtype,
                           head_weight.device(), ce_stream_desc, false);
    pi::tensorlib::TraceTensor dy_targets_chunk_gpu =
        graph.createTensor({streaming_chunk, batch_size}, pi::tensorlib::DataType::UINT32, head_weight.device(),
                           ce_stream_desc, false);

    const pi::tensorlib::TraceTensor head_weight_T = head_weight.transpose(graph, {1, 0});

    const pi::tensorlib::StreamingDySupplier dy_supplier =
        [action_targets_host, loss_scale, y_chunk_gpu, dy_targets_chunk_gpu, head_weight, head_weight_T,
         head_weight_dtype, head_weight_grad, vocab_size, batch_size, n_embed, seq_len, streaming_chunk, ce_stream_desc,
         output_host](pi::tensorlib::OpGraph &hook_graph, const pi::tensorlib::TraceTensor &dy_dst,
                      const uint64_t chunk_start,
                      const uint64_t chunk_steps) -> std::optional<pi::tensorlib::GpuStreamDescriptor>
    {
        // Validate chunk window.
        if (chunk_start + chunk_steps > seq_len)
        {
            throw std::out_of_range("dy supplier slice: chunk_start=" + std::to_string(chunk_start) +
                                    " chunk_steps=" + std::to_string(chunk_steps) +
                                    " exceeds total_steps=" + std::to_string(seq_len));
        }
        if (chunk_steps > streaming_chunk)
        {
            throw std::out_of_range("dy supplier chunk_steps exceeds streaming_chunk size");
        }

        // Stage y/targets for this chunk onto the CE stream.
        pi::tensorlib::TraceTensor y_host_slice = output_host.slice(hook_graph, 0, chunk_start, chunk_steps);
        pi::tensorlib::TraceTensor y_chunk_slice = y_chunk_gpu.slice(hook_graph, 0, 0, chunk_steps);
        pi::tensorlib::DeviceCopy(
            hook_graph, y_host_slice, y_chunk_slice,
            ce_stream_desc);

        pi::tensorlib::TraceTensor targets_host_slice =
            action_targets_host.slice(hook_graph, 0, chunk_start, chunk_steps);
        pi::tensorlib::TraceTensor targets_slice = dy_targets_chunk_gpu.slice(hook_graph, 0, 0, chunk_steps);
        pi::tensorlib::DeviceCopy(
            hook_graph, targets_host_slice, targets_slice,
            ce_stream_desc);

        // Compute logits and CE gradient for the chunk (cast to head dtype if needed).
        pi::tensorlib::TraceTensor head_input = y_chunk_slice;
        if (head_input.dtype() != head_weight_dtype)
        {
            head_input = head_input.to(hook_graph, head_weight_dtype, ce_stream_desc);
        }
        pi::tensorlib::TraceTensor flat_y = head_input.view(hook_graph, {chunk_steps * batch_size, n_embed});
        pi::tensorlib::TraceTensor flat_logits =
            hook_graph.createTensor({flat_y.shape()[0], vocab_size}, head_weight_dtype, head_weight.device(),
                                    ce_stream_desc, false);
        hook_graph.recordOperation(pi::tensorlib::OperationEntry{.type = pi::tensorlib::OpType::MATMUL,
                                                                 .inputs = {flat_y, head_weight},
                                                                 .outputs = {flat_logits},
                                                                 .is_useful = false,
                                                                 .attributes = {},
                                                                 .gpu_stream_desc = ce_stream_desc});

        pi::tensorlib::TraceTensor vocab_logits = flat_logits.view(hook_graph, {chunk_steps, batch_size, vocab_size});
        pi::tensorlib::TraceTensor upstream_scalar = hook_graph.createTensor(
            {1}, pi::tensorlib::DataType::FLOAT32, vocab_logits.device(), ce_stream_desc, false);
        pi::tensorlib::FillConstant(hook_graph, upstream_scalar, loss_scale, ce_stream_desc);

        pi::tensorlib::TraceTensor grad_logits =
            CrossEntropyOnTargetsBackward(hook_graph, vocab_logits, targets_slice, upstream_scalar,
                                          pi::tensorlib::Reduction::ADD, ce_stream_desc,
                                          /*reduce_over_rows=*/true);

        // Project dLogits back to dY and write into dy_dst.
        pi::tensorlib::TraceTensor grad_logits_flat =
            grad_logits.view(hook_graph, {chunk_steps * batch_size, vocab_size});

        if (head_weight_grad.has_value())
        {
            pi::tensorlib::TraceTensor flat_y_T = flat_y.transpose(hook_graph, {1, 0});
            std::unordered_map<std::string, std::any> matmul_attrs{};
            matmul_attrs.emplace("accumulate_output", true);
            hook_graph.recordOperation(pi::tensorlib::OperationEntry{.type = pi::tensorlib::OpType::MATMUL,
                                                                     .inputs = {flat_y_T, grad_logits_flat},
                                                                     .outputs = {head_weight_grad.value()},
                                                                     .attributes = matmul_attrs,
                                                                     .gpu_stream_desc = ce_stream_desc});
        }

        pi::tensorlib::TraceTensor dy_flat = dy_dst.view(hook_graph, {chunk_steps * batch_size, n_embed});
        hook_graph.recordOperation(pi::tensorlib::OperationEntry{.type = pi::tensorlib::OpType::MATMUL,
                                                                 .inputs = {grad_logits_flat, head_weight_T},
                                                                 .outputs = {dy_flat},
                                                                 .attributes = {},
                                                                 .gpu_stream_desc = ce_stream_desc});

        hook_graph.deleteTensor(flat_logits);
        hook_graph.deleteTensor(grad_logits);
        return ce_stream_desc;
    };

    pi::tensorlib::StreamingLstmBackwardInput upstream_grad{};
    upstream_grad.upstream_supplier = dy_supplier;
    upstream_grad.upstream_supplier_stream_desc = ce_stream_desc;
    upstream_grad.upstream_dy_dtype = head_weight_dtype;

    lstm_.buildBackward(graph, upstream_grad, parameter_gradients, operand_gradients);
    lstm_.clearChunkHook();
}
