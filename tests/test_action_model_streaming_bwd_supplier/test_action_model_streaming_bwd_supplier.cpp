#include "testing.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{
    constexpr pi::tensorlib::Device DEVICE_CPU{pi::tensorlib::DeviceType::CPU, 0};
    constexpr pi::tensorlib::Device DEVICE_GPU{pi::tensorlib::DeviceType::GPU, 0};
    constexpr int SUPPLIER_STREAM_ID = 1;
    constexpr int SUPPLIER_STREAM_PRIORITY = 1000;

    void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<MatmulFusePass>());
        passes.emplace_back(std::make_unique<MatmulImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<FillUniformImplPass>());
        passes.emplace_back(std::make_unique<ActImplPass>());
        passes.emplace_back(std::make_unique<CastImplPass>());
        passes.emplace_back(std::make_unique<ContiguousImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
        passes.emplace_back(std::make_unique<ReduceSumImplPass>());
        passes.emplace_back(std::make_unique<LstmCellImplPass>());
        passes.emplace_back(std::make_unique<LstmCellBwdImplPass>());
        passes.emplace_back(std::make_unique<CrossEntropyOnTargetsImplPass>());
        passes.emplace_back(std::make_unique<GatherImplPass>());
        passes.emplace_back(std::make_unique<DivImplPass>());
        passes.emplace_back(std::make_unique<ReduceSumPartialImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);
    }
} // namespace

int main()
{
    using namespace pi::tensorlib;

    ExecutionBackend &backend = ExecutionBackend::getInstance();
    const auto main_stream_desc = GpuStreamDescriptors::Main;
    const auto supplier_stream_desc = GpuStreamDescriptor{StreamKind::Compute, SUPPLIER_STREAM_ID};
    if (SUPPLIER_STREAM_ID != 0)
    {
        ExecutionBackend::SetComputeStreamPriority(DEVICE_GPU, supplier_stream_desc, SUPPLIER_STREAM_PRIORITY);
    }
    const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();

    const auto ref_tensors = safetensors::Load("reference.safetensors", true);
    const auto fetch = [&ref_tensors](const char *name)
    {
        const auto it = ref_tensors.find(name);
        if (it == ref_tensors.end())
        {
            throw std::runtime_error(std::string("missing tensor: ") + name);
        }
        return it->second;
    };

    const auto x_host = fetch("x");
    const auto h0_host = fetch("h0");
    const auto c0_host = fetch("c0");
    const auto w_ih_host = fetch("w_ih");
    const auto w_hh_host = fetch("w_hh");
    const auto b_ih_host = fetch("b_ih");
    const auto b_hh_host = fetch("b_hh");
    const auto head_weight_host = fetch("head_weight");
    const auto targets_host_float = fetch("targets");
    const auto expected_loss_sum = fetch("loss_sum");
    const auto expected_loss_mean = fetch("loss_mean");

    auto upstream_scalar_host = RealTensor::Allocate({1}, DataType::FLOAT32, DEVICE_CPU);
    *static_cast<float *>(upstream_scalar_host->dataptr()) = 1.0f;

    // Prepare device copies.
    auto h0_gpu = h0_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
    auto c0_gpu = c0_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
    auto w_ih_gpu = w_ih_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
    auto w_hh_gpu = w_hh_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
    auto b_ih_gpu = b_ih_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
    auto b_hh_gpu = b_hh_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
    const auto head_weight_shape = head_weight_host->shape().dims();
    if (head_weight_shape.size() != 2)
    {
        throw std::runtime_error("head_weight must be 2D");
    }
    auto head_weight_gpu = RealTensor::AllocateOnStream({head_weight_shape[0], head_weight_shape[1]},
                                                        head_weight_host->dtype(), DEVICE_GPU, supplier_stream_desc,
                                                        false);
    head_weight_gpu->storage()->copyFrom(*head_weight_host->storage(), main_stream_desc);

    const auto seq_len = x_host->shape()[0];
    const auto batch = x_host->shape()[1];
    const auto hidden = x_host->shape()[2];
    const auto vocab = head_weight_host->shape()[1];

    auto targets_host_uint32 = RealTensor::Allocate({seq_len, batch}, DataType::UINT32, DEVICE_CPU, true);
    {
        auto src = static_cast<const float *>(targets_host_float->dataptr());
        auto *dst = static_cast<uint32_t *>(targets_host_uint32->dataptr());
        const auto count = targets_host_uint32->shape().numel();
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = static_cast<uint32_t>(src[i]);
        }
    }

    auto loss_denominator_host = RealTensor::Allocate({1}, DataType::FLOAT32, DEVICE_CPU);
    *static_cast<float *>(loss_denominator_host->dataptr()) = static_cast<float>(seq_len * batch);
    auto loss_sum_tensor = RealTensor::Allocate({1}, DataType::FLOAT32, DEVICE_GPU);
    auto loss_mean_tensor = RealTensor::Allocate({1}, DataType::FLOAT32, DEVICE_GPU);

    std::vector<GraphInputDescriptor> inputs{};
    inputs.emplace_back(GraphInputDescriptor{
        .name = "x",
        .tensor = TraceTensor::Create(x_host->shape().dims(), x_host->dtype(), DEVICE_CPU, main_stream_desc,
                                      /*pinned=*/true)});
    inputs.emplace_back(GraphInputDescriptor{
        .name = "h0", .tensor = TraceTensor::Create(h0_host->shape().dims(), DataType::FLOAT32, DEVICE_GPU,
                                                    main_stream_desc)});
    inputs.emplace_back(GraphInputDescriptor{
        .name = "c0", .tensor = TraceTensor::Create(c0_host->shape().dims(), DataType::FLOAT32, DEVICE_GPU,
                                                    main_stream_desc)});
    inputs.emplace_back(GraphInputDescriptor{
        .name = "targets",
        .tensor = TraceTensor::Create(targets_host_uint32->shape().dims(), DataType::UINT32, DEVICE_CPU,
                                      main_stream_desc, true)});
    inputs.emplace_back(GraphInputDescriptor{.name = "loss_denominator",
                                             .tensor = TraceTensor::Create({1}, DataType::FLOAT32, DEVICE_CPU,
                                                                           main_stream_desc)});
    inputs.emplace_back(
        GraphInputDescriptor{.name = "loss_sum", .tensor = TraceTensor::Create({1}, DataType::FLOAT32, DEVICE_GPU,
                                                                               main_stream_desc)});
    inputs.emplace_back(
        GraphInputDescriptor{.name = "loss_mean", .tensor = TraceTensor::Create({1}, DataType::FLOAT32, DEVICE_GPU,
                                                                                main_stream_desc)});
    inputs.emplace_back(
        GraphInputDescriptor{.name = "upstream", .tensor = TraceTensor::Create({1}, DataType::FLOAT32, DEVICE_CPU,
                                                                               main_stream_desc)});

    std::vector<GraphInputDescriptor> params{};
    params.emplace_back(GraphInputDescriptor{
        .name = "w_ih", .tensor = TraceTensor::Create(w_ih_host->shape().dims(), DataType::FLOAT32, DEVICE_GPU,
                                                      main_stream_desc)});
    params.emplace_back(GraphInputDescriptor{
        .name = "w_hh", .tensor = TraceTensor::Create(w_hh_host->shape().dims(), DataType::FLOAT32, DEVICE_GPU,
                                                      main_stream_desc)});
    params.emplace_back(GraphInputDescriptor{
        .name = "b_ih", .tensor = TraceTensor::Create(b_ih_host->shape().dims(), DataType::FLOAT32, DEVICE_GPU,
                                                      main_stream_desc)});
    params.emplace_back(GraphInputDescriptor{
        .name = "b_hh", .tensor = TraceTensor::Create(b_hh_host->shape().dims(), DataType::FLOAT32, DEVICE_GPU,
                                                      main_stream_desc)});
    params.emplace_back(GraphInputDescriptor{
        .name = "head_weight",
        .tensor = TraceTensor::Create(head_weight_host->shape().dims(), head_weight_host->dtype(), DEVICE_GPU,
                                      main_stream_desc)});

    for (auto &entry : inputs)
    {
        entry.tensor.markRetained();
    }
    for (auto &entry : params)
    {
        entry.tensor.markRetained();
    }

    OpGraph graph(inputs, params);

    TraceTensor loss_sum = inputs[5].tensor;
    TraceTensor loss_mean = inputs[6].tensor;

    const auto streaming_chunk = 8;

    // Forward LSTM (streaming) without chunk hook; caches are returned for backward.
    auto lstm_result = StreamingLstmFwd(graph, inputs[0].tensor, inputs[1].tensor, inputs[2].tensor, params[0].tensor,
                                        params[1].tensor, params[2].tensor, params[3].tensor, DataType::FLOAT32,
                                        /*recompute_interval=*/1, streaming_chunk, main_stream_desc);

    TraceTensor loss_denominator_gpu = graph.createTensor({1}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
    DeviceCopy(graph, inputs[4].tensor, loss_denominator_gpu, main_stream_desc);

    TraceTensor y_gpu =
        graph.createTensor(lstm_result.output.shape().dims(), DataType::FLOAT16, DEVICE_GPU, supplier_stream_desc,
                           false);
    DeviceCopy(graph, lstm_result.output, y_gpu, main_stream_desc);
    TraceTensor flat_y = y_gpu.view(graph, {seq_len * batch, hidden});
    TraceTensor logits =
        graph.createTensor({flat_y.shape()[0], vocab}, head_weight_host->dtype(), DEVICE_GPU, main_stream_desc, false);
    graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                         .inputs = {flat_y, params[4].tensor},
                                         .outputs = {logits},
                                         .attributes = {},
                                         .gpu_stream_desc = main_stream_desc});
    TraceTensor vocab_logits = logits.view(graph, {seq_len, batch, vocab});

    TraceTensor targets_gpu =
        graph.createTensor(inputs[3].tensor.shape().dims(), inputs[3].tensor.dtype(), DEVICE_GPU, supplier_stream_desc,
                           false);
    DeviceCopy(graph, inputs[3].tensor, targets_gpu, main_stream_desc);

    FillZeros(graph, loss_sum, main_stream_desc);
    TraceTensor ce_sum = CrossEntropyOnTargets(graph, vocab_logits, targets_gpu, Reduction::ADD,
                                               main_stream_desc, /*reduce_over_rows=*/true);
    InplaceAdd(graph, loss_sum, ce_sum, main_stream_desc);
    graph.deleteTensor(ce_sum);

    Div(graph, loss_sum, loss_denominator_gpu, loss_mean, main_stream_desc);

    // Backward: CE -> matmul head -> LSTM, streamed per chunk on a separate compute stream.
    TraceTensor upstream_scalar =
        graph.createTensor({1}, DataType::FLOAT32, DEVICE_GPU, supplier_stream_desc, false);
    const float loss_scale = 1.0f / static_cast<float>(seq_len * batch);
    FillConstant(graph, upstream_scalar, loss_scale, supplier_stream_desc);

    TraceTensor grad_head_weight =
        graph.createTensor(head_weight_host->shape().dims(), DataType::FLOAT16, DEVICE_GPU, supplier_stream_desc, false);
    FillZeros(graph, grad_head_weight, supplier_stream_desc);

    const TraceTensor head_weight = params[4].tensor;
    const TraceTensor head_weight_T = head_weight.transpose(graph, {1, 0});
    const TraceTensor &y_host_trace = y_gpu;
    const auto head_weight_dtype = head_weight_host->dtype();

    TraceTensor y_chunk_gpu =
        graph.createTensor({static_cast<uint64_t>(streaming_chunk), batch, hidden}, DataType::FLOAT16, DEVICE_GPU,
                           supplier_stream_desc, false);
    std::unordered_map<std::string, std::any> accumulate_attr{};
    accumulate_attr.emplace("accumulate_output", true);

    StreamingDySupplier dy_supplier = [&](OpGraph &hook_graph, const TraceTensor &dy_dst, const uint64_t chunk_start,
                                          const uint64_t chunk_steps) -> std::optional<GpuStreamDescriptor>
    {
        const uint64_t total_steps = y_host_trace.shape()[0];
        if (chunk_start + chunk_steps > total_steps)
        {
            throw std::out_of_range("dy supplier slice out of range");
        }
        if (chunk_steps > static_cast<uint64_t>(streaming_chunk))
        {
            throw std::out_of_range("dy supplier chunk exceeds streaming_chunk size");
        }

        TraceTensor y_host_slice = y_host_trace.slice(hook_graph, 0, chunk_start, chunk_steps);
        TraceTensor y_chunk_slice = y_chunk_gpu.slice(hook_graph, 0, 0, chunk_steps);
        DeviceCopy(hook_graph, y_host_slice, y_chunk_slice, supplier_stream_desc);

        TraceTensor targets_chunk_slice = targets_gpu.slice(hook_graph, 0, chunk_start, chunk_steps);

        const uint64_t chunk_steps_u64 = chunk_steps;
        TraceTensor flat_y = y_chunk_slice.view(hook_graph, {chunk_steps_u64 * batch, hidden});
        TraceTensor flat_logits =
            hook_graph.createTensor({flat_y.shape()[0], vocab}, head_weight_dtype, DEVICE_GPU, supplier_stream_desc,
                                    false);
        hook_graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                                  .inputs = {flat_y, head_weight},
                                                  .outputs = {flat_logits},
                                                  .is_useful = false,
                                                  .attributes = {},
                                                  .gpu_stream_desc = supplier_stream_desc});

        TraceTensor vocab_logits = flat_logits.view(hook_graph, {chunk_steps_u64, batch, vocab});

        TraceTensor grad_logits =
            CrossEntropyOnTargetsBackward(hook_graph, vocab_logits, targets_chunk_slice, upstream_scalar,
                                          Reduction::ADD, supplier_stream_desc, /*reduce_over_rows=*/true);

        TraceTensor grad_logits_flat = grad_logits.view(hook_graph, {chunk_steps_u64 * batch, vocab});

        TraceTensor dy_flat = dy_dst.view(hook_graph, {chunk_steps_u64 * batch, hidden});
        hook_graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                                  .inputs = {grad_logits_flat, head_weight_T},
                                                  .outputs = {dy_flat},
                                                  .attributes = {},
                                                  .gpu_stream_desc = supplier_stream_desc});

        TraceTensor flat_y_T = flat_y.transpose(hook_graph, {1, 0});
        hook_graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                                  .inputs = {flat_y_T, grad_logits_flat},
                                                  .outputs = {grad_head_weight},
                                                  .attributes = accumulate_attr,
                                                  .gpu_stream_desc = supplier_stream_desc});

        hook_graph.deleteTensor(flat_logits);
        hook_graph.deleteTensor(grad_logits);
        return supplier_stream_desc;
    };

    auto lstm_bwd = StreamingLstmBwdStreamedDy(
        graph, inputs[0].tensor, inputs[1].tensor, inputs[2].tensor, params[0].tensor, params[1].tensor,
        params[2].tensor, params[3].tensor, lstm_result.output, lstm_result.h_cache, lstm_result.c_cache, dy_supplier,
        supplier_stream_desc, DataType::FLOAT16, std::nullopt, std::nullopt, /*recompute_interval=*/1, streaming_chunk,
        std::nullopt, main_stream_desc);

    graph.finalize();

    std::vector<GraphExecutionInputDescriptor> exec_inputs{};
    exec_inputs.push_back(GraphExecutionInputDescriptor{.name = "x", .tensor = x_host});
    exec_inputs.push_back(GraphExecutionInputDescriptor{.name = "h0", .tensor = h0_gpu});
    exec_inputs.push_back(GraphExecutionInputDescriptor{.name = "c0", .tensor = c0_gpu});
    exec_inputs.push_back(GraphExecutionInputDescriptor{.name = "targets", .tensor = targets_host_uint32});
    exec_inputs.push_back(GraphExecutionInputDescriptor{.name = "loss_denominator", .tensor = loss_denominator_host});
    exec_inputs.push_back(GraphExecutionInputDescriptor{.name = "loss_sum", .tensor = loss_sum_tensor});
    exec_inputs.push_back(GraphExecutionInputDescriptor{.name = "loss_mean", .tensor = loss_mean_tensor});
    exec_inputs.push_back(GraphExecutionInputDescriptor{.name = "upstream", .tensor = upstream_scalar_host});

    std::vector<GraphExecutionInputDescriptor> exec_params{};
    exec_params.push_back(GraphExecutionInputDescriptor{.name = "w_ih", .tensor = w_ih_gpu});
    exec_params.push_back(GraphExecutionInputDescriptor{.name = "w_hh", .tensor = w_hh_gpu});
    exec_params.push_back(GraphExecutionInputDescriptor{.name = "b_ih", .tensor = b_ih_gpu});
    exec_params.push_back(GraphExecutionInputDescriptor{.name = "b_hh", .tensor = b_hh_gpu});
    exec_params.push_back(GraphExecutionInputDescriptor{.name = "head_weight", .tensor = head_weight_gpu});

    ExecutionPlan plan = ExecutionPlan::FromGraph(graph, exec_inputs, exec_params);
    ApplyDefaultPasses(plan);

    Executor executor{plan, backend, 0};
    executor.execute(allocator_registry);

    const auto actual_loss_sum = executor.getOutput(loss_sum);
    const auto actual_loss_mean = executor.getOutput(loss_mean);
    const auto grad_x = executor.getOutput(lstm_bwd.grad_x);
    const auto grad_h0 = executor.getOutput(lstm_bwd.grad_h0);
    const auto grad_c0 = executor.getOutput(lstm_bwd.grad_c0);
    const auto grad_w_ih = executor.getOutput(lstm_bwd.grad_w_ih);
    const auto grad_w_hh = executor.getOutput(lstm_bwd.grad_w_hh);
    const auto grad_b_ih = executor.getOutput(lstm_bwd.grad_b_ih);
    const auto grad_b_hh = executor.getOutput(lstm_bwd.grad_b_hh);
    const auto grad_head = executor.getOutput(grad_head_weight);

    if (!actual_loss_sum || !actual_loss_mean || !grad_x || !grad_h0 || !grad_c0 || !grad_w_ih || !grad_w_hh ||
        !grad_b_ih || !grad_b_hh || !grad_head)
    {
        throw std::runtime_error("missing outputs");
    }

    const bool skip_asserts = std::getenv("FBAMTRAIN_LSTM_BWD_SKIP_ASSERT") != nullptr;
    if (!skip_asserts)
    {
        const auto assert_named = [&](const char *name, const std::shared_ptr<RealTensor> &expected,
                                      const std::shared_ptr<RealTensor> &actual, const double tolerance)
        {
            try
            {
                testing::AssertSimilar(expected, actual, tolerance);
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error(std::string(name) + ": " + e.what());
            }
        };

        assert_named("loss_sum", expected_loss_sum, *actual_loss_sum, 1e-3);
        assert_named("loss_mean", expected_loss_mean, *actual_loss_mean, 1e-4);
        assert_named("grad_x", fetch("grad_x"), *grad_x, 5e-3);
        assert_named("grad_h0", fetch("grad_h0"), *grad_h0, 5e-3);
        assert_named("grad_c0", fetch("grad_c0"), *grad_c0, 5e-3);
        assert_named("grad_w_ih", fetch("grad_w_ih"), *grad_w_ih, 5e-3);
        assert_named("grad_w_hh", fetch("grad_w_hh"), *grad_w_hh, 5e-3);
        assert_named("grad_b_ih", fetch("grad_b_ih"), *grad_b_ih, 5e-3);
        assert_named("grad_b_hh", fetch("grad_b_hh"), *grad_b_hh, 5e-3);
        assert_named("grad_head_weight", fetch("grad_head_weight"), *grad_head, 5e-3);
    }

    if (const char *dump_path = std::getenv("FBAMTRAIN_LSTM_BWD_DUMP_PATH"))
    {
        std::map<std::string, std::shared_ptr<RealTensor>> dump{};
        auto to_cpu = [](const std::optional<std::shared_ptr<RealTensor>> &t) -> std::shared_ptr<RealTensor>
        {
            if (!t.has_value())
            {
                throw std::runtime_error("missing tensor for dump");
            }
            const auto &tensor = t.value();
            if (tensor->device().device_type == DEVICE_CPU.device_type)
            {
                return tensor;
            }
            return tensor->to(DEVICE_CPU, pi::tensorlib::GpuStreamDescriptors::Main);
        };
        dump.emplace("loss_sum", to_cpu(actual_loss_sum));
        dump.emplace("loss_mean", to_cpu(actual_loss_mean));
        dump.emplace("grad_x", to_cpu(grad_x));
        dump.emplace("grad_h0", to_cpu(grad_h0));
        dump.emplace("grad_c0", to_cpu(grad_c0));
        dump.emplace("grad_w_ih", to_cpu(grad_w_ih));
        dump.emplace("grad_w_hh", to_cpu(grad_w_hh));
        dump.emplace("grad_b_ih", to_cpu(grad_b_ih));
        dump.emplace("grad_b_hh", to_cpu(grad_b_hh));
        dump.emplace("grad_head_weight", to_cpu(grad_head));
        dump.emplace("y", to_cpu(executor.getOutput(lstm_result.output)));
        dump.emplace("x", x_host);
        dump.emplace("h0", h0_host);
        dump.emplace("c0", c0_host);
        dump.emplace("w_ih", w_ih_host);
        dump.emplace("w_hh", w_hh_host);
        dump.emplace("b_ih", b_ih_host);
        dump.emplace("b_hh", b_hh_host);
        dump.emplace("head_weight", head_weight_host);
        dump.emplace("targets", targets_host_float);
        safetensors::SaveToFile(dump_path, dump);
    }

    return 0;
}
