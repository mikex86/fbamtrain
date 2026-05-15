#include "testing.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

namespace
{
    constexpr pi::tensorlib::Device DEVICE_CPU{pi::tensorlib::DeviceType::CPU, 0};
    constexpr pi::tensorlib::Device DEVICE_GPU{pi::tensorlib::DeviceType::GPU, 0};

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
        pi::tensorlib::passes::Transform(plan, passes);
    }
} // namespace

int main()
{
    using namespace pi::tensorlib;

    ExecutionBackend &backend = ExecutionBackend::getInstance();
    // Pinned host allocations in this test require an active CUDA context.
    (void)ExecutionBackend::GetStreamBundle(DEVICE_GPU);
    const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();
    const auto main_stream_desc = GpuStreamDescriptors::Main;

    const auto ref_tensors = safetensors::Load("reference.safetensors", true);
    const auto fetch = [&ref_tensors](const char *name)
    {
        auto it = ref_tensors.find(name);
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
    const auto dy_host = fetch("dy");

    const auto expected_grad_x = fetch("grad_x");
    const auto expected_grad_h0 = fetch("grad_h0");
    const auto expected_grad_c0 = fetch("grad_c0");
    const auto expected_grad_w_ih = fetch("grad_w_ih");
    const auto expected_grad_w_hh = fetch("grad_w_hh");
    const auto expected_grad_b_ih = fetch("grad_b_ih");
    const auto expected_grad_b_hh = fetch("grad_b_hh");

    constexpr double kTolerance = 4e-2;
    const auto assert_named = [&](const char *name, const std::shared_ptr<RealTensor> &expected,
                                  const std::shared_ptr<RealTensor> &actual)
    {
        try
        {
            testing::AssertSimilar(expected, actual, kTolerance);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error(std::string(name) + " mismatch: " + e.what());
        }
    };

    auto h0_gpu = h0_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
    auto c0_gpu = c0_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
    auto w_ih_gpu = w_ih_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
    auto w_hh_gpu = w_hh_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
    auto b_ih_gpu = b_ih_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
    auto b_hh_gpu = b_hh_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

    auto dy_gpu = dy_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

    const auto batch = x_host->shape()[1];
    const auto hidden_size = h0_host->shape()[0];
    auto make_zero_state = [&](uint64_t rows, uint64_t cols)
    {
        auto zero_host = RealTensor::Allocate({rows, cols}, DataType::FLOAT32, DEVICE_CPU);
        std::memset(zero_host->dataptr(), 0, zero_host->shape().numel() * sizeof(float));
        return zero_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
    };

    auto dh_n_gpu = make_zero_state(batch, hidden_size);
    auto dc_n_gpu = make_zero_state(batch, hidden_size);

    std::vector<GraphInputDescriptor> inputs{};
    inputs.emplace_back(GraphInputDescriptor{
        .name = "x", .tensor = TraceTensor::Create(x_host->shape().dims(), x_host->dtype(), DEVICE_CPU,
                                                   main_stream_desc)});
    inputs.emplace_back(GraphInputDescriptor{
        .name = "h0", .tensor = TraceTensor::Create(h0_host->shape().dims(), DataType::FLOAT32, DEVICE_GPU,
                                                    main_stream_desc)});
    inputs.emplace_back(GraphInputDescriptor{
        .name = "c0", .tensor = TraceTensor::Create(c0_host->shape().dims(), DataType::FLOAT32, DEVICE_GPU,
                                                    main_stream_desc)});
    inputs.emplace_back(GraphInputDescriptor{
        .name = "dy", .tensor = TraceTensor::Create(dy_host->shape().dims(), DataType::FLOAT32, DEVICE_CPU,
                                                    main_stream_desc)});
    inputs.emplace_back(GraphInputDescriptor{
        .name = "dh_n", .tensor = TraceTensor::Create({batch, hidden_size}, DataType::FLOAT32, DEVICE_GPU,
                                                      main_stream_desc)});
    inputs.emplace_back(GraphInputDescriptor{
        .name = "dc_n", .tensor = TraceTensor::Create({batch, hidden_size}, DataType::FLOAT32, DEVICE_GPU,
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

    for (auto &entry : inputs)
    {
        entry.tensor.markRetained();
    }
    for (auto &entry : params)
    {
        entry.tensor.markRetained();
    }

    const std::vector<std::pair<int, int>> scenarios = {
        {1, 1}, {1, 2}, {1, 4}, {1, 8}, // baseline streaming sizes
        {2, 1}, {2, 2}, {2, 4},         // interval equals chunk or greater
        {3, 1}, {3, 2}, {3, 4},         // interval > chunk cases
        {4, 1}, {4, 2}, {4, 4},         // interval aligned with chunk divisor
        {5, 1}, {5, 2}, {5, 4},         // interval larger than chunk
        {8, 1}, {8, 2}, {8, 4}, {8, 8}  // interval spans the full sequence
    };

    for (const auto [recompute_interval, streaming_chunk_size] : scenarios)
    {
        OpGraph graph(inputs, params);

        auto fwd = StreamingLstmFwd(graph, inputs[0].tensor, inputs[1].tensor, inputs[2].tensor, params[0].tensor,
                                    params[1].tensor, params[2].tensor, params[3].tensor, DataType::FLOAT32,
                                    recompute_interval, streaming_chunk_size, main_stream_desc);

        auto bwd = StreamingLstmBwd(graph, inputs[0].tensor, inputs[1].tensor, inputs[2].tensor, params[0].tensor,
                                    params[1].tensor, params[2].tensor, params[3].tensor, fwd.output, fwd.h_cache,
                                    fwd.c_cache, inputs[3].tensor, inputs[4].tensor, inputs[5].tensor,
                                    recompute_interval, streaming_chunk_size, std::nullopt, main_stream_desc);

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                      {{.name = "x", .tensor = x_host},
                                                       {.name = "h0", .tensor = h0_gpu},
                                                       {.name = "c0", .tensor = c0_gpu},
                                                       {.name = "dy", .tensor = dy_host},
                                                       {.name = "dh_n", .tensor = dh_n_gpu},
                                                       {.name = "dc_n", .tensor = dc_n_gpu}},
                                                      {{.name = "w_ih", .tensor = w_ih_gpu},
                                                       {.name = "w_hh", .tensor = w_hh_gpu},
                                                       {.name = "b_ih", .tensor = b_ih_gpu},
                                                       {.name = "b_hh", .tensor = b_hh_gpu}});

        ApplyDefaultPasses(plan);

        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);

        const auto grad_x = executor.getOutput(bwd.grad_x);
        const auto grad_h0 = executor.getOutput(bwd.grad_h0);
        const auto grad_c0 = executor.getOutput(bwd.grad_c0);
        const auto grad_w_ih = executor.getOutput(bwd.grad_w_ih);
        const auto grad_w_hh = executor.getOutput(bwd.grad_w_hh);
        const auto grad_b_ih = executor.getOutput(bwd.grad_b_ih);
        const auto grad_b_hh = executor.getOutput(bwd.grad_b_hh);

        if (!grad_x || !grad_h0 || !grad_c0 || !grad_w_ih || !grad_w_hh || !grad_b_ih || !grad_b_hh)
        {
            throw std::runtime_error("missing gradient outputs");
        }

        const bool skip_asserts = std::getenv("FBAMTRAIN_LSTM_BWD_SKIP_ASSERT") != nullptr;
        if (!skip_asserts)
        {
            try
            {
                assert_named("grad_x", expected_grad_x, *grad_x);
                assert_named("grad_h0", expected_grad_h0, *grad_h0);
                assert_named("grad_c0", expected_grad_c0, *grad_c0);
                assert_named("grad_w_ih", expected_grad_w_ih, *grad_w_ih);
                assert_named("grad_w_hh", expected_grad_w_hh, *grad_w_hh);
                assert_named("grad_b_ih", expected_grad_b_ih, *grad_b_ih);
                assert_named("grad_b_hh", expected_grad_b_hh, *grad_b_hh);
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error("scenario recompute_interval=" + std::to_string(recompute_interval) +
                                         ", streaming_chunk_size=" + std::to_string(streaming_chunk_size) +
                                         " failed: " + e.what());
            }
        }
    }

    return 0;
}
