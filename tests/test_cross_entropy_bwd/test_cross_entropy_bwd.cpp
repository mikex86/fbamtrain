#include "testing.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    constexpr pi::tensorlib::Device DEVICE_CPU{pi::tensorlib::DeviceType::CPU, 0};
    constexpr pi::tensorlib::Device DEVICE_GPU{pi::tensorlib::DeviceType::GPU, 0};

    constexpr std::array TARGET_VALUES{1U, 0U, 5U, 3U, 2U, 6U};

    void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<CrossEntropyOnTargetsImplPass>());
        passes.emplace_back(std::make_unique<MeanImplPass>());
        passes.emplace_back(std::make_unique<ReduceSumPartialImplPass>());
        passes.emplace_back(std::make_unique<CastImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
        passes.emplace_back(std::make_unique<DivImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<FillUniformImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);
    }
} // namespace

int main()
{
    using namespace pi::tensorlib;

    ExecutionBackend &backend = ExecutionBackend::getInstance();
    auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();

    const auto ref_tensors = safetensors::Load("reference.safetensors");
    const auto fetch = [&ref_tensors](const char *name)
    {
        auto it = ref_tensors.find(name);
        if (it == ref_tensors.end())
        {
            throw std::runtime_error(std::string("missing tensor: ") + name);
        }
        return it->second;
    };

    const auto logits_fp16 = fetch("logits_fp16");
    const auto logits_bf16 = fetch("logits_bf16");
    const auto upstream_rows = fetch("upstream_rows");
    const auto upstream_scalar = fetch("upstream_scalar");
    const auto expected_rows_fp16 = fetch("grad_rows_fp16");
    const auto expected_rows_bf16 = fetch("grad_rows_bf16");
    const auto expected_scalar_fp16 = fetch("grad_scalar_fp16");
    const auto expected_scalar_bf16 = fetch("grad_scalar_bf16");

    auto make_targets = []
    {
        auto targets_cpu = RealTensor::Allocate({3, 2}, DataType::UINT32, DEVICE_CPU);
        auto *dst = static_cast<uint32_t *>(targets_cpu->dataptr());
        for (size_t i = 0; i < TARGET_VALUES.size(); ++i)
        {
            dst[i] = TARGET_VALUES[i];
        }
        return targets_cpu;
    };

    const auto run_case = [&](const std::shared_ptr<RealTensor> &logits_host,
                              const std::shared_ptr<RealTensor> &upstream_host, const bool reduce_over_rows,
                              const std::shared_ptr<RealTensor> &expected_grad)
    {
        const auto main_stream_desc = GpuStreamDescriptors::Main;
        std::vector<GraphInputDescriptor> inputs{};
        inputs.emplace_back(GraphInputDescriptor{
            .name = "logits",
            .tensor = TraceTensor::Create(logits_host->shape().dims(), logits_host->dtype(), DEVICE_GPU,
                                          main_stream_desc)});
        inputs.emplace_back(GraphInputDescriptor{.name = "targets",
                                                 .tensor = TraceTensor::Create({3, 2}, DataType::UINT32, DEVICE_GPU,
                                                                               main_stream_desc)});
        inputs.emplace_back(GraphInputDescriptor{
            .name = "upstream",
            .tensor = TraceTensor::Create(upstream_host->shape().dims(), DataType::FLOAT32, DEVICE_GPU,
                                          main_stream_desc)});

        for (auto &entry : inputs)
        {
            entry.tensor.markRetained();
        }

        OpGraph graph(inputs, {});
        TraceTensor grad = CrossEntropyOnTargetsBackward(graph, inputs[0].tensor, inputs[1].tensor, inputs[2].tensor,
                                                         Reduction::MEAN, main_stream_desc,
                                                         reduce_over_rows);
        graph.finalize();

        auto logits_gpu = logits_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        auto targets_cpu = make_targets();
        auto targets_gpu = targets_cpu->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        auto upstream_gpu = upstream_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                      {{.name = "logits", .tensor = logits_gpu},
                                                       {.name = "targets", .tensor = targets_gpu},
                                                       {.name = "upstream", .tensor = upstream_gpu}},
                                                      {});

        ApplyDefaultPasses(plan);
        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);

        const auto actual_grad = executor.getOutput(grad);
        if (!actual_grad)
        {
            throw std::runtime_error("failed to fetch grad output");
        }
        testing::AssertSimilar(expected_grad, *actual_grad, 5e-3);
    };

    run_case(logits_fp16, upstream_rows, /*reduce_over_rows=*/false, expected_rows_fp16);
    run_case(logits_fp16, upstream_scalar, /*reduce_over_rows=*/true, expected_scalar_fp16);
    run_case(logits_bf16, upstream_rows, /*reduce_over_rows=*/false, expected_rows_bf16);
    run_case(logits_bf16, upstream_scalar, /*reduce_over_rows=*/true, expected_scalar_bf16);

    return 0;
}
