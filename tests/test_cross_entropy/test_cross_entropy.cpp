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
#include <vector>

namespace
{
    constexpr uint32_t SEQ = 3;
    constexpr uint32_t BATCH = 2;
    constexpr uint32_t VOCAB = 8;

    constexpr std::array<uint32_t, SEQ * BATCH> TARGET_VALUES{
        1U, 0U, 5U, 3U, 2U, 6U,
    };

    void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &execution_plan)
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
        pi::tensorlib::passes::Transform(execution_plan, passes);
    }

} // namespace

static std::shared_ptr<pi::tensorlib::RealTensor>
FetchTensor(const std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &tensors, const char *name)
{
    auto it = tensors.find(name);
    if (it == tensors.end())
    {
        throw std::runtime_error(std::string("Missing tensor in reference.safetensors: ") + name);
    }
    return it->second;
}

int main()
{
    using namespace pi::tensorlib;

    constexpr Device DEVICE_CPU{DeviceType::CPU, 0};
    constexpr Device DEVICE_GPU{DeviceType::GPU, 0};

    ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
    auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();

    const auto reference_tensors = safetensors::Load("reference.safetensors");

    const auto logits_fp16 = FetchTensor(reference_tensors, "logits_fp16");
    const auto logits_bf16 = FetchTensor(reference_tensors, "logits_bf16");
    const auto expected_loss_fp16 = FetchTensor(reference_tensors, "loss_fp16");
    const auto expected_loss_bf16 = FetchTensor(reference_tensors, "loss_bf16");
    const auto expected_sum_fp32 = FetchTensor(reference_tensors, "sum_fp32");
    const auto expected_sum_bf32 = FetchTensor(reference_tensors, "sum_bf32");

    auto targets_cpu = RealTensor::Allocate({SEQ, BATCH}, DataType::UINT32, DEVICE_CPU);
    auto *targets_ptr = static_cast<uint32_t *>(targets_cpu->dataptr());
    for (size_t i = 0; i < TARGET_VALUES.size(); ++i)
    {
        targets_ptr[i] = TARGET_VALUES[i];
    }
    auto targets_gpu = targets_cpu->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

    const auto run_case = [&](const std::shared_ptr<RealTensor> &logits_host,
                              const std::shared_ptr<RealTensor> &expected_loss,
                              const std::shared_ptr<RealTensor> &expected_sum)
    {
        const auto main_stream_desc = GpuStreamDescriptors::Main;
        TraceTensor logits_trace =
            TraceTensor::Create(logits_host->shape().dims(), logits_host->dtype(), DEVICE_GPU, main_stream_desc);
        TraceTensor targets_trace =
            TraceTensor::Create(targets_gpu->shape().dims(), targets_gpu->dtype(), DEVICE_GPU, main_stream_desc);
        logits_trace.markRetained();
        targets_trace.markRetained();

        OpGraph graph{{{.name = "logits", .tensor = logits_trace}, {.name = "targets", .tensor = targets_trace}}, {}};
        TraceTensor loss =
            CrossEntropyOnTargets(graph, logits_trace, targets_trace, Reduction::MEAN, main_stream_desc);
        TraceTensor ce_rows =
            CrossEntropyOnTargets(graph, logits_trace, targets_trace, Reduction::ADD, main_stream_desc);
        TraceTensor ce_sum_scalar =
            CrossEntropyOnTargets(graph, logits_trace, targets_trace, Reduction::ADD, main_stream_desc,
                                  /*reduce_over_rows=*/true);
        TraceTensor ce_mean_scalar =
            CrossEntropyOnTargets(graph, logits_trace, targets_trace, Reduction::MEAN, main_stream_desc,
                                  /*reduce_over_rows=*/true);
        auto reduce_to_scalar = [&](TraceTensor current)
        {
            while (current.shape().numel() > 1)
            {
                current = ReduceSumPartial(graph, current, main_stream_desc, 128);
            }
            return current;
        };
        TraceTensor sum_accum = graph.createTensor({1}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        FillZeros(graph, sum_accum, main_stream_desc);
        TraceTensor sum_once = reduce_to_scalar(ce_rows);
        InplaceAdd(graph, sum_accum, sum_once, main_stream_desc);
        TraceTensor sum_second = reduce_to_scalar(ce_rows);
        InplaceAdd(graph, sum_accum, sum_second, main_stream_desc);
        graph.finalize();

        auto logits_gpu = logits_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                      {
                                                          {.name = "logits", .tensor = logits_gpu},
                                                          {.name = "targets", .tensor = targets_gpu},
                                                      },
                                                      {});

        ApplyDefaultPasses(plan);
        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto actual_loss = *executor.getOutput(loss);
        const auto actual_sum = *executor.getOutput(sum_accum);
        const auto actual_sum_scalar = *executor.getOutput(ce_sum_scalar);
        const auto actual_mean_scalar = *executor.getOutput(ce_mean_scalar);

        pi::tensorlib::testing::AssertSimilar(expected_loss, actual_loss, 5e-3);

        auto expected_accum = RealTensor::Allocate({1}, DataType::FLOAT32, expected_sum->device());
        const float expected_sum_value = static_cast<const float *>(expected_sum->dataptr())[0];
        static_cast<float *>(expected_accum->dataptr())[0] = expected_sum_value * 2.0f;
        pi::tensorlib::testing::AssertSimilar(expected_accum, actual_sum, 5e-3);

        pi::tensorlib::testing::AssertSimilar(expected_sum, actual_sum_scalar, 5e-3);

        auto expected_mean = RealTensor::Allocate({1}, DataType::FLOAT32, expected_sum->device());
        static_cast<float *>(expected_mean->dataptr())[0] = expected_sum_value / static_cast<float>(SEQ * BATCH);
        pi::tensorlib::testing::AssertSimilar(expected_mean, actual_mean_scalar, 5e-3);
    };

    run_case(logits_fp16, expected_loss_fp16, expected_sum_fp32);
    run_case(logits_bf16, expected_loss_bf16, expected_sum_bf32);

    return 0;
}
