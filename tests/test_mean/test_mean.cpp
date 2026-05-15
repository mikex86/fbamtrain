#include "testing.h"

#include <allocator.h>

#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>

static void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &execution_plan)
{
    std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
    passes.emplace_back(std::make_unique<MeanImplPass>());
    passes.emplace_back(std::make_unique<FillZerosImplPass>());
    passes.emplace_back(std::make_unique<FillConstantImplPass>());
    passes.emplace_back(std::make_unique<FillUniformImplPass>());
    pi::tensorlib::passes::Transform(execution_plan, passes);
}

constexpr uint32_t BATCH = 4;
constexpr uint32_t ROWS = 5;
constexpr uint32_t COLS = 6;
constexpr float FILL_LOW = -0.75f;
constexpr float FILL_HIGH = 0.75f;
constexpr uint32_t SEED = 1234u;

int main()
{
    pi::tensorlib::ExecutionBackend &execution_backend = pi::tensorlib::ExecutionBackend::getInstance();
    auto &allocator_registry = pi::tensorlib::allocator::DefaultAllocatorRegistry::instance();

    constexpr pi::tensorlib::Device device{
        .device_type = pi::tensorlib::DeviceType::GPU,
        .ordinal = 0,
    };
    const auto main_stream_desc = pi::tensorlib::GpuStreamDescriptors::Main;

    pi::tensorlib::OpGraph init_graph{{}, {}};

    pi::tensorlib::TraceTensor input =
        init_graph.createTensor({BATCH, ROWS, COLS}, pi::tensorlib::DataType::BFLOAT16, device, main_stream_desc, false);
    input.markRetained();

    pi::tensorlib::FillUniform(init_graph, input, FILL_LOW, FILL_HIGH, SEED, main_stream_desc);

    init_graph.finalize();

    pi::tensorlib::ExecutionPlan init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
    ApplyDefaultPasses(init_plan);
    pi::tensorlib::Executor init_executor{init_plan, execution_backend, 0};
    init_executor.execute(allocator_registry);

    pi::tensorlib::OpGraph graph{{
                                     {.name = "input", .tensor = input},
                                 },
                                 {}};

    pi::tensorlib::TraceTensor mean_keep = pi::tensorlib::Mean(graph, input, -1, true, main_stream_desc);
    pi::tensorlib::TraceTensor mean_no_keep = pi::tensorlib::Mean(graph, input, -1, false, main_stream_desc);
    pi::tensorlib::TraceTensor mean_time = pi::tensorlib::Mean(graph, input, 1, false, main_stream_desc);

    graph.finalize();

    std::shared_ptr<pi::tensorlib::RealTensor> input_value = *init_executor.getOutput(input);

    pi::tensorlib::ExecutionPlan plan =
        pi::tensorlib::ExecutionPlan::FromGraph(graph,
                                                {
                                                    {.name = "input", .tensor = input_value},
                                                },
                                                {});

    ApplyDefaultPasses(plan);
    pi::tensorlib::Executor executor{plan, execution_backend, 0};
    executor.execute(allocator_registry);

    std::shared_ptr<pi::tensorlib::RealTensor> actual_keep = *executor.getOutput(mean_keep);
    std::shared_ptr<pi::tensorlib::RealTensor> actual_no_keep = *executor.getOutput(mean_no_keep);
    std::shared_ptr<pi::tensorlib::RealTensor> actual_time = *executor.getOutput(mean_time);

    const auto expected_tensors = pi::tensorlib::safetensors::Load("reference.safetensors");

    const auto expected_keep_it = expected_tensors.find("mean_keep");
    if (expected_keep_it == expected_tensors.end())
    {
        throw std::runtime_error("Expected tensor 'mean_keep' not found in reference.safetensors");
    }
    const auto expected_no_keep_it = expected_tensors.find("mean_no_keep");
    if (expected_no_keep_it == expected_tensors.end())
    {
        throw std::runtime_error("Expected tensor 'mean_no_keep' not found in reference.safetensors");
    }
    const auto expected_time_it = expected_tensors.find("mean_bt");
    if (expected_time_it == expected_tensors.end())
    {
        throw std::runtime_error("Expected tensor 'mean_bt' not found in reference.safetensors");
    }

    pi::tensorlib::testing::AssertSimilar(expected_keep_it->second, actual_keep, 2e-3f);
    pi::tensorlib::testing::AssertSimilar(expected_no_keep_it->second, actual_no_keep, 2e-3f);
    pi::tensorlib::testing::AssertSimilar(expected_time_it->second, actual_time, 2e-3f);

    return 0;
}
