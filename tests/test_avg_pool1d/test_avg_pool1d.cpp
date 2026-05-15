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
    passes.emplace_back(std::make_unique<AvgPool1dImplPass>());
    passes.emplace_back(std::make_unique<FillZerosImplPass>());
    passes.emplace_back(std::make_unique<FillConstantImplPass>());
    passes.emplace_back(std::make_unique<FillUniformImplPass>());
    pi::tensorlib::passes::Transform(execution_plan, passes);
}

#define BATCH 8
#define CHANNELS 16
#define LENGTH 128
#define KERNEL_SIZE 3
#define STRIDE 2
#define SEED 1337

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
        init_graph.createTensor({BATCH, CHANNELS, LENGTH}, pi::tensorlib::DataType::BFLOAT16, device,
                                main_stream_desc, false);
    input.markRetained();

    pi::tensorlib::FillUniform(init_graph, input, -1.0f, 1.0f, SEED, main_stream_desc);
    init_graph.finalize();

    pi::tensorlib::ExecutionPlan init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
    ApplyDefaultPasses(init_plan);
    pi::tensorlib::Executor init_executor{init_plan, execution_backend, 0};
    init_executor.execute(allocator_registry);

    pi::tensorlib::OpGraph graph{{
                                     {.name = "input", .tensor = input},
                                 },
                                 {}};

    pi::tensorlib::TraceTensor strided = input.transpose(graph, {0, 2, 1});
    pi::tensorlib::TraceTensor pooled =
        pi::tensorlib::AvgPool1d(graph, strided, KERNEL_SIZE, STRIDE, main_stream_desc, 1);
    pi::tensorlib::TraceTensor output = pooled.transpose(graph, {0, 2, 1});
    graph.finalize();

    std::shared_ptr<pi::tensorlib::RealTensor> input_real = *init_executor.getOutput(input);

    pi::tensorlib::ExecutionPlan plan =
        pi::tensorlib::ExecutionPlan::FromGraph(graph,
                                                {
                                                    {.name = "input", .tensor = input_real},
                                                },
                                                {});

    ApplyDefaultPasses(plan);
    pi::tensorlib::Executor executor{plan, execution_backend, 0};
    executor.execute(allocator_registry);

    std::shared_ptr<pi::tensorlib::RealTensor> actual_output = *executor.getOutput(output);

    const auto expected_tensors = pi::tensorlib::safetensors::Load("reference.safetensors");
    const auto expected_output_it = expected_tensors.find("output");
    if (expected_output_it == expected_tensors.end())
    {
        throw std::runtime_error("Expected output tensor not found in reference.safetensors");
    }
    const auto &expected_output = expected_output_it->second;

    pi::tensorlib::testing::AssertSimilar(expected_output, actual_output, 2e-3f);
}
