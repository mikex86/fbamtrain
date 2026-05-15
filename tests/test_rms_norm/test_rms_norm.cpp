#include "testing.h"

#include <allocator.h>

#include "../common/test_dtype_utils.h"
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>

#include <string>

static void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &execution_plan)
{
    std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
    passes.emplace_back(std::make_unique<RmsNormImplPass>());
    passes.emplace_back(std::make_unique<FillZerosImplPass>());
    passes.emplace_back(std::make_unique<FillConstantImplPass>());
    passes.emplace_back(std::make_unique<FillUniformImplPass>());
    pi::tensorlib::passes::Transform(execution_plan, passes);
}

#define B 16
#define S 256
#define C 256
#define E 1e-5f
#define SEED 2024

int main()
{
    constexpr float TOLERANCE_BF16 = 1e-2f;
    constexpr float TOLERANCE_FP16 = 2e-2f;

    pi::tensorlib::ExecutionBackend &execution_backend = pi::tensorlib::ExecutionBackend::getInstance();
    auto &allocator_registry = pi::tensorlib::allocator::DefaultAllocatorRegistry::instance();

    constexpr pi::tensorlib::Device device{
        .device_type = pi::tensorlib::DeviceType::GPU,
        .ordinal = 0,
    };
    const auto main_stream_desc = pi::tensorlib::GpuStreamDescriptors::Main;

    const auto dtype = test_utils::GetTestDtype();

    pi::tensorlib::OpGraph init_graph{{}, {}};

    pi::tensorlib::TraceTensor input = init_graph.createTensor({B, S, C}, dtype, device, main_stream_desc, false);
    pi::tensorlib::TraceTensor weight = init_graph.createTensor({C}, dtype, device, main_stream_desc, false);
    input.markRetained();
    weight.markRetained();

    pi::tensorlib::FillUniform(init_graph, input, -0.5f, 0.5f, SEED, main_stream_desc);
    pi::tensorlib::FillUniform(init_graph, weight, 0.9f, 1.1f, SEED + 1, main_stream_desc);

    init_graph.finalize();

    pi::tensorlib::ExecutionPlan init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
    ApplyDefaultPasses(init_plan);
    pi::tensorlib::Executor init_executor{init_plan, execution_backend, 0};
    init_executor.execute(allocator_registry);

    pi::tensorlib::OpGraph graph{{
                                     {.name = "input", .tensor = input},
                                     {.name = "weight", .tensor = weight},
                                 },
                                 {}};

    pi::tensorlib::TraceTensor output = pi::tensorlib::RmsNormFwd(graph, input, weight, E, main_stream_desc);
    graph.finalize();

    auto input_real = *init_executor.getOutput(input);
    auto weight_real = *init_executor.getOutput(weight);

    pi::tensorlib::ExecutionPlan plan =
        pi::tensorlib::ExecutionPlan::FromGraph(graph,
                                                {
                                                    {.name = "input", .tensor = input_real},
                                                    {.name = "weight", .tensor = weight_real},
                                                },
                                                {});

    ApplyDefaultPasses(plan);
    pi::tensorlib::Executor executor{plan, execution_backend, 0};
    executor.execute(allocator_registry);

    auto actual_output = *executor.getOutput(output);

    const std::string reference_path = test_utils::ReferenceFileName(dtype);
    const auto expected_tensors = pi::tensorlib::safetensors::Load(reference_path);
    const auto expected_output_it = expected_tensors.find("output");
    if (expected_output_it == expected_tensors.end())
    {
        throw std::runtime_error("Expected output tensor not found in reference file: " + reference_path);
    }
    const auto &expected_output = expected_output_it->second;

    const float tolerance = test_utils::SelectTolerance(dtype, TOLERANCE_BF16, TOLERANCE_FP16);
    pi::tensorlib::testing::AssertSimilar(expected_output, actual_output, tolerance);

    init_executor.execute(allocator_registry);
    input_real = *init_executor.getOutput(input);
    weight_real = *init_executor.getOutput(weight);

    pi::tensorlib::OpGraph inplace_graph{{
                                             {.name = "input", .tensor = input},
                                             {.name = "weight", .tensor = weight},
                                         },
                                         {}};

    pi::tensorlib::RmsNormFwdInplace(inplace_graph, input, weight, E, main_stream_desc);
    inplace_graph.finalize();

    pi::tensorlib::ExecutionPlan inplace_plan =
        pi::tensorlib::ExecutionPlan::FromGraph(inplace_graph,
                                                {
                                                    {.name = "input", .tensor = input_real},
                                                    {.name = "weight", .tensor = weight_real},
                                                },
                                                {});

    ApplyDefaultPasses(inplace_plan);
    pi::tensorlib::Executor inplace_executor{inplace_plan, execution_backend, 0};
    inplace_executor.execute(allocator_registry);

    auto actual_inplace_output = *inplace_executor.getOutput(input);

    pi::tensorlib::testing::AssertSimilar(expected_output, actual_inplace_output, tolerance);
}
