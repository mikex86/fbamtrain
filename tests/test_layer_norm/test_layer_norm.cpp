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
    passes.emplace_back(std::make_unique<LayerNormImplPass>());
    passes.emplace_back(std::make_unique<FillZerosImplPass>());
    passes.emplace_back(std::make_unique<FillConstantImplPass>());
    passes.emplace_back(std::make_unique<FillUniformImplPass>());
    pi::tensorlib::passes::Transform(execution_plan, passes);
}

#define B 16
#define S 256
#define H 256
#define E 1e-5f

int main()
{
    constexpr float TOLERANCE_BF16 = 2e-3f;
    constexpr float TOLERANCE_FP16 = 5e-3f;

    pi::tensorlib::ExecutionBackend &execution_backend = pi::tensorlib::ExecutionBackend::getInstance();
    auto &allocator_registry = pi::tensorlib::allocator::DefaultAllocatorRegistry::instance();

    const pi::tensorlib::Device device{
        .device_type = pi::tensorlib::DeviceType::GPU,
        .ordinal = 0,
    };
    const auto main_stream_desc = pi::tensorlib::GpuStreamDescriptors::Main;

    const auto dtype = test_utils::GetTestDtype();

    pi::tensorlib::OpGraph init_graph{{}, {}};

    pi::tensorlib::TraceTensor input = init_graph.createTensor({B, S, H}, dtype, device, main_stream_desc, false);
    pi::tensorlib::TraceTensor weight = init_graph.createTensor({H}, dtype, device, main_stream_desc, false);
    pi::tensorlib::TraceTensor bias = init_graph.createTensor({H}, dtype, device, main_stream_desc, false);
    input.markRetained();
    weight.markRetained();
    bias.markRetained();

    uint32_t seed = 42;
    pi::tensorlib::FillUniform(init_graph, input, -0.5f, 0.5f, seed++, main_stream_desc);
    pi::tensorlib::FillUniform(init_graph, weight, -0.5f, 0.5f, seed++, main_stream_desc);
    pi::tensorlib::FillUniform(init_graph, bias, -0.5f, 0.5f, seed++, main_stream_desc);

    init_graph.finalize();

    pi::tensorlib::ExecutionPlan init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
    ApplyDefaultPasses(init_plan);
    pi::tensorlib::Executor init_executor{init_plan, execution_backend, 0};
    init_executor.execute(allocator_registry);

    pi::tensorlib::OpGraph graph{{
                                     {.name = "input", .tensor = input},
                                     {.name = "weight", .tensor = weight},
                                     {.name = "bias", .tensor = bias},
                                 },
                                 {}};

    pi::tensorlib::TraceTensor output =
        pi::tensorlib::LayerNormFwd(graph, input, weight, bias, E, main_stream_desc);
    graph.finalize();

    std::shared_ptr<pi::tensorlib::RealTensor> input_real = *init_executor.getOutput(input);
    std::shared_ptr<pi::tensorlib::RealTensor> weight_real = *init_executor.getOutput(weight);
    std::shared_ptr<pi::tensorlib::RealTensor> bias_real = *init_executor.getOutput(bias);

    pi::tensorlib::ExecutionPlan plan =
        pi::tensorlib::ExecutionPlan::FromGraph(graph,
                                                {
                                                    {.name = "input", .tensor = input_real},
                                                    {.name = "weight", .tensor = weight_real},
                                                    {.name = "bias", .tensor = bias_real},
                                                },
                                                {});

    ApplyDefaultPasses(plan);
    pi::tensorlib::Executor executor{plan, execution_backend, 0};
    executor.execute(allocator_registry);

    std::shared_ptr<pi::tensorlib::RealTensor> actual_output = *executor.getOutput(output);

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
}
