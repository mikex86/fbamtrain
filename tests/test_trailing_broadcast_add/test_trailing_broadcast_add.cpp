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
    passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
    passes.emplace_back(std::make_unique<FillZerosImplPass>());
    passes.emplace_back(std::make_unique<FillConstantImplPass>());
    passes.emplace_back(std::make_unique<FillUniformImplPass>());
    pi::tensorlib::passes::Transform(execution_plan, passes);
}

#define ROWS 2048
#define COLS 512

int main()
{
    constexpr float TOLERANCE_BF16 = 1e-3f;
    constexpr float TOLERANCE_FP16 = 2e-3f;

    pi::tensorlib::ExecutionBackend &execution_backend = pi::tensorlib::ExecutionBackend::getInstance();
    auto &allocator_registry = pi::tensorlib::allocator::DefaultAllocatorRegistry::instance();

    constexpr pi::tensorlib::Device device{
        .device_type = pi::tensorlib::DeviceType::GPU,
        .ordinal = 0,
    };
    const auto main_stream_desc = pi::tensorlib::GpuStreamDescriptors::Main;

    const auto dtype = test_utils::GetTestDtype();

    pi::tensorlib::OpGraph init_graph{{}, {}};

    pi::tensorlib::TraceTensor activation =
        init_graph.createTensor({ROWS, COLS}, dtype, device, main_stream_desc, false);
    pi::tensorlib::TraceTensor bias = init_graph.createTensor({1, COLS}, dtype, device, main_stream_desc, false);
    activation.markRetained();
    bias.markRetained();

    uint32_t seed = 123;
    pi::tensorlib::FillUniform(init_graph, activation, -0.5f, 0.5f, seed++, main_stream_desc);
    pi::tensorlib::FillUniform(init_graph, bias, -0.5f, 0.5f, seed++, main_stream_desc);

    init_graph.finalize();

    pi::tensorlib::ExecutionPlan init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
    ApplyDefaultPasses(init_plan);
    pi::tensorlib::Executor init_executor{init_plan, execution_backend, 0};
    init_executor.execute(allocator_registry);

    pi::tensorlib::OpGraph graph{{
                                     {.name = "activation", .tensor = activation},
                                     {.name = "bias", .tensor = bias},
                                 },
                                 {}};

    // Make the leading broadcast dimension explicit to hit the dedicated pass.
    pi::tensorlib::InplaceAdd(graph, activation, bias, main_stream_desc);

    graph.finalize();

    const auto activation_real = init_executor.getOutput(activation);
    const auto bias_real = init_executor.getOutput(bias);
    if (!activation_real || !bias_real)
    {
        throw std::runtime_error("Failed to retrieve initialization tensors");
    }

    pi::tensorlib::ExecutionPlan plan =
        pi::tensorlib::ExecutionPlan::FromGraph(graph,
                                                {
                                                    {.name = "activation", .tensor = *activation_real},
                                                    {.name = "bias", .tensor = *bias_real},
                                                },
                                                {});

    ApplyDefaultPasses(plan);

    pi::tensorlib::Executor executor{plan, execution_backend, 0};
    executor.execute(allocator_registry);

    const auto actual_output = executor.getOutput(activation);
    if (!actual_output)
    {
        throw std::runtime_error("Expected activation output tensor not found");
    }

    const std::string reference_path = test_utils::ReferenceFileName(dtype);
    const auto expected_tensors = pi::tensorlib::safetensors::Load(reference_path);
    const auto expected_output_it = expected_tensors.find("output");
    if (expected_output_it == expected_tensors.end())
    {
        throw std::runtime_error("Expected output tensor not found in reference file: " + reference_path);
    }

    const float tolerance = test_utils::SelectTolerance(dtype, TOLERANCE_BF16, TOLERANCE_FP16);
    pi::tensorlib::testing::AssertSimilar(expected_output_it->second, *actual_output, tolerance);
}
