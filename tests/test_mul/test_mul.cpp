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
    passes.emplace_back(std::make_unique<TrailingBroadcastMulImplPass>());
    passes.emplace_back(std::make_unique<FillZerosImplPass>());
    passes.emplace_back(std::make_unique<FillConstantImplPass>());
    passes.emplace_back(std::make_unique<FillUniformImplPass>());
    pi::tensorlib::passes::Transform(execution_plan, passes);
}

constexpr int64_t ROWS = 2048;
constexpr int64_t COLS = 512;
constexpr int64_t NUMEL = ROWS * COLS;

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
    pi::tensorlib::TraceTensor scale = init_graph.createTensor({1, COLS}, dtype, device, main_stream_desc, false);
    pi::tensorlib::TraceTensor elem_lhs = init_graph.createTensor({NUMEL}, dtype, device, main_stream_desc, false);
    pi::tensorlib::TraceTensor elem_rhs = init_graph.createTensor({NUMEL}, dtype, device, main_stream_desc, false);
    activation.markRetained();
    scale.markRetained();
    elem_lhs.markRetained();
    elem_rhs.markRetained();

    uint32_t seed = 123;
    pi::tensorlib::FillUniform(init_graph, activation, -0.5f, 0.5f, seed++, main_stream_desc);
    pi::tensorlib::FillUniform(init_graph, scale, -0.5f, 0.5f, seed++, main_stream_desc);
    pi::tensorlib::FillUniform(init_graph, elem_lhs, -0.5f, 0.5f, seed++, main_stream_desc);
    pi::tensorlib::FillUniform(init_graph, elem_rhs, -0.5f, 0.5f, seed++, main_stream_desc);

    init_graph.finalize();

    pi::tensorlib::ExecutionPlan init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
    ApplyDefaultPasses(init_plan);
    pi::tensorlib::Executor init_executor{init_plan, execution_backend, 0};
    init_executor.execute(allocator_registry);

    const auto activation_real = init_executor.getOutput(activation);
    const auto scale_real = init_executor.getOutput(scale);
    const auto elem_lhs_real = init_executor.getOutput(elem_lhs);
    const auto elem_rhs_real = init_executor.getOutput(elem_rhs);
    if (!activation_real || !scale_real || !elem_lhs_real || !elem_rhs_real)
    {
        throw std::runtime_error("Failed to retrieve initialization tensors");
    }

    pi::tensorlib::OpGraph graph{{
                                     {.name = "activation", .tensor = activation},
                                     {.name = "scale", .tensor = scale},
                                     {.name = "elem_lhs", .tensor = elem_lhs},
                                     {.name = "elem_rhs", .tensor = elem_rhs},
                                 },
                                 {}};

    pi::tensorlib::InplaceMul(graph, activation, scale, main_stream_desc);
    pi::tensorlib::InplaceMul(graph, elem_lhs, elem_rhs, main_stream_desc);

    graph.finalize();

    pi::tensorlib::ExecutionPlan plan =
        pi::tensorlib::ExecutionPlan::FromGraph(graph,
                                                {
                                                    {.name = "activation", .tensor = *activation_real},
                                                    {.name = "scale", .tensor = *scale_real},
                                                    {.name = "elem_lhs", .tensor = *elem_lhs_real},
                                                    {.name = "elem_rhs", .tensor = *elem_rhs_real},
                                                },
                                                {});

    ApplyDefaultPasses(plan);

    pi::tensorlib::Executor executor{plan, execution_backend, 0};
    executor.execute(allocator_registry);

    const auto leading_broadcast_output = executor.getOutput(activation);
    const auto elementwise_output = executor.getOutput(elem_lhs);
    if (!leading_broadcast_output || !elementwise_output)
    {
        throw std::runtime_error("Expected output tensors not found");
    }

    const std::string reference_path = test_utils::ReferenceFileName(dtype);
    const auto expected_tensors = pi::tensorlib::safetensors::Load(reference_path);
    const auto leading_it = expected_tensors.find("leading_broadcast_output");
    if (leading_it == expected_tensors.end())
    {
        throw std::runtime_error("Expected leading broadcast output tensor not found in reference file: " +
                                 reference_path);
    }
    const auto elementwise_it = expected_tensors.find("elementwise_output");
    if (elementwise_it == expected_tensors.end())
    {
        throw std::runtime_error("Expected elementwise output tensor not found in reference file: " + reference_path);
    }

    const float tolerance = test_utils::SelectTolerance(dtype, TOLERANCE_BF16, TOLERANCE_FP16);
    pi::tensorlib::testing::AssertSimilar(leading_it->second, *leading_broadcast_output, tolerance);
    pi::tensorlib::testing::AssertSimilar(elementwise_it->second, *elementwise_output, tolerance);
}
