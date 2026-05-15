#include "testing.h"

#include "../common/test_dtype_utils.h"
#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <linear.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    constexpr int64_t NUMEL = 2048 * 512;
    constexpr float INPUT_LOW = -0.5f;
    constexpr float INPUT_HIGH = 0.5f;
    constexpr uint32_t RNG_SEED = 123;
    constexpr float TOLERANCE_BF16 = 1e-3f;
    constexpr float TOLERANCE_FP16 = 3e-3f;

    void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &execution_plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<ActImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<FillUniformImplPass>());
        pi::tensorlib::passes::Transform(execution_plan, passes);
    }

    void RunActivationTest(const pi::tensorlib::DataType dtype)
    {
        using namespace pi::tensorlib;

        const float tolerance = test_utils::SelectTolerance(dtype, TOLERANCE_BF16, TOLERANCE_FP16);

        ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
        auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();

        const Device device{
            .device_type = DeviceType::GPU,
            .ordinal = 0,
        };
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        OpGraph init_graph{{}, {}};
        TraceTensor gelu_input = init_graph.createTensor({NUMEL}, dtype, device, main_stream_desc, false);
        TraceTensor relu_input = init_graph.createTensor({NUMEL}, dtype, device, main_stream_desc, false);
        TraceTensor gelu_inplace_input = init_graph.createTensor({NUMEL}, dtype, device, main_stream_desc, false);
        TraceTensor relu_inplace_input = init_graph.createTensor({NUMEL}, dtype, device, main_stream_desc, false);
        gelu_input.markRetained();
        relu_input.markRetained();
        gelu_inplace_input.markRetained();
        relu_inplace_input.markRetained();
        uint32_t seed = RNG_SEED;
        FillUniform(init_graph, gelu_input, INPUT_LOW, INPUT_HIGH, seed++, main_stream_desc);
        FillUniform(init_graph, relu_input, INPUT_LOW, INPUT_HIGH, seed++, main_stream_desc);
        FillUniform(init_graph, gelu_inplace_input, INPUT_LOW, INPUT_HIGH, seed++, main_stream_desc);
        FillUniform(init_graph, relu_inplace_input, INPUT_LOW, INPUT_HIGH, seed++, main_stream_desc);
        init_graph.finalize();

        ExecutionPlan init_plan = ExecutionPlan::FromGraph(init_graph, {}, {});
        ApplyDefaultPasses(init_plan);
        Executor init_executor{init_plan, execution_backend, 0};
        init_executor.execute(allocator_registry);

        const auto gelu_input_real = init_executor.getOutput(gelu_input);
        const auto relu_input_real = init_executor.getOutput(relu_input);
        const auto gelu_inplace_real = init_executor.getOutput(gelu_inplace_input);
        const auto relu_inplace_real = init_executor.getOutput(relu_inplace_input);
        if (!gelu_input_real || !relu_input_real || !gelu_inplace_real || !relu_inplace_real)
        {
            throw std::runtime_error("Failed to retrieve activation initialization tensors");
        }

        OpGraph graph{{
                          {.name = "gelu_input", .tensor = gelu_input},
                          {.name = "relu_input", .tensor = relu_input},
                          {.name = "gelu_inplace_input", .tensor = gelu_inplace_input},
                          {.name = "relu_inplace_input", .tensor = relu_inplace_input},
                      },
                      {}};

        TraceTensor gelu_output = Gelu(graph, gelu_input, main_stream_desc);
        TraceTensor relu_output = Relu(graph, relu_input, main_stream_desc);
        GeluInplace(graph, gelu_inplace_input, main_stream_desc);
        ReluInplace(graph, relu_inplace_input, main_stream_desc);
        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                      {
                                                          {.name = "gelu_input", .tensor = *gelu_input_real},
                                                          {.name = "relu_input", .tensor = *relu_input_real},
                                                          {.name = "gelu_inplace_input", .tensor = *gelu_inplace_real},
                                                          {.name = "relu_inplace_input", .tensor = *relu_inplace_real},
                                                      },
                                                      {});

        ApplyDefaultPasses(plan);

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto gelu_output_real = executor.getOutput(gelu_output);
        const auto relu_output_real = executor.getOutput(relu_output);
        const auto gelu_inplace_output_real = executor.getOutput(gelu_inplace_input);
        const auto relu_inplace_output_real = executor.getOutput(relu_inplace_input);
        if (!gelu_output_real || !relu_output_real || !gelu_inplace_output_real || !relu_inplace_output_real)
        {
            throw std::runtime_error("Expected activation output tensor not found");
        }

        std::string reference_path{"reference_"};
        reference_path.append(test_utils::GetDtypeSuffix(dtype));
        reference_path.append(".safetensors");

        const auto expected_tensors = safetensors::Load(reference_path);
        const auto expected_gelu = expected_tensors.find("gelu_output");
        const auto expected_relu = expected_tensors.find("relu_output");
        const auto expected_gelu_inplace = expected_tensors.find("gelu_inplace_output");
        const auto expected_relu_inplace = expected_tensors.find("relu_inplace_output");

        if (expected_gelu == expected_tensors.end() || expected_relu == expected_tensors.end() ||
            expected_gelu_inplace == expected_tensors.end() || expected_relu_inplace == expected_tensors.end())
        {
            throw std::runtime_error("Expected activation outputs not found in reference file: " + reference_path);
        }

        testing::AssertSimilar(expected_gelu->second, *gelu_output_real, tolerance);
        testing::AssertSimilar(expected_relu->second, *relu_output_real, tolerance);
        testing::AssertSimilar(expected_gelu_inplace->second, *gelu_inplace_output_real, tolerance);
        testing::AssertSimilar(expected_relu_inplace->second, *relu_inplace_output_real, tolerance);
    }
} // namespace

int main()
{
    const auto dtype = test_utils::GetTestDtype();
    RunActivationTest(dtype);
}
