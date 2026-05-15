#include "testing.h"

#include <allocator.h>

#include <iostream>

#include <attention.h>
#include <cmath>
#include <execution_backend.h>
#include <executor.h>
#include <string>
#include <tensorlib.h>

#include "../common/test_dtype_utils.h"
#include <passes.h>
#include <safe_tensors.h>

static void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &execution_plan)
{
    std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
    passes.emplace_back(std::make_unique<MhaAttentionImplPass>());
    passes.emplace_back(std::make_unique<MatmulFusePass>());
    passes.emplace_back(std::make_unique<FillZerosImplPass>());
    passes.emplace_back(std::make_unique<FillConstantImplPass>());
    passes.emplace_back(std::make_unique<FillUniformImplPass>());
    passes.emplace_back(std::make_unique<ContiguousImplPass>());
    pi::tensorlib::passes::Transform(execution_plan, passes);
}

#define B 32
#define H 8
#define T 4096
#define HS 128
#define EMBED (H * HS)

namespace
{
    constexpr float TOLERANCE_BF16 = 1e-3f;
    constexpr float TOLERANCE_FP16 = 3e-3f;

    void RunMhaReshapeTest(const pi::tensorlib::DataType dtype)
    {
        pi::tensorlib::ExecutionBackend &execution_backend = pi::tensorlib::ExecutionBackend::getInstance();
        auto &allocator_registry = pi::tensorlib::allocator::DefaultAllocatorRegistry::instance();

        const pi::tensorlib::Device device{
            .device_type = pi::tensorlib::DeviceType::GPU,
            .ordinal = 0,
        };
        const auto main_stream_desc = pi::tensorlib::GpuStreamDescriptors::Main;
        pi::tensorlib::OpGraph init_graph{{}, {}};

        uint32_t rng_seed = 42;

        const auto mha =
            std::make_shared<pi::tensorlib::FullMhaAttention>("attn", EMBED, H, device, dtype, init_graph, rng_seed,
                                                              main_stream_desc);

        pi::tensorlib::TraceTensor x = init_graph.createTensor({B, T, EMBED}, dtype, device, main_stream_desc, false);
        pi::tensorlib::FillUniform(init_graph, x, -0.5f, 0.5f, rng_seed++, main_stream_desc);

        init_graph.finalize();

        pi::tensorlib::ExecutionPlan init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
        ApplyDefaultPasses(init_plan);

        pi::tensorlib::Executor init_executor{init_plan, execution_backend, 0};
        init_executor.execute(allocator_registry);

        std::vector<pi::tensorlib::GraphInputDescriptor> parameter_descriptors{};
        for (const auto &[name, tensor] : mha->parameters())
        {
            auto retained_tensor = tensor;
            retained_tensor.markRetained();
            parameter_descriptors.push_back(pi::tensorlib::GraphInputDescriptor{.name = name, .tensor = retained_tensor});
        }

        x.markRetained();
        pi::tensorlib::OpGraph graph{{
                                         pi::tensorlib::GraphInputDescriptor{.name = "x", .tensor = x},
                                     },
                                     parameter_descriptors};

        pi::tensorlib::TraceTensor output = mha->buildForward(graph, {x});
        graph.finalize();

        std::vector<pi::tensorlib::GraphExecutionInputDescriptor> parameter_real_descriptors{};
        for (const auto &[name, tensor] : mha->parameters())
        {
            const auto real_tensor_opt = init_executor.getOutput(tensor);
            if (!real_tensor_opt)
            {
                throw std::runtime_error("Parameter tensor not found after init graph execution");
            }
            parameter_real_descriptors.push_back(
                pi::tensorlib::GraphExecutionInputDescriptor{.name = name, .tensor = *real_tensor_opt});
        }

        const auto x_real_opt = init_executor.getOutput(x);
        if (!x_real_opt)
        {
            throw std::runtime_error("Input tensor not found after init graph execution");
        }
        std::shared_ptr<pi::tensorlib::RealTensor> x_real = *x_real_opt;

        pi::tensorlib::ExecutionPlan plan = pi::tensorlib::ExecutionPlan::FromGraph(
            graph, {{.name = "x", .tensor = x_real}}, parameter_real_descriptors);
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
} // namespace

int main()
{
    const auto dtype = test_utils::GetTestDtype();
    RunMhaReshapeTest(dtype);
}
