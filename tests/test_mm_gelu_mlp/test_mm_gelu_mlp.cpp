#include "testing.h"

#include <allocator.h>

#include <iostream>
#include <string>

#include <execution_backend.h>
#include <executor.h>
#include <linear.h>
#include <module_list.h>
#include <tensorlib.h>

#include "../common/test_dtype_utils.h"
#include <passes.h>
#include <safe_tensors.h>

static void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &execution_plan)
{
    std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
    passes.emplace_back(std::make_unique<MatmulFusePass>());
    passes.emplace_back(std::make_unique<MatmulImplPass>());
    passes.emplace_back(std::make_unique<ActImplPass>());
    passes.emplace_back(std::make_unique<FillZerosImplPass>());
    passes.emplace_back(std::make_unique<FillConstantImplPass>());
    passes.emplace_back(std::make_unique<FillUniformImplPass>());
    pi::tensorlib::passes::Transform(execution_plan, passes);
}

namespace
{
    constexpr float TOLERANCE_BF16 = 1e-3f;
    constexpr float TOLERANCE_FP16 = 3e-3f;

    void RunGeluMlpTest(const pi::tensorlib::DataType dtype)
    {
        pi::tensorlib::ExecutionBackend &execution_backend = pi::tensorlib::ExecutionBackend::getInstance();
        auto &allocator_registry = pi::tensorlib::allocator::DefaultAllocatorRegistry::instance();

        // create the model
        const pi::tensorlib::Device device{
            .device_type = pi::tensorlib::DeviceType::GPU,
            .ordinal = 0,
        };
        const auto main_stream_desc = pi::tensorlib::GpuStreamDescriptors::Main;
        pi::tensorlib::OpGraph init_graph{{}, {}};

        uint32_t rng_seed = 42;
        const auto l1 =
            std::make_shared<pi::tensorlib::Linear>("l1", 10, 5, device, dtype, pi::tensorlib::ActivationFunction::GELU,
                                                    init_graph, rng_seed, main_stream_desc, false /*has_bias*/);
        const auto l2 =
            std::make_shared<pi::tensorlib::Linear>("l2", 5, 2, device, dtype, pi::tensorlib::ActivationFunction::GELU,
                                                    init_graph, rng_seed, main_stream_desc, false /*has_bias*/);

        pi::tensorlib::ModuleList module_list{"modules", {l1, l2}};

        pi::tensorlib::TraceTensor x = init_graph.createTensor({32, 10}, dtype, device, main_stream_desc, false);
        pi::tensorlib::FillUniform(init_graph, x, -0.5f, 0.5f, rng_seed++, main_stream_desc);
        init_graph.finalize();

        // actually execute the init graph
        pi::tensorlib::ExecutionPlan init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
        ApplyDefaultPasses(init_plan);

        pi::tensorlib::Executor init_executor{init_plan, execution_backend, 0};
        init_executor.execute(allocator_registry);

        std::vector<pi::tensorlib::GraphInputDescriptor> parameter_descriptors{};
        for (const auto &[name, tensor] : module_list.parameters())
        {
            auto retained_tensor = tensor;
            retained_tensor.markRetained();
            parameter_descriptors.push_back(pi::tensorlib::GraphInputDescriptor{.name = name, .tensor = retained_tensor});
        }
        x.markRetained();
        pi::tensorlib::OpGraph main_graph{{
                                              {.name = "x", .tensor = x},
                                          },
                                          parameter_descriptors};

        const pi::tensorlib::TraceTensor output = module_list.buildForward(main_graph, {x}, false);
        main_graph.finalize();

        std::cout << "dtype=" << pi::tensorlib::GetDataTypeName(dtype) << " x: " << x << std::endl;
        std::cout << "dtype=" << pi::tensorlib::GetDataTypeName(dtype) << " output: " << output << std::endl;

        // allocate parameters
        std::vector<pi::tensorlib::GraphExecutionInputDescriptor> parameter_real_descriptors{};
        for (const auto &[name, trace_tensor] : module_list.parameters())
        {
            const auto tensor = init_executor.getOutput(trace_tensor);
            if (!tensor)
            {
                throw std::runtime_error("Parameter tensor not found after init graph execution");
            }
            parameter_real_descriptors.emplace_back(
                pi::tensorlib::GraphExecutionInputDescriptor{.name = name, .tensor = *tensor});
        }

        std::shared_ptr<pi::tensorlib::RealTensor> x_real{};
        if (const auto x_real_opt = init_executor.getOutput(x); x_real_opt)
        {
            x_real = *x_real_opt;
        }
        else
        {
            throw std::runtime_error("Input tensor not found after init graph execution");
        }

        // execute the main graph
        pi::tensorlib::ExecutionPlan main_plan = pi::tensorlib::ExecutionPlan::FromGraph(
            main_graph, {{.name = "x", .tensor = x_real}}, parameter_real_descriptors);
        ApplyDefaultPasses(main_plan);

        pi::tensorlib::Executor executor{main_plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto output_real_opt = executor.getOutput(output);
        if (!output_real_opt)
        {
            throw std::runtime_error("Output tensor not found after main graph execution");
        }
        const auto &actual_output = *output_real_opt;

        std::string reference_path{"reference_"};
        reference_path.append(test_utils::GetDtypeSuffix(dtype));
        reference_path.append(".safetensors");

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
    RunGeluMlpTest(dtype);
}
