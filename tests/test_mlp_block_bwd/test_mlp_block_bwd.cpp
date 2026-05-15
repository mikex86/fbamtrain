#include "testing.h"

#include "../common/test_dtype_utils.h"
#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <mlp.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    constexpr pi::tensorlib::Device DEVICE_GPU{pi::tensorlib::DeviceType::GPU, 0};

    bool IsEnvFlagEnabled(const char *name)
    {
        const char *env = std::getenv(name);
        return env != nullptr && env[0] != '\0';
    }

    void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        if (!IsEnvFlagEnabled("MLP_TEST_DISABLE_FUSE"))
        {
            passes.emplace_back(std::make_unique<MatmulFusePass>());
        }
        passes.emplace_back(std::make_unique<MatmulImplPass>());
        passes.emplace_back(std::make_unique<ActImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
        passes.emplace_back(std::make_unique<ReduceSumImplPass>());
        passes.emplace_back(std::make_unique<ContiguousImplPass>());
        passes.emplace_back(std::make_unique<CastImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);
    }

    template <typename MapT>
    std::shared_ptr<pi::tensorlib::RealTensor> FetchTensor(const MapT &ref, const std::string &name)
    {
        auto it = ref.find(name);
        if (it == ref.end())
        {
            throw std::runtime_error("Missing tensor in reference file: " + name);
        }
        return it->second;
    }

    void RunMlpBlockBackwardCase(const pi::tensorlib::DataType dtype)
    {
        using namespace pi::tensorlib;

        ExecutionBackend &backend = ExecutionBackend::getInstance();
        auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        const std::string reference_path = test_utils::ReferenceFileName(dtype);
        const auto ref = safetensors::Load(reference_path);

        const auto x_host = FetchTensor(ref, "x");
        const auto w1_host = FetchTensor(ref, "w1");
        const auto b1_host = FetchTensor(ref, "b1");
        const auto w2_host = FetchTensor(ref, "w2");
        const auto b2_host = FetchTensor(ref, "b2");
        const auto upstream_host = FetchTensor(ref, "upstream");

        const auto expected_dx = FetchTensor(ref, "grad_x");
        const auto expected_dw1 = FetchTensor(ref, "grad_w1");
        const auto expected_db1 = FetchTensor(ref, "grad_b1");
        const auto expected_dw2 = FetchTensor(ref, "grad_w2");
        const auto expected_db2 = FetchTensor(ref, "grad_b2");

        OpGraph init_graph{{}, {}};
        uint32_t seed = 0;
        const uint32_t embed_dim = static_cast<uint32_t>(x_host->shape()[1]);
        const uint32_t hidden_dim = static_cast<uint32_t>(w1_host->shape()[1]);
        MlpBlock mlp_block{"mlp", embed_dim, hidden_dim, DEVICE_GPU, dtype, init_graph, seed, main_stream_desc};

        TraceTensor x = TraceTensor::Create(x_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor upstream = TraceTensor::Create(upstream_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        x.markRetained();
        upstream.markRetained();
        const auto params = mlp_block.parameters();
        auto retained_w1 = params[0].tensor;
        auto retained_b1 = params[1].tensor;
        auto retained_w2 = params[2].tensor;
        auto retained_b2 = params[3].tensor;
        retained_w1.markRetained();
        retained_b1.markRetained();
        retained_w2.markRetained();
        retained_b2.markRetained();

        OpGraph graph(
            {
                {.name = "x", .tensor = x},
                {.name = "upstream", .tensor = upstream},
                {.name = params[0].name, .tensor = retained_w1},
                {.name = params[1].name, .tensor = retained_b1},
                {.name = params[2].name, .tensor = retained_w2},
                {.name = params[3].name, .tensor = retained_b2},
            },
            {});

        (void)mlp_block.buildForward(graph, {x}, /*save_input_for_backward=*/true);

        std::unordered_map<std::string, TraceTensor> parameter_grads{};
        std::unordered_map<std::string, TraceTensor> operand_grads{};
        TraceTensor w1_grad = graph.createTensor(w1_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
        TraceTensor b1_grad = graph.createTensor(b1_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
        TraceTensor w2_grad = graph.createTensor(w2_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
        TraceTensor b2_grad = graph.createTensor(b2_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dx_grad = graph.createTensor(x_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
        FillZeros(graph, w1_grad, main_stream_desc);
        FillZeros(graph, b1_grad, main_stream_desc);
        FillZeros(graph, w2_grad, main_stream_desc);
        FillZeros(graph, b2_grad, main_stream_desc);
        FillZeros(graph, dx_grad, main_stream_desc);
        parameter_grads.emplace(params[0].name, w1_grad);
        parameter_grads.emplace(params[1].name, b1_grad);
        parameter_grads.emplace(params[2].name, w2_grad);
        parameter_grads.emplace(params[3].name, b2_grad);
        operand_grads.emplace("input", dx_grad);

        mlp_block.buildBackward(graph, upstream, parameter_grads, operand_grads);
        graph.finalize();

        const auto x_gpu = x_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto upstream_gpu = upstream_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto w1_gpu = w1_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto b1_gpu = b1_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto w2_gpu = w2_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto b2_gpu = b2_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

        std::vector<GraphExecutionInputDescriptor> inputs{
            {.name = "x", .tensor = x_gpu},
            {.name = "upstream", .tensor = upstream_gpu},
        };
        for (const auto &param : params)
        {
            if (param.name.find("fc1.weight") != std::string::npos)
            {
                inputs.push_back({.name = param.name, .tensor = w1_gpu});
            }
            else if (param.name.find("fc1.bias") != std::string::npos)
            {
                inputs.push_back({.name = param.name, .tensor = b1_gpu});
            }
            else if (param.name.find("fc2.weight") != std::string::npos)
            {
                inputs.push_back({.name = param.name, .tensor = w2_gpu});
            }
            else if (param.name.find("fc2.bias") != std::string::npos)
            {
                inputs.push_back({.name = param.name, .tensor = b2_gpu});
            }
            else
            {
                throw std::runtime_error("Unexpected parameter name: " + param.name);
            }
        }

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, inputs, {});
        ApplyDefaultPasses(plan);
        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);

        const auto actual_w1 = executor.getOutput(w1_grad);
        const auto actual_b1 = executor.getOutput(b1_grad);
        const auto actual_w2 = executor.getOutput(w2_grad);
        const auto actual_b2 = executor.getOutput(b2_grad);
        const auto actual_dx = executor.getOutput(dx_grad);
        if (!actual_w1 || !actual_b1 || !actual_w2 || !actual_b2 || !actual_dx)
        {
            throw std::runtime_error("Failed to retrieve MlpBlock backward gradients");
        }

        const float tolerance = test_utils::SelectTolerance(dtype, 1e-2f, 7e-3f);
        testing::AssertSimilar(expected_dw1, *actual_w1, tolerance);
        testing::AssertSimilar(expected_db1, *actual_b1, tolerance);
        testing::AssertSimilar(expected_dw2, *actual_w2, tolerance);
        testing::AssertSimilar(expected_db2, *actual_b2, tolerance);
        testing::AssertSimilar(expected_dx, *actual_dx, tolerance);
    }

    void AssertAlignedGeluBwdKernel(const pi::tensorlib::DataType dtype, const uint64_t m_dim)
    {
        using namespace pi::tensorlib;

        OpGraph init_graph{{}, {}};
        uint32_t seed = 0;
        constexpr uint32_t embed_dim = 16;
        constexpr uint32_t hidden_dim = 64;
        const auto main_stream_desc = GpuStreamDescriptors::Main;
        MlpBlock mlp_block{"mlp", embed_dim, hidden_dim, DEVICE_GPU, dtype, init_graph, seed, main_stream_desc};

        TraceTensor x = TraceTensor::Create({m_dim, embed_dim}, dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor upstream = TraceTensor::Create({m_dim, embed_dim}, dtype, DEVICE_GPU, main_stream_desc);
        x.markRetained();
        upstream.markRetained();
        const auto params = mlp_block.parameters();
        auto retained_w1 = params[0].tensor;
        auto retained_b1 = params[1].tensor;
        auto retained_w2 = params[2].tensor;
        auto retained_b2 = params[3].tensor;
        retained_w1.markRetained();
        retained_b1.markRetained();
        retained_w2.markRetained();
        retained_b2.markRetained();

        OpGraph graph(
            {
                {.name = "x", .tensor = x},
                {.name = "upstream", .tensor = upstream},
                {.name = params[0].name, .tensor = retained_w1},
                {.name = params[1].name, .tensor = retained_b1},
                {.name = params[2].name, .tensor = retained_w2},
                {.name = params[3].name, .tensor = retained_b2},
            },
            {});

        (void)mlp_block.buildForward(graph, {x}, /*save_input_for_backward=*/true);

        std::unordered_map<std::string, TraceTensor> parameter_grads{};
        std::unordered_map<std::string, TraceTensor> operand_grads{};
        TraceTensor w1_grad = graph.createTensor({embed_dim, hidden_dim}, dtype, DEVICE_GPU, main_stream_desc, false);
        TraceTensor b1_grad = graph.createTensor({hidden_dim}, dtype, DEVICE_GPU, main_stream_desc, false);
        TraceTensor w2_grad = graph.createTensor({hidden_dim, embed_dim}, dtype, DEVICE_GPU, main_stream_desc, false);
        TraceTensor b2_grad = graph.createTensor({embed_dim}, dtype, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dx_grad = graph.createTensor({m_dim, embed_dim}, dtype, DEVICE_GPU, main_stream_desc, false);
        FillZeros(graph, w1_grad, main_stream_desc);
        FillZeros(graph, b1_grad, main_stream_desc);
        FillZeros(graph, w2_grad, main_stream_desc);
        FillZeros(graph, b2_grad, main_stream_desc);
        FillZeros(graph, dx_grad, main_stream_desc);
        parameter_grads.emplace(params[0].name, w1_grad);
        parameter_grads.emplace(params[1].name, b1_grad);
        parameter_grads.emplace(params[2].name, w2_grad);
        parameter_grads.emplace(params[3].name, b2_grad);
        operand_grads.emplace("input", dx_grad);

        mlp_block.buildBackward(graph, upstream, parameter_grads, operand_grads);
        graph.finalize();

        std::vector<GraphExecutionInputDescriptor> inputs{
            {.name = "x", .tensor = RealTensor::Allocate({m_dim, embed_dim}, dtype, DEVICE_GPU)},
            {.name = "upstream", .tensor = RealTensor::Allocate({m_dim, embed_dim}, dtype, DEVICE_GPU)},
        };
        inputs.reserve(inputs.size() + params.size());
        for (const auto &param : params)
        {
            inputs.push_back({.name = param.name, .tensor = RealTensor::CreateLike(param.tensor)});
        }

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, inputs, {});
        ApplyDefaultPasses(plan);

        bool found = false;
        bool aligned = true;
        for (const auto &entry : plan.entries)
        {
            if (!entry.kernel_descriptor.has_value())
            {
                continue;
            }
            const auto &kernel_name = entry.kernel_descriptor->kernel_name;
            if (kernel_name.rfind("matmul_gelu_bwd", 0) == 0 || kernel_name.rfind("cutlass_matmul_gelu_bwd", 0) == 0)
            {
                found = true;
                if (kernel_name.find("unaligned") != std::string::npos)
                {
                    aligned = false;
                }
            }
        }

        if (!found)
        {
            throw std::runtime_error("Expected matmul_gelu_bwd kernel in execution plan");
        }
        if (m_dim >= 16 && !aligned)
        {
            throw std::runtime_error("Expected aligned matmul_gelu_bwd kernel for M >= 16");
        }
    }
} // namespace

int main()
{
    const auto dtype = test_utils::GetTestDtype();
    RunMlpBlockBackwardCase(dtype);
    AssertAlignedGeluBwdKernel(dtype, 16);
    return 0;
}
