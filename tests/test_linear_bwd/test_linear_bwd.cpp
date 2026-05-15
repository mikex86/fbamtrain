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
#include <unordered_map>
#include <vector>

namespace
{
    constexpr pi::tensorlib::Device DEVICE_GPU{pi::tensorlib::DeviceType::GPU, 0};

    void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<MatmulFusePass>());
        passes.emplace_back(std::make_unique<MatmulImplPass>());
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

    void RunLinearBackwardCase(const pi::tensorlib::DataType dtype)
    {
        using namespace pi::tensorlib;

        ExecutionBackend &backend = ExecutionBackend::getInstance();
        auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        const std::string reference_path = test_utils::ReferenceFileName(dtype);
        const auto ref = safetensors::Load(reference_path);

        const auto x_host = FetchTensor(ref, "x");
        const auto weight_host = FetchTensor(ref, "weight");
        const auto bias_host = FetchTensor(ref, "bias");
        const auto upstream_host = FetchTensor(ref, "upstream");
        const auto expected_dx = FetchTensor(ref, "grad_x");
        const auto expected_dw = FetchTensor(ref, "grad_w");
        const auto expected_db = FetchTensor(ref, "grad_b");

        OpGraph init_graph{{}, {}};
        uint32_t seed = 0;
        Linear linear{"linear",
                      static_cast<size_t>(x_host->shape()[1]),
                      static_cast<size_t>(weight_host->shape()[1]),
                      DEVICE_GPU,
                      dtype,
                      ActivationFunction::NONE,
                      init_graph,
                      seed,
                      main_stream_desc};

        TraceTensor x = TraceTensor::Create(x_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor upstream = TraceTensor::Create(upstream_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        x.markRetained();
        upstream.markRetained();
        const auto params = linear.parameters();
        auto retained_weight = params[0].tensor;
        auto retained_bias = params[1].tensor;
        retained_weight.markRetained();
        retained_bias.markRetained();

        OpGraph graph({
                          {.name = "x", .tensor = x},
                          {.name = "upstream", .tensor = upstream},
                          {.name = params[0].name, .tensor = retained_weight},
                          {.name = params[1].name, .tensor = retained_bias},
                      },
                      {});

        (void)linear.buildForward(graph, {x}, /*save_input_for_backward=*/true);

        std::unordered_map<std::string, TraceTensor> parameter_grads{};
        std::unordered_map<std::string, TraceTensor> operand_grads{};
        TraceTensor w_grad = graph.createTensor(weight_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
        TraceTensor b_grad = graph.createTensor(bias_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dx_grad = graph.createTensor(x_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
        FillZeros(graph, w_grad, main_stream_desc);
        FillZeros(graph, b_grad, main_stream_desc);
        FillZeros(graph, dx_grad, main_stream_desc);
        parameter_grads.emplace(params[0].name, w_grad);
        parameter_grads.emplace(params[1].name, b_grad);
        operand_grads.emplace("input", dx_grad);

        linear.buildBackward(graph, upstream, parameter_grads, operand_grads);
        graph.finalize();

        const auto x_gpu = x_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto upstream_gpu = upstream_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto weight_gpu = weight_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto bias_gpu = bias_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

        std::vector<GraphExecutionInputDescriptor> inputs{
            {.name = "x", .tensor = x_gpu},
            {.name = "upstream", .tensor = upstream_gpu},
            {.name = params[0].name, .tensor = weight_gpu},
            {.name = params[1].name, .tensor = bias_gpu},
        };

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, inputs, {});

        ApplyDefaultPasses(plan);
        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);

        const auto actual_w = executor.getOutput(w_grad);
        const auto actual_b = executor.getOutput(b_grad);
        const auto actual_dx = executor.getOutput(dx_grad);
        if (!actual_w || !actual_b || !actual_dx)
        {
            throw std::runtime_error("Failed to retrieve linear backward gradients");
        }

        const float tolerance = test_utils::SelectTolerance(dtype, 8e-3f, 5e-3f);
        testing::AssertSimilar(expected_dw, *actual_w, tolerance);
        testing::AssertSimilar(expected_db, *actual_b, tolerance);
        testing::AssertSimilar(expected_dx, *actual_dx, tolerance);
    }
} // namespace

int main()
{
    const auto dtype = test_utils::GetTestDtype();
    RunLinearBackwardCase(dtype);
    return 0;
}
