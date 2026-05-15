#include "testing.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <mean.h>
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
        passes.emplace_back(std::make_unique<MeanImplPass>());
        passes.emplace_back(std::make_unique<DivAddFusePass>());
        passes.emplace_back(std::make_unique<DivImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
        passes.emplace_back(std::make_unique<ContiguousImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);
    }

    template <typename MapT>
    std::shared_ptr<pi::tensorlib::RealTensor> FetchTensor(const MapT &ref, const std::string &name)
    {
        auto it = ref.find(name);
        if (it == ref.end())
        {
            throw std::runtime_error("Missing tensor in reference.safetensors: " + name);
        }
        return it->second;
    }

    void RunCase(const std::shared_ptr<pi::tensorlib::RealTensor> &x_host, const std::string &upstream_name,
                 const std::string &grad_name, const int64_t dim, const bool keepdim)
    {
        using namespace pi::tensorlib;

        ExecutionBackend &backend = ExecutionBackend::getInstance();
        auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        const auto ref = safetensors::Load("reference.safetensors");
        const auto upstream_host = FetchTensor(ref, upstream_name);
        const auto expected_grad = FetchTensor(ref, grad_name);

        const auto dtype = x_host->dtype();

        TraceTensor x = TraceTensor::Create(x_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor upstream = TraceTensor::Create(upstream_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        x.markRetained();
        upstream.markRetained();

        const int64_t ndims = static_cast<int64_t>(x_host->shape().ndims());
        int64_t reduction_dim = dim;
        if (reduction_dim < 0)
        {
            reduction_dim += ndims;
        }
        const uint64_t reduction_size = x_host->shape()[reduction_dim];

        OpGraph graph{
            {
                {.name = "x", .tensor = x},
                {.name = "upstream", .tensor = upstream},
            },
            {},
        };
        MeanModule mean_op{"mean", dim, keepdim, reduction_size, graph, dtype, DEVICE_GPU, main_stream_desc};
        (void)mean_op.buildForward(graph, {x}, /*save_input_for_backward=*/true);
        std::unordered_map<std::string, TraceTensor> parameter_grads{};
        std::unordered_map<std::string, TraceTensor> operand_grads{};
        TraceTensor dx_target = graph.createTensor(x.shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
        FillZeros(graph, dx_target, main_stream_desc);
        operand_grads.emplace("input", dx_target);
        mean_op.buildBackward(graph, upstream, parameter_grads, operand_grads);

        graph.finalize();

        const auto x_gpu = x_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto upstream_gpu = upstream_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

        std::vector<GraphExecutionInputDescriptor> inputs{
            {.name = "x", .tensor = x_gpu},
            {.name = "upstream", .tensor = upstream_gpu},
        };
        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, inputs, {});

        ApplyDefaultPasses(plan);
        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);

        const auto actual_dx = executor.getOutput(dx_target);
        if (!actual_dx)
        {
            throw std::runtime_error("Expected mean backward output not found");
        }

        constexpr float tolerance = 2e-3f;
        testing::AssertSimilar(expected_grad, *actual_dx, tolerance);
    }
} // namespace

int main()
{
    const auto ref = pi::tensorlib::safetensors::Load("reference.safetensors");
    const auto x_host = FetchTensor(ref, "x");

    RunCase(x_host, "upstream_keep", "grad_keep", /*dim=*/-1, /*keepdim=*/true);
    RunCase(x_host, "upstream_time", "grad_time", /*dim=*/1, /*keepdim=*/false);

    return 0;
}
