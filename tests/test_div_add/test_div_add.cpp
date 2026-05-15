#include "testing.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
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

    void ApplyPasses(pi::tensorlib::ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<DivAddFusePass>());
        passes.emplace_back(std::make_unique<DivImplPass>());
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

    void AssertKernelHit(const pi::tensorlib::ExecutionPlan &plan, const std::string &kernel_prefix)
    {
        for (const auto &entry : plan.entries)
        {
            if (!entry.kernel_descriptor.has_value())
            {
                continue;
            }
            const auto &kernel_name = entry.kernel_descriptor->kernel_name;
            if (kernel_name.rfind(kernel_prefix, 0) == 0)
            {
                return;
            }
        }
        throw std::runtime_error("Expected kernel not found in execution plan: " + kernel_prefix);
    }

    void RunCase(const std::string &lhs_name, const std::string &rhs_name, const std::string &denom_name,
                 const std::string &out_name, const std::string &kernel_prefix)
    {
        using namespace pi::tensorlib;

        ExecutionBackend &backend = ExecutionBackend::getInstance();
        auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        const auto ref = safetensors::Load("reference.safetensors");
        const auto lhs_host = FetchTensor(ref, lhs_name);
        const auto rhs_host = FetchTensor(ref, rhs_name);
        const auto denom_host = FetchTensor(ref, denom_name);
        const auto expected_out = FetchTensor(ref, out_name);

        const auto dtype = lhs_host->dtype();
        TraceTensor lhs = TraceTensor::Create(lhs_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor rhs = TraceTensor::Create(rhs_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor denom = TraceTensor::Create(denom_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        lhs.markRetained();
        rhs.markRetained();
        denom.markRetained();

        OpGraph graph(
            {
                {.name = "lhs", .tensor = lhs},
                {.name = "rhs", .tensor = rhs},
                {.name = "denom", .tensor = denom},
            },
            {});

        InplaceDiv(graph, rhs, denom, main_stream_desc);
        InplaceAdd(graph, lhs, rhs, main_stream_desc);

        graph.finalize();

        const auto lhs_gpu = lhs_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto rhs_gpu = rhs_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto denom_gpu = denom_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

        std::vector<GraphExecutionInputDescriptor> inputs{
            {.name = "lhs", .tensor = lhs_gpu},
            {.name = "rhs", .tensor = rhs_gpu},
            {.name = "denom", .tensor = denom_gpu},
        };
        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, inputs, {});
        ApplyPasses(plan);
        AssertKernelHit(plan, kernel_prefix);

        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);

        const auto actual_out = executor.getOutput(lhs);
        if (!actual_out)
        {
            throw std::runtime_error("Expected output tensor not found");
        }

        constexpr float tolerance = 2e-3f;
        testing::AssertSimilar(expected_out, *actual_out, tolerance);
    }

    void RunBroadcastCase()
    {
        using namespace pi::tensorlib;

        ExecutionBackend &backend = ExecutionBackend::getInstance();
        auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        const auto ref = safetensors::Load("reference.safetensors");
        const auto lhs_host = FetchTensor(ref, "lhs_bcast");
        const auto rhs_host = FetchTensor(ref, "rhs_bcast");
        const auto denom_host = FetchTensor(ref, "denom_bcast");
        const auto expected_out = FetchTensor(ref, "out_bcast");

        const auto dtype = lhs_host->dtype();
        TraceTensor lhs = TraceTensor::Create(lhs_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor rhs = TraceTensor::Create(rhs_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor denom = TraceTensor::Create(denom_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        lhs.markRetained();
        rhs.markRetained();
        denom.markRetained();

        OpGraph graph(
            {
                {.name = "lhs", .tensor = lhs},
                {.name = "rhs", .tensor = rhs},
                {.name = "denom", .tensor = denom},
            },
            {});

        const auto broadcast_dim = 1;
        TraceTensor rhs_bcast = rhs.broadcast(graph, broadcast_dim, lhs.shape()[broadcast_dim]);
        InplaceDiv(graph, rhs_bcast, denom, main_stream_desc);
        InplaceAdd(graph, lhs, rhs_bcast, main_stream_desc);

        graph.finalize();

        const auto lhs_gpu = lhs_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto rhs_gpu = rhs_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto denom_gpu = denom_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

        std::vector<GraphExecutionInputDescriptor> inputs{
            {.name = "lhs", .tensor = lhs_gpu},
            {.name = "rhs", .tensor = rhs_gpu},
            {.name = "denom", .tensor = denom_gpu},
        };
        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, inputs, {});
        ApplyPasses(plan);
        AssertKernelHit(plan, "div_scalar_add_broadcast_");

        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);

        const auto actual_out = executor.getOutput(lhs);
        if (!actual_out)
        {
            throw std::runtime_error("Expected output tensor not found");
        }

        constexpr float tolerance = 2e-3f;
        testing::AssertSimilar(expected_out, *actual_out, tolerance);
    }
} // namespace

int main()
{
    RunCase("lhs_scalar", "rhs_scalar", "denom_scalar", "out_scalar", "div_scalar_add_");
    RunCase("lhs_elem", "rhs_elem", "denom_elem", "out_elem", "div_add_");
    RunBroadcastCase();
    return 0;
}
