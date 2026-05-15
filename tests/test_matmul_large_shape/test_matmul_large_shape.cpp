#include "testing.h"

#include "../common/test_dtype_utils.h"
#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>
#include <utils.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace pi::tensorlib;

namespace
{
    constexpr Device DEVICE_GPU{DeviceType::GPU, 0};

    struct MatmulCase
    {
        const char *name;
        const char *a_key;
        const char *b_key;
        const char *expected_key;
        bool transpose_a{};
        bool transpose_b{};
    };

    TraceTensor TransposeIfNeeded(OpGraph &graph, const TraceTensor &base, const bool transpose_flag)
    {
        return transpose_flag ? base.transpose(graph, {1, 0}) : base;
    }

    template <typename MapT>
    std::shared_ptr<RealTensor> FetchTensor(const MapT &ref, const std::string &name)
    {
        auto it = ref.find(name);
        if (it == ref.end())
        {
            throw std::runtime_error("Missing tensor in reference file: " + name);
        }
        return it->second;
    }

    template <typename MapT>
    void RunCase(const MatmulCase &cfg, const MapT &tensors, const float tolerance)
    {
        const auto &a_host = FetchTensor(tensors, cfg.a_key);
        const auto &b_host = FetchTensor(tensors, cfg.b_key);
        const auto &expected = FetchTensor(tensors, cfg.expected_key);

        const auto main_stream_desc = GpuStreamDescriptors::Main;
        const std::shared_ptr<RealTensor> a_device = a_host->to(DEVICE_GPU, main_stream_desc);
        const std::shared_ptr<RealTensor> b_device = b_host->to(DEVICE_GPU, main_stream_desc);

        TraceTensor a_trace = TraceTensor::Create(a_device->shape().dims(), a_device->dtype(), DEVICE_GPU,
                                                  main_stream_desc);
        TraceTensor b_trace = TraceTensor::Create(b_device->shape().dims(), b_device->dtype(), DEVICE_GPU,
                                                  main_stream_desc);
        a_trace.markRetained();
        b_trace.markRetained();

        OpGraph graph({{.name = "a_base", .tensor = a_trace}, {.name = "b_base", .tensor = b_trace}}, {});
        TraceTensor a_view = TransposeIfNeeded(graph, graph.getInputDescriptors()[0].tensor, cfg.transpose_a);
        TraceTensor b_view = TransposeIfNeeded(graph, graph.getInputDescriptors()[1].tensor, cfg.transpose_b);

        TraceTensor out =
            graph.createTensor(expected->shape().dims(), expected->dtype(), DEVICE_GPU, main_stream_desc, false);
        graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                             .inputs = {a_view, b_view},
                                             .outputs = {out},
                                             .attributes = {},
                                             .gpu_stream_desc = main_stream_desc});
        graph.finalize();

        ExecutionPlan plan =
            ExecutionPlan::FromGraph(graph, {{.name = "a_base", .tensor = a_device}, {.name = "b_base", .tensor = b_device}}, {});

        std::vector<std::unique_ptr<passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<MatmulImplPass>());
        passes::Transform(plan, passes);

        ExecutionBackend &backend = ExecutionBackend::getInstance();
        const auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();
        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);

        const auto out_rt = executor.getOutput(out);
        if (!out_rt)
        {
            throw std::runtime_error(std::string("Failed to fetch matmul output for case: ") + cfg.name);
        }
        testing::AssertSimilar(expected, *out_rt, tolerance);
    }
} // namespace

int main()
{
    const auto dtype = test_utils::GetTestDtype();
    const std::string reference_path = test_utils::ReferenceFileName(dtype);
    const auto ref = safetensors::Load(reference_path);

    const float tolerance = test_utils::SelectTolerance(dtype, 1e-2f, 1e-2f);

    const std::vector<MatmulCase> cases{
        MatmulCase{.name = "ta",
                   .a_key = "a_ta_base",
                   .b_key = "b_ta_base",
                   .expected_key = "out_ta",
                   .transpose_a = true,
                   .transpose_b = false},
        MatmulCase{.name = "tb",
                   .a_key = "a_tb_base",
                   .b_key = "b_tb_base",
                   .expected_key = "out_tb",
                   .transpose_a = false,
                   .transpose_b = true},
    };

    for (const auto &cfg : cases)
    {
        try
        {
            RunCase(cfg, ref, tolerance);
        }
        catch (const std::exception &ex)
        {
            throw std::runtime_error(std::string("Matmul case '") + cfg.name + "' failed: " + ex.what());
        }
    }

    return 0;
}
