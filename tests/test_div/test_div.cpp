#include "testing.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>

#include "../common/test_dtype_utils.h"

namespace
{
    void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &execution_plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<DivImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<FillUniformImplPass>());
        pi::tensorlib::passes::Transform(execution_plan, passes);
    }

    constexpr int64_t ROWS = 2048;
    constexpr int64_t COLS = 512;
    constexpr int64_t NUMEL = ROWS * COLS;
} // namespace

int main()
{
    constexpr float TOLERANCE_BF16 = 1e-3f;
    constexpr float TOLERANCE_FP16 = 2e-3f;

    using namespace pi::tensorlib;

    ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
    auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();

    const Device device{DeviceType::GPU, 0};
    const auto main_stream_desc = GpuStreamDescriptors::Main;
    const auto dtype = test_utils::GetTestDtype();

    OpGraph init_graph{{}, {}};

    TraceTensor lhs = init_graph.createTensor({NUMEL}, dtype, device, main_stream_desc, false);
    TraceTensor rhs = init_graph.createTensor({NUMEL}, dtype, device, main_stream_desc, false);
    lhs.markRetained();
    rhs.markRetained();

    uint32_t seed = 321;
    FillUniform(init_graph, lhs, -0.5f, 0.5f, seed++, main_stream_desc);
    // keep denominators away from zero to avoid inf/nan in reference
    FillUniform(init_graph, rhs, 0.5f, 1.5f, seed++, main_stream_desc);

    init_graph.finalize();

    ExecutionPlan init_plan = ExecutionPlan::FromGraph(init_graph, {}, {});
    ApplyDefaultPasses(init_plan);
    Executor init_executor{init_plan, execution_backend, 0};
    init_executor.execute(allocator_registry);

    const auto lhs_real = init_executor.getOutput(lhs);
    const auto rhs_real = init_executor.getOutput(rhs);
    if (!lhs_real || !rhs_real)
    {
        throw std::runtime_error("Failed to retrieve initialization tensors");
    }

    OpGraph graph{{{.name = "lhs", .tensor = lhs}, {.name = "rhs", .tensor = rhs}}, {}};

    InplaceDiv(graph, lhs, rhs, main_stream_desc);

    graph.finalize();

    ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                  {
                                                      {.name = "lhs", .tensor = *lhs_real},
                                                      {.name = "rhs", .tensor = *rhs_real},
                                                  },
                                                  {});

    ApplyDefaultPasses(plan);

    Executor executor{plan, execution_backend, 0};
    executor.execute(allocator_registry);

    const auto output = executor.getOutput(lhs);
    if (!output)
    {
        throw std::runtime_error("Expected output tensor not found");
    }

    const std::string reference_path = test_utils::ReferenceFileName(dtype);
    const auto expected_tensors = safetensors::Load(reference_path);
    const auto it = expected_tensors.find("elementwise_output");
    if (it == expected_tensors.end())
    {
        throw std::runtime_error("Expected elementwise output tensor not found in reference file: " + reference_path);
    }

    const float tolerance = test_utils::SelectTolerance(dtype, TOLERANCE_BF16, TOLERANCE_FP16);
    testing::AssertSimilar(it->second, *output, tolerance);
}
