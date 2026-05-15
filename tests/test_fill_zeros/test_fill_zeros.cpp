#include "testing.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <tensorlib.h>

#include "../common/test_dtype_utils.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    constexpr int64_t NUMEL = 1024 * 16;

    void ApplyFillUniformPasses(pi::tensorlib::ExecutionPlan &execution_plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<FillUniformImplPass>());
        pi::tensorlib::passes::Transform(execution_plan, passes);
    }

    void ApplyFillZerosPass(pi::tensorlib::ExecutionPlan &execution_plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        pi::tensorlib::passes::Transform(execution_plan, passes);
    }

    void AssertTensorAllZero(const std::shared_ptr<pi::tensorlib::RealTensor> &tensor)
    {
        if (!tensor)
        {
            throw std::runtime_error("null tensor provided to AssertTensorAllZero");
        }

        auto storage = tensor->storage();
        if (storage->device().device_type != pi::tensorlib::DeviceType::CPU)
        {
            storage = storage->toCPU();
        }

        const size_t elem_size = pi::tensorlib::GetDataTypeSize(tensor->dtype());
        const size_t byte_count = tensor->shape().numel() * elem_size;
        const auto *data = static_cast<const uint8_t *>(storage->dataptr()) + tensor->storageOffset() * elem_size;

        for (size_t i = 0; i < byte_count; ++i)
        {
            if (data[i] != 0)
            {
                throw std::runtime_error("FillZeros produced non-zero data at byte offset " + std::to_string(i));
            }
        }
    }
} // namespace

int main()
{
    using namespace pi::tensorlib;

    ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
    auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();

    const Device device{
        .device_type = DeviceType::GPU,
        .ordinal = 0,
    };
    const auto main_stream_desc = GpuStreamDescriptors::Main;

    const DataType dtype = test_utils::GetTestDtype();

    // Initialize the tensor with non-zero data.
    OpGraph init_graph{{}, {}};
    TraceTensor tensor = init_graph.createTensor({NUMEL}, dtype, device, main_stream_desc, false);
    tensor.markRetained();
    FillUniform(init_graph, tensor, -1.0f, 1.0f, 1337, main_stream_desc);
    init_graph.finalize();

    ExecutionPlan init_plan = ExecutionPlan::FromGraph(init_graph, {}, {});
    ApplyFillUniformPasses(init_plan);

    Executor init_executor{init_plan, execution_backend, 0};
    init_executor.execute(allocator_registry);

    const auto tensor_real = init_executor.getOutput(tensor);
    if (!tensor_real)
    {
        throw std::runtime_error("Failed to retrieve initialized tensor");
    }

    // Apply FillZeros to the tensor and validate all bytes are zero after execution.
    OpGraph graph{
        {
            {.name = "tensor", .tensor = tensor},
        },
        {},
    };
    FillZeros(graph, tensor, main_stream_desc);
    graph.finalize();

    ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                  {
                                                      {.name = "tensor", .tensor = *tensor_real},
                                                  },
                                                  {});
    ApplyFillZerosPass(plan);

    Executor executor{plan, execution_backend, 0};
    executor.execute(allocator_registry);

    const auto zeroed_tensor = executor.getOutput(tensor);
    if (!zeroed_tensor)
    {
        throw std::runtime_error("FillZeros output tensor not found");
    }

    AssertTensorAllZero(*zeroed_tensor);
    return 0;
}
