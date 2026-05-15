#include "testing.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <tensorlib.h>
#include <utils.h>

#include "../common/test_dtype_utils.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace
{
    constexpr int64_t NUMEL = 1024 * 16;
    constexpr float kFillValue = 1.2345f;

    void ApplyFillConstantPass(pi::tensorlib::ExecutionPlan &execution_plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        pi::tensorlib::passes::Transform(execution_plan, passes);
    }

    void AssertTensorAllConstant(const std::shared_ptr<pi::tensorlib::RealTensor> &tensor, float value, float tol)
    {
        using namespace pi::tensorlib;
        if (!tensor)
        {
            throw std::runtime_error("null tensor provided to AssertTensorAllConstant");
        }

        auto storage = tensor->storage();
        if (storage->device().device_type != DeviceType::CPU)
        {
            storage = storage->toCPU();
        }

        const size_t elem_size = GetDataTypeSize(tensor->dtype());
        const size_t numel = tensor->shape().numel();
        const auto *base_ptr = static_cast<const uint8_t *>(storage->dataptr()) + tensor->storageOffset() * elem_size;

        for (size_t i = 0; i < numel; ++i)
        {
            float actual{};
            switch (tensor->dtype())
            {
                case DataType::FLOAT32:
                    actual = *(reinterpret_cast<const float *>(base_ptr) + i);
                    break;
                case DataType::BFLOAT16:
                    actual = utils::Fp32FromBf16(*(reinterpret_cast<const uint16_t *>(base_ptr) + i));
                    break;
                case DataType::FLOAT16:
                    actual = utils::Fp32FromFp16(*(reinterpret_cast<const uint16_t *>(base_ptr) + i));
                    break;
                default:
                    throw std::runtime_error("unsupported dtype in AssertTensorAllConstant");
            }

            if (std::fabs(actual - value) > tol)
            {
                throw std::runtime_error("FillConstant produced unexpected value at element " + std::to_string(i) +
                                         ": expected " + std::to_string(value) + ", got " + std::to_string(actual));
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

    OpGraph graph{{}, {}};
    TraceTensor tensor = graph.createTensor({NUMEL}, dtype, device, main_stream_desc, false);
    FillConstant(graph, tensor, kFillValue, main_stream_desc);
    graph.finalize();

    ExecutionPlan plan = ExecutionPlan::FromGraph(graph, {}, {});
    ApplyFillConstantPass(plan);

    Executor executor{plan, execution_backend, 0};
    executor.execute(allocator_registry);

    const auto tensor_real = executor.getOutput(tensor);
    if (!tensor_real)
    {
        throw std::runtime_error("FillConstant output tensor not found");
    }

    const float tol = test_utils::SelectTolerance(dtype, /*bf16_tol=*/5e-2f, /*fp16_tol=*/5e-2f);
    AssertTensorAllConstant(*tensor_real, kFillValue, tol);
    return 0;
}
