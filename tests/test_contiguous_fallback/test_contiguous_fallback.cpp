#include "testing.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <tensorlib.h>
#include <utils.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

using namespace pi::tensorlib;

namespace
{
    constexpr Device DEVICE_CPU{DeviceType::CPU, 0};
    constexpr Device DEVICE_GPU{DeviceType::GPU, 0};

    std::shared_ptr<RealTensor> MakeHostTensor(const Shape &shape, const DataType dtype)
    {
        auto tensor = RealTensor::Allocate({shape[0], shape[1]}, dtype, DEVICE_CPU);
        auto *raw = reinterpret_cast<uint16_t *>(tensor->dataptr());
        for (uint64_t i = 0; i < shape.numel(); ++i)
        {
            const float value = static_cast<float>(i % 17) - 3.0f;
            if (dtype == DataType::BFLOAT16)
            {
                raw[i] = utils::Bf16FromFp32(value);
            }
            else if (dtype == DataType::FLOAT16)
            {
                raw[i] = utils::Fp16FromFp32(value);
            }
            else
            {
                throw std::runtime_error("Unsupported dtype for test_contiguous_fallback");
            }
        }
        return tensor;
    }

    void ApplyPasses(ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<ContiguousImplPass>());
        passes::Transform(plan, passes);
    }

    void RunCase(const DataType dtype)
    {
        ExecutionBackend &backend = ExecutionBackend::getInstance();
        const auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        const Shape input_shape(std::vector<uint64_t>{3, 6});
        constexpr int slice_start = 1;
        constexpr int slice_len = 4;
        auto host_input = MakeHostTensor(input_shape, dtype);

        // Expected is the contiguous copy of the sliced view.
        auto expected = RealTensor::Allocate({input_shape[0], static_cast<uint64_t>(slice_len)}, dtype, DEVICE_CPU);
        const auto *src = reinterpret_cast<const uint16_t *>(host_input->dataptr());
        auto *dst = reinterpret_cast<uint16_t *>(expected->dataptr());
        const auto src_stride = input_shape[1];
        for (uint64_t r = 0; r < input_shape[0]; ++r)
        {
            const auto *src_row = src + r * src_stride + slice_start;
            auto *dst_row = dst + r * slice_len;
            std::copy(src_row, src_row + slice_len, dst_row);
        }

        TraceTensor input_cpu = TraceTensor::Create(input_shape.dims(), dtype, DEVICE_CPU, main_stream_desc);
        input_cpu.markRetained();

        OpGraph graph({{.name = "input", .tensor = input_cpu}}, {});
        TraceTensor input_gpu = input_cpu.to(graph, DEVICE_GPU, GpuStreamDescriptors::Main);
        TraceTensor sliced = input_gpu.slice(graph, /*dim=*/1, /*start=*/slice_start, /*length=*/slice_len);
        TraceTensor contiguous = sliced.contiguous(graph, main_stream_desc);
        TraceTensor output_cpu = contiguous.to(graph, DEVICE_CPU, GpuStreamDescriptors::Main);
        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, {{.name = "input", .tensor = host_input}}, {});
        ApplyPasses(plan);
        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);

        const auto output_opt = executor.getOutput(output_cpu);
        if (!output_opt)
        {
            throw std::runtime_error("Failed to fetch output from contiguous fallback test");
        }

        testing::AssertSimilar(expected, *output_opt, /*atol=*/0.0f);
    }
} // namespace

int main()
{
    RunCase(DataType::BFLOAT16);
    RunCase(DataType::FLOAT16);
    return 0;
}
