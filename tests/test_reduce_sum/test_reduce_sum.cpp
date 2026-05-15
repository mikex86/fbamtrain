#include "testing.h"

#include <allocator.h>

#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <tensorlib.h>
#include <utils.h>

#include <array>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pi::tensorlib;

namespace
{
    void ApplyDefaultPasses(ExecutionPlan &execution_plan)
    {
        std::vector<std::unique_ptr<passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<FuseMulReducePass>());
        passes.emplace_back(std::make_unique<ReduceSumImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastMulImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<FillUniformImplPass>());
        passes::Transform(execution_plan, passes);
    }

    bool PlanHasKernel(const ExecutionPlan &execution_plan, const std::string &kernel_name)
    {
        for (const auto &entry : execution_plan.entries)
        {
            if (entry.kernel_descriptor.has_value() && entry.kernel_descriptor->kernel_name == kernel_name)
            {
                return true;
            }
        }
        return false;
    }

    std::string SplitSumKernelName(const DataType dtype)
    {
        switch (dtype)
        {
            case DataType::BFLOAT16:
                return "sum_reduce_column_split_partials_bf16_out_fp32";
            case DataType::FLOAT16:
                return "sum_reduce_column_split_partials_fp16_out_fp32";
            case DataType::FLOAT32:
                return "sum_reduce_column_split_partials_fp32_out_fp32";
            default:
                throw std::runtime_error("Unsupported dtype for split reduce-sum test");
        }
    }

    float ReadElement(const DataType dtype, const void *data, const size_t offset)
    {
        switch (dtype)
        {
            case DataType::FLOAT32:
                return static_cast<const float *>(data)[offset];
            case DataType::BFLOAT16:
                return utils::Fp32FromBf16(static_cast<const uint16_t *>(data)[offset]);
            case DataType::FLOAT16:
                return utils::Fp32FromFp16(static_cast<const uint16_t *>(data)[offset]);
            default:
                throw std::runtime_error("Unsupported dtype for ReduceSum reference");
        }
    }

    std::shared_ptr<RealTensor> ComputeSumKeepLast(const std::shared_ptr<RealTensor> &input, const Device device_cpu,
                                                   const uint64_t batch, const uint64_t rows, const uint64_t cols)
    {
        auto expected = RealTensor::Allocate({batch, rows, 1}, DataType::FLOAT32, device_cpu);

        const auto input_cpu = input->to(device_cpu, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto dtype = input_cpu->dtype();
        const auto &in_strides = input_cpu->strides().strides();
        const auto &out_strides = expected->strides().strides();
        const void *in_data = input_cpu->dataptr();

        auto *out_ptr = static_cast<float *>(expected->dataptr());

        for (uint64_t b = 0; b < batch; ++b)
        {
            for (uint64_t r = 0; r < rows; ++r)
            {
                float sum = 0.0f;
                for (uint64_t c = 0; c < cols; ++c)
                {
                    const auto in_offset =
                        static_cast<size_t>(b * in_strides[0] + r * in_strides[1] + c * in_strides[2]);
                    sum += ReadElement(dtype, in_data, in_offset);
                }
                const auto out_offset = static_cast<size_t>(b * out_strides[0] + r * out_strides[1]);
                out_ptr[out_offset] = sum;
            }
        }

        return expected;
    }

    std::shared_ptr<RealTensor> ComputeSumDim0NoKeep(const std::shared_ptr<RealTensor> &input, const Device device_cpu,
                                                     const uint64_t batch, const uint64_t rows, const uint64_t cols)
    {
        auto expected = RealTensor::Allocate({rows, cols}, DataType::FLOAT32, device_cpu);

        const auto input_cpu = input->to(device_cpu, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto dtype = input_cpu->dtype();
        const auto &in_strides = input_cpu->strides().strides();
        const auto &out_strides = expected->strides().strides();
        const void *in_data = input_cpu->dataptr();

        auto *out_ptr = static_cast<float *>(expected->dataptr());

        for (uint64_t r = 0; r < rows; ++r)
        {
            for (uint64_t c = 0; c < cols; ++c)
            {
                float sum = 0.0f;
                for (uint64_t b = 0; b < batch; ++b)
                {
                    const auto in_offset =
                        static_cast<size_t>(b * in_strides[0] + r * in_strides[1] + c * in_strides[2]);
                    sum += ReadElement(dtype, in_data, in_offset);
                }
                const auto out_offset = static_cast<size_t>(r * out_strides[0] + c * out_strides[1]);
                out_ptr[out_offset] = sum;
            }
        }

        return expected;
    }

    std::shared_ptr<RealTensor> ComputeMulSumDim0NoKeep(const std::shared_ptr<RealTensor> &lhs,
                                                        const std::shared_ptr<RealTensor> &rhs, const Device device_cpu,
                                                        const uint64_t batch, const uint64_t rows, const uint64_t cols)
    {
        auto expected = RealTensor::Allocate({rows, cols}, DataType::FLOAT32, device_cpu);

        const auto lhs_cpu = lhs->to(device_cpu, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto rhs_cpu = rhs->to(device_cpu, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto dtype = lhs_cpu->dtype();
        const auto &in_strides = lhs_cpu->strides().strides();
        const auto &out_strides = expected->strides().strides();
        const void *lhs_data = lhs_cpu->dataptr();
        const void *rhs_data = rhs_cpu->dataptr();

        auto *out_ptr = static_cast<float *>(expected->dataptr());

        for (uint64_t r = 0; r < rows; ++r)
        {
            for (uint64_t c = 0; c < cols; ++c)
            {
                float sum = 0.0f;
                for (uint64_t b = 0; b < batch; ++b)
                {
                    const auto offset = static_cast<size_t>(b * in_strides[0] + r * in_strides[1] + c * in_strides[2]);
                    sum += ReadElement(dtype, lhs_data, offset) * ReadElement(dtype, rhs_data, offset);
                }
                const auto out_offset = static_cast<size_t>(r * out_strides[0] + c * out_strides[1]);
                out_ptr[out_offset] = sum;
            }
        }

        return expected;
    }

    void RunCase(const DataType dtype, const float tolerance, const uint64_t batch, const uint64_t rows,
                 const uint64_t cols, const float fill_low, const float fill_high, const uint32_t seed,
                 ExecutionBackend &execution_backend, allocator::AllocatorRegistry &allocator_registry)
    {
        constexpr Device DEVICE_GPU{DeviceType::GPU, 0};
        constexpr Device DEVICE_CPU{DeviceType::CPU, 0};
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        OpGraph init_graph{{}, {}};
        TraceTensor input = init_graph.createTensor({batch, rows, cols}, dtype, DEVICE_GPU, main_stream_desc, false);
        input.markRetained();
        FillUniform(init_graph, input, fill_low, fill_high, seed, main_stream_desc);
        init_graph.finalize();

        ExecutionPlan init_plan = ExecutionPlan::FromGraph(init_graph, {}, {});
        ApplyDefaultPasses(init_plan);

        Executor init_executor{init_plan, execution_backend, 0};
        init_executor.execute(allocator_registry);

        const auto input_real_opt = init_executor.getOutput(input);
        if (!input_real_opt.has_value())
        {
            throw std::runtime_error("Failed to get input tensor");
        }
        const auto &input_real = input_real_opt.value();

        const auto expected_keep = ComputeSumKeepLast(input_real, DEVICE_CPU, batch, rows, cols);
        const auto expected_dim0 = ComputeSumDim0NoKeep(input_real, DEVICE_CPU, batch, rows, cols);

        OpGraph graph{{{.name = "input", .tensor = input}}, {}};

        TraceTensor sum_keep = ReduceSum(graph, input, -1, true, DataType::FLOAT32, main_stream_desc);
        TraceTensor sum_dim0 = ReduceSum(graph, input, 0, false, DataType::FLOAT32, main_stream_desc);

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, {{.name = "input", .tensor = input_real}}, {});
        ApplyDefaultPasses(plan);

        const bool split_reduce_enabled = std::getenv("FBAMTRAIN_DISABLE_SPLIT_REDUCE") == nullptr;
        const bool expects_split_dim0_reduce = split_reduce_enabled && batch >= 1024 && rows * cols >= 64;
        if (expects_split_dim0_reduce && !PlanHasKernel(plan, SplitSumKernelName(dtype)))
        {
            throw std::runtime_error("Expected large dim-0 reduce-sum to use " + SplitSumKernelName(dtype));
        }

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto sum_keep_opt = executor.getOutput(sum_keep);
        const auto sum_dim0_opt = executor.getOutput(sum_dim0);
        if (!sum_keep_opt || !sum_dim0_opt)
        {
            throw std::runtime_error("Expected output tensors not found");
        }

        if (sum_keep_opt.value()->dtype() != DataType::FLOAT32 || sum_dim0_opt.value()->dtype() != DataType::FLOAT32)
        {
            throw std::runtime_error("ReduceSum expected FLOAT32 outputs");
        }

        testing::AssertSimilar(expected_keep, *sum_keep_opt, tolerance);
        testing::AssertSimilar(expected_dim0, *sum_dim0_opt, tolerance);
    }

    void RunMulReduceCase(const DataType dtype, const float tolerance, const uint64_t batch, const uint64_t rows,
                          const uint64_t cols, const float fill_low, const float fill_high, const uint32_t seed,
                          ExecutionBackend &execution_backend, allocator::AllocatorRegistry &allocator_registry)
    {
        constexpr Device DEVICE_GPU{DeviceType::GPU, 0};
        constexpr Device DEVICE_CPU{DeviceType::CPU, 0};
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        OpGraph init_graph{{}, {}};
        TraceTensor lhs = init_graph.createTensor({batch, rows, cols}, dtype, DEVICE_GPU, main_stream_desc, false);
        TraceTensor rhs = init_graph.createTensor({batch, rows, cols}, dtype, DEVICE_GPU, main_stream_desc, false);
        lhs.markRetained();
        rhs.markRetained();
        FillUniform(init_graph, lhs, fill_low, fill_high, seed, main_stream_desc);
        FillUniform(init_graph, rhs, fill_low, fill_high, seed + 1U, main_stream_desc);
        init_graph.finalize();

        ExecutionPlan init_plan = ExecutionPlan::FromGraph(init_graph, {}, {});
        ApplyDefaultPasses(init_plan);

        Executor init_executor{init_plan, execution_backend, 0};
        init_executor.execute(allocator_registry);

        const auto lhs_real_opt = init_executor.getOutput(lhs);
        const auto rhs_real_opt = init_executor.getOutput(rhs);
        if (!lhs_real_opt.has_value() || !rhs_real_opt.has_value())
        {
            throw std::runtime_error("Failed to get initialized mul_reduce inputs");
        }

        const auto expected_dim0 =
            ComputeMulSumDim0NoKeep(lhs_real_opt.value(), rhs_real_opt.value(), DEVICE_CPU, batch, rows, cols);

        OpGraph graph{{{.name = "lhs", .tensor = lhs}, {.name = "rhs", .tensor = rhs}}, {}};

        TraceTensor mul_tmp = graph.createTensor({batch, rows, cols}, dtype, DEVICE_GPU, main_stream_desc, false);
        graph.recordOperation(OperationEntry{.type = OpType::MUL,
                                             .inputs = {lhs, rhs},
                                             .outputs = {mul_tmp},
                                             .attributes = {},
                                             .gpu_stream_desc = main_stream_desc});
        TraceTensor sum_dim0 = ReduceSum(graph, mul_tmp, 0, false, DataType::FLOAT32, main_stream_desc);
        graph.deleteTensor(mul_tmp);
        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(
            graph, {{.name = "lhs", .tensor = lhs_real_opt.value()}, {.name = "rhs", .tensor = rhs_real_opt.value()}},
            {});
        ApplyDefaultPasses(plan);

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto sum_dim0_opt = executor.getOutput(sum_dim0);
        if (!sum_dim0_opt)
        {
            throw std::runtime_error("Expected mul_reduce output tensor not found");
        }
        if (sum_dim0_opt.value()->dtype() != DataType::FLOAT32)
        {
            throw std::runtime_error("MulReduce expected FLOAT32 output");
        }

        testing::AssertSimilar(expected_dim0, *sum_dim0_opt, tolerance);
    }
} // namespace

int main()
{
    constexpr uint64_t BATCH = 4;
    constexpr uint64_t ROWS = 5;
    constexpr uint64_t COLS = 6;
    constexpr uint64_t LARGE_BATCH = 1152;
    constexpr uint64_t LARGE_ROWS = 4;
    constexpr uint64_t LARGE_COLS = 64;
    constexpr float FILL_LOW = -0.5f;
    constexpr float FILL_HIGH = 0.5f;
    constexpr float LARGE_FILL_LOW = -0.1f;
    constexpr float LARGE_FILL_HIGH = 0.1f;
    constexpr uint32_t SEED = 4321u;

    ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
    auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();

    const std::array cases{
        std::pair{DataType::BFLOAT16, 2e-3f},
        std::pair{DataType::FLOAT16, 2e-3f},
        std::pair{DataType::FLOAT32, 1e-4f},
    };

    uint32_t seed = SEED;
    for (const auto &[dtype, tolerance] : cases)
    {
        RunCase(dtype, tolerance, BATCH, ROWS, COLS, FILL_LOW, FILL_HIGH, seed++, execution_backend,
                allocator_registry);
        const float large_tolerance = dtype == DataType::FLOAT32 ? 3e-4f : 1e-2f;
        RunCase(dtype, large_tolerance, LARGE_BATCH, LARGE_ROWS, LARGE_COLS, LARGE_FILL_LOW, LARGE_FILL_HIGH, seed++,
                execution_backend, allocator_registry);
        if (dtype != DataType::FLOAT32)
        {
            RunMulReduceCase(dtype, 1e-2f, LARGE_BATCH, LARGE_ROWS, LARGE_COLS, LARGE_FILL_LOW, LARGE_FILL_HIGH, seed++,
                             execution_backend, allocator_registry);
        }
    }

    return 0;
}
