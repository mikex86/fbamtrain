#include "loss_accumulation_ring_buffer.h"

#include "device_copy.h"
#include "functional.h"
#include "passes.h"

#include <allocator.h>
#include <executor.h>
#include <op_graph.h>

#include <chrono>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{
    void ApplyLossAccumulationPasses(pi::tensorlib::ExecutionPlan &execution_plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
        pi::tensorlib::passes::Transform(execution_plan, passes);
    }
} // namespace

struct LossAccumulationRingBuffer::Impl
{
    struct SlotState
    {
        bool pending{false};
        std::shared_ptr<pi::tensorlib::GpuEvent> ready_event{};
    };

    Impl(const pi::tensorlib::Device &device_gpu, const pi::tensorlib::Device &device_cpu, const int device_ordinal,
         const int loss_accum_stream_id, const int loss_copy_stream_id,
         pi::tensorlib::ExecutionBackend &execution_backend,
         const pi::tensorlib::allocator::AllocatorRegistry &allocator_registry)
        : device_gpu_(device_gpu), device_cpu_(device_cpu), device_ordinal_(device_ordinal),
          execution_backend_(execution_backend), allocator_registry_(allocator_registry),
          loss_accum_stream_desc_(
              pi::tensorlib::GpuStreamDescriptor{pi::tensorlib::StreamKind::Compute, loss_accum_stream_id})
    {
        if (LossAccumulationRingBuffer::kBufferSize < 2)
        {
            throw std::runtime_error("Loss accumulation ring buffer size must be at least 2.");
        }

        const auto stream_bundle = pi::tensorlib::ExecutionBackend::GetStreamBundle(device_gpu_);
        if (!stream_bundle)
        {
            throw std::runtime_error("Failed to get stream bundle for loss accumulation ring buffer.");
        }
        loss_accum_stream_ = stream_bundle->getComputeStream(loss_accum_stream_id);
        loss_copy_stream_ = stream_bundle->getComputeStream(loss_copy_stream_id);

        slot_states_.resize(LossAccumulationRingBuffer::kBufferSize);
        InitializeStorage();

        zero_plan_ = BuildZeroPlan(gpu_slots_.front());
        add_plan_ = BuildAddPlan(gpu_slots_.front(), gpu_slots_[1]);
    }

    void BeginStep()
    {
        size_t slot_idx = 0;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (active_slot_idx_.has_value())
            {
                throw std::runtime_error("Loss accumulation step already active.");
            }
            slot_idx = write_index_.load(std::memory_order_acquire);
            auto &slot = slot_states_[slot_idx];
            if (slot.pending)
            {
                throw std::runtime_error("Loss accumulation ring buffer overflow (size=" +
                                         std::to_string(LossAccumulationRingBuffer::kBufferSize) +
                                         "). Async logger is too slow. Increase the ring size constant.");
            }
            slot.pending = true;
            slot.ready_event.reset();
            active_slot_idx_ = slot_idx;
        }

        zero_plan_.updateInputDescriptors({pi::tensorlib::GraphExecutionInputDescriptor{
            .name = "slot", .tensor = gpu_slots_[slot_idx]}});
        pi::tensorlib::Executor zero_executor{zero_plan_, execution_backend_, device_ordinal_};
        zero_executor.execute(allocator_registry_, false);
    }

    void AccumulateMicroStepLoss(const std::shared_ptr<pi::tensorlib::RealTensor> &loss_mean_tensor,
                                 const pi::tensorlib::gpustream::GpuStream ce_stream)
    {
        size_t slot_idx = 0;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (!active_slot_idx_.has_value())
            {
                throw std::runtime_error("Loss accumulation step is not active.");
            }
            slot_idx = *active_slot_idx_;
        }

        // Gate accumulation on CE completion via an explicit event to avoid transient
        // stream-to-stream wait event setup in GpuStreamWaitFor.
        auto ce_done_event = pi::tensorlib::ExecutionBackend::CreateEvent(device_gpu_);
        ce_done_event.record(ce_stream);
        pi::tensorlib::ExecutionBackend::GpuStreamWaitForEvent(ce_done_event, loss_accum_stream_);

        add_plan_.updateInputDescriptors(
            {pi::tensorlib::GraphExecutionInputDescriptor{.name = "slot", .tensor = gpu_slots_[slot_idx]},
             pi::tensorlib::GraphExecutionInputDescriptor{.name = "loss_mean", .tensor = loss_mean_tensor}});
        pi::tensorlib::Executor add_executor{add_plan_, execution_backend_, device_ordinal_};
        add_executor.execute(allocator_registry_, false);
    }

    void FinalizeStep()
    {
        size_t slot_idx = 0;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (!active_slot_idx_.has_value())
            {
                throw std::runtime_error("Loss accumulation step is not active.");
            }
            slot_idx = *active_slot_idx_;
            active_slot_idx_.reset();
        }

        auto ready_event = pi::tensorlib::ExecutionBackend::CreateEvent(device_gpu_);
        ready_event.record(loss_accum_stream_);

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            auto &slot = slot_states_[slot_idx];
            slot.ready_event = std::make_shared<pi::tensorlib::GpuEvent>(std::move(ready_event));
            const size_t next_slot = (slot_idx + 1) % LossAccumulationRingBuffer::kBufferSize;
            write_index_.store(next_slot, std::memory_order_release);
        }
    }

    [[nodiscard]] float ConsumeNextLoss()
    {
        while (true)
        {
            size_t slot_idx = LossAccumulationRingBuffer::kBufferSize;
            std::shared_ptr<pi::tensorlib::GpuEvent> ready_event{};
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                const size_t candidate_idx = read_index_.load(std::memory_order_acquire);
                const auto &slot = slot_states_[candidate_idx];
                if (slot.pending && slot.ready_event)
                {
                    slot_idx = candidate_idx;
                    ready_event = slot.ready_event;
                }
            }
            if (slot_idx == LossAccumulationRingBuffer::kBufferSize)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            // Gate the copy stream on the per-slot completion event recorded after the last accumulation.
            pi::tensorlib::ExecutionBackend::GpuStreamWaitForEvent(*ready_event, loss_copy_stream_);
            pi::tensorlib::internal::device_copy::PerformDeviceCopy(gpu_slots_[slot_idx], cpu_staging_, loss_copy_stream_);
            auto copy_done_event = pi::tensorlib::ExecutionBackend::CreateEvent(device_gpu_);
            copy_done_event.record(loss_copy_stream_);
            copy_done_event.synchronize();
            const float loss_value = static_cast<const float *>(cpu_staging_->dataptr())[0];

            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                auto &slot = slot_states_[slot_idx];
                if (!(slot.pending && slot.ready_event))
                {
                    throw std::runtime_error("Loss ring slot state changed unexpectedly during consumption.");
                }
                slot.pending = false;
                slot.ready_event.reset();
                read_index_.store((slot_idx + 1) % LossAccumulationRingBuffer::kBufferSize, std::memory_order_release);
            }
            return loss_value;
        }
    }

  private:
    void InitializeStorage()
    {
        pi::tensorlib::OpGraph init_graph{{}, {}};
        std::vector<pi::tensorlib::TraceTensor> gpu_slot_traces{};
        gpu_slot_traces.reserve(LossAccumulationRingBuffer::kBufferSize);
        for (size_t idx = 0; idx < LossAccumulationRingBuffer::kBufferSize; ++idx)
        {
            auto slot_trace =
                init_graph.createTensor({1}, pi::tensorlib::DataType::FLOAT32, device_gpu_, loss_accum_stream_desc_,
                                        false);
            slot_trace.markRetained();
            pi::tensorlib::FillZeros(init_graph, slot_trace, loss_accum_stream_desc_);
            gpu_slot_traces.push_back(slot_trace);
        }

        auto cpu_staging_trace = init_graph.createTensor({1}, pi::tensorlib::DataType::FLOAT32, device_cpu_,
                                                         pi::tensorlib::GpuStreamDescriptors::Main, true);
        cpu_staging_trace.markRetained();
        init_graph.finalize();

        auto init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
        ApplyLossAccumulationPasses(init_plan);
        pi::tensorlib::Executor init_executor{init_plan, execution_backend_, device_ordinal_};
        init_executor.execute(allocator_registry_, false);
        init_executor.await();

        gpu_slots_.reserve(LossAccumulationRingBuffer::kBufferSize);
        for (const auto &slot_trace : gpu_slot_traces)
        {
            const auto slot_opt = init_executor.getOutput(slot_trace);
            if (!slot_opt.has_value())
            {
                throw std::runtime_error("Failed to initialize GPU loss ring slot tensor.");
            }
            gpu_slots_.push_back(slot_opt.value());
        }

        const auto cpu_staging_opt = init_executor.getOutput(cpu_staging_trace);
        if (!cpu_staging_opt.has_value())
        {
            throw std::runtime_error("Failed to initialize CPU loss staging tensor.");
        }
        cpu_staging_ = cpu_staging_opt.value();
    }

    [[nodiscard]] pi::tensorlib::ExecutionPlan
    BuildZeroPlan(const std::shared_ptr<pi::tensorlib::RealTensor> &example_slot) const
    {
        auto slot_input = pi::tensorlib::TraceTensor::Create({1}, pi::tensorlib::DataType::FLOAT32, device_gpu_,
                                                              loss_accum_stream_desc_, false);
        slot_input.markRetained();
        pi::tensorlib::OpGraph graph{{pi::tensorlib::GraphInputDescriptor{.name = "slot", .tensor = slot_input}}, {}};
        pi::tensorlib::FillZeros(graph, slot_input, loss_accum_stream_desc_);
        graph.finalize();

        auto plan = pi::tensorlib::ExecutionPlan::FromGraph(
            graph, {pi::tensorlib::GraphExecutionInputDescriptor{.name = "slot", .tensor = example_slot}}, {});
        ApplyLossAccumulationPasses(plan);
        return plan;
    }

    [[nodiscard]] pi::tensorlib::ExecutionPlan
    BuildAddPlan(const std::shared_ptr<pi::tensorlib::RealTensor> &example_slot,
                 const std::shared_ptr<pi::tensorlib::RealTensor> &example_loss_input) const
    {
        auto slot_input = pi::tensorlib::TraceTensor::Create({1}, pi::tensorlib::DataType::FLOAT32, device_gpu_,
                                                              loss_accum_stream_desc_, false);
        auto loss_input = pi::tensorlib::TraceTensor::Create({1}, pi::tensorlib::DataType::FLOAT32, device_gpu_,
                                                              loss_accum_stream_desc_, false);
        slot_input.markRetained();
        loss_input.markRetained();
        pi::tensorlib::OpGraph graph{{pi::tensorlib::GraphInputDescriptor{.name = "slot", .tensor = slot_input},
                                      pi::tensorlib::GraphInputDescriptor{.name = "loss_mean", .tensor = loss_input}},
                                     {}};
        pi::tensorlib::InplaceAdd(graph, slot_input, loss_input, loss_accum_stream_desc_);
        graph.finalize();

        auto plan = pi::tensorlib::ExecutionPlan::FromGraph(
            graph,
            {pi::tensorlib::GraphExecutionInputDescriptor{.name = "slot", .tensor = example_slot},
             pi::tensorlib::GraphExecutionInputDescriptor{.name = "loss_mean", .tensor = example_loss_input}},
            {});
        ApplyLossAccumulationPasses(plan);
        return plan;
    }

    pi::tensorlib::Device device_gpu_{};
    pi::tensorlib::Device device_cpu_{};
    int device_ordinal_{-1};
    pi::tensorlib::ExecutionBackend &execution_backend_;
    const pi::tensorlib::allocator::AllocatorRegistry &allocator_registry_;
    pi::tensorlib::GpuStreamDescriptor loss_accum_stream_desc_{};
    pi::tensorlib::gpustream::GpuStream loss_accum_stream_{};
    pi::tensorlib::gpustream::GpuStream loss_copy_stream_{};

    std::shared_ptr<pi::tensorlib::RealTensor> cpu_staging_{};
    std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> gpu_slots_{};
    std::vector<SlotState> slot_states_{};

    std::atomic<size_t> write_index_{0};
    std::atomic<size_t> read_index_{0};
    std::optional<size_t> active_slot_idx_{};
    std::mutex state_mutex_{};

    pi::tensorlib::ExecutionPlan zero_plan_{};
    pi::tensorlib::ExecutionPlan add_plan_{};
};

LossAccumulationRingBuffer::LossAccumulationRingBuffer(
    const pi::tensorlib::Device &device_gpu, const pi::tensorlib::Device &device_cpu, const int device_ordinal,
    const int loss_accum_stream_id, const int loss_copy_stream_id, pi::tensorlib::ExecutionBackend &execution_backend,
    const pi::tensorlib::allocator::AllocatorRegistry &allocator_registry)
    : impl_(std::make_unique<Impl>(device_gpu, device_cpu, device_ordinal, loss_accum_stream_id, loss_copy_stream_id,
                                   execution_backend, allocator_registry))
{
}

LossAccumulationRingBuffer::~LossAccumulationRingBuffer() = default;

LossAccumulationRingBuffer::LossAccumulationRingBuffer(LossAccumulationRingBuffer &&) noexcept = default;

LossAccumulationRingBuffer &LossAccumulationRingBuffer::operator=(LossAccumulationRingBuffer &&) noexcept = default;

void LossAccumulationRingBuffer::BeginStep() { impl_->BeginStep(); }

void LossAccumulationRingBuffer::AccumulateMicroStepLoss(
    const std::shared_ptr<pi::tensorlib::RealTensor> &loss_mean_tensor, const pi::tensorlib::gpustream::GpuStream ce_stream)
{
    impl_->AccumulateMicroStepLoss(loss_mean_tensor, ce_stream);
}

void LossAccumulationRingBuffer::FinalizeStep() { impl_->FinalizeStep(); }

float LossAccumulationRingBuffer::ConsumeNextLoss() { return impl_->ConsumeNextLoss(); }
