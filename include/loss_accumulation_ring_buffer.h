#pragma once

#include <execution_backend.h>

#include <cstddef>
#include <memory>

namespace pi::tensorlib
{
    class RealTensor;
    class ExecutionBackend;
    namespace allocator
    {
        class AllocatorRegistry;
    } // namespace allocator
} // namespace pi::tensorlib

class LossAccumulationRingBuffer final
{
  public:
    static constexpr size_t kBufferSize = 128;

    LossAccumulationRingBuffer(const pi::tensorlib::Device &device_gpu, const pi::tensorlib::Device &device_cpu,
                               int device_ordinal, int loss_accum_stream_id, int loss_copy_stream_id,
                               pi::tensorlib::ExecutionBackend &execution_backend,
                               const pi::tensorlib::allocator::AllocatorRegistry &allocator_registry);

    ~LossAccumulationRingBuffer();

    LossAccumulationRingBuffer(const LossAccumulationRingBuffer &) = delete;
    LossAccumulationRingBuffer &operator=(const LossAccumulationRingBuffer &) = delete;
    LossAccumulationRingBuffer(LossAccumulationRingBuffer &&) noexcept;
    LossAccumulationRingBuffer &operator=(LossAccumulationRingBuffer &&) noexcept;

    void BeginStep();

    void AccumulateMicroStepLoss(const std::shared_ptr<pi::tensorlib::RealTensor> &loss_mean_tensor,
                                 pi::tensorlib::gpustream::GpuStream ce_stream);

    void FinalizeStep();

    [[nodiscard]] float ConsumeNextLoss();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_{};
};
