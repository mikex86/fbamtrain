#pragma once

#include <execution_backend.h>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>
#include <variant>

class LossAccumulationRingBuffer;

struct StepTimingLogRequest
{
    uint64_t step{};
    bool emit_metrics_json{};
    bool dropped{};
    pi::tensorlib::GpuEvent step_start_event;
    pi::tensorlib::GpuEvent step_end_event;

    StepTimingLogRequest(const uint64_t step_num, const bool emit_json, pi::tensorlib::GpuEvent &&start_event,
                         pi::tensorlib::GpuEvent &&end_event, const bool dropped_step = false)
        : step(step_num), emit_metrics_json(emit_json), dropped(dropped_step), step_start_event(std::move(start_event)),
          step_end_event(std::move(end_event))
    {
    }

    StepTimingLogRequest(StepTimingLogRequest &&) noexcept = default;
    StepTimingLogRequest &operator=(StepTimingLogRequest &&) noexcept = default;
    StepTimingLogRequest(const StepTimingLogRequest &) = delete;
    StepTimingLogRequest &operator=(const StepTimingLogRequest &) = delete;
};

struct FramePhaseStartLogRequest
{
    bool is_backward{};
    size_t seq_len{};
    uint32_t batch_size{};
    pi::tensorlib::GpuEvent phase_start_event;

    FramePhaseStartLogRequest(const bool backward, const size_t sequence_len, const uint32_t batch,
                              pi::tensorlib::GpuEvent &&start_event)
        : is_backward(backward), seq_len(sequence_len), batch_size(batch), phase_start_event(std::move(start_event))
    {
    }

    FramePhaseStartLogRequest(FramePhaseStartLogRequest &&) noexcept = default;
    FramePhaseStartLogRequest &operator=(FramePhaseStartLogRequest &&) noexcept = default;
    FramePhaseStartLogRequest(const FramePhaseStartLogRequest &) = delete;
    FramePhaseStartLogRequest &operator=(const FramePhaseStartLogRequest &) = delete;
};

struct FrameProgressEventLogRequest
{
    size_t completed_idx{};
    pi::tensorlib::GpuEvent completion_event;

    FrameProgressEventLogRequest(const size_t idx, pi::tensorlib::GpuEvent &&event)
        : completed_idx(idx), completion_event(std::move(event))
    {
    }

    FrameProgressEventLogRequest(FrameProgressEventLogRequest &&) noexcept = default;
    FrameProgressEventLogRequest &operator=(FrameProgressEventLogRequest &&) noexcept = default;
    FrameProgressEventLogRequest(const FrameProgressEventLogRequest &) = delete;
    FrameProgressEventLogRequest &operator=(const FrameProgressEventLogRequest &) = delete;
};

struct FramePhaseEndLogRequest
{};

using AsyncLogRequest =
    std::variant<StepTimingLogRequest, FramePhaseStartLogRequest, FrameProgressEventLogRequest, FramePhaseEndLogRequest>;

class AsyncStepLogger final
{
  public:
    explicit AsyncStepLogger(size_t seq_len, LossAccumulationRingBuffer *loss_accumulation_ring);

    ~AsyncStepLogger();

    void enqueue(StepTimingLogRequest &&request);

    void enqueueFramePhaseStart(FramePhaseStartLogRequest &&request);

    void enqueueFrameProgressEvent(FrameProgressEventLogRequest &&request);

    void enqueueFramePhaseEnd();

    void stop();

  private:
    void RunLoggerThread();

    struct FramePhaseState
    {
        bool is_backward{};
        size_t seq_len{};
        uint32_t batch_size{};
        pi::tensorlib::GpuEvent phase_start_event;

        FramePhaseState(const bool backward, const size_t sequence_len, const uint32_t batch,
                        pi::tensorlib::GpuEvent &&start_event)
            : is_backward(backward), seq_len(sequence_len), batch_size(batch), phase_start_event(std::move(start_event))
        {
        }

        FramePhaseState(FramePhaseState &&) noexcept = default;
        FramePhaseState &operator=(FramePhaseState &&) noexcept = default;
        FramePhaseState(const FramePhaseState &) = delete;
        FramePhaseState &operator=(const FramePhaseState &) = delete;
    };

    std::mutex mutex_{};
    std::condition_variable cv_{};
    std::deque<AsyncLogRequest> pending_logs_{};
    std::optional<FramePhaseState> active_frame_phase_{};
    size_t seq_len_{};
    LossAccumulationRingBuffer *loss_accumulation_ring_{nullptr};
    bool stop_requested_{false};
    std::thread logger_thread_{};
};
