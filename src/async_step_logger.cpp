#include "async_step_logger.h"

#include "logger.h"
#include "loss_accumulation_ring_buffer.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

AsyncStepLogger::AsyncStepLogger(const size_t seq_len, LossAccumulationRingBuffer *loss_accumulation_ring)
    : seq_len_(seq_len), loss_accumulation_ring_(loss_accumulation_ring),
      logger_thread_([this] { RunLoggerThread(); })
{
}

AsyncStepLogger::~AsyncStepLogger() { stop(); }

void AsyncStepLogger::enqueue(StepTimingLogRequest &&request)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_logs_.emplace_back(std::move(request));
    }
    cv_.notify_one();
}

void AsyncStepLogger::enqueueFramePhaseStart(FramePhaseStartLogRequest &&request)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_logs_.emplace_back(std::move(request));
    }
    cv_.notify_one();
}

void AsyncStepLogger::enqueueFrameProgressEvent(FrameProgressEventLogRequest &&request)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_logs_.emplace_back(std::move(request));
    }
    cv_.notify_one();
}

void AsyncStepLogger::enqueueFramePhaseEnd()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_logs_.emplace_back(FramePhaseEndLogRequest{});
    }
    cv_.notify_one();
}

void AsyncStepLogger::stop()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_requested_ = true;
    }
    cv_.notify_one();
    if (logger_thread_.joinable())
    {
        logger_thread_.join();
    }
}

void AsyncStepLogger::RunLoggerThread()
{
    while (true)
    {
        std::optional<AsyncLogRequest> request{};
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]()
                     { return stop_requested_ || !pending_logs_.empty(); });

            if (pending_logs_.empty())
            {
                if (stop_requested_)
                {
                    break;
                }
                continue;
            }

            request.emplace(std::move(pending_logs_.front()));
            pending_logs_.pop_front();
        }

        if (auto *phase_start = std::get_if<FramePhaseStartLogRequest>(&request.value()))
        {
            if (active_frame_phase_.has_value())
            {
                LOG(WARN) << "Received frame phase start while a phase is active; replacing previous phase.";
            }
            active_frame_phase_.emplace(phase_start->is_backward, phase_start->seq_len, phase_start->batch_size,
                                        std::move(phase_start->phase_start_event));
            continue;
        }

        if (auto *progress = std::get_if<FrameProgressEventLogRequest>(&request.value()))
        {
            if (!active_frame_phase_.has_value())
            {
                LOG(WARN) << "Received frame progress event without an active frame phase.";
                continue;
            }
            auto &phase = active_frame_phase_.value();
            while (!progress->completion_event.isComplete())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            const auto elapsed_seconds = progress->completion_event.elapsedMsSince(phase.phase_start_event) / 1000.0;
            const auto frames_processed = static_cast<double>((progress->completed_idx + 1) * phase.batch_size);
            const auto fps = elapsed_seconds > 0.0 ? frames_processed / elapsed_seconds : 0.0;
            if (phase.is_backward)
            {
                LOG(INFO) << "Backward sequence position " << progress->completed_idx << "/" << phase.seq_len
                          << " processed | FPS: " << fps;
            }
            else
            {
                LOG(INFO) << "Sequence position " << progress->completed_idx << "/" << phase.seq_len
                          << " processed | FPS: " << fps;
            }
            continue;
        }

        if (std::holds_alternative<FramePhaseEndLogRequest>(request.value()))
        {
            if (!active_frame_phase_.has_value())
            {
                LOG(WARN) << "Received frame phase end without an active frame phase.";
            }
            active_frame_phase_.reset();
            continue;
        }

        if (auto *step_request = std::get_if<StepTimingLogRequest>(&request.value()))
        {
            while (!step_request->step_end_event.isComplete())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            std::optional<float> loss_value{};
            if (loss_accumulation_ring_ != nullptr)
            {
                loss_value = loss_accumulation_ring_->ConsumeNextLoss();
            }

            const double step_time_sec = step_request->step_end_event.elapsedMsSince(step_request->step_start_event) / 1000.0;
            const double seq_len_per_sec = step_time_sec > 0.0 ? static_cast<double>(seq_len_) / step_time_sec : 0.0;
            if (step_request->dropped)
            {
                LOG(WARN) << "Dropped step " << step_request->step << " after " << step_time_sec
                          << " seconds; accumulated loss was discarded.";
                continue;
            }

            std::ostringstream step_log{};
            step_log << "Step " << step_request->step << " took " << step_time_sec
                     << " seconds; seqlen/time = " << seq_len_per_sec;
            if (loss_value.has_value())
            {
                step_log << "; loss/train = " << *loss_value;
            }
            LOG(INFO) << step_log.str();

            if (step_request->emit_metrics_json)
            {
                std::cout << std::setprecision(10) << "{\"step\":" << step_request->step
                          << ",\"step_time_sec\":" << step_time_sec;
                if (loss_value.has_value())
                {
                    std::cout << ",\"loss/train\":" << *loss_value;
                }
                std::cout
                          << "}\n"
                          << std::flush;
            }
        }
    }
}
