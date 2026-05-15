#include "dataset_iterator.h"

#include "random.h"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <termstreamxz/termstream.h>
#include <thread>
#include <utility>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <stdexcept>
#include <vector>

struct fbamtrain::RecordingDatasetInternalState
{
    struct StreamBuffer
    {
        struct PooledStream
        {
            size_t file_index{};
            file_input_stream input;
            TermInflateStream inflate;

            PooledStream(const std::string &path, size_t index) : file_index(index), input(path), inflate(input) {}
        };

        std::vector<std::vector<std::unique_ptr<PooledStream>>> pool{};
        std::vector<std::unique_ptr<PooledStream>> in_use{};

        void initialize(size_t num_files)
        {
            pool.clear();
            pool.resize(num_files);
            in_use.clear();
        }

        void releaseInUse()
        {
            for (auto &handle : in_use)
            {
                pool[handle->file_index].push_back(std::move(handle));
            }
            in_use.clear();
        }

        TermInflateStream *acquire(size_t file_index, const std::string &path)
        {
            auto &bucket = pool[file_index];
            std::unique_ptr<PooledStream> handle{};
            if (!bucket.empty())
            {
                handle = std::move(bucket.back());
                bucket.pop_back();
            }
            else
            {
                handle = std::make_unique<PooledStream>(path, file_index);
            }
            TermInflateStream *stream = &handle->inflate;
            in_use.push_back(std::move(handle));
            return stream;
        }
    };

    std::vector<std::string> recording_file_paths{};
    std::vector<uint64_t> recording_total_frames{};
    StreamBuffer buffers[2]{};

    std::vector<TermInflateStream *> prepared_batch_streams{};
    std::vector<TermInflateStream *> staging_batch_streams{};

    uint32_t batch_size{};
    uint64_t sequence_length{};

    // rng state (owned by background worker)
    Random random{0};
    uint64_t prepared_rng_state{0};
    std::atomic<uint64_t> cursor_rng_state{0};

    std::vector<uint32_t> prepared_indices{};
    std::vector<uint64_t> prepared_start_positions{};

    // Double-buffer state
    size_t prefetch_buffer_index{0};

    mutable std::mutex mutex{};
    mutable std::condition_variable cv_prefetch_request{};
    mutable std::condition_variable cv_prefetch_ready{};

    bool stop_worker{false};
    bool prefetch_requested{false};
    bool prefetch_ready{false};

    std::thread worker{};

    RecordingDatasetInternalState(const uint32_t batch_size_, const uint64_t sequence_length_,
                                  const uint64_t seed)
        : batch_size(batch_size_), sequence_length(sequence_length_), random(seed)
    {
        cursor_rng_state.store(seed, std::memory_order_release);
        prepared_batch_streams.resize(batch_size);
        staging_batch_streams.resize(batch_size);
    }

    ~RecordingDatasetInternalState() { stopWorkerAndJoin(); }

    void startWorker()
    {
        worker = std::thread([this] { workerLoop(); });
    }

    void stopWorkerAndJoin()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop_worker = true;
            prefetch_requested = false;
            cv_prefetch_request.notify_all();
            cv_prefetch_ready.notify_all();
        }
        if (worker.joinable())
        {
            worker.join();
        }
    }

    void requestPrefetch()
    {
        std::unique_lock lock(mutex);
        if (stop_worker)
        {
            return;
        }
        prefetch_requested = true;
        lock.unlock();
        cv_prefetch_request.notify_one();
    }

  private:
    void workerLoop()
    {
        std::unique_lock lock(mutex);
        while (true)
        {
            cv_prefetch_request.wait(lock, [this] { return stop_worker || prefetch_requested; });
            if (stop_worker)
            {
                break;
            }

            prefetch_requested = false;
            const size_t buffer_idx = prefetch_buffer_index;
            auto &buffer = buffers[buffer_idx];
            buffer.releaseInUse();

            const size_t num_streams = recording_file_paths.size();
            if (num_streams == 0)
            {
                throw std::runtime_error("No recordings available for dataset iterator.");
            }

            std::vector<uint32_t> selected_indices(batch_size);
            std::vector<uint64_t> start_positions(batch_size);

            for (uint32_t i = 0; i < batch_size; ++i)
            {
                const auto stream_idx = static_cast<uint32_t>(random.randint(0, static_cast<int64_t>(num_streams)));
                selected_indices[i] = stream_idx;

                const auto total_frames = static_cast<int64_t>(recording_total_frames[stream_idx]);
                const auto max_start = std::max<int64_t>(0, total_frames - static_cast<int64_t>(sequence_length));
                const uint64_t start_position =
                    total_frames <= 0 ? 0 : static_cast<uint64_t>(random.randint(0, max_start + 1));
                start_positions[i] = start_position;
            }

            lock.unlock();
            for (uint32_t i = 0; i < batch_size; ++i)
            {
                const size_t file_index = selected_indices[i];
                TermInflateStream *stream = buffer.acquire(file_index, recording_file_paths[file_index]);
                stream->seek(start_positions[i]);
                staging_batch_streams[i] = stream;
            }
            lock.lock();

            prepared_batch_streams = staging_batch_streams;
            prepared_indices = selected_indices;
            prepared_start_positions = start_positions;
            prepared_rng_state = random.getState();
            prefetch_ready = true;

            prefetch_buffer_index = 1 - buffer_idx;

            cv_prefetch_ready.notify_one();
        }
    }
};

fbamtrain::TerminalFrameCopy fbamtrain::TerminalFrameCopy::FromFrame(const TerminalFrame &frame)
{
    std::vector<Cell> cells{};
    cells.resize(frame.width * frame.height);
    for (size_t y = 0; y < frame.height; ++y)
    {
        for (size_t x = 0; x < frame.width; ++x)
        {
            cells[y * frame.width + x] = frame.cells[y * frame.width + x];
        }
    }
    return TerminalFrameCopy{frame.width, frame.height, cells};
}

fbamtrain::RecordingDatasetIterator::RecordingDatasetIterator(const std::string &recordings_folder_path,
                                                              const uint32_t batch_size, const uint64_t sequence_length,
                                                              const uint64_t seed)
    : internal_state_(new RecordingDatasetInternalState(batch_size, sequence_length, seed)), batch_size_(batch_size),
      sequence_length_(sequence_length), seed_(seed)
{
    if (!std::filesystem::exists(recordings_folder_path) || !std::filesystem::is_directory(recordings_folder_path))
    {
        throw std::runtime_error("Recordings folder path does not exist or is not a directory: " +
                                 recordings_folder_path);
    }
    std::vector<std::filesystem::path> texz_files{};
    for (const auto &entry : std::filesystem::directory_iterator(recordings_folder_path))
    {
        if (!entry.is_directory() && entry.path().extension() == ".texz")
        {
            texz_files.push_back(entry.path());
        }
    }
    std::ranges::sort(texz_files);
    if (texz_files.empty())
    {
        throw std::runtime_error("No .texz recordings found in folder: " + recordings_folder_path);
    }

    std::vector<std::string> file_paths{};
    file_paths.reserve(texz_files.size());
    for (const auto &path : texz_files)
    {
        file_paths.push_back(path.string());
    }

    internal_state_->recording_file_paths = file_paths;
    internal_state_->recording_total_frames.clear();
    internal_state_->recording_total_frames.reserve(file_paths.size());
    for (const auto &path : file_paths)
    {
        file_input_stream input(path);
        TermInflateStream inflate(input);
        internal_state_->recording_total_frames.push_back(inflate.getTotalNumFrames());
    }
    for (auto &buffer : internal_state_->buffers)
    {
        buffer.initialize(file_paths.size());
    }

    internal_state_->startWorker();
    internal_state_->requestPrefetch();
}

fbamtrain::BatchIterator fbamtrain::RecordingDatasetIterator::nextBatch() const
{
    std::unique_lock lock(internal_state_->mutex);
    internal_state_->cv_prefetch_ready.wait(lock, [state = internal_state_]
                                            { return state->prefetch_ready || state->stop_worker; });

    if (!internal_state_->prefetch_ready)
    {
        return {};
    }

    auto batch_streams = internal_state_->prepared_batch_streams;
    auto batch_start_positions = internal_state_->prepared_start_positions;
    internal_state_->cursor_rng_state.store(internal_state_->prepared_rng_state, std::memory_order_release);
    internal_state_->prefetch_ready = false;
    lock.unlock();

    internal_state_->requestPrefetch();

    return BatchIterator{std::move(batch_streams), std::move(batch_start_positions)};
}

fbamtrain::DatasetIteratorState fbamtrain::RecordingDatasetIterator::cursorCheckpoint() const
{
    DatasetIteratorState state{};
    state.rng_state = internal_state_->cursor_rng_state.load(std::memory_order_acquire);
    state.batch_size = batch_size_;
    state.prefetch_buffer_index = 0;
    return state;
}

fbamtrain::DatasetIteratorState fbamtrain::RecordingDatasetIterator::checkpoint() const
{
    std::unique_lock lock(internal_state_->mutex);
    internal_state_->cv_prefetch_ready.wait(lock, [state = internal_state_]
                                            { return state->prefetch_ready || state->stop_worker; });
    if (!internal_state_->prefetch_ready)
    {
        throw std::runtime_error("Dataset iterator has no prefetched batch to checkpoint.");
    }

    DatasetIteratorState state{};
    state.rng_state = internal_state_->prepared_rng_state;
    state.batch_size = batch_size_;
    state.prefetch_buffer_index = static_cast<uint32_t>(internal_state_->prefetch_buffer_index);
    state.prepared_indices = internal_state_->prepared_indices;
    state.prepared_start_positions = internal_state_->prepared_start_positions;
    return state;
}

void fbamtrain::RecordingDatasetIterator::restore(const DatasetIteratorState &state)
{
    if (state.batch_size != batch_size_)
    {
        throw std::runtime_error("Checkpoint batch size does not match dataset iterator batch size.");
    }
    if (state.prefetch_buffer_index > 1)
    {
        throw std::runtime_error("Checkpoint prefetch_buffer_index must be 0 or 1.");
    }

    const bool has_prefetch_indices = !state.prepared_indices.empty();
    const bool has_prefetch_starts = !state.prepared_start_positions.empty();
    if (has_prefetch_indices != has_prefetch_starts)
    {
        throw std::runtime_error("Checkpoint prefetch state is incomplete.");
    }

    const bool has_prefetch_state = has_prefetch_indices && has_prefetch_starts;
    if (has_prefetch_state)
    {
        if (state.prepared_indices.size() != batch_size_)
        {
            throw std::runtime_error("Checkpoint prepared_indices size mismatch.");
        }
        if (state.prepared_start_positions.size() != batch_size_)
        {
            throw std::runtime_error("Checkpoint prepared_start_positions size mismatch.");
        }
    }

    internal_state_->stopWorkerAndJoin();

    {
        std::lock_guard<std::mutex> lock(internal_state_->mutex);
        internal_state_->stop_worker = false;
        internal_state_->prefetch_requested = false;
        internal_state_->prefetch_ready = false;
        internal_state_->prefetch_buffer_index = state.prefetch_buffer_index;
        internal_state_->prepared_indices = state.prepared_indices;
        internal_state_->prepared_start_positions = state.prepared_start_positions;
        internal_state_->prepared_rng_state = state.rng_state;
        internal_state_->cursor_rng_state.store(state.rng_state, std::memory_order_release);
        internal_state_->random.setState(state.rng_state);

        internal_state_->buffers[0].releaseInUse();
        internal_state_->buffers[1].releaseInUse();

        if (!has_prefetch_state)
        {
            std::fill(internal_state_->prepared_batch_streams.begin(), internal_state_->prepared_batch_streams.end(),
                      nullptr);
            std::fill(internal_state_->staging_batch_streams.begin(), internal_state_->staging_batch_streams.end(),
                      nullptr);
        }
    }

    if (!has_prefetch_state)
    {
        // Compatibility path for checkpoints that only store RNG state.
        // Request a fresh prefetch so the next nextBatch() call can proceed.
        internal_state_->startWorker();
        internal_state_->requestPrefetch();
        return;
    }

    const size_t prepared_buffer_index = 1 - internal_state_->prefetch_buffer_index;
    auto &buffer = internal_state_->buffers[prepared_buffer_index];

    {
        std::lock_guard<std::mutex> lock(internal_state_->mutex);
        for (uint32_t i = 0; i < batch_size_; ++i)
        {
            const uint32_t file_index = internal_state_->prepared_indices[i];
            if (file_index >= internal_state_->recording_file_paths.size())
            {
                throw std::runtime_error("Checkpoint file index out of range for dataset iterator.");
            }
            TermInflateStream *stream = buffer.acquire(file_index, internal_state_->recording_file_paths[file_index]);
            stream->seek(internal_state_->prepared_start_positions[i]);
            internal_state_->staging_batch_streams[i] = stream;
        }
        internal_state_->prepared_batch_streams = internal_state_->staging_batch_streams;
        internal_state_->prefetch_ready = true;
    }

    internal_state_->startWorker();
}

fbamtrain::RecordingDatasetIterator::~RecordingDatasetIterator() { delete internal_state_; }
