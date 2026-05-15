#include "dataset_cursor.h"

#include "logger.h"

#include <nlohmann/json.hpp>
#include <threadpark.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace
{
    constexpr uint32_t kDatasetCursorVersion = 1;

    void ThrowErrno(const std::string &action, const std::filesystem::path &path)
    {
        throw std::runtime_error(action + " failed for " + path.string() + ": " + std::strerror(errno));
    }

    void WriteAll(const int fd, const char *data, const size_t size, const std::filesystem::path &path)
    {
        size_t written = 0;
        while (written < size)
        {
            const ssize_t rc = ::write(fd, data + written, size - written);
            if (rc < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }
                ThrowErrno("write", path);
            }
            written += static_cast<size_t>(rc);
        }
    }

    void FsyncDirectory(const std::filesystem::path &path)
    {
        const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (fd < 0)
        {
            ThrowErrno("open directory", path);
        }
        if (::fsync(fd) != 0)
        {
            const int saved_errno = errno;
            ::close(fd);
            errno = saved_errno;
            ThrowErrno("fsync directory", path);
        }
        if (::close(fd) != 0)
        {
            ThrowErrno("close directory", path);
        }
    }

    void WriteDurableFile(const std::filesystem::path &path, const std::string &contents)
    {
        const auto parent = path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
        std::filesystem::create_directories(parent);

        const auto tmp_path = parent / (path.filename().string() + ".tmp." + std::to_string(::getpid()));
        const int fd = ::open(tmp_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH | O_CLOEXEC);
        if (fd < 0)
        {
            ThrowErrno("open", tmp_path);
        }

        bool fd_open = true;
        try
        {
            WriteAll(fd, contents.data(), contents.size(), tmp_path);
            if (::fsync(fd) != 0)
            {
                ThrowErrno("fsync", tmp_path);
            }
            const int close_result = ::close(fd);
            fd_open = false;
            if (close_result != 0)
            {
                ThrowErrno("close", tmp_path);
            }
            if (::rename(tmp_path.c_str(), path.c_str()) != 0)
            {
                ThrowErrno("rename", path);
            }
            FsyncDirectory(parent);
        }
        catch (...)
        {
            if (fd_open)
            {
                ::close(fd);
            }
            std::error_code ignored;
            std::filesystem::remove(tmp_path, ignored);
            throw;
        }
    }

    std::string Serialize(const fbamtrain::DatasetCursorSnapshot &snapshot)
    {
        return nlohmann::json{{"version", kDatasetCursorVersion},
                              {"step_count", snapshot.step_count},
                              {"rng_state", snapshot.rng_state},
                              {"batch_size", snapshot.batch_size},
                              {"sequence_length", snapshot.sequence_length}}
            .dump(2);
    }

    uint64_t RequireU64(const nlohmann::json &json, const char *key)
    {
        const auto it = json.find(key);
        if (it == json.end() || !it->is_number_unsigned())
        {
            throw std::runtime_error(std::string("Dataset cursor file is missing unsigned field: ") + key);
        }
        return it->get<uint64_t>();
    }
} // namespace

struct fbamtrain::DatasetCursorCommitter::Impl
{
    explicit Impl(std::string path) : file_path(std::move(path)), park_handle(tparkCreateHandle())
    {
        if (file_path.empty())
        {
            throw std::runtime_error("Dataset cursor path must be non-empty.");
        }
        if (park_handle == nullptr)
        {
            throw std::runtime_error("Failed to create dataset cursor threadpark handle.");
        }
        worker = std::thread([this] { workerLoop(); });
    }

    ~Impl()
    {
        stop.store(true, std::memory_order_release);
        tparkWake(park_handle);
        if (worker.joinable())
        {
            worker.join();
        }
        tparkDestroyHandle(park_handle);
    }

    void commit(const DatasetCursorSnapshot &snapshot)
    {
        uint64_t seq = sequence.load(std::memory_order_relaxed);
        for (;;)
        {
            while ((seq & 1U) != 0U)
            {
                std::this_thread::yield();
                seq = sequence.load(std::memory_order_acquire);
            }
            if (sequence.compare_exchange_weak(seq, seq + 1, std::memory_order_acq_rel,
                                               std::memory_order_acquire))
            {
                break;
            }
        }

        pending_step_count.store(snapshot.step_count, std::memory_order_relaxed);
        pending_rng_state.store(snapshot.rng_state, std::memory_order_relaxed);
        pending_batch_size.store(snapshot.batch_size, std::memory_order_relaxed);
        pending_sequence_length.store(snapshot.sequence_length, std::memory_order_relaxed);

        // Publish the payload with a release store. The wake below is only a doorbell; readers synchronize through
        // this generation value before trusting the payload atomics.
        sequence.store(seq + 2, std::memory_order_release);
        tparkWake(park_handle);
    }

    void workerLoop()
    {
        uint64_t last_seen_sequence = sequence.load(std::memory_order_acquire);
        for (;;)
        {
            DatasetCursorSnapshot snapshot{};
            if (tryLoadSnapshot(last_seen_sequence, snapshot))
            {
                try
                {
                    WriteDurableFile(file_path, Serialize(snapshot));
                }
                catch (const std::exception &ex)
                {
                    LOG(ERR) << "Failed to commit dataset cursor: " << ex.what();
                }
                continue;
            }
            if (stop.load(std::memory_order_acquire))
            {
                break;
            }

            tparkBeginPark(park_handle);
            if (sequence.load(std::memory_order_acquire) != last_seen_sequence ||
                stop.load(std::memory_order_acquire))
            {
                tparkEndPark(park_handle);
                continue;
            }
            tparkWait(park_handle, true);
            tparkEndPark(park_handle);

            // threadpark is only the sleep/wake primitive. If wake visibility beats payload visibility on this core,
            // acquire-load the generation until the publisher's release store becomes observable.
            for (size_t spin = 0; spin < 8192 &&
                                  sequence.load(std::memory_order_acquire) == last_seen_sequence &&
                                  !stop.load(std::memory_order_acquire);
                 ++spin)
            {
                std::this_thread::yield();
            }
        }
    }

    bool tryLoadSnapshot(uint64_t &last_seen_sequence, DatasetCursorSnapshot &snapshot) const
    {
        for (;;)
        {
            const uint64_t seq_before = sequence.load(std::memory_order_acquire);
            if (seq_before == last_seen_sequence)
            {
                return false;
            }
            if ((seq_before & 1U) != 0U)
            {
                std::this_thread::yield();
                continue;
            }

            DatasetCursorSnapshot candidate{
                .step_count = pending_step_count.load(std::memory_order_relaxed),
                .rng_state = pending_rng_state.load(std::memory_order_relaxed),
                .batch_size = pending_batch_size.load(std::memory_order_relaxed),
                .sequence_length = pending_sequence_length.load(std::memory_order_relaxed),
            };

            const uint64_t seq_after = sequence.load(std::memory_order_acquire);
            if (seq_before == seq_after)
            {
                last_seen_sequence = seq_after;
                snapshot = candidate;
                return true;
            }
        }
    }

    std::filesystem::path file_path;
    tpark_handle_t *park_handle{};
    std::thread worker{};
    std::atomic<uint64_t> sequence{0};
    std::atomic<uint64_t> pending_step_count{0};
    std::atomic<uint64_t> pending_rng_state{0};
    std::atomic<uint32_t> pending_batch_size{0};
    std::atomic<uint64_t> pending_sequence_length{0};
    std::atomic<bool> stop{false};
};

fbamtrain::DatasetIteratorState fbamtrain::DatasetCursorSnapshot::toDatasetIteratorState() const
{
    DatasetIteratorState state{};
    state.rng_state = rng_state;
    state.batch_size = batch_size;
    state.prefetch_buffer_index = 0;
    return state;
}

fbamtrain::DatasetCursorSnapshot
fbamtrain::DatasetCursorSnapshot::FromIterator(const uint64_t step_count,
                                               const RecordingDatasetIterator &iterator,
                                               const uint64_t sequence_length)
{
    const auto state = iterator.cursorCheckpoint();
    return DatasetCursorSnapshot{.step_count = step_count,
                                 .rng_state = state.rng_state,
                                 .batch_size = state.batch_size,
                                 .sequence_length = sequence_length};
}

fbamtrain::DatasetCursorCommitter::DatasetCursorCommitter(std::string file_path)
    : impl_(new Impl(std::move(file_path)))
{
}

fbamtrain::DatasetCursorCommitter::~DatasetCursorCommitter()
{
    delete impl_;
}

void fbamtrain::DatasetCursorCommitter::commit(const DatasetCursorSnapshot &snapshot)
{
    impl_->commit(snapshot);
}

std::optional<fbamtrain::DatasetCursorSnapshot>
fbamtrain::DatasetCursorCommitter::Load(const std::string &file_path)
{
    if (file_path.empty() || !std::filesystem::exists(file_path))
    {
        return std::nullopt;
    }

    std::ifstream stream(file_path);
    if (!stream.is_open())
    {
        throw std::runtime_error("Failed to open dataset cursor file: " + file_path);
    }
    const auto json = nlohmann::json::parse(stream, nullptr, false);
    if (json.is_discarded() || !json.is_object())
    {
        throw std::runtime_error("Failed to parse dataset cursor file: " + file_path);
    }
    const auto version = RequireU64(json, "version");
    if (version != kDatasetCursorVersion)
    {
        throw std::runtime_error("Unsupported dataset cursor version in " + file_path + ": " +
                                 std::to_string(version));
    }

    const auto batch_size = RequireU64(json, "batch_size");
    if (batch_size > UINT32_MAX)
    {
        throw std::runtime_error("Dataset cursor batch_size exceeds uint32 range.");
    }

    return DatasetCursorSnapshot{.step_count = RequireU64(json, "step_count"),
                                 .rng_state = RequireU64(json, "rng_state"),
                                 .batch_size = static_cast<uint32_t>(batch_size),
                                 .sequence_length = RequireU64(json, "sequence_length")};
}
