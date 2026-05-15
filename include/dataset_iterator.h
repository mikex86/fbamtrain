#pragma once

#include <termstreamxz/termstream.h>

#include <memory>
#include <cstdint>
#include <vector>

namespace fbamtrain
{
    struct RecordingDatasetInternalState;

    struct TerminalFrameCopy
    {
        int width{};
        int height{};
        std::vector<Cell> cells{};

        static TerminalFrameCopy FromFrame(const TerminalFrame &frame);
    };

    struct BatchIterator
    {
        std::vector<TermInflateStream *> streams{};
        std::vector<uint64_t> start_positions{};
    };

    struct DatasetIteratorState
    {
        uint64_t rng_state{};
        uint32_t batch_size{};
        uint32_t prefetch_buffer_index{};
        std::vector<uint32_t> prepared_indices{};
        std::vector<uint64_t> prepared_start_positions{};
    };

    class RecordingDatasetIterator
    {
        RecordingDatasetInternalState *internal_state_;

        uint32_t batch_size_{};
        uint64_t sequence_length_{};
        uint64_t seed_{};

      public:
        explicit RecordingDatasetIterator(const std::string &recordings_folder_path, uint32_t batch_size,
                                          uint64_t sequence_length, uint64_t seed);

        [[nodiscard]] BatchIterator nextBatch() const;

        /**
         * @brief Returns the lightweight dataset cursor after the last consumed batch.
         *
         * The returned state intentionally contains only the RNG cursor and shape metadata. Restoring it requests a
         * fresh prefetch, which resumes at the next batch without reading the prefetch worker's transient buffers.
         */
        [[nodiscard]] DatasetIteratorState cursorCheckpoint() const;

        [[nodiscard]] DatasetIteratorState checkpoint() const;

        void restore(const DatasetIteratorState &state);

        ~RecordingDatasetIterator();
    };
} // namespace fbamtrain
