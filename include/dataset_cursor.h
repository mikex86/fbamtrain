#pragma once

#include "dataset_iterator.h"

#include <cstdint>
#include <optional>
#include <string>

namespace fbamtrain
{
    struct DatasetCursorSnapshot
    {
        uint64_t step_count{};
        uint64_t rng_state{};
        uint32_t batch_size{};
        uint64_t sequence_length{};

        [[nodiscard]] DatasetIteratorState toDatasetIteratorState() const;

        [[nodiscard]] static DatasetCursorSnapshot FromIterator(uint64_t step_count,
                                                                const RecordingDatasetIterator &iterator,
                                                                uint64_t sequence_length);
    };

    class DatasetCursorCommitter final
    {
      public:
        explicit DatasetCursorCommitter(std::string file_path);
        ~DatasetCursorCommitter();

        DatasetCursorCommitter(const DatasetCursorCommitter &) = delete;
        DatasetCursorCommitter &operator=(const DatasetCursorCommitter &) = delete;

        void commit(const DatasetCursorSnapshot &snapshot);

        [[nodiscard]] static std::optional<DatasetCursorSnapshot> Load(const std::string &file_path);

      private:
        struct Impl;
        Impl *impl_{};
    };
} // namespace fbamtrain
