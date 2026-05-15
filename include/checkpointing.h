#pragma once

#include "config.h"
#include "dataset_iterator.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace pi::tensorlib
{
    class RealTensor;
}

namespace fbamtrain::checkpointing
{
    struct CheckpointInfo
    {
        std::string path{};
        uint64_t step_count{};
    };

    struct CheckpointLoadResult
    {
        uint64_t step_count{};
        std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> tensors{};
        DatasetIteratorState dataset_state{};
    };

    class CheckpointManager
    {
      public:
        CheckpointManager(const CheckpointingConfiguration &config, uint64_t max_training_steps);

        [[nodiscard]] bool enabled() const;
        [[nodiscard]] std::optional<CheckpointInfo> findLatestCheckpoint() const;

        [[nodiscard]] CheckpointLoadResult load(const std::string &path, uint32_t batch_size) const;

        [[nodiscard]] std::string save(
            uint64_t step_count,
            const std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &tensors,
            const std::optional<DatasetIteratorState> &dataset_state) const;

      private:
        [[nodiscard]] std::string BuildCheckpointPath(uint64_t step_count) const;

        CheckpointingConfiguration config_{};
        uint64_t max_training_steps_{};
        size_t step_width_{};
    };

    // All other helpers are implementation details and intentionally not exposed.
} // namespace fbamtrain::checkpointing
