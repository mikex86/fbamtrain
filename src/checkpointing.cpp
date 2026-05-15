#include "checkpointing.h"

#include <safe_tensors.h>
#include <tensorlib.h>

#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <cstring>

namespace
{
    constexpr const char *kCheckpointPrefix = "checkpoint_step_";
    constexpr const char *kCheckpointSuffix = ".safetensors";

    bool StartsWith(const std::string &value, const std::string &prefix)
    {
        return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
    }

    bool EndsWith(const std::string &value, const std::string &suffix)
    {
        return value.size() >= suffix.size() &&
               value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    std::optional<uint64_t> ParseCheckpointStep(const std::string &filename, const size_t width,
                                                const uint64_t max_training_steps)
    {
        if (!StartsWith(filename, kCheckpointPrefix) || !EndsWith(filename, kCheckpointSuffix))
        {
            return std::nullopt;
        }
        const size_t number_start = std::strlen(kCheckpointPrefix);
        const size_t number_len = filename.size() - number_start - std::strlen(kCheckpointSuffix);
        if (number_len == 0)
        {
            return std::nullopt;
        }
        if (width > 0 && number_len != width)
        {
            return std::nullopt;
        }
        const std::string number_str = filename.substr(number_start, number_len);
        for (char c : number_str)
        {
            if (c < '0' || c > '9')
            {
                return std::nullopt;
            }
        }
        uint64_t step_count = 0;
        try
        {
            step_count = static_cast<uint64_t>(std::stoull(number_str));
        }
        catch (const std::exception &)
        {
            return std::nullopt;
        }
        if (step_count > max_training_steps)
        {
            return std::nullopt;
        }
        return step_count;
    }

    uint64_t ReadScalarU64(const std::shared_ptr<pi::tensorlib::RealTensor> &tensor,
                           const std::string &name)
    {
        if (!tensor)
        {
            throw std::runtime_error("Missing tensor '" + name + "' in checkpoint.");
        }
        if (tensor->dtype() != pi::tensorlib::DataType::UINT64)
        {
            throw std::runtime_error("Checkpoint tensor '" + name + "' has dtype " +
                                     pi::tensorlib::GetDataTypeName(tensor->dtype()) + ", expected uint64.");
        }
        if (tensor->shape().numel() != 1)
        {
            throw std::runtime_error("Checkpoint tensor '" + name +
                                     "' must be a scalar (numel=1).");
        }
        if (tensor->device().device_type != pi::tensorlib::DeviceType::CPU)
        {
            throw std::runtime_error("Checkpoint tensor '" + name + "' must reside on CPU.");
        }
        return static_cast<const uint64_t *>(tensor->dataptr())[0];
    }

    std::shared_ptr<pi::tensorlib::RealTensor> CreateCpuU32Tensor(const std::vector<uint32_t> &data)
    {
        constexpr auto device_cpu = pi::tensorlib::Device{.device_type = pi::tensorlib::DeviceType::CPU, .ordinal = 0};
        auto tensor =
            pi::tensorlib::RealTensor::Allocate({static_cast<uint64_t>(data.size())}, pi::tensorlib::DataType::UINT32,
                                                 device_cpu, false);
        if (!data.empty())
        {
            std::memcpy(tensor->dataptr(), data.data(), data.size() * sizeof(uint32_t));
        }
        return tensor;
    }

    std::shared_ptr<pi::tensorlib::RealTensor> CreateCpuU64Tensor(const std::vector<uint64_t> &data)
    {
        constexpr auto device_cpu = pi::tensorlib::Device{.device_type = pi::tensorlib::DeviceType::CPU, .ordinal = 0};
        auto tensor =
            pi::tensorlib::RealTensor::Allocate({static_cast<uint64_t>(data.size())}, pi::tensorlib::DataType::UINT64,
                                                 device_cpu, false);
        if (!data.empty())
        {
            std::memcpy(tensor->dataptr(), data.data(), data.size() * sizeof(uint64_t));
        }
        return tensor;
    }

    std::vector<uint32_t> ReadCpuU32Vector(const std::shared_ptr<pi::tensorlib::RealTensor> &tensor,
                                           const std::string &name, const size_t expected_size)
    {
        if (!tensor)
        {
            throw std::runtime_error("Missing checkpoint tensor: " + name);
        }
        if (tensor->dtype() != pi::tensorlib::DataType::UINT32)
        {
            throw std::runtime_error("Checkpoint tensor '" + name + "' has dtype " +
                                     pi::tensorlib::GetDataTypeName(tensor->dtype()) + ", expected uint32.");
        }
        if (tensor->shape().numel() != expected_size)
        {
            throw std::runtime_error("Checkpoint tensor '" + name + "' has unexpected size.");
        }
        if (tensor->device().device_type != pi::tensorlib::DeviceType::CPU)
        {
            throw std::runtime_error("Checkpoint tensor '" + name + "' must be on CPU.");
        }
        const auto *data = static_cast<const uint32_t *>(tensor->dataptr());
        return std::vector<uint32_t>(data, data + expected_size);
    }

    std::vector<uint64_t> ReadCpuU64Vector(const std::shared_ptr<pi::tensorlib::RealTensor> &tensor,
                                           const std::string &name, const size_t expected_size)
    {
        if (!tensor)
        {
            throw std::runtime_error("Missing checkpoint tensor: " + name);
        }
        if (tensor->dtype() != pi::tensorlib::DataType::UINT64)
        {
            throw std::runtime_error("Checkpoint tensor '" + name + "' has dtype " +
                                     pi::tensorlib::GetDataTypeName(tensor->dtype()) + ", expected uint64.");
        }
        if (tensor->shape().numel() != expected_size)
        {
            throw std::runtime_error("Checkpoint tensor '" + name + "' has unexpected size.");
        }
        if (tensor->device().device_type != pi::tensorlib::DeviceType::CPU)
        {
            throw std::runtime_error("Checkpoint tensor '" + name + "' must be on CPU.");
        }
        const auto *data = static_cast<const uint64_t *>(tensor->dataptr());
        return std::vector<uint64_t>(data, data + expected_size);
    }
    std::shared_ptr<pi::tensorlib::RealTensor>
    ConsumeTensor(std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &tensors,
                  const std::string &key)
    {
        const auto it = tensors.find(key);
        if (it == tensors.end())
        {
            throw std::runtime_error("Missing checkpoint tensor: " + key);
        }
        auto tensor = it->second;
        tensors.erase(it);
        return tensor;
    }

    fbamtrain::DatasetIteratorState
    DecodeDatasetState(std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &tensors,
                       const uint32_t batch_size)
    {
        auto rng_tensor = ConsumeTensor(tensors, "dataset_rng_state");

        struct TensorCleanup
        {
            std::shared_ptr<pi::tensorlib::RealTensor> tensor;
            ~TensorCleanup()
            {
                if (tensor)
                {
                    tensor->free();
                }
            }
        };
        TensorCleanup rng_cleanup{rng_tensor};

        const auto rng_vec = ReadCpuU64Vector(rng_tensor, "dataset_rng_state", 1);
        fbamtrain::DatasetIteratorState state{};
        state.rng_state = rng_vec[0];
        state.batch_size = batch_size;

        const bool has_indices = tensors.contains("dataset_prefetch_indices");
        const bool has_starts = tensors.contains("dataset_prefetch_starts");
        const bool has_buffer_index = tensors.contains("dataset_prefetch_buffer_index");

        if (!has_indices && !has_starts && !has_buffer_index)
        {
            // Compatibility path for checkpoints that only store RNG state.
            state.prefetch_buffer_index = 0;
            return state;
        }

        if (!(has_indices && has_starts && has_buffer_index))
        {
            throw std::runtime_error(
                "Checkpoint dataset state is incomplete. Expected either only 'dataset_rng_state' "
                "or all prefetch tensors ('dataset_prefetch_indices', 'dataset_prefetch_starts', "
                "'dataset_prefetch_buffer_index').");
        }

        auto indices_tensor = ConsumeTensor(tensors, "dataset_prefetch_indices");
        auto starts_tensor = ConsumeTensor(tensors, "dataset_prefetch_starts");
        auto buffer_tensor = ConsumeTensor(tensors, "dataset_prefetch_buffer_index");
        TensorCleanup indices_cleanup{indices_tensor};
        TensorCleanup starts_cleanup{starts_tensor};
        TensorCleanup buffer_cleanup{buffer_tensor};

        state.prepared_indices = ReadCpuU32Vector(indices_tensor, "dataset_prefetch_indices", batch_size);
        state.prepared_start_positions = ReadCpuU64Vector(starts_tensor, "dataset_prefetch_starts", batch_size);
        auto buffer_index_vec = ReadCpuU32Vector(buffer_tensor, "dataset_prefetch_buffer_index", 1);
        state.prefetch_buffer_index = buffer_index_vec[0];
        return state;
    }
} // namespace

namespace fbamtrain::checkpointing
{
    CheckpointManager::CheckpointManager(const CheckpointingConfiguration &config, const uint64_t max_training_steps)
        : config_(config), max_training_steps_(max_training_steps)
    {
        if (max_training_steps_ == 0)
        {
            step_width_ = 1;
        }
        else
        {
            step_width_ = std::to_string(max_training_steps_).size();
        }
    }

    bool CheckpointManager::enabled() const
    {
        return config_.checkpoint_interval > 0 && !config_.checkpoint_directory.empty();
    }

    std::string CheckpointManager::BuildCheckpointPath(const uint64_t step_count) const
    {
        std::ostringstream oss;
        oss << kCheckpointPrefix << std::setw(static_cast<int>(step_width_)) << std::setfill('0') << step_count
            << kCheckpointSuffix;
        const std::filesystem::path dir_path(config_.checkpoint_directory);
        return (dir_path / oss.str()).string();
    }

    std::optional<CheckpointInfo> CheckpointManager::findLatestCheckpoint() const
    {
        if (!enabled())
        {
            return std::nullopt;
        }
        const std::filesystem::path dir_path(config_.checkpoint_directory);
        if (!std::filesystem::exists(dir_path))
        {
            return std::nullopt;
        }
        if (!std::filesystem::is_directory(dir_path))
        {
            throw std::runtime_error("Checkpoint directory is not a directory: " + config_.checkpoint_directory);
        }

        std::optional<CheckpointInfo> best{};
        for (const auto &entry : std::filesystem::directory_iterator(dir_path))
        {
            if (entry.is_directory())
            {
                continue;
            }
            const std::string filename = entry.path().filename().string();
            const auto step_opt = ParseCheckpointStep(filename, step_width_, max_training_steps_);
            if (!step_opt.has_value())
            {
                continue;
            }
            const uint64_t step_count = step_opt.value();
            if (!best.has_value() || step_count > best->step_count)
            {
                best = CheckpointInfo{.path = entry.path().string(), .step_count = step_count};
            }
        }
        return best;
    }

    CheckpointLoadResult CheckpointManager::load(const std::string &path, const uint32_t batch_size) const
    {
        const std::filesystem::path file_path(path);
        const std::string filename = file_path.filename().string();
        const auto step_opt = ParseCheckpointStep(filename, step_width_, max_training_steps_);
        if (!step_opt.has_value())
        {
            throw std::runtime_error("Checkpoint filename does not match expected pattern: " + filename);
        }
        const uint64_t step_count_from_name = step_opt.value();

        auto tensors = pi::tensorlib::safetensors::Load(path, false);
        auto step_tensor = ConsumeTensor(tensors, "step_count");
        struct TensorCleanup
        {
            std::shared_ptr<pi::tensorlib::RealTensor> tensor;
            ~TensorCleanup()
            {
                if (tensor)
                {
                    tensor->free();
                }
            }
        };
        TensorCleanup step_cleanup{step_tensor};
        const uint64_t step_count_tensor = ReadScalarU64(step_tensor, "step_count");
        if (step_count_tensor != step_count_from_name)
        {
            throw std::runtime_error("Checkpoint step_count mismatch: filename says " +
                                     std::to_string(step_count_from_name) + " but tensor has " +
                                     std::to_string(step_count_tensor));
        }

        const auto dataset_state = DecodeDatasetState(tensors, batch_size);

        return CheckpointLoadResult{.step_count = step_count_from_name,
                                    .tensors = std::move(tensors),
                                    .dataset_state = dataset_state};
    }

    std::string CheckpointManager::save(
        const uint64_t step_count,
        const std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &tensors,
        const std::optional<DatasetIteratorState> &dataset_state) const
    {
        if (!enabled())
        {
            return {};
        }
        if (step_count > max_training_steps_)
        {
            throw std::runtime_error("Checkpoint step_count exceeds max_training_steps: " + std::to_string(step_count));
        }

        std::filesystem::path dir_path(config_.checkpoint_directory);
        if (!std::filesystem::exists(dir_path))
        {
            std::filesystem::create_directories(dir_path);
        }
        if (!std::filesystem::is_directory(dir_path))
        {
            throw std::runtime_error("Checkpoint directory is not a directory: " + config_.checkpoint_directory);
        }

        std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> tensors_out = tensors;
        std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> temporary_tensors{};
        struct TensorListCleanup
        {
            std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> *tensors;
            ~TensorListCleanup()
            {
                if (!tensors)
                {
                    return;
                }
                for (const auto &tensor : *tensors)
                {
                    if (tensor)
                    {
                        tensor->free();
                    }
                }
            }
        };
        TensorListCleanup temp_cleanup{&temporary_tensors};
        if (tensors_out.contains("step_count"))
        {
            throw std::runtime_error("Checkpoint tensor map already contains 'step_count'.");
        }

        auto step_tensor = CreateCpuU64Tensor({step_count});
        temporary_tensors.push_back(step_tensor);
        tensors_out.emplace("step_count", step_tensor);

        if (dataset_state.has_value())
        {
            const auto &state = dataset_state.value();
            auto rng_tensor = CreateCpuU64Tensor({state.rng_state});
            temporary_tensors.push_back(rng_tensor);
            tensors_out.emplace("dataset_rng_state", rng_tensor);

            auto indices_tensor = CreateCpuU32Tensor(state.prepared_indices);
            temporary_tensors.push_back(indices_tensor);
            tensors_out.emplace("dataset_prefetch_indices", indices_tensor);

            auto starts_tensor = CreateCpuU64Tensor(state.prepared_start_positions);
            temporary_tensors.push_back(starts_tensor);
            tensors_out.emplace("dataset_prefetch_starts", starts_tensor);

            auto buffer_tensor = CreateCpuU32Tensor({state.prefetch_buffer_index});
            temporary_tensors.push_back(buffer_tensor);
            tensors_out.emplace("dataset_prefetch_buffer_index", buffer_tensor);
        }

        const std::string output_path = BuildCheckpointPath(step_count);
        pi::tensorlib::safetensors::SaveToFile(output_path, tensors_out);
        return output_path;
    }
} // namespace fbamtrain::checkpointing
