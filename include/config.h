#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <string>

namespace fbamtrain
{
    enum class FrameHeadReductionStrategy
    {
        Mean,
        LastPos
    };

    enum class ModelDType
    {
        Float16,
        BFloat16
    };

    enum class CheckpointResumeBehavior
    {
        LoadLatest,
        None
    };

    struct DatasetConfiguration
    {
        /// The path to the directory containing the texz recordings.
        /// All recordings in this directory will be used.
        std::string recordings_directory_path;

        /// The iteration seed used for "shuffling" the dataset.
        /// We note that the dataset is never explicitly shuffled, but instead the seed is used to
        /// determine the iteration order on the fly.
        uint64_t iteration_seed;

        [[nodiscard]] static DatasetConfiguration FromJson(const nlohmann::json &json_object);

        [[nodiscard]] std::string jsonstr() const;
    };

    struct FbamModelConfiguration
    {
        uint32_t frame_rows;
        uint32_t frame_cols;
        uint32_t n_embed;
        uint32_t n_layer;
        uint32_t n_head;
        uint32_t downsample_blocks;
        uint32_t max_code_point;
        float rms_norm_eps;
        bool bias;
        ModelDType dtype{ModelDType::Float16};
        std::string downsample_conv_mode;
        uint32_t downsample_conv_dilation;
        uint64_t model_init_seed;
        bool use_fp16_accumulation;
        uint32_t streaming_chunk_size;
        uint32_t recompute_interval;
        FrameHeadReductionStrategy frame_head_reduction_strategy{FrameHeadReductionStrategy::Mean};

        [[nodiscard]] static FbamModelConfiguration FromJson(const nlohmann::json &json_object);

        [[nodiscard]] std::string jsonstr() const;
    };

    struct OptimizerConfiguration
    {
        std::string type;
        float learning_rate;
        float weight_decay;
        float beta1;
        float beta2;
        float eps;
        float momentum;
        bool nesterov;

        [[nodiscard]] static OptimizerConfiguration FromJson(const nlohmann::json &json_object);

        [[nodiscard]] std::string jsonstr() const;
    };

    struct CheckpointingConfiguration
    {
        uint64_t checkpoint_interval{};
        std::string checkpoint_directory{};
        CheckpointResumeBehavior resume_behavior{CheckpointResumeBehavior::LoadLatest};
        std::string dataset_cursor_path{};

        [[nodiscard]] static CheckpointingConfiguration FromJson(const nlohmann::json &json_object);

        [[nodiscard]] std::string jsonstr() const;
    };

    struct RunConfiguration
    {
        DatasetConfiguration train_data_config{};
        DatasetConfiguration validation_data_config{};
        FbamModelConfiguration model_config{};
        OptimizerConfiguration optimizer_config{};
        CheckpointingConfiguration checkpointing{};

        uint32_t micro_batch_size{};
        uint32_t total_batch_size{};
        uint64_t train_sequence_length{};
        uint64_t validation_sequence_length{};
        uint64_t validation_interval{};
        bool enable_validation{true};
        uint64_t max_training_steps{};
        uint64_t frame_rows{};
        uint64_t frame_cols{};
        std::string tokenizer_file_path{};

        [[nodiscard]] static RunConfiguration FromJson(const nlohmann::json &json_object);

        [[nodiscard]] static std::optional<RunConfiguration> FromJsonFile(const std::string &file_path);

        [[nodiscard]] std::string jsonstr() const;
    };

    struct ParallelEndpointConfiguration
    {
        std::string ip{};
        uint32_t port{};

        [[nodiscard]] static ParallelEndpointConfiguration FromJson(const nlohmann::json &json_object,
                                                                    const std::string &context_name);

        [[nodiscard]] std::string jsonstr() const;
    };

    struct FrameHeadParallelConfiguration
    {
        bool use_frame_head_parallel{};
        std::string transport{};
        ParallelEndpointConfiguration ccl_rendezvous{};

        [[nodiscard]] static FrameHeadParallelConfiguration FromJson(const nlohmann::json &json_object);

        [[nodiscard]] std::string jsonstr() const;
    };

    struct DdpParallelConfiguration
    {
        bool use_ddp_parallel{};
        std::string transport{};
        std::optional<ParallelEndpointConfiguration> ccl_rendezvous{};
        std::optional<ParallelEndpointConfiguration> ftccl_master{};

        [[nodiscard]] static DdpParallelConfiguration FromJson(const nlohmann::json &json_object);

        [[nodiscard]] std::string jsonstr() const;
    };

    struct ParallelConfiguration
    {
        FrameHeadParallelConfiguration frame_head_parallel{};
        DdpParallelConfiguration ddp_parallel{};

        [[nodiscard]] static ParallelConfiguration FromJson(const nlohmann::json &json_object);

        [[nodiscard]] static std::optional<ParallelConfiguration> FromJsonFile(const std::string &file_path);

        [[nodiscard]] std::string jsonstr() const;
    };
} // namespace fbamtrain
