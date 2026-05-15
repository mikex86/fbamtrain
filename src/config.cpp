#include "config.h"

#include <algorithm>
#include <cctype>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#define FBAMTRAIN_JSON_VALIDATE(CONDITION, MSG)                                                                        \
    if (!(CONDITION))                                                                                                  \
    {                                                                                                                  \
        throw std::runtime_error(MSG);                                                                                 \
    }

namespace
{
    std::string NormalizeToLower(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    fbamtrain::FrameHeadReductionStrategy ParseFrameHeadReductionStrategy(const std::string &value)
    {
        const std::string normalized = NormalizeToLower(value);
        if (normalized == "mean")
        {
            return fbamtrain::FrameHeadReductionStrategy::Mean;
        }
        if (normalized == "last_pos")
        {
            return fbamtrain::FrameHeadReductionStrategy::LastPos;
        }
        throw std::runtime_error("Invalid frame_head_reduction_strategy: " + value +
                                 " (expected 'mean' or 'last_pos').");
    }

    fbamtrain::ModelDType ParseModelDType(const std::string &value)
    {
        const std::string normalized = NormalizeToLower(value);
        if (normalized == "fp16")
        {
            return fbamtrain::ModelDType::Float16;
        }
        if (normalized == "bf16")
        {
            return fbamtrain::ModelDType::BFloat16;
        }
        throw std::runtime_error("Invalid dtype: " + value + " (expected 'fp16' or 'bf16').");
    }

    std::string FrameHeadReductionStrategyToString(fbamtrain::FrameHeadReductionStrategy strategy)
    {
        switch (strategy)
        {
            case fbamtrain::FrameHeadReductionStrategy::Mean:
                return "mean";
            case fbamtrain::FrameHeadReductionStrategy::LastPos:
                return "last_pos";
        }
        throw std::runtime_error("Unsupported frame head reduction strategy enum.");
    }

    std::string ModelDTypeToString(fbamtrain::ModelDType dtype)
    {
        switch (dtype)
        {
            case fbamtrain::ModelDType::Float16:
                return "fp16";
            case fbamtrain::ModelDType::BFloat16:
                return "bf16";
        }
        throw std::runtime_error("Unsupported model dtype enum.");
    }

    fbamtrain::CheckpointResumeBehavior ParseCheckpointResumeBehavior(const std::string &value)
    {
        const std::string normalized = NormalizeToLower(value);
        if (normalized == "load_latest")
        {
            return fbamtrain::CheckpointResumeBehavior::LoadLatest;
        }
        if (normalized == "none" || normalized == "disabled")
        {
            return fbamtrain::CheckpointResumeBehavior::None;
        }
        throw std::runtime_error("Invalid resume_behavior: " + value + " (expected 'load_latest' or 'none').");
    }

    std::string CheckpointResumeBehaviorToString(fbamtrain::CheckpointResumeBehavior behavior)
    {
        switch (behavior)
        {
            case fbamtrain::CheckpointResumeBehavior::LoadLatest:
                return "load_latest";
            case fbamtrain::CheckpointResumeBehavior::None:
                return "none";
        }
        throw std::runtime_error("Unsupported checkpoint resume behavior enum.");
    }
} // namespace

fbamtrain::DatasetConfiguration fbamtrain::DatasetConfiguration::FromJson(const nlohmann::json &json_object)
{
    const auto recordings_directory_path_property = json_object.find("recordings_directory_path");
    FBAMTRAIN_JSON_VALIDATE(recordings_directory_path_property != json_object.end(),
                            "Missing 'recordings_directory_path' property in dataset configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(recordings_directory_path_property->is_string(),
                            "'recordings_directory_path' property must be a string.");

    const auto iteration_seed_property = json_object.find("iteration_seed");
    FBAMTRAIN_JSON_VALIDATE(iteration_seed_property != json_object.end(),
                            "Missing 'iteration_seed' property in dataset configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(iteration_seed_property->is_number_unsigned(),
                            "'iteration_seed' property must be an unsigned integer.");

    const std::string recordings_directory_path = recordings_directory_path_property->get<std::string>();
    const uint64_t iteration_seed = iteration_seed_property->get<uint64_t>();
    return DatasetConfiguration{.recordings_directory_path = recordings_directory_path,
                                .iteration_seed = iteration_seed};
}

std::string fbamtrain::DatasetConfiguration::jsonstr() const
{

    return nlohmann::json{{"recordings_directory_path", recordings_directory_path}, {"iteration_seed", iteration_seed}}
        .dump(4);
}

fbamtrain::FbamModelConfiguration fbamtrain::FbamModelConfiguration::FromJson(const nlohmann::json &json_object)
{
    const auto frame_rows = json_object.find("frame_rows");
    FBAMTRAIN_JSON_VALIDATE(frame_rows != json_object.end(),
                            "Missing 'frame_rows' property in model configuration JSON.");

    const auto frame_cols = json_object.find("frame_cols");
    FBAMTRAIN_JSON_VALIDATE(frame_cols != json_object.end(),
                            "Missing 'frame_cols' property in model configuration JSON.");

    const auto n_embed = json_object.find("n_embed");
    FBAMTRAIN_JSON_VALIDATE(n_embed != json_object.end(), "Missing 'n_embed' property in model configuration JSON.");

    const auto n_layer = json_object.find("n_layer");
    FBAMTRAIN_JSON_VALIDATE(n_layer != json_object.end(), "Missing 'n_layer' property in model configuration JSON.");

    const auto n_head = json_object.find("n_head");
    FBAMTRAIN_JSON_VALIDATE(n_head != json_object.end(), "Missing 'n_head' property in model configuration JSON.");

    const auto downsample_blocks = json_object.find("downsample_blocks");
    FBAMTRAIN_JSON_VALIDATE(downsample_blocks != json_object.end(),
                            "Missing 'downsample_blocks' property in model configuration JSON.");

    const auto max_code_point = json_object.find("max_code_point");
    FBAMTRAIN_JSON_VALIDATE(max_code_point != json_object.end(),
                            "Missing 'max_code_point' property in model configuration JSON.");

    const auto rms_norm_eps = json_object.find("rms_norm_eps");
    FBAMTRAIN_JSON_VALIDATE(rms_norm_eps != json_object.end(),
                            "Missing 'rms_norm_eps' property in model configuration JSON.");

    const auto bias = json_object.find("bias");
    FBAMTRAIN_JSON_VALIDATE(bias != json_object.end(), "Missing 'bias' property in model configuration JSON.");

    const auto downsample_conv_mode = json_object.find("downsample_conv_mode");
    FBAMTRAIN_JSON_VALIDATE(downsample_conv_mode != json_object.end(),
                            "Missing 'downsample_conv_mode' property in model configuration JSON.");

    const auto downsample_conv_dilation = json_object.find("downsample_conv_dilation");
    FBAMTRAIN_JSON_VALIDATE(downsample_conv_dilation != json_object.end(),
                            "Missing 'downsample_conv_dilation' property in model configuration JSON.");

    const auto model_init_seed = json_object.find("model_init_seed");
    FBAMTRAIN_JSON_VALIDATE(model_init_seed != json_object.end(),
                            "Missing 'model_init_seed' property in model configuration JSON.");
    const auto use_fp16_accumulation = json_object.find("use_fp16_accumulation");
    FBAMTRAIN_JSON_VALIDATE(use_fp16_accumulation != json_object.end(),
                            "Missing or invalid 'use_fp16_accumulation' property in model configuration JSON.");
    const auto dtype = json_object.find("dtype");
    const auto streaming_chunk_size = json_object.find("streaming_chunk_size");
    FBAMTRAIN_JSON_VALIDATE(streaming_chunk_size != json_object.end(),
                            "Missing 'streaming_chunk_size' property in model configuration JSON.");
    const auto recompute_interval = json_object.find("recompute_interval");
    FBAMTRAIN_JSON_VALIDATE(recompute_interval != json_object.end(),
                            "Missing 'recompute_interval' property in model configuration JSON.");
    const auto frame_head_reduction_strategy = json_object.find("frame_head_reduction_strategy");

    FBAMTRAIN_JSON_VALIDATE(n_embed->is_number_unsigned(),
                            "'n_embed' property must be an unsigned integer in model configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(n_layer->is_number_unsigned(),
                            "'n_layer' property must be an unsigned integer in model configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(n_head->is_number_unsigned(),
                            "'n_head' property must be an unsigned integer in model configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(downsample_blocks->is_number_unsigned(),
                            "'downsample_blocks' property must be an unsigned integer in model configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(max_code_point->is_number_unsigned(),
                            "'max_code_point' property must be an unsigned integer in model configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(rms_norm_eps->is_number(),
                            "'rms_norm_eps' property must be a floating point number in model configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(bias->is_boolean(), "'bias' property must be a boolean in model configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(downsample_conv_mode->is_string(),
                            "'downsample_conv_mode' property must be a string in model configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(
        downsample_conv_dilation->is_number_unsigned(),
        "'downsample_conv_dilation' property must be an unsigned integer in model configuration JSON.");
    const auto downsample_conv_mode_value = downsample_conv_mode->get<std::string>();
    FBAMTRAIN_JSON_VALIDATE(!downsample_conv_mode_value.empty(),
                            "'downsample_conv_mode' property must be a non-empty string in model configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(model_init_seed->is_number_unsigned(),
                            "'model_init_seed' property must be an unsigned integer in model configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(use_fp16_accumulation->is_boolean(),
                            "'use_fp16_accumulation' property must be a boolean in model configuration JSON.");
    if (dtype != json_object.end())
    {
        FBAMTRAIN_JSON_VALIDATE(dtype->is_string(), "'dtype' property must be a string in model configuration JSON.");
    }
    FBAMTRAIN_JSON_VALIDATE(streaming_chunk_size->is_number_unsigned(),
                            "'streaming_chunk_size' property must be an unsigned integer in model configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(recompute_interval->is_number_unsigned(),
                            "'recompute_interval' property must be an unsigned integer in model configuration JSON.");
    if (frame_head_reduction_strategy != json_object.end())
    {
        FBAMTRAIN_JSON_VALIDATE(frame_head_reduction_strategy->is_string(),
                                "'frame_head_reduction_strategy' property must be a string in model configuration JSON.");
    }

    fbamtrain::ModelDType dtype_value = fbamtrain::ModelDType::Float16;
    if (dtype != json_object.end())
    {
        const std::string dtype_raw = dtype->get<std::string>();
        FBAMTRAIN_JSON_VALIDATE(!dtype_raw.empty(),
                                "'dtype' property must be a non-empty string in model configuration JSON.");
        dtype_value = ParseModelDType(dtype_raw);
    }
    if (dtype_value == fbamtrain::ModelDType::BFloat16 && use_fp16_accumulation->get<bool>())
    {
        throw std::runtime_error("use_fp16_accumulation cannot be true when dtype is bf16.");
    }

    FrameHeadReductionStrategy reduction_strategy_value = FrameHeadReductionStrategy::Mean;
    if (frame_head_reduction_strategy != json_object.end())
    {
        const std::string reduction_strategy_raw = frame_head_reduction_strategy->get<std::string>();
        FBAMTRAIN_JSON_VALIDATE(!reduction_strategy_raw.empty(),
                                "'frame_head_reduction_strategy' property must be a non-empty string in model configuration JSON.");
        reduction_strategy_value = ParseFrameHeadReductionStrategy(reduction_strategy_raw);
    }

    return FbamModelConfiguration{.frame_rows = frame_rows->get<uint32_t>(),
                                  .frame_cols = frame_cols->get<uint32_t>(),
                                  .n_embed = n_embed->get<uint32_t>(),
                                  .n_layer = n_layer->get<uint32_t>(),
                                  .n_head = n_head->get<uint32_t>(),
                                  .downsample_blocks = downsample_blocks->get<uint32_t>(),
                                  .max_code_point = max_code_point->get<uint32_t>(),
                                  .rms_norm_eps = rms_norm_eps->get<float>(),
                                  .bias = bias->get<bool>(),
                                  .dtype = dtype_value,
                                  .downsample_conv_mode = downsample_conv_mode_value,
                                  .downsample_conv_dilation = downsample_conv_dilation->get<uint32_t>(),
                                  .model_init_seed = model_init_seed->get<uint64_t>(),
                                  .use_fp16_accumulation = use_fp16_accumulation->get<bool>(),
                                  .streaming_chunk_size = streaming_chunk_size->get<uint32_t>(),
                                  .recompute_interval = recompute_interval->get<uint32_t>(),
                                  .frame_head_reduction_strategy = reduction_strategy_value};
}

std::string fbamtrain::FbamModelConfiguration::jsonstr() const
{
    return nlohmann::json{{"frame_rows", frame_rows},
                          {"frame_cols", frame_cols},
                          {"n_embed", n_embed},
                          {"n_layer", n_layer},
                          {"n_head", n_head},
                          {"downsample_blocks", downsample_blocks},
                          {"max_code_point", max_code_point},
                          {"rms_norm_eps", rms_norm_eps},
                          {"bias", bias},
                          {"dtype", ModelDTypeToString(dtype)},
                          {"downsample_conv_mode", downsample_conv_mode},
                          {"downsample_conv_dilation", downsample_conv_dilation},
                          {"model_init_seed", model_init_seed},
                          {"use_fp16_accumulation", use_fp16_accumulation},
                          {"streaming_chunk_size", streaming_chunk_size},
                          {"recompute_interval", recompute_interval},
                          {"frame_head_reduction_strategy", FrameHeadReductionStrategyToString(frame_head_reduction_strategy)}}
        .dump(4);
}

fbamtrain::OptimizerConfiguration fbamtrain::OptimizerConfiguration::FromJson(const nlohmann::json &json_object)
{
    const auto type = json_object.find("type");
    FBAMTRAIN_JSON_VALIDATE(type != json_object.end(), "Missing 'type' property in optimizer configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(type->is_string(), "'type' property must be a string in optimizer configuration JSON.");

    const auto learning_rate = json_object.find("learning_rate");
    FBAMTRAIN_JSON_VALIDATE(learning_rate != json_object.end(),
                            "Missing 'learning_rate' property in optimizer configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(learning_rate->is_number(),
                            "'learning_rate' property must be a number in optimizer configuration JSON.");

    const auto weight_decay = json_object.find("weight_decay");
    FBAMTRAIN_JSON_VALIDATE(weight_decay != json_object.end(),
                            "Missing 'weight_decay' property in optimizer configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(weight_decay->is_number(),
                            "'weight_decay' property must be a number in optimizer configuration JSON.");

    const auto beta1 = json_object.find("beta1");
    FBAMTRAIN_JSON_VALIDATE(beta1 != json_object.end(), "Missing 'beta1' property in optimizer configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(beta1->is_number(), "'beta1' property must be a number in optimizer configuration JSON.");

    const auto beta2 = json_object.find("beta2");
    FBAMTRAIN_JSON_VALIDATE(beta2 != json_object.end(), "Missing 'beta2' property in optimizer configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(beta2->is_number(), "'beta2' property must be a number in optimizer configuration JSON.");

    const auto eps = json_object.find("eps");
    FBAMTRAIN_JSON_VALIDATE(eps != json_object.end(), "Missing 'eps' property in optimizer configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(eps->is_number(), "'eps' property must be a number in optimizer configuration JSON.");

    const auto momentum = json_object.find("momentum");
    FBAMTRAIN_JSON_VALIDATE(momentum != json_object.end(),
                            "Missing 'momentum' property in optimizer configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(momentum->is_number(),
                            "'momentum' property must be a number in optimizer configuration JSON.");

    const auto nesterov = json_object.find("nesterov");
    FBAMTRAIN_JSON_VALIDATE(nesterov != json_object.end(),
                            "Missing 'nesterov' property in optimizer configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(nesterov->is_boolean(),
                            "'nesterov' property must be a boolean in optimizer configuration JSON.");

    const auto type_value = type->get<std::string>();
    FBAMTRAIN_JSON_VALIDATE(!type_value.empty(), "'type' property must be non-empty in optimizer configuration JSON.");

    return OptimizerConfiguration{.type = type_value,
                                  .learning_rate = learning_rate->get<float>(),
                                  .weight_decay = weight_decay->get<float>(),
                                  .beta1 = beta1->get<float>(),
                                  .beta2 = beta2->get<float>(),
                                  .eps = eps->get<float>(),
                                  .momentum = momentum->get<float>(),
                                  .nesterov = nesterov->get<bool>()};
}

std::string fbamtrain::OptimizerConfiguration::jsonstr() const
{
    return nlohmann::json{{"type", type},
                          {"learning_rate", learning_rate},
                          {"weight_decay", weight_decay},
                          {"beta1", beta1},
                          {"beta2", beta2},
                          {"eps", eps},
                          {"momentum", momentum},
                          {"nesterov", nesterov}}
        .dump(4);
}

fbamtrain::CheckpointingConfiguration fbamtrain::CheckpointingConfiguration::FromJson(const nlohmann::json &json_object)
{
    FBAMTRAIN_JSON_VALIDATE(json_object.is_object(),
                            "'checkpointing' property must be a JSON object in run configuration JSON.");

    const auto checkpoint_interval = json_object.find("checkpoint_interval");
    FBAMTRAIN_JSON_VALIDATE(checkpoint_interval != json_object.end(),
                            "Missing 'checkpoint_interval' property in checkpointing configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(
        checkpoint_interval->is_number_unsigned(),
        "'checkpoint_interval' property must be an unsigned integer in checkpointing configuration JSON.");

    const auto checkpoint_directory = json_object.find("checkpoint_directory");
    FBAMTRAIN_JSON_VALIDATE(checkpoint_directory != json_object.end(),
                            "Missing 'checkpoint_directory' property in checkpointing configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(checkpoint_directory->is_string(),
                            "'checkpoint_directory' property must be a string in checkpointing configuration JSON.");

    const auto resume_behavior = json_object.find("resume_behavior");
    FBAMTRAIN_JSON_VALIDATE(resume_behavior != json_object.end(),
                            "Missing 'resume_behavior' property in checkpointing configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(resume_behavior->is_string(),
                            "'resume_behavior' property must be a string in checkpointing configuration JSON.");

    const uint64_t checkpoint_interval_value = checkpoint_interval->get<uint64_t>();
    FBAMTRAIN_JSON_VALIDATE(checkpoint_interval_value > 0,
                            "'checkpoint_interval' must be a positive integer in checkpointing configuration JSON.");

    const std::string checkpoint_directory_value = checkpoint_directory->get<std::string>();
    FBAMTRAIN_JSON_VALIDATE(!checkpoint_directory_value.empty(),
                            "'checkpoint_directory' property must be a non-empty string in checkpointing configuration JSON.");

    const std::string resume_behavior_raw = resume_behavior->get<std::string>();
    FBAMTRAIN_JSON_VALIDATE(!resume_behavior_raw.empty(),
                            "'resume_behavior' property must be a non-empty string in checkpointing configuration JSON.");
    const auto resume_behavior_value = ParseCheckpointResumeBehavior(resume_behavior_raw);

    std::string dataset_cursor_path_value{};
    const auto dataset_cursor_path = json_object.find("dataset_cursor_path");
    if (dataset_cursor_path != json_object.end())
    {
        FBAMTRAIN_JSON_VALIDATE(dataset_cursor_path->is_string(),
                                "'dataset_cursor_path' property must be a string in checkpointing configuration JSON.");
        dataset_cursor_path_value = dataset_cursor_path->get<std::string>();
    }

    return CheckpointingConfiguration{.checkpoint_interval = checkpoint_interval_value,
                                      .checkpoint_directory = checkpoint_directory_value,
                                      .resume_behavior = resume_behavior_value,
                                      .dataset_cursor_path = dataset_cursor_path_value};
}

std::string fbamtrain::CheckpointingConfiguration::jsonstr() const
{
    return nlohmann::json{{"checkpoint_interval", checkpoint_interval},
                          {"checkpoint_directory", checkpoint_directory},
                          {"resume_behavior", CheckpointResumeBehaviorToString(resume_behavior)},
                          {"dataset_cursor_path", dataset_cursor_path}}
        .dump(4);
}

fbamtrain::RunConfiguration fbamtrain::RunConfiguration::FromJson(const nlohmann::json &json_object)
{
    const auto train = json_object.find("train_data_config");
    const auto validation = json_object.find("validation_data_config");
    FBAMTRAIN_JSON_VALIDATE(train != json_object.end(),
                            "Missing 'train_data_config' property in run configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(train->is_object(), "'train_data_config' property must be a JSON object.");

    FBAMTRAIN_JSON_VALIDATE(validation != json_object.end(),
                            "Missing 'validation_data_config' property in run configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(validation->is_object(), "'validation_data_config' property must be a JSON object.");

    const auto model_config = json_object.find("model_config");
    FBAMTRAIN_JSON_VALIDATE(model_config != json_object.end(),
                            "Missing 'model_config' property in run configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(train->is_object(), "'train_data_config' property must be a JSON object.");

    const auto micro_batch_size = json_object.find("micro_batch_size");
    FBAMTRAIN_JSON_VALIDATE(micro_batch_size != json_object.end(),
                            "Missing 'micro_batch_size' property in run configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(micro_batch_size->is_number_unsigned(),
                            "'micro_batch_size' property must be an unsigned integer.");
    const auto total_batch_size = json_object.find("total_batch_size");
    FBAMTRAIN_JSON_VALIDATE(total_batch_size != json_object.end(),
                            "Missing 'total_batch_size' property in run configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(total_batch_size->is_number_unsigned(),
                            "'total_batch_size' property must be an unsigned integer.");
    const auto train_sequence_length = json_object.find("train_sequence_length");
    FBAMTRAIN_JSON_VALIDATE(train_sequence_length != json_object.end(),
                            "Missing 'train_sequence_length' property in run configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(train_sequence_length->is_number_unsigned(),
                            "'train_sequence_length' property must be an unsigned integer.");
    const auto validation_sequence_length = json_object.find("validation_sequence_length");
    FBAMTRAIN_JSON_VALIDATE(validation_sequence_length != json_object.end(),
                            "Missing 'validation_sequence_length' property in run configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(validation_sequence_length->is_number_unsigned(),
                            "'validation_sequence_length' property must be an unsigned integer.");
    const auto validation_interval = json_object.find("validation_interval");
    uint64_t validation_interval_value = 0;
    if (validation_interval != json_object.end())
    {
        FBAMTRAIN_JSON_VALIDATE(validation_interval->is_number_unsigned(),
                                "'validation_interval' property must be an unsigned integer.");
        validation_interval_value = validation_interval->get<uint64_t>();
    }
    const auto enable_validation = json_object.find("enable_validation");
    bool enable_validation_value = true;
    if (enable_validation != json_object.end())
    {
        FBAMTRAIN_JSON_VALIDATE(enable_validation->is_boolean(),
                                "'enable_validation' property must be a boolean.");
        enable_validation_value = enable_validation->get<bool>();
    }
    const auto max_training_steps = json_object.find("max_training_steps");
    FBAMTRAIN_JSON_VALIDATE(max_training_steps != json_object.end(),
                            "Missing 'max_training_steps' property in run configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(max_training_steps->is_number_unsigned(),
                            "'max_training_steps' property must be an unsigned integer.");

    const auto frame_rows = json_object.find("frame_rows");
    FBAMTRAIN_JSON_VALIDATE(frame_rows != json_object.end(),
                            "Missing 'frame_rows' property in run configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(frame_rows->is_number_unsigned(), "'frame_rows' property must be an unsigned integer.");

    const auto frame_cols = json_object.find("frame_cols");
    FBAMTRAIN_JSON_VALIDATE(frame_cols != json_object.end(),
                            "Missing 'frame_cols' property in run configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(frame_cols->is_number_unsigned(), "'frame_cols' property must be an unsigned integer.");
    const auto tokenizer_file_path = json_object.find("tokenizer_file_path");
    FBAMTRAIN_JSON_VALIDATE(tokenizer_file_path != json_object.end(),
                            "Missing 'tokenizer_file_path' property in run configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(tokenizer_file_path->is_string(), "'tokenizer_file_path' property must be a string.");
    const auto optimizer_config = json_object.find("optimizer");
    FBAMTRAIN_JSON_VALIDATE(optimizer_config != json_object.end(),
                            "Missing 'optimizer' property in run configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(optimizer_config->is_object(), "'optimizer' property must be a JSON object.");
    const auto checkpointing_config = json_object.find("checkpointing");
    FBAMTRAIN_JSON_VALIDATE(checkpointing_config != json_object.end(),
                            "Missing 'checkpointing' property in run configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(checkpointing_config->is_object(), "'checkpointing' property must be a JSON object.");
    const auto micro_batch_size_value = micro_batch_size->get<uint32_t>();
    const auto total_batch_size_value = total_batch_size->get<uint32_t>();
    FBAMTRAIN_JSON_VALIDATE(total_batch_size_value > 0, "'total_batch_size' must be greater than zero.");
    FBAMTRAIN_JSON_VALIDATE(total_batch_size_value % micro_batch_size_value == 0,
                            "'total_batch_size' must be a multiple of 'micro_batch_size'.");

    const auto checkpointing_value = CheckpointingConfiguration::FromJson(*checkpointing_config);

    return RunConfiguration{.train_data_config = DatasetConfiguration::FromJson(*train),
                            .validation_data_config = DatasetConfiguration::FromJson(*validation),
                            .model_config = FbamModelConfiguration::FromJson(*model_config),
                            .optimizer_config = OptimizerConfiguration::FromJson(*optimizer_config),
                            .checkpointing = checkpointing_value,
                            .micro_batch_size = micro_batch_size_value,
                            .total_batch_size = total_batch_size_value,
                            .train_sequence_length = train_sequence_length->get<uint32_t>(),
                            .validation_sequence_length = validation_sequence_length->get<uint32_t>(),
                            .validation_interval = validation_interval_value,
                            .enable_validation = enable_validation_value,
                            .max_training_steps = max_training_steps->get<uint64_t>(),
                            .frame_rows = frame_rows->get<uint32_t>(),
                            .frame_cols = frame_cols->get<uint32_t>(),
                            .tokenizer_file_path = tokenizer_file_path->get<std::string>()};
}

std::optional<fbamtrain::RunConfiguration> fbamtrain::RunConfiguration::FromJsonFile(const std::string &file_path)
{
    if (!std::filesystem::exists(file_path))
    {
        return std::nullopt;
    }
    std::ifstream file_stream(file_path);
    if (!file_stream.is_open())
    {
        if (!file_stream.is_open())
        {
            throw std::runtime_error("Failed to open configuration file: " + file_path);
        }
        return std::nullopt;
    }
    const nlohmann::json json_content = nlohmann::json::parse(file_stream, nullptr, false);
    if (json_content.is_discarded())
    {
        if (!file_stream.eof())
        {
            throw std::runtime_error("Failed to parse JSON from configuration file: " + file_path);
        }
        return std::nullopt;
    }
    return FromJson(json_content);
}

std::string fbamtrain::RunConfiguration::jsonstr() const
{
    return nlohmann::json{{"train_data_config",
                           {{"recordings_directory_path", train_data_config.recordings_directory_path},
                            {"iteration_seed", train_data_config.iteration_seed}}},
                          {"validation_data_config",
                           {{"recordings_directory_path", validation_data_config.recordings_directory_path},
                            {"iteration_seed", validation_data_config.iteration_seed}}},
                          {"micro_batch_size", micro_batch_size},
                          {"total_batch_size", total_batch_size},
                          {"train_sequence_length", train_sequence_length},
                          {"validation_sequence_length", validation_sequence_length},
                          {"validation_interval", validation_interval},
                          {"enable_validation", enable_validation},
                          {"max_training_steps", max_training_steps},
                          {"frame_rows", frame_rows},
                          {"frame_cols", frame_cols},
                          {"tokenizer_file_path", tokenizer_file_path},
                          {"checkpointing",
                           {{"checkpoint_interval", checkpointing.checkpoint_interval},
                            {"checkpoint_directory", checkpointing.checkpoint_directory},
                            {"resume_behavior", CheckpointResumeBehaviorToString(checkpointing.resume_behavior)},
                            {"dataset_cursor_path", checkpointing.dataset_cursor_path}}},
                          {"optimizer",
                           {{"type", optimizer_config.type},
                            {"learning_rate", optimizer_config.learning_rate},
                            {"weight_decay", optimizer_config.weight_decay},
                            {"beta1", optimizer_config.beta1},
                            {"beta2", optimizer_config.beta2},
                            {"eps", optimizer_config.eps},
                            {"momentum", optimizer_config.momentum},
                            {"nesterov", optimizer_config.nesterov}}},
                          {"model_config",
                           {{"frame_rows", model_config.frame_rows},
                            {"frame_cols", model_config.frame_cols},
                            {"n_embed", model_config.n_embed},
                            {"n_layer", model_config.n_layer},
                            {"n_head", model_config.n_head},
                            {"downsample_blocks", model_config.downsample_blocks},
                            {"max_code_point", model_config.max_code_point},
                            {"rms_norm_eps", model_config.rms_norm_eps},
                            {"bias", model_config.bias},
                            {"downsample_conv_mode", model_config.downsample_conv_mode},
                            {"downsample_conv_dilation", model_config.downsample_conv_dilation},
                            {"model_init_seed", model_config.model_init_seed},
                            {"use_fp16_accumulation", model_config.use_fp16_accumulation},
                            {"streaming_chunk_size", model_config.streaming_chunk_size},
                            {"recompute_interval", model_config.recompute_interval}}}}
        .dump(4);
}

fbamtrain::ParallelEndpointConfiguration
fbamtrain::ParallelEndpointConfiguration::FromJson(const nlohmann::json &json_object,
                                                   const std::string &context_name)
{
    const auto ip = json_object.find("ip");
    FBAMTRAIN_JSON_VALIDATE(ip != json_object.end(),
                            "Missing 'ip' property in " + context_name + " configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(ip->is_string(),
                            "'ip' property must be a string in " + context_name + " configuration JSON.");
    const auto ip_value = ip->get<std::string>();
    FBAMTRAIN_JSON_VALIDATE(!ip_value.empty(),
                            "'ip' property must be non-empty in " + context_name + " configuration JSON.");

    const auto port = json_object.find("port");
    FBAMTRAIN_JSON_VALIDATE(port != json_object.end(),
                            "Missing 'port' property in " + context_name + " configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(port->is_number_unsigned(),
                            "'port' property must be an unsigned integer in " + context_name +
                                " configuration JSON.");
    const auto port_value = port->get<uint32_t>();
    FBAMTRAIN_JSON_VALIDATE(port_value <= 65535,
                            "'port' property must fit in uint16_t in " + context_name + " configuration JSON.");

    return ParallelEndpointConfiguration{.ip = ip_value, .port = port_value};
}

std::string fbamtrain::ParallelEndpointConfiguration::jsonstr() const
{
    return nlohmann::json{{"ip", ip}, {"port", port}}.dump(4);
}

fbamtrain::FrameHeadParallelConfiguration
fbamtrain::FrameHeadParallelConfiguration::FromJson(const nlohmann::json &json_object)
{
    const auto use_frame_head_parallel = json_object.find("use_frame_head_parallel");
    FBAMTRAIN_JSON_VALIDATE(use_frame_head_parallel != json_object.end(),
                            "Missing 'use_frame_head_parallel' property in frame_head_parallel configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(
        use_frame_head_parallel->is_boolean(),
        "'use_frame_head_parallel' property must be a boolean in frame_head_parallel configuration JSON.");

    const auto transport = json_object.find("transport");
    FBAMTRAIN_JSON_VALIDATE(transport != json_object.end(),
                            "Missing 'transport' property in frame_head_parallel configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(transport->is_string(),
                            "'transport' property must be a string in frame_head_parallel configuration JSON.");
    const auto transport_value = transport->get<std::string>();
    FBAMTRAIN_JSON_VALIDATE(transport_value == "ccl",
                            "frame_head_parallel.transport must be \"ccl\".");

    const auto ccl_rendezvous = json_object.find("ccl_rendezvous");
    FBAMTRAIN_JSON_VALIDATE(ccl_rendezvous != json_object.end(),
                            "Missing 'ccl_rendezvous' property in frame_head_parallel configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(ccl_rendezvous->is_object(),
                            "'ccl_rendezvous' property must be a JSON object in frame_head_parallel "
                            "configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(json_object.find("ftccl_master") == json_object.end(),
                            "'ftccl_master' is illegal in frame_head_parallel when transport is \"ccl\".");

    return FrameHeadParallelConfiguration{
        .use_frame_head_parallel = use_frame_head_parallel->get<bool>(),
        .transport = transport_value,
        .ccl_rendezvous =
            ParallelEndpointConfiguration::FromJson(*ccl_rendezvous, "frame_head_parallel.ccl_rendezvous")};
}

std::string fbamtrain::FrameHeadParallelConfiguration::jsonstr() const
{
    return nlohmann::json{{"use_frame_head_parallel", use_frame_head_parallel},
                          {"transport", transport},
                          {"ccl_rendezvous", {{"ip", ccl_rendezvous.ip}, {"port", ccl_rendezvous.port}}}}
        .dump(4);
}

fbamtrain::DdpParallelConfiguration fbamtrain::DdpParallelConfiguration::FromJson(const nlohmann::json &json_object)
{
    const auto use_ddp_parallel = json_object.find("use_ddp_parallel");
    FBAMTRAIN_JSON_VALIDATE(use_ddp_parallel != json_object.end(),
                            "Missing 'use_ddp_parallel' property in ddp_parallel configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(use_ddp_parallel->is_boolean(),
                            "'use_ddp_parallel' property must be a boolean in ddp_parallel configuration JSON.");

    if (!use_ddp_parallel->get<bool>())
    {
        FBAMTRAIN_JSON_VALIDATE(json_object.find("transport") == json_object.end(),
                                "'transport' is illegal in ddp_parallel when use_ddp_parallel is false.");
        FBAMTRAIN_JSON_VALIDATE(json_object.find("ccl_rendezvous") == json_object.end(),
                                "'ccl_rendezvous' is illegal in ddp_parallel when use_ddp_parallel is false.");
        FBAMTRAIN_JSON_VALIDATE(json_object.find("ftccl_master") == json_object.end(),
                                "'ftccl_master' is illegal in ddp_parallel when use_ddp_parallel is false.");
        return DdpParallelConfiguration{.use_ddp_parallel = false};
    }

    const auto transport = json_object.find("transport");
    FBAMTRAIN_JSON_VALIDATE(transport != json_object.end(),
                            "Missing 'transport' property in ddp_parallel configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(transport->is_string(),
                            "'transport' property must be a string in ddp_parallel configuration JSON.");
    const auto transport_value = transport->get<std::string>();
    FBAMTRAIN_JSON_VALIDATE(transport_value == "ccl" || transport_value == "ftccl",
                            "ddp_parallel.transport must be \"ccl\" or \"ftccl\".");

    const auto ccl_rendezvous = json_object.find("ccl_rendezvous");
    const auto ftccl_master = json_object.find("ftccl_master");
    std::optional<ParallelEndpointConfiguration> ccl_endpoint{};
    std::optional<ParallelEndpointConfiguration> ftccl_endpoint{};
    if (transport_value == "ccl")
    {
        FBAMTRAIN_JSON_VALIDATE(ccl_rendezvous != json_object.end(),
                                "Missing 'ccl_rendezvous' property in ddp_parallel configuration JSON.");
        FBAMTRAIN_JSON_VALIDATE(ccl_rendezvous->is_object(),
                                "'ccl_rendezvous' property must be a JSON object in ddp_parallel configuration JSON.");
        FBAMTRAIN_JSON_VALIDATE(ftccl_master == json_object.end(),
                                "'ftccl_master' is illegal in ddp_parallel when transport is \"ccl\".");
        ccl_endpoint = ParallelEndpointConfiguration::FromJson(*ccl_rendezvous, "ddp_parallel.ccl_rendezvous");
    }
    else
    {
        FBAMTRAIN_JSON_VALIDATE(ftccl_master != json_object.end(),
                                "Missing 'ftccl_master' property in ddp_parallel configuration JSON.");
        FBAMTRAIN_JSON_VALIDATE(ftccl_master->is_object(),
                                "'ftccl_master' property must be a JSON object in ddp_parallel configuration JSON.");
        FBAMTRAIN_JSON_VALIDATE(ccl_rendezvous == json_object.end(),
                                "'ccl_rendezvous' is illegal in ddp_parallel when transport is \"ftccl\".");
        ftccl_endpoint = ParallelEndpointConfiguration::FromJson(*ftccl_master, "ddp_parallel.ftccl_master");
    }

    return DdpParallelConfiguration{
        .use_ddp_parallel = use_ddp_parallel->get<bool>(),
        .transport = transport_value,
        .ccl_rendezvous = ccl_endpoint,
        .ftccl_master = ftccl_endpoint};
}

std::string fbamtrain::DdpParallelConfiguration::jsonstr() const
{
    if (!use_ddp_parallel)
    {
        return nlohmann::json{{"use_ddp_parallel", false}}.dump(4);
    }

    nlohmann::json json{{"use_ddp_parallel", use_ddp_parallel}, {"transport", transport}};
    if (ccl_rendezvous.has_value())
    {
        json["ccl_rendezvous"] = {{"ip", ccl_rendezvous->ip}, {"port", ccl_rendezvous->port}};
    }
    if (ftccl_master.has_value())
    {
        json["ftccl_master"] = {{"ip", ftccl_master->ip}, {"port", ftccl_master->port}};
    }
    return json.dump(4);
}

fbamtrain::ParallelConfiguration fbamtrain::ParallelConfiguration::FromJson(const nlohmann::json &json_object)
{
    const auto frame_head_parallel = json_object.find("frame_head_parallel");
    FBAMTRAIN_JSON_VALIDATE(frame_head_parallel != json_object.end(),
                            "Missing 'frame_head_parallel' property in parallel configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(frame_head_parallel->is_object(),
                            "'frame_head_parallel' property must be a JSON object in parallel configuration JSON.");

    const auto ddp_parallel = json_object.find("ddp_parallel");
    FBAMTRAIN_JSON_VALIDATE(ddp_parallel != json_object.end(),
                            "Missing 'ddp_parallel' property in parallel configuration JSON.");
    FBAMTRAIN_JSON_VALIDATE(ddp_parallel->is_object(),
                            "'ddp_parallel' property must be a JSON object in parallel configuration JSON.");

    return ParallelConfiguration{
        .frame_head_parallel = FrameHeadParallelConfiguration::FromJson(*frame_head_parallel),
        .ddp_parallel = DdpParallelConfiguration::FromJson(*ddp_parallel)};
}

std::optional<fbamtrain::ParallelConfiguration>
fbamtrain::ParallelConfiguration::FromJsonFile(const std::string &file_path)
{
    if (!std::filesystem::exists(file_path))
    {
        return std::nullopt;
    }
    std::ifstream file_stream(file_path);
    if (!file_stream.is_open())
    {
        if (!file_stream.is_open())
        {
            throw std::runtime_error("Failed to open parallel configuration file: " + file_path);
        }
        return std::nullopt;
    }
    const nlohmann::json json_content = nlohmann::json::parse(file_stream, nullptr, false);
    if (json_content.is_discarded())
    {
        if (!file_stream.eof())
        {
            throw std::runtime_error("Failed to parse JSON from parallel configuration file: " + file_path);
        }
        return std::nullopt;
    }
    return FromJson(json_content);
}

std::string fbamtrain::ParallelConfiguration::jsonstr() const
{
    nlohmann::json ddp_parallel_json{{"use_ddp_parallel", ddp_parallel.use_ddp_parallel}};
    if (ddp_parallel.use_ddp_parallel)
    {
        ddp_parallel_json["transport"] = ddp_parallel.transport;
        if (ddp_parallel.ccl_rendezvous.has_value())
        {
            ddp_parallel_json["ccl_rendezvous"] = {
                {"ip", ddp_parallel.ccl_rendezvous->ip}, {"port", ddp_parallel.ccl_rendezvous->port}};
        }
        if (ddp_parallel.ftccl_master.has_value())
        {
            ddp_parallel_json["ftccl_master"] = {
                {"ip", ddp_parallel.ftccl_master->ip}, {"port", ddp_parallel.ftccl_master->port}};
        }
    }

    return nlohmann::json{{"frame_head_parallel",
                           {{"use_frame_head_parallel", frame_head_parallel.use_frame_head_parallel},
                            {"transport", frame_head_parallel.transport},
                            {"ccl_rendezvous",
                             {{"ip", frame_head_parallel.ccl_rendezvous.ip},
                              {"port", frame_head_parallel.ccl_rendezvous.port}}}}},
                          {"ddp_parallel", ddp_parallel_json}}
        .dump(4);
}
