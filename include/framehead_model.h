#pragma once

#include "config.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mean.h>
#include <module.h>
#include <stream_descriptor.h>
#include <tensorlib.h>

namespace pi::tensorlib
{
    class Embedding;
}

namespace fbamtrain
{
    class FrameHeadAttentionBlock;
    class FrameHeadConvBlock;

    class FrameHeadModule final : public pi::tensorlib::Module<>
    {
      public:
        explicit FrameHeadModule(const FbamModelConfiguration &config, pi::tensorlib::OpGraph &graph,
                                 pi::tensorlib::Device device, pi::tensorlib::DataType dtype,
                                 const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor);

        [[nodiscard]] pi::tensorlib::TraceTensor buildForward(pi::tensorlib::OpGraph &graph,
                                                              std::initializer_list<pi::tensorlib::TraceTensor> inputs,
                                                              bool save_input_for_backward) override;

        [[nodiscard]] std::vector<pi::tensorlib::ParameterEntry> parameters() const override;

        void buildBackward(pi::tensorlib::OpGraph &graph, const pi::tensorlib::TraceTensor &backward_input,
                      const std::unordered_map<std::string, pi::tensorlib::TraceTensor> &parameter_gradients,
                      const std::unordered_map<std::string, pi::tensorlib::TraceTensor> &operand_gradients) override;

        [[nodiscard]] const pi::tensorlib::TraceTensor &codepointEmbedding() const;
        [[nodiscard]] const pi::tensorlib::TraceTensor &positionEmbedding() const;
        [[nodiscard]] const pi::tensorlib::TraceTensor &fgREmbedding() const;
        [[nodiscard]] const pi::tensorlib::TraceTensor &fgGEmbedding() const;
        [[nodiscard]] const pi::tensorlib::TraceTensor &fgBEmbedding() const;
        [[nodiscard]] const pi::tensorlib::TraceTensor &bgREmbedding() const;
        [[nodiscard]] const pi::tensorlib::TraceTensor &bgGEmbedding() const;
        [[nodiscard]] const pi::tensorlib::TraceTensor &bgBEmbedding() const;

      private:
        uint32_t rows_;
        uint32_t cols_;
        uint32_t vocab_size_;

        FbamModelConfiguration config_;
        pi::tensorlib::Device device_;
        pi::tensorlib::DataType dtype_;
        pi::tensorlib::GpuStreamDescriptor compute_stream_descriptor_;
        FrameHeadReductionStrategy reduction_strategy_;

        std::shared_ptr<pi::tensorlib::Embedding> position_embedding_;
        std::vector<std::shared_ptr<FrameHeadConvBlock>> downsample_blocks_;
        std::vector<std::shared_ptr<FrameHeadAttentionBlock>> blocks_;
        pi::tensorlib::MeanModule mean_;

        std::optional<pi::tensorlib::TraceTensor> position_embed_;
        std::optional<pi::tensorlib::TraceTensor> codepoint_embed_;
        std::optional<pi::tensorlib::TraceTensor> fg_r_embed_;
        std::optional<pi::tensorlib::TraceTensor> fg_g_embed_;
        std::optional<pi::tensorlib::TraceTensor> fg_b_embed_;
        std::optional<pi::tensorlib::TraceTensor> bg_r_embed_;
        std::optional<pi::tensorlib::TraceTensor> bg_g_embed_;
        std::optional<pi::tensorlib::TraceTensor> bg_b_embed_;
    };
} // namespace fbamtrain
