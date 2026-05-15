#pragma once

#include "config.h"
#include "module.h"
#include "tensorlib.h"

#include "streaming_lstm.h"
#include "linear.h"
#include <stream_descriptor.h>

// forward declare StreamingLstm
namespace pi::tensorlib
{
    class StreamingLstm;
};

namespace fbamtrain
{
    struct ActionModelBackwardInput
    {
        pi::tensorlib::TraceTensor action_targets_host;
        pi::tensorlib::TraceTensor output_host;
        pi::tensorlib::GpuStreamDescriptor cross_entropy_stream_desc;
        // Scale applied to cross-entropy gradients (e.g., 1 / (seq_len * total_batch_size)).
        float loss_scale{};
    };

    class ActionModelModule final
        : public pi::tensorlib::Module<pi::tensorlib::LstmForwardStreamingResult, ActionModelBackwardInput>
    {
        FbamModelConfiguration config_;
        uint32_t rng_seed_;
        pi::tensorlib::GpuStreamDescriptor compute_stream_descriptor_;
        pi::tensorlib::GpuStreamDescriptor head_stream_descriptor_;

      public:
        pi::tensorlib::StreamingLstm lstm_;

      private:
        pi::tensorlib::Linear head_;

      public:
        ActionModelModule(FbamModelConfiguration config, uint32_t batch_size, size_t vocab_size,
                          pi::tensorlib::OpGraph &graph, pi::tensorlib::Device device,
                          pi::tensorlib::DataType dtype, const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor,
                          const pi::tensorlib::GpuStreamDescriptor &head_stream_descriptor);

        [[nodiscard]] pi::tensorlib::LstmForwardStreamingResult
        buildForward(pi::tensorlib::OpGraph &graph, std::initializer_list<pi::tensorlib::TraceTensor> inputs, bool save_input_for_backward) override;

        void buildBackward(pi::tensorlib::OpGraph &graph, const ActionModelBackwardInput &backward_input,
                      const std::unordered_map<std::string, pi::tensorlib::TraceTensor> &parameter_gradients,
                      const std::unordered_map<std::string, pi::tensorlib::TraceTensor> &operand_gradients) override;

        [[nodiscard]] pi::tensorlib::Linear &head() { return head_; }
        [[nodiscard]] const pi::tensorlib::Linear &head() const { return head_; }

        [[nodiscard]] std::vector<pi::tensorlib::ParameterEntry> parameters() const override;
    };
} // namespace fbamtrain
