#pragma once

#include <any>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "functional.h"
#include "module.h"
#include "op_graph.h"
#include "tensorlib.h"

namespace pi::tensorlib
{
    class Embedding final : public Module<>
    {
        Device device_;
        DataType dtype_;
        size_t num_embeddings_;
        size_t embedding_dim_;
        TraceTensor weight_;
        GpuStreamDescriptor compute_stream_descriptor_;

      public:
        Embedding(const std::string &name, size_t num_embeddings, size_t embedding_dim, const Device device,
                  const DataType dtype, OpGraph &graph, const uint32_t &init_rng_seed,
                  const GpuStreamDescriptor &compute_stream_descriptor)
            : Module(name), device_(device), dtype_(dtype), num_embeddings_(num_embeddings),
              embedding_dim_(embedding_dim),
              weight_(graph.createTensor({num_embeddings, embedding_dim}, dtype, device, compute_stream_descriptor, false)),
              compute_stream_descriptor_(compute_stream_descriptor)
        {
            if (dtype != DataType::BFLOAT16 && dtype != DataType::FLOAT16)
            {
                throw std::invalid_argument("Embedding currently supports BFLOAT16 or FLOAT16 weights");
            }
            if (num_embeddings_ == 0 || embedding_dim_ == 0)
            {
                throw std::invalid_argument("Embedding dimensions must be greater than zero");
            }

            FillNormal(graph, weight_, 0.0f, 1.0f, init_rng_seed, compute_stream_descriptor_);
        }

        [[nodiscard]] TraceTensor buildForward(OpGraph &graph, const std::initializer_list<TraceTensor> inputs,
                                               bool /*save_input_for_backward*/ = false) override
        {
            const auto &indices = inputs.begin()[0];
            if (indices.device() != device_)
            {
                throw std::invalid_argument("Embedding indices must be on the same device as the module");
            }
            if (indices.dtype() != DataType::UINT32)
            {
                throw std::invalid_argument("Embedding indices must be UINT32 tensors");
            }

            std::vector<uint64_t> output_shape = indices.shape().dims();
            output_shape.push_back(embedding_dim_);
            TraceTensor output = graph.createTensor(output_shape, dtype_, device_, compute_stream_descriptor_, false);

            std::unordered_map<std::string, std::any> attributes{};
            attributes.emplace("axis", static_cast<uint32_t>(0));

            graph.recordOperation(OperationEntry{.type = OpType::GATHER,
                                                 .inputs = {weight_, indices},
                                                 .outputs = {output},
                                                 .attributes = attributes,
                                                 .gpu_stream_desc = compute_stream_descriptor_});
            return output;
        }

        void buildBackward(OpGraph & /*graph*/, const TraceTensor & /*backward_input*/,
                           const std::unordered_map<std::string, TraceTensor> &,
                           const std::unordered_map<std::string, TraceTensor> &) override
        {
            throw std::runtime_error("Embedding backward is not implemented.");
        }

        [[nodiscard]] std::vector<ParameterEntry> parameters() const override
        {
            return {ParameterEntry{.name = name_ + ".weight", .tensor = weight_}};
        }
    };
} // namespace pi::tensorlib
