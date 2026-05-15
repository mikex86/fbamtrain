#pragma once

#include "config.h"

#include <memory>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <tensorlib.h>

namespace fbamtrain::optim
{
    struct ParameterGrad
    {
        std::string name;
        std::shared_ptr<pi::tensorlib::RealTensor> param;
        std::shared_ptr<pi::tensorlib::RealTensor> grad;
    };

    class Optimizer
    {
      public:
        virtual ~Optimizer() = default;

        virtual void step() = 0;

        /**
         * Appends optimizer-owned state tensors to an output state map.
         *
         * This only covers optimizer internals such as momentum, Adam moments, master weights, and optimizer-local step
         * tensors. Model parameters are owned by modules and should be appended via Module::appendParamState instead.
         *
         * @param out Output map receiving state tensors.
         * @param prefix Key prefix used for all optimizer state entries.
         */
        virtual void appendOptimState(std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &out,
                                      const std::string &prefix) = 0;

        /**
         * @brief Restores optimizer-owned state tensors previously emitted by appendOptimState.
         *
         * @param in Input map containing optimizer state tensors.
         * @param prefix Key prefix used for all optimizer state entries.
         */
        virtual void loadState(const std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &in,
                               const std::string &prefix) = 0;

        virtual void setStep(uint64_t step) = 0;
    };

    std::unique_ptr<Optimizer> CreateOptimizer(const OptimizerConfiguration &config,
                                               const std::vector<ParameterGrad> &parameters,
                                               const pi::tensorlib::Device &device,
                                               const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor);
} // namespace fbamtrain
