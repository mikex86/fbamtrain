#pragma once

#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>

#include "module.h"

namespace pi::tensorlib
{
    class ModuleList final : public Module<>
    {
        std::vector<std::shared_ptr<Module>> modules_;

      public:
        explicit ModuleList(const std::string &name, std::vector<std::shared_ptr<Module>> modules)
            : Module(name), modules_(std::move(modules))
        {
        }

        [[nodiscard]] TraceTensor buildForward(OpGraph &graph, const std::initializer_list<TraceTensor> inputs,
                                               const bool save_input_for_backward) override
        {
            const auto input = inputs.begin()[0];
            TraceTensor h = input;
            for (const auto &module : modules_)
            {
                const TraceTensor output = module->buildForward(graph, {h}, save_input_for_backward);
                graph.deleteTensor(h);
                h = output;
            }
            return h;
        }

        void buildBackward(OpGraph & /*graph*/, const TraceTensor & /*backward_input*/,
                           const std::unordered_map<std::string, TraceTensor> &,
                           const std::unordered_map<std::string, TraceTensor> &) override
        {
            throw std::runtime_error("ModuleList backward is not implemented.");
        }

        [[nodiscard]] std::vector<ParameterEntry> parameters() const override
        {
            std::vector<ParameterEntry> params{};
            for (const auto &module : modules_)
            {
                for (auto module_params = module->parameters(); const auto &param : module_params)
                {
                    params.emplace_back(param);
                }
            }
            return params;
        }
    };
} // namespace pi::tensorlib
