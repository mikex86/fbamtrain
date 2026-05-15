#pragma once

#include <string>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tensorlib.h>

#include "op_graph.h"

namespace pi::tensorlib
{

    class OpGraph;

    struct ParameterEntry
    {
        std::string name;
        const TraceTensor &tensor;
    };

    /**
     * Context to save tensors during the forward pass for use in the backward pass.
     */
    class BackwardContext
    {
        std::unordered_map<std::string, TraceTensor> saved{};

      public:
        void saveForBackward(const std::string &name, TraceTensor &tensor);

        void clear() { saved.clear(); }

        void release(OpGraph &graph);

        [[nodiscard]] bool hasSaved(const std::string &name) const;

        [[nodiscard]] const TraceTensor &getSaved(const std::string &name) const;
    };

    template <typename F = TraceTensor, typename B = TraceTensor> class Module
    {
      protected:
        std::string name_{};
        BackwardContext bw_context_{};

        explicit Module(std::string name) : name_(std::move(name)) {}

      public:
        [[nodiscard]] virtual F buildForward(OpGraph &graph, std::initializer_list<TraceTensor> inputs,
                                             bool save_input_for_backward) = 0;

        /**
         * Called to build the backward pass for this module.
         * The function will compute the local gradients for the parameters of the module.
         * The computed gradients will be inplace-added to the corresponding entries in parameter_gradients.
         * The parameter gradients map is asserted to be pre-populated with either zero tensors or pre-existing
         * gradients to accumulated as needed per the chain rule.
         * @param graph the operation graph
         * @param backward_input a templated input representing the input to the backward pass
         * @param parameter_gradients a map of parameter names to their corresponding gradient tensors
         */
        virtual void buildBackward(OpGraph &graph, const B &backward_input,
                                   const std::unordered_map<std::string, TraceTensor> &parameter_gradients,
                                   const std::unordered_map<std::string, TraceTensor> &operand_gradients) = 0;

        [[nodiscard]] virtual std::vector<ParameterEntry> parameters() const = 0;

        /**
         * Appends this module's parameter tensors to an output state map.
         *
         * The module owns the logical parameter list, while `real_tensors` maps those logical names to the concrete
         * runtime tensors created by an execution plan. Optimizer internals are intentionally excluded and should be
         * appended through the optimizer API.
         *
         * @param out Output map receiving parameter tensors under param/<parameter-name> keys.
         * @param real_tensors Runtime tensors keyed by logical parameter name.
         * @param context Description used in validation error messages.
         */
        void appendParamState(std::map<std::string, std::shared_ptr<RealTensor>> &out,
                              const std::unordered_map<std::string, std::shared_ptr<RealTensor>> &real_tensors,
                              const std::string_view context) const
        {
            for (const auto &param : parameters())
            {
                const auto it = real_tensors.find(param.name);
                if (it == real_tensors.end())
                {
                    throw std::runtime_error(std::string(context) + " is missing parameter tensor: " + param.name);
                }
                if (!it->second)
                {
                    throw std::runtime_error(std::string(context) + " parameter tensor is null: " + param.name);
                }
                const auto [_, inserted] = out.emplace("param/" + param.name, it->second);
                if (!inserted)
                {
                    throw std::runtime_error(std::string(context) + " has duplicate parameter tensor key: param/" +
                                             param.name);
                }
            }
        }

        [[nodiscard]] std::optional<ParameterEntry> getParameter(const TraceTensor &weight_tensor) const
        {
            for (const auto &param : parameters())
            {
                if (param.tensor.id() == weight_tensor.id())
                {
                    return param;
                }
            }
            return std::nullopt;
        }

        virtual ~Module() = default;
    };
} // namespace pi::tensorlib
