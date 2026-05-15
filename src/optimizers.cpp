#include "optimizers.h"

#include "executor.h"
#include "functional.h"
#include "passes.h"
#include "stream_descriptor.h"

#include <allocator.h>
#include <device_copy.h>
#include <gputx.h>
#include <shape_utils.h>
#include <stream_utils.h>
#include <tensorlib.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    std::string BuildStateKey(const std::string &prefix, const std::string &param_name, const std::string &state_name)
    {
        if (prefix.empty())
        {
            return param_name + "/" + state_name;
        }
        return prefix + "/" + param_name + "/" + state_name;
    }

    void AddStateTensor(std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &out, const std::string &key,
                        const std::shared_ptr<pi::tensorlib::RealTensor> &tensor)
    {
        if (!tensor)
        {
            return;
        }
        if (out.contains(key))
        {
            throw std::runtime_error("Duplicate optimizer state tensor key: " + key);
        }
        out.emplace(key, tensor);
    }

    void LoadStateTensor(const std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &in,
                         const std::string &key, const std::shared_ptr<pi::tensorlib::RealTensor> &target,
                         const pi::tensorlib::GpuStreamDescriptor &stream_desc)
    {
        if (!target)
        {
            return;
        }
        const auto it = in.find(key);
        if (it == in.end())
        {
            throw std::runtime_error("Missing optimizer state tensor in checkpoint: " + key);
        }
        const auto &source = it->second;
        if (!source)
        {
            throw std::runtime_error("Checkpoint tensor is null for key: " + key);
        }
        if (source->dtype() != target->dtype())
        {
            throw std::runtime_error("Optimizer state dtype mismatch for '" + key + "': expected " +
                                     pi::tensorlib::GetDataTypeName(target->dtype()) + ", got " +
                                     pi::tensorlib::GetDataTypeName(source->dtype()));
        }
        if (source->shape() != target->shape())
        {
            throw std::runtime_error("Optimizer state shape mismatch for '" + key + "'.");
        }
        target->storage()->copyFrom(*source->storage(), stream_desc);
    }

    void ValidateParamGrad(const fbamtrain::optim::ParameterGrad &parameter)
    {
        if (!parameter.param || !parameter.grad)
        {
            throw std::runtime_error("Optimizer parameter must provide both param and grad tensors.");
        }
        if (parameter.param->shape() != parameter.grad->shape())
        {
            throw std::runtime_error("Optimizer param/grad shape mismatch for: " + parameter.name);
        }
        if (parameter.param->dtype() != parameter.grad->dtype())
        {
            const auto param_dtype = parameter.param->dtype();
            const auto grad_dtype = parameter.grad->dtype();
            const bool allow_fp32_grad =
                grad_dtype == pi::tensorlib::DataType::FLOAT32 &&
                (param_dtype == pi::tensorlib::DataType::FLOAT16 || param_dtype == pi::tensorlib::DataType::BFLOAT16);
            if (!allow_fp32_grad)
            {
                throw std::runtime_error("Optimizer param/grad dtype mismatch for: " + parameter.name);
            }
        }
        if (parameter.param->device() != parameter.grad->device())
        {
            throw std::runtime_error("Optimizer param/grad device mismatch for: " + parameter.name);
        }
    }

    void ApplyInitPasses(pi::tensorlib::ExecutionPlan &execution_plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<CastImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        pi::tensorlib::passes::Transform(execution_plan, passes);
    }

    void ApplyOptimizerPass(pi::tensorlib::ExecutionPlan &execution_plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<CastImplPass>());
        passes.emplace_back(std::make_unique<OptimizerImplPass>());
        pi::tensorlib::passes::Transform(execution_plan, passes);
    }

    void ApplyMuonPasses(pi::tensorlib::ExecutionPlan &execution_plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<MatmulImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<CastImplPass>());
        passes.emplace_back(std::make_unique<ReduceSumImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastMulImplPass>());
        passes.emplace_back(std::make_unique<DivImplPass>());
        passes.emplace_back(std::make_unique<SqrtImplPass>());
        passes.emplace_back(std::make_unique<ContiguousImplPass>());
        pi::tensorlib::passes::Transform(execution_plan, passes);
    }

    pi::tensorlib::TraceTensor MulTensor(pi::tensorlib::OpGraph &graph, const pi::tensorlib::TraceTensor &lhs,
                                         const pi::tensorlib::TraceTensor &rhs,
                                         const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor)
    {
        auto output =
            graph.createTensor(lhs.shape().dims(), lhs.dtype(), lhs.device(), compute_stream_descriptor, false);
        graph.recordOperation(pi::tensorlib::OperationEntry{.type = pi::tensorlib::OpType::MUL,
                                                            .inputs = {lhs, rhs},
                                                            .outputs = {output},
                                                            .attributes = {},
                                                            .gpu_stream_desc = compute_stream_descriptor});
        return output;
    }

} // namespace

namespace fbamtrain::optim
{
    class SgdOptimizer final : public Optimizer
    {
      public:
        SgdOptimizer(const OptimizerConfiguration &config, const std::vector<ParameterGrad> &parameters,
                     const pi::tensorlib::Device &device,
                     const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor)
            : learning_rate_(config.learning_rate), weight_decay_(config.weight_decay), momentum_(config.momentum),
              nesterov_(config.nesterov), device_(device), compute_stream_descriptor_(compute_stream_descriptor),
              parameters_(parameters)
        {
            if (parameters_.empty())
            {
                return;
            }
            BuildDescriptors();
            InitializeState();
            BuildExecutor();
        }

        void step() override
        {
            if (!executor_)
            {
                return;
            }
            GPUTX_RANGE("fbamtrain::optimizer::sgd_step");
            const auto &allocator_registry = pi::tensorlib::allocator::DefaultAllocatorRegistry::instance();
            executor_->execute(allocator_registry, false);
            executor_->releaseTensors();
        }

        void appendOptimState(std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &out,
                              const std::string &prefix) override
        {
            if (entries_.empty())
            {
                return;
            }
            size_t master_index = 0;
            for (size_t idx = 0; idx < entries_.size(); ++idx)
            {
                const auto &entry = entries_[idx];
                const auto &name = parameters_[idx].name;
                AddStateTensor(out, BuildStateKey(prefix, name, "velocity"), velocity_real_[idx]);
                if (entry.use_master)
                {
                    AddStateTensor(out, BuildStateKey(prefix, name, "master"), master_params_[master_index]);
                    ++master_index;
                }
            }
        }

        void loadState(const std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &in,
                       const std::string &prefix) override
        {
            if (entries_.empty())
            {
                return;
            }
            size_t master_index = 0;
            for (size_t idx = 0; idx < entries_.size(); ++idx)
            {
                const auto &entry = entries_[idx];
                const auto &name = parameters_[idx].name;
                LoadStateTensor(in, BuildStateKey(prefix, name, "velocity"), velocity_real_[idx],
                                compute_stream_descriptor_);
                if (entry.use_master)
                {
                    LoadStateTensor(in, BuildStateKey(prefix, name, "master"), master_params_[master_index],
                                    compute_stream_descriptor_);
                    ++master_index;
                }
            }
        }

        void setStep(const uint64_t /*step*/) override {}

      private:
        struct SgdEntry
        {
            pi::tensorlib::TraceTensor param;
            pi::tensorlib::TraceTensor grad;
            std::optional<pi::tensorlib::TraceTensor> master;
            pi::tensorlib::TraceTensor velocity;
            pi::tensorlib::DataType param_dtype{};
            bool use_master{};
        };

        void BuildDescriptors()
        {
            entries_.reserve(parameters_.size());
            master_params_.reserve(parameters_.size());
            init_input_descriptors_.reserve(parameters_.size() * 2);
            optimizer_input_descriptors_.reserve(parameters_.size() * 4);
            init_exec_parameters_.reserve(parameters_.size() * 2);

            for (const auto &parameter : parameters_)
            {
                ValidateParamGrad(parameter);
                const auto param_dtype = parameter.param->dtype();
                const bool use_master = (param_dtype != pi::tensorlib::DataType::FLOAT32);
                auto param_trace = pi::tensorlib::TraceTensor::Create(
                    parameter.param->shape().dims(), parameter.param->dtype(), parameter.param->device(),
                    compute_stream_descriptor_, parameter.param->pinned());
                auto grad_trace = pi::tensorlib::TraceTensor::Create(
                    parameter.grad->shape().dims(), parameter.grad->dtype(), parameter.grad->device(),
                    compute_stream_descriptor_, parameter.grad->pinned());
                param_trace.markRetained();
                grad_trace.markRetained();
                auto velocity_trace = pi::tensorlib::TraceTensor::Create(
                    parameter.param->shape().dims(), pi::tensorlib::DataType::FLOAT32, parameter.param->device(),
                    compute_stream_descriptor_, parameter.param->pinned());
                velocity_trace.markRetained();

                std::optional<pi::tensorlib::TraceTensor> master_trace{};
                if (use_master)
                {
                    auto master = pi::tensorlib::TraceTensor::Create(
                        parameter.param->shape().dims(), pi::tensorlib::DataType::FLOAT32, parameter.param->device(),
                        compute_stream_descriptor_, parameter.param->pinned());
                    master.markRetained();
                    master_trace = master;
                }

                entries_.push_back(SgdEntry{.param = param_trace,
                                            .grad = grad_trace,
                                            .master = master_trace,
                                            .velocity = velocity_trace,
                                            .param_dtype = param_dtype,
                                            .use_master = use_master});

                init_input_descriptors_.push_back(
                    pi::tensorlib::GraphInputDescriptor{.name = parameter.name, .tensor = param_trace});
                init_input_descriptors_.push_back(
                    pi::tensorlib::GraphInputDescriptor{.name = parameter.name + "_grad", .tensor = grad_trace});

                optimizer_input_descriptors_.push_back(
                    pi::tensorlib::GraphInputDescriptor{.name = parameter.name, .tensor = param_trace});
                optimizer_input_descriptors_.push_back(
                    pi::tensorlib::GraphInputDescriptor{.name = parameter.name + "_grad", .tensor = grad_trace});
                optimizer_input_descriptors_.push_back(pi::tensorlib::GraphInputDescriptor{
                    .name = parameter.name + "_velocity", .tensor = velocity_trace});
                if (master_trace)
                {
                    optimizer_input_descriptors_.push_back(pi::tensorlib::GraphInputDescriptor{
                        .name = parameter.name + "_master", .tensor = master_trace.value()});
                }

                init_exec_parameters_.push_back(
                    pi::tensorlib::GraphExecutionInputDescriptor{.name = parameter.name, .tensor = parameter.param});
                init_exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
                    .name = parameter.name + "_grad", .tensor = parameter.grad});
            }
        }

        void InitializeState()
        {
            pi::tensorlib::OpGraph init_graph{init_input_descriptors_, {}};
            for (auto &entry : entries_)
            {
                init_graph.createTensor(entry.velocity);
                pi::tensorlib::FillZeros(init_graph, entry.velocity, compute_stream_descriptor_);
                if (entry.use_master && entry.master)
                {
                    init_graph.createTensor(entry.master.value());
                    auto param_fp32 =
                        entry.param.to(init_graph, pi::tensorlib::DataType::FLOAT32, compute_stream_descriptor_);
                    entry.master->populate(init_graph, param_fp32, compute_stream_descriptor_);
                }
            }
            init_graph.finalize();

            auto init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, init_exec_parameters_, {});
            ApplyInitPasses(init_plan);

            auto &backend = pi::tensorlib::ExecutionBackend::getInstance();
            pi::tensorlib::Executor init_executor{std::move(init_plan), backend, device_.ordinal};
            const auto &allocator_registry = pi::tensorlib::allocator::DefaultAllocatorRegistry::instance();
            init_executor.execute(allocator_registry);
            init_executor.await();

            velocity_real_.clear();
            master_params_.clear();
            velocity_real_.reserve(entries_.size());
            master_params_.reserve(entries_.size());
            exec_parameters_.clear();
            exec_parameters_.reserve(entries_.size() * 4);

            for (size_t idx = 0; idx < entries_.size(); ++idx)
            {
                const auto &entry = entries_[idx];
                auto velocity_real = init_executor.getOutput(entry.velocity);
                if (!velocity_real.has_value())
                {
                    throw std::runtime_error("Failed to resolve SGD velocity tensor");
                }
                velocity_real_.push_back(velocity_real.value());

                std::shared_ptr<pi::tensorlib::RealTensor> master_real{};
                if (entry.use_master && entry.master)
                {
                    auto resolved = init_executor.getOutput(entry.master.value());
                    if (!resolved.has_value())
                    {
                        throw std::runtime_error("Failed to resolve SGD master tensor");
                    }
                    master_real = resolved.value();
                    master_params_.push_back(master_real);
                }

                exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
                    .name = parameters_[idx].name, .tensor = parameters_[idx].param});
                exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
                    .name = parameters_[idx].name + "_grad", .tensor = parameters_[idx].grad});
                exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
                    .name = parameters_[idx].name + "_velocity", .tensor = velocity_real_.back()});
                if (entry.use_master && entry.master)
                {
                    exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
                        .name = parameters_[idx].name + "_master", .tensor = master_real});
                }
            }
        }

        void BuildExecutor()
        {
            pi::tensorlib::OpGraph graph{{}, optimizer_input_descriptors_};
            for (auto &entry : entries_)
            {
                auto &param_for_update = (entry.use_master && entry.master) ? entry.master.value() : entry.param;
                auto grad_for_update = entry.use_master ? entry.grad.to(graph, pi::tensorlib::DataType::FLOAT32,
                                                                        compute_stream_descriptor_)
                                                        : entry.grad;

                pi::tensorlib::OptimizerSgd(graph, param_for_update, grad_for_update, entry.velocity, learning_rate_,
                                            momentum_, weight_decay_, nesterov_, compute_stream_descriptor_);

                if (entry.use_master && entry.master)
                {
                    auto param_cast = param_for_update.to(graph, entry.param_dtype, compute_stream_descriptor_);
                    entry.param.populate(graph, param_cast, compute_stream_descriptor_);
                }
            }
            graph.finalize();

            auto execution_plan = pi::tensorlib::ExecutionPlan::FromGraph(graph, {}, exec_parameters_);
            ApplyOptimizerPass(execution_plan);

            auto &backend = pi::tensorlib::ExecutionBackend::getInstance();
            executor_ = std::make_unique<pi::tensorlib::Executor>(std::move(execution_plan), backend, device_.ordinal);
        }

        float learning_rate_{};
        float weight_decay_{};
        float momentum_{};
        bool nesterov_{};

        pi::tensorlib::Device device_;

        pi::tensorlib::GpuStreamDescriptor compute_stream_descriptor_;
        std::vector<ParameterGrad> parameters_{};
        std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> velocity_real_{};
        std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> master_params_{};
        std::vector<SgdEntry> entries_{};
        std::vector<pi::tensorlib::GraphInputDescriptor> init_input_descriptors_{};
        std::vector<pi::tensorlib::GraphInputDescriptor> optimizer_input_descriptors_{};
        std::vector<pi::tensorlib::GraphExecutionInputDescriptor> init_exec_parameters_{};
        std::vector<pi::tensorlib::GraphExecutionInputDescriptor> exec_parameters_{};
        std::unique_ptr<pi::tensorlib::Executor> executor_{};
    };

    class AdamwOptimizer final : public Optimizer
    {
      public:
        AdamwOptimizer(const OptimizerConfiguration &config, const std::vector<ParameterGrad> &parameters,
                       const pi::tensorlib::Device &device,
                       const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor)
            : learning_rate_(config.learning_rate), weight_decay_(config.weight_decay), beta1_(config.beta1),
              beta2_(config.beta2), eps_(config.eps), device_(device),
              compute_stream_descriptor_(compute_stream_descriptor), parameters_(parameters)
        {
            if (parameters_.empty())
            {
                return;
            }
            BuildDescriptors();
            InitializeState();
            BuildExecutor();
        }

        void step() override
        {
            if (!executor_)
            {
                return;
            }
            GPUTX_RANGE("fbamtrain::optimizer::adamw_step");
            const uint64_t step = ReadStepTensor() + 1;
            WriteStepTensor(step);
            const float step_f = static_cast<float>(step);
            const float bias_correction1 = 1.0f - std::pow(beta1_, step_f);
            const float bias_correction2 = 1.0f - std::pow(beta2_, step_f);
            UpdateBiasCorrections(bias_correction1, bias_correction2);
            const auto &allocator_registry = pi::tensorlib::allocator::DefaultAllocatorRegistry::instance();
            executor_->execute(allocator_registry, false);
            executor_->releaseTensors();
        }

        void appendOptimState(std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &out,
                              const std::string &prefix) override
        {
            if (entries_.empty())
            {
                return;
            }
            AddStateTensor(out, BuildStateKey(prefix, "__optimizer", "step"), step_host_);
            for (size_t idx = 0; idx < entries_.size(); ++idx)
            {
                const auto &entry = entries_[idx];
                const auto &state = state_[idx];
                const auto &name = parameters_[idx].name;
                AddStateTensor(out, BuildStateKey(prefix, name, "m"), state.m);
                AddStateTensor(out, BuildStateKey(prefix, name, "v"), state.v);
                if (entry.use_master)
                {
                    AddStateTensor(out, BuildStateKey(prefix, name, "master"), state.master);
                }
            }
        }

        void loadState(const std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &in,
                       const std::string &prefix) override
        {
            if (entries_.empty())
            {
                return;
            }
            const auto step_key = BuildStateKey(prefix, "__optimizer", "step");
            if (in.contains(step_key))
            {
                LoadStateTensor(in, step_key, step_host_, compute_stream_descriptor_);
            }
            for (size_t idx = 0; idx < entries_.size(); ++idx)
            {
                const auto &entry = entries_[idx];
                auto &state = state_[idx];
                const auto &name = parameters_[idx].name;
                const auto m_key = BuildStateKey(prefix, name, "m");
                const auto v_key = BuildStateKey(prefix, name, "v");
                if (in.contains(m_key))
                {
                    LoadStateTensor(in, m_key, state.m, compute_stream_descriptor_);
                }
                if (in.contains(v_key))
                {
                    LoadStateTensor(in, v_key, state.v, compute_stream_descriptor_);
                }
                if (entry.use_master)
                {
                    const auto master_key = BuildStateKey(prefix, name, "master");
                    if (in.contains(master_key))
                    {
                        LoadStateTensor(in, master_key, state.master, compute_stream_descriptor_);
                    }
                }
            }
        }

        void setStep(const uint64_t step) override
        {
            WriteStepTensor(step);
        }

      private:
        struct AdamwEntry
        {
            pi::tensorlib::TraceTensor param;
            pi::tensorlib::TraceTensor grad;
            std::optional<pi::tensorlib::TraceTensor> master;
            pi::tensorlib::TraceTensor m;
            pi::tensorlib::TraceTensor v;
            pi::tensorlib::DataType param_dtype{};
            bool use_master{};
        };

        struct AdamwState
        {
            std::shared_ptr<pi::tensorlib::RealTensor> master;
            std::shared_ptr<pi::tensorlib::RealTensor> m;
            std::shared_ptr<pi::tensorlib::RealTensor> v;
        };

        void BuildDescriptors()
        {
            entries_.reserve(parameters_.size());
            state_.reserve(parameters_.size());
            init_input_descriptors_.reserve(parameters_.size() * 2);
            optimizer_input_descriptors_.reserve(parameters_.size() * 5 + 2);
            init_exec_parameters_.reserve(parameters_.size() * 2);

            const auto &first_param = parameters_.front();
            ValidateParamGrad(first_param);
            const auto &device = first_param.param->device();
            constexpr auto device_cpu =
                pi::tensorlib::Device{.device_type = pi::tensorlib::DeviceType::CPU, .ordinal = 0};

            bias_correction1_host_ =
                pi::tensorlib::RealTensor::Allocate({1}, pi::tensorlib::DataType::FLOAT32, device_cpu, true);
            bias_correction2_host_ =
                pi::tensorlib::RealTensor::Allocate({1}, pi::tensorlib::DataType::FLOAT32, device_cpu, true);

            bias_correction1_trace_ = pi::tensorlib::TraceTensor::Create({1}, pi::tensorlib::DataType::FLOAT32, device,
                                                                         compute_stream_descriptor_, false);
            bias_correction2_trace_ = pi::tensorlib::TraceTensor::Create({1}, pi::tensorlib::DataType::FLOAT32, device,
                                                                         compute_stream_descriptor_, false);
            bias_correction1_trace_->markRetained();
            bias_correction2_trace_->markRetained();

            optimizer_input_descriptors_.push_back(pi::tensorlib::GraphInputDescriptor{
                .name = "adamw_bias_correction1", .tensor = bias_correction1_trace_.value()});
            optimizer_input_descriptors_.push_back(pi::tensorlib::GraphInputDescriptor{
                .name = "adamw_bias_correction2", .tensor = bias_correction2_trace_.value()});

            for (const auto &parameter : parameters_)
            {
                ValidateParamGrad(parameter);
                const auto param_dtype = parameter.param->dtype();
                const bool use_master = (param_dtype != pi::tensorlib::DataType::FLOAT32);
                auto m_trace = pi::tensorlib::TraceTensor::Create(
                    parameter.param->shape().dims(), pi::tensorlib::DataType::FLOAT32, parameter.param->device(),
                    compute_stream_descriptor_, parameter.param->pinned());
                auto v_trace = pi::tensorlib::TraceTensor::Create(
                    parameter.param->shape().dims(), pi::tensorlib::DataType::FLOAT32, parameter.param->device(),
                    compute_stream_descriptor_, parameter.param->pinned());
                m_trace.markRetained();
                v_trace.markRetained();

                auto param_trace = pi::tensorlib::TraceTensor::Create(
                    parameter.param->shape().dims(), parameter.param->dtype(), parameter.param->device(),
                    compute_stream_descriptor_, parameter.param->pinned());
                auto grad_trace = pi::tensorlib::TraceTensor::Create(
                    parameter.grad->shape().dims(), parameter.grad->dtype(), parameter.grad->device(),
                    compute_stream_descriptor_, parameter.grad->pinned());
                param_trace.markRetained();
                grad_trace.markRetained();

                std::optional<pi::tensorlib::TraceTensor> master_trace{};
                if (use_master)
                {
                    auto master = pi::tensorlib::TraceTensor::Create(
                        parameter.param->shape().dims(), pi::tensorlib::DataType::FLOAT32, parameter.param->device(),
                        compute_stream_descriptor_, parameter.param->pinned());
                    master.markRetained();
                    master_trace = master;
                }

                entries_.push_back(AdamwEntry{.param = param_trace,
                                              .grad = grad_trace,
                                              .master = master_trace,
                                              .m = m_trace,
                                              .v = v_trace,
                                              .param_dtype = param_dtype,
                                              .use_master = use_master});

                init_input_descriptors_.push_back(
                    pi::tensorlib::GraphInputDescriptor{.name = parameter.name, .tensor = param_trace});
                init_input_descriptors_.push_back(
                    pi::tensorlib::GraphInputDescriptor{.name = parameter.name + "_grad", .tensor = grad_trace});

                optimizer_input_descriptors_.push_back(
                    pi::tensorlib::GraphInputDescriptor{.name = parameter.name, .tensor = param_trace});
                optimizer_input_descriptors_.push_back(
                    pi::tensorlib::GraphInputDescriptor{.name = parameter.name + "_grad", .tensor = grad_trace});
                optimizer_input_descriptors_.push_back(
                    pi::tensorlib::GraphInputDescriptor{.name = parameter.name + "_m", .tensor = m_trace});
                optimizer_input_descriptors_.push_back(
                    pi::tensorlib::GraphInputDescriptor{.name = parameter.name + "_v", .tensor = v_trace});
                if (master_trace)
                {
                    optimizer_input_descriptors_.push_back(pi::tensorlib::GraphInputDescriptor{
                        .name = parameter.name + "_master", .tensor = master_trace.value()});
                }

                init_exec_parameters_.push_back(
                    pi::tensorlib::GraphExecutionInputDescriptor{.name = parameter.name, .tensor = parameter.param});
                init_exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
                    .name = parameter.name + "_grad", .tensor = parameter.grad});
            }
        }

        void InitializeState()
        {
            step_host_ = pi::tensorlib::RealTensor::Allocate({1}, pi::tensorlib::DataType::UINT64,
                                                             pi::tensorlib::Device{
                                                                 .device_type = pi::tensorlib::DeviceType::CPU,
                                                                 .ordinal = 0},
                                                             true);
            WriteStepTensor(0);

            pi::tensorlib::OpGraph init_graph{init_input_descriptors_, {}};
            init_graph.createTensor(bias_correction1_trace_.value());
            init_graph.createTensor(bias_correction2_trace_.value());
            pi::tensorlib::FillZeros(init_graph, bias_correction1_trace_.value(), compute_stream_descriptor_);
            pi::tensorlib::FillZeros(init_graph, bias_correction2_trace_.value(), compute_stream_descriptor_);
            for (auto &entry : entries_)
            {
                init_graph.createTensor(entry.m);
                init_graph.createTensor(entry.v);
                pi::tensorlib::FillZeros(init_graph, entry.m, compute_stream_descriptor_);
                pi::tensorlib::FillZeros(init_graph, entry.v, compute_stream_descriptor_);
                if (entry.use_master && entry.master)
                {
                    init_graph.createTensor(entry.master.value());
                    auto param_fp32 =
                        entry.param.to(init_graph, pi::tensorlib::DataType::FLOAT32, compute_stream_descriptor_);
                    entry.master->populate(init_graph, param_fp32, compute_stream_descriptor_);
                }
            }
            init_graph.finalize();

            auto init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, init_exec_parameters_, {});
            ApplyInitPasses(init_plan);

            auto &backend = pi::tensorlib::ExecutionBackend::getInstance();
            pi::tensorlib::Executor init_executor{std::move(init_plan), backend, device_.ordinal};
            const auto &allocator_registry = pi::tensorlib::allocator::DefaultAllocatorRegistry::instance();
            init_executor.execute(allocator_registry);
            init_executor.await();

            bias_correction1_ = init_executor.getOutput(bias_correction1_trace_.value()).value();
            bias_correction2_ = init_executor.getOutput(bias_correction2_trace_.value()).value();

            state_.clear();
            exec_parameters_.clear();
            exec_parameters_.reserve(parameters_.size() * 5 + 2);

            exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{.name = "adamw_bias_correction1",
                                                                                    .tensor = bias_correction1_});
            exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{.name = "adamw_bias_correction2",
                                                                                    .tensor = bias_correction2_});

            for (size_t idx = 0; idx < entries_.size(); ++idx)
            {
                const auto &entry = entries_[idx];
                auto m_real = init_executor.getOutput(entry.m);
                auto v_real = init_executor.getOutput(entry.v);
                if (!m_real.has_value() || !v_real.has_value())
                {
                    throw std::runtime_error("Failed to resolve AdamW state tensors");
                }
                std::shared_ptr<pi::tensorlib::RealTensor> master{};
                if (entry.use_master && entry.master)
                {
                    auto master_real = init_executor.getOutput(entry.master.value());
                    if (!master_real.has_value())
                    {
                        throw std::runtime_error("Failed to resolve AdamW master tensor");
                    }
                    master = master_real.value();
                }
                state_.push_back(AdamwState{.master = master, .m = m_real.value(), .v = v_real.value()});
                exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
                    .name = parameters_[idx].name, .tensor = parameters_[idx].param});
                exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
                    .name = parameters_[idx].name + "_grad", .tensor = parameters_[idx].grad});
                exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
                    .name = parameters_[idx].name + "_m", .tensor = state_.back().m});
                exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
                    .name = parameters_[idx].name + "_v", .tensor = state_.back().v});
                if (entry.use_master && entry.master)
                {
                    exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
                        .name = parameters_[idx].name + "_master", .tensor = master});
                }
            }
        }

        void UpdateBiasCorrections(const float bias_correction1, const float bias_correction2)
        {
            if (!bias_correction1_host_ || !bias_correction2_host_ || !bias_correction1_ || !bias_correction2_)
            {
                throw std::runtime_error("AdamW bias correction tensors are not initialized.");
            }
            *static_cast<float *>(bias_correction1_host_->dataptr()) = bias_correction1;
            *static_cast<float *>(bias_correction2_host_->dataptr()) = bias_correction2;
            const auto stream_bundle = pi::tensorlib::ExecutionBackend::GetStreamBundle(bias_correction1_->device());
            if (!stream_bundle)
            {
                throw std::runtime_error("Failed to resolve stream bundle for AdamW bias correction update.");
            }
            const auto stream = pi::tensorlib::streamutils::GetStream(stream_bundle, compute_stream_descriptor_);
            pi::tensorlib::internal::device_copy::PerformDeviceCopy(bias_correction1_host_, bias_correction1_, stream);
            pi::tensorlib::internal::device_copy::PerformDeviceCopy(bias_correction2_host_, bias_correction2_, stream);
        }

        uint64_t ReadStepTensor() const
        {
            if (!step_host_)
            {
                return 0;
            }
            return *static_cast<const uint64_t *>(step_host_->dataptr());
        }

        void WriteStepTensor(const uint64_t step)
        {
            if (!step_host_)
            {
                return;
            }
            *static_cast<uint64_t *>(step_host_->dataptr()) = step;
        }

        void BuildExecutor()
        {
            pi::tensorlib::OpGraph graph{{}, optimizer_input_descriptors_};
            for (auto &entry : entries_)
            {
                auto &param_for_update = (entry.use_master && entry.master) ? entry.master.value() : entry.param;
                auto grad_for_update = entry.use_master ? entry.grad.to(graph, pi::tensorlib::DataType::FLOAT32,
                                                                        compute_stream_descriptor_)
                                                        : entry.grad;

                pi::tensorlib::OptimizerAdamw(graph, param_for_update, grad_for_update, entry.m, entry.v,
                                              bias_correction1_trace_.value(), bias_correction2_trace_.value(),
                                              learning_rate_, beta1_, beta2_, eps_, weight_decay_,
                                              compute_stream_descriptor_);

                if (entry.use_master && entry.master)
                {
                    auto param_cast = param_for_update.to(graph, entry.param_dtype, compute_stream_descriptor_);
                    entry.param.populate(graph, param_cast, compute_stream_descriptor_);
                }
            }
            graph.finalize();

            auto execution_plan = pi::tensorlib::ExecutionPlan::FromGraph(graph, {}, exec_parameters_);
            ApplyOptimizerPass(execution_plan);

            auto &backend = pi::tensorlib::ExecutionBackend::getInstance();
            executor_ = std::make_unique<pi::tensorlib::Executor>(std::move(execution_plan), backend, device_.ordinal);
        }

        float learning_rate_{};
        float weight_decay_{};
        float beta1_{};
        float beta2_{};
        float eps_{};

        pi::tensorlib::Device device_{};
        pi::tensorlib::GpuStreamDescriptor compute_stream_descriptor_;

        std::vector<ParameterGrad> parameters_{};
        std::vector<AdamwEntry> entries_{};
        std::vector<AdamwState> state_{};
        std::shared_ptr<pi::tensorlib::RealTensor> bias_correction1_{};
        std::shared_ptr<pi::tensorlib::RealTensor> bias_correction2_{};
        std::shared_ptr<pi::tensorlib::RealTensor> bias_correction1_host_{};
        std::shared_ptr<pi::tensorlib::RealTensor> bias_correction2_host_{};
        std::shared_ptr<pi::tensorlib::RealTensor> step_host_{};
        std::optional<pi::tensorlib::TraceTensor> bias_correction1_trace_{};
        std::optional<pi::tensorlib::TraceTensor> bias_correction2_trace_{};
        std::vector<pi::tensorlib::GraphInputDescriptor> init_input_descriptors_{};
        std::vector<pi::tensorlib::GraphInputDescriptor> optimizer_input_descriptors_{};
        std::vector<pi::tensorlib::GraphExecutionInputDescriptor> init_exec_parameters_{};
        std::vector<pi::tensorlib::GraphExecutionInputDescriptor> exec_parameters_{};
        std::unique_ptr<pi::tensorlib::Executor> executor_{};
    };

#define NUM_NEWTON_SCHULZ_ITERS 5

    class MuonOptimizer final : public Optimizer
    {
      public:
        MuonOptimizer(const OptimizerConfiguration &config, const std::vector<ParameterGrad> &parameters,
                      const pi::tensorlib::Device &device,
                      const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor)
            : learning_rate_(config.learning_rate), weight_decay_(config.weight_decay), momentum_(config.momentum),
              device_(device), compute_stream_descriptor_(compute_stream_descriptor), parameters_(parameters)
        {
            if (parameters_.empty())
            {
                return;
            }
            BuildDescriptors();
            InitializeState();
            BuildPreExecutor();
            BuildMainExecutor();
        }

        void step() override
        {
            if (!pre_executor_ || !main_executor_)
            {
                return;
            }
            GPUTX_RANGE("fbamtrain::optimizer::muon_step");
            const auto &allocator_registry = pi::tensorlib::allocator::DefaultAllocatorRegistry::instance();
            pre_executor_->execute(allocator_registry, false);
            main_executor_->execute(allocator_registry, false);
            pre_executor_->releaseTensors();
            main_executor_->releaseTensors();
        }

        void appendOptimState(std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &out,
                              const std::string &prefix) override
        {
            if (entries_.empty())
            {
                return;
            }
            for (size_t idx = 0; idx < entries_.size(); ++idx)
            {
                const auto &entry = entries_[idx];
                const auto &state = state_[idx];
                const auto &name = parameters_[idx].name;
                AddStateTensor(out, BuildStateKey(prefix, name, "momentum"), state.momentum);
                if (entry.use_master)
                {
                    AddStateTensor(out, BuildStateKey(prefix, name, "master"), state.master);
                }
            }
        }

        void loadState(const std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &in,
                       const std::string &prefix) override
        {
            if (entries_.empty())
            {
                return;
            }
            for (size_t idx = 0; idx < entries_.size(); ++idx)
            {
                const auto &entry = entries_[idx];
                auto &state = state_[idx];
                const auto &name = parameters_[idx].name;
                const auto momentum_key = BuildStateKey(prefix, name, "momentum");
                if (in.contains(momentum_key))
                {
                    LoadStateTensor(in, momentum_key, state.momentum, compute_stream_descriptor_);
                }
                if (entry.use_master)
                {
                    const auto master_key = BuildStateKey(prefix, name, "master");
                    if (in.contains(master_key))
                    {
                        LoadStateTensor(in, master_key, state.master, compute_stream_descriptor_);
                    }
                }
            }
        }

        void setStep(const uint64_t /*step*/) override {}

      private:
        struct MuonEntry
        {
            pi::tensorlib::TraceTensor param;
            pi::tensorlib::TraceTensor grad;
            std::optional<pi::tensorlib::TraceTensor> master;
            pi::tensorlib::TraceTensor momentum;
            pi::tensorlib::TraceTensor update_bf16;
            pi::tensorlib::TraceTensor norm;
            pi::tensorlib::DataType param_dtype{};
            bool use_master{};
            bool needs_flatten{};
            bool transpose{};
            bool scale_transpose{};
            uint64_t rows{};
            uint64_t cols{};
            float scale{};
        };

        struct MuonState
        {
            std::shared_ptr<pi::tensorlib::RealTensor> master;
            std::shared_ptr<pi::tensorlib::RealTensor> momentum;
            std::shared_ptr<pi::tensorlib::RealTensor> update_bf16;
            std::shared_ptr<pi::tensorlib::RealTensor> norm;
        };

        /**
         * Muon is sensitive to weight matrix layout; We are not storing in pytorch default layout.
         * Hence, we need to virtually transpose certain weight matrices during the newton schulz iterations.
         */
        static bool ShouldTransposeScale(const std::string &name, const pi::tensorlib::Shape &shape)
        {
            if (shape.ndims() != 2)
            {
                return false;
            }
            if (name.find("embedding") != std::string::npos)
            {
                return false;
            }
            if (name.find(".weights_ih") != std::string::npos || name.find(".weights_hh") != std::string::npos)
            {
                return true;
            }
            if (name.find(".w_qkv") != std::string::npos || name.find(".w_proj") != std::string::npos)
            {
                return true;
            }
            if (name.find(".mlp.fc1.weight") != std::string::npos || name.find(".mlp.fc2.weight") != std::string::npos)
            {
                return true;
            }
            if (name.find(".head.weight") != std::string::npos)
            {
                return true;
            }
            return false;
        }

        void BuildDescriptors()
        {
            entries_.reserve(parameters_.size());
            state_.reserve(parameters_.size());
            init_input_descriptors_.reserve(parameters_.size() * 2);
            pre_input_descriptors_.reserve(parameters_.size() * 4);
            main_input_descriptors_.reserve(parameters_.size() * 4);
            init_exec_parameters_.reserve(parameters_.size() * 2);

            for (const auto &parameter : parameters_)
            {
                ValidateParamGrad(parameter);
                if (parameter.param->shape().ndims() < 2)
                {
                    throw std::runtime_error("Muon optimizer requires parameters with at least 2 dimensions: " +
                                             parameter.name);
                }

                const auto param_dtype = parameter.param->dtype();
                const bool use_master = (param_dtype != pi::tensorlib::DataType::FLOAT32);

                auto param_trace = pi::tensorlib::TraceTensor::Create(
                    parameter.param->shape().dims(), parameter.param->dtype(), parameter.param->device(),
                    compute_stream_descriptor_, parameter.param->pinned());
                auto grad_trace = pi::tensorlib::TraceTensor::Create(
                    parameter.grad->shape().dims(), parameter.grad->dtype(), parameter.grad->device(),
                    compute_stream_descriptor_, parameter.grad->pinned());
                param_trace.markRetained();
                grad_trace.markRetained();

                auto momentum_trace = pi::tensorlib::TraceTensor::Create(
                    parameter.param->shape().dims(), pi::tensorlib::DataType::FLOAT32, parameter.param->device(),
                    compute_stream_descriptor_, parameter.param->pinned());
                momentum_trace.markRetained();

                auto update_bf16_trace = pi::tensorlib::TraceTensor::Create(
                    parameter.param->shape().dims(), pi::tensorlib::DataType::BFLOAT16, parameter.param->device(),
                    compute_stream_descriptor_, parameter.param->pinned());
                update_bf16_trace.markRetained();

                auto norm_trace =
                    pi::tensorlib::TraceTensor::Create({1, 1}, pi::tensorlib::DataType::FLOAT32,
                                                       parameter.param->device(), compute_stream_descriptor_, false);
                norm_trace.markRetained();

                std::optional<pi::tensorlib::TraceTensor> master_trace{};
                if (use_master)
                {
                    auto master = pi::tensorlib::TraceTensor::Create(
                        parameter.param->shape().dims(), pi::tensorlib::DataType::FLOAT32, parameter.param->device(),
                        compute_stream_descriptor_, parameter.param->pinned());
                    master.markRetained();
                    master_trace = master;
                }

                const auto &shape = parameter.param->shape();
                const auto ndims = shape.ndims();

                bool needs_flatten = false;

                // Stored matrix dimensions (after any 4D flatten); this matches actual memory layout.
                uint64_t stored_rows = shape[0];
                uint64_t stored_cols = shape[1];
                if (ndims == 4)
                {
                    needs_flatten = true;
                    stored_rows = shape[0];
                    stored_cols = shape[1] * shape[2] * shape[3];
                }
                else if (ndims != 2)
                {
                    throw std::runtime_error("Muon optimizer currently supports only 2D or 4D parameters: " +
                                             parameter.name);
                }
                // Newton–Schulz expects to operate on the matrix with rows <= cols,
                // so we transpose for the iteration when needed.
                const bool transpose = stored_rows > stored_cols;

                // For scaling, we must use logical rows/cols (PyTorch layout).
                // Some parameters are stored transposed relative to PyTorch, so swap for the scale ratio.
                const bool scale_transpose = ShouldTransposeScale(parameter.name, shape);
                const uint64_t scale_rows = scale_transpose ? stored_cols : stored_rows;
                const uint64_t scale_cols = scale_transpose ? stored_rows : stored_cols;
                const float ratio = static_cast<float>(scale_rows) / static_cast<float>(scale_cols);
                const float scale = std::sqrt(std::max(1.0f, ratio));

                entries_.push_back(MuonEntry{.param = param_trace,
                                             .grad = grad_trace,
                                             .master = master_trace,
                                             .momentum = momentum_trace,
                                             .update_bf16 = update_bf16_trace,
                                             .norm = norm_trace,
                                             .param_dtype = param_dtype,
                                             .use_master = use_master,
                                             .needs_flatten = needs_flatten,
                                             .transpose = transpose,
                                             .scale_transpose = scale_transpose,
                                             .rows = stored_rows,
                                             .cols = stored_cols,
                                             .scale = scale});

                init_input_descriptors_.push_back(
                    pi::tensorlib::GraphInputDescriptor{.name = parameter.name, .tensor = param_trace});
                init_input_descriptors_.push_back(
                    pi::tensorlib::GraphInputDescriptor{.name = parameter.name + "_grad", .tensor = grad_trace});

                pre_input_descriptors_.push_back(
                    pi::tensorlib::GraphInputDescriptor{.name = parameter.name + "_grad", .tensor = grad_trace});
                pre_input_descriptors_.push_back(pi::tensorlib::GraphInputDescriptor{
                    .name = parameter.name + "_muon_momentum", .tensor = momentum_trace});
                pre_input_descriptors_.push_back(pi::tensorlib::GraphInputDescriptor{
                    .name = parameter.name + "_muon_update", .tensor = update_bf16_trace});
                pre_input_descriptors_.push_back(
                    pi::tensorlib::GraphInputDescriptor{.name = parameter.name + "_muon_norm", .tensor = norm_trace});

                main_input_descriptors_.push_back(
                    pi::tensorlib::GraphInputDescriptor{.name = parameter.name, .tensor = param_trace});
                main_input_descriptors_.push_back(pi::tensorlib::GraphInputDescriptor{
                    .name = parameter.name + "_muon_update", .tensor = update_bf16_trace});
                main_input_descriptors_.push_back(
                    pi::tensorlib::GraphInputDescriptor{.name = parameter.name + "_muon_norm", .tensor = norm_trace});
                if (master_trace)
                {
                    main_input_descriptors_.push_back(pi::tensorlib::GraphInputDescriptor{
                        .name = parameter.name + "_master", .tensor = master_trace.value()});
                }

                init_exec_parameters_.push_back(
                    pi::tensorlib::GraphExecutionInputDescriptor{.name = parameter.name, .tensor = parameter.param});
                init_exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
                    .name = parameter.name + "_grad", .tensor = parameter.grad});
            }
        }

        void InitializeState()
        {
            pi::tensorlib::OpGraph init_graph{init_input_descriptors_, {}};
            for (auto &entry : entries_)
            {
                init_graph.createTensor(entry.momentum);
                pi::tensorlib::FillZeros(init_graph, entry.momentum, compute_stream_descriptor_);

                init_graph.createTensor(entry.update_bf16);
                pi::tensorlib::FillZeros(init_graph, entry.update_bf16, compute_stream_descriptor_);

                init_graph.createTensor(entry.norm);
                pi::tensorlib::FillZeros(init_graph, entry.norm, compute_stream_descriptor_);

                if (entry.use_master && entry.master)
                {
                    init_graph.createTensor(entry.master.value());
                    auto param_fp32 =
                        entry.param.to(init_graph, pi::tensorlib::DataType::FLOAT32, compute_stream_descriptor_);
                    entry.master->populate(init_graph, param_fp32, compute_stream_descriptor_);
                }
            }
            init_graph.finalize();

            auto init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, init_exec_parameters_, {});
            ApplyInitPasses(init_plan);

            auto &backend = pi::tensorlib::ExecutionBackend::getInstance();
            pi::tensorlib::Executor init_executor{std::move(init_plan), backend, device_.ordinal};
            const auto &allocator_registry = pi::tensorlib::allocator::DefaultAllocatorRegistry::instance();
            init_executor.execute(allocator_registry);
            init_executor.await();

            state_.clear();
            pre_exec_parameters_.clear();
            main_exec_parameters_.clear();
            pre_exec_parameters_.reserve(entries_.size() * 3);
            main_exec_parameters_.reserve(entries_.size() * 4);

            for (size_t idx = 0; idx < entries_.size(); ++idx)
            {
                const auto &entry = entries_[idx];
                auto momentum_real = init_executor.getOutput(entry.momentum);
                auto update_real = init_executor.getOutput(entry.update_bf16);
                auto norm_real = init_executor.getOutput(entry.norm);
                if (!momentum_real.has_value() || !update_real.has_value() || !norm_real.has_value())
                {
                    throw std::runtime_error("Failed to resolve Muon state tensors");
                }

                std::shared_ptr<pi::tensorlib::RealTensor> master{};
                if (entry.use_master && entry.master)
                {
                    auto master_real = init_executor.getOutput(entry.master.value());
                    if (!master_real.has_value())
                    {
                        throw std::runtime_error("Failed to resolve Muon master tensor");
                    }
                    master = master_real.value();
                }

                state_.push_back(MuonState{.master = master,
                                           .momentum = momentum_real.value(),
                                           .update_bf16 = update_real.value(),
                                           .norm = norm_real.value()});

                pre_exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
                    .name = parameters_[idx].name + "_grad", .tensor = parameters_[idx].grad});
                pre_exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
                    .name = parameters_[idx].name + "_muon_momentum", .tensor = state_.back().momentum});
                pre_exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
                    .name = parameters_[idx].name + "_muon_update", .tensor = state_.back().update_bf16});
                pre_exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
                    .name = parameters_[idx].name + "_muon_norm", .tensor = state_.back().norm});

                main_exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
                    .name = parameters_[idx].name, .tensor = parameters_[idx].param});
                main_exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
                    .name = parameters_[idx].name + "_muon_update", .tensor = state_.back().update_bf16});
                main_exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
                    .name = parameters_[idx].name + "_muon_norm", .tensor = state_.back().norm});
                if (entry.use_master && entry.master)
                {
                    main_exec_parameters_.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
                        .name = parameters_[idx].name + "_master", .tensor = master});
                }
            }
        }

        void BuildPreExecutor()
        {
            pi::tensorlib::OpGraph graph{{}, pre_input_descriptors_};

            if (!entries_.empty())
            {
                const auto &device = entries_.front().param.device();
                auto beta_scalar = graph.createTensor({1}, pi::tensorlib::DataType::FLOAT32, device,
                                                      compute_stream_descriptor_, false);
                auto one_minus_beta_scalar = graph.createTensor({1}, pi::tensorlib::DataType::FLOAT32, device,
                                                                compute_stream_descriptor_, false);
                pi::tensorlib::FillConstant(graph, beta_scalar, momentum_, compute_stream_descriptor_);
                pi::tensorlib::FillConstant(graph, one_minus_beta_scalar, 1.0f - momentum_, compute_stream_descriptor_);
                auto eps = graph.createTensor({1, 1}, pi::tensorlib::DataType::FLOAT32, device,
                                              compute_stream_descriptor_, false);
                pi::tensorlib::FillConstant(graph, eps, 1.0e-7f, compute_stream_descriptor_);

                for (auto &entry : entries_)
                {
                    auto grad_fp32 = entry.grad;
                    if (grad_fp32.dtype() != pi::tensorlib::DataType::FLOAT32)
                    {
                        grad_fp32 = grad_fp32.to(graph, pi::tensorlib::DataType::FLOAT32, compute_stream_descriptor_);
                    }

                    auto nesterov_update_buffer =
                        MulTensor(graph, grad_fp32, one_minus_beta_scalar, compute_stream_descriptor_);
                    // nesterov_update_buffer currently holds scaled_grad.

                    pi::tensorlib::InplaceMul(graph, entry.momentum, beta_scalar, compute_stream_descriptor_);
                    pi::tensorlib::InplaceAdd(graph, entry.momentum, nesterov_update_buffer, compute_stream_descriptor_);

                    auto momentum_scaled = MulTensor(graph, entry.momentum, beta_scalar, compute_stream_descriptor_);
                    pi::tensorlib::InplaceAdd(graph, nesterov_update_buffer, momentum_scaled,
                                              compute_stream_descriptor_);
                    // nesterov_update_buffer now holds update_pre_fp32.

                    auto update_view = nesterov_update_buffer;
                    if (entry.needs_flatten)
                    {
                        update_view = update_view.view(graph, std::vector<int64_t>{static_cast<int64_t>(entry.rows),
                                                                                   static_cast<int64_t>(entry.cols)});
                    }

                    auto update_view_bf16 =
                        update_view.to(graph, pi::tensorlib::DataType::BFLOAT16, compute_stream_descriptor_);
                    auto update_sq = MulTensor(graph, update_view_bf16, update_view_bf16, compute_stream_descriptor_);
                    auto sum1 = pi::tensorlib::ReduceSum(graph, update_sq, -1, true, pi::tensorlib::DataType::FLOAT32,
                                                         compute_stream_descriptor_);
                    auto sum2 = pi::tensorlib::ReduceSum(graph, sum1, -2, true, pi::tensorlib::DataType::FLOAT32,
                                                         compute_stream_descriptor_);
                    auto sqrt_sum = pi::tensorlib::Sqrt(graph, sum2, compute_stream_descriptor_);
                    graph.recordOperation(pi::tensorlib::OperationEntry{.type = pi::tensorlib::OpType::PLUS,
                                                                        .inputs = {sqrt_sum, eps},
                                                                        .outputs = {entry.norm},
                                                                        .attributes = {},
                                                                        .gpu_stream_desc = compute_stream_descriptor_});

                    auto norm_bf16 =
                        entry.norm.to(graph, pi::tensorlib::DataType::BFLOAT16, compute_stream_descriptor_);
                    auto update_output = entry.update_bf16;
                    if (entry.needs_flatten)
                    {
                        update_output =
                            update_output.view(graph, std::vector<int64_t>{static_cast<int64_t>(entry.rows),
                                                                           static_cast<int64_t>(entry.cols)});
                    }
                    // Write normalized update directly into the retained update buffer to avoid a BF16->BF16 copy.
                    graph.recordOperation(pi::tensorlib::OperationEntry{.type = pi::tensorlib::OpType::DIV,
                                                                        .inputs = {update_view_bf16, norm_bf16},
                                                                        .outputs = {update_output},
                                                                        .attributes = {},
                                                                        .gpu_stream_desc = compute_stream_descriptor_});
                }
            }

            graph.finalize();

            auto execution_plan = pi::tensorlib::ExecutionPlan::FromGraph(graph, {}, pre_exec_parameters_);
            ApplyMuonPasses(execution_plan);

            auto &backend = pi::tensorlib::ExecutionBackend::getInstance();
            pre_executor_ =
                std::make_unique<pi::tensorlib::Executor>(std::move(execution_plan), backend, device_.ordinal);
        }

        void BuildMainExecutor()
        {
            pi::tensorlib::OpGraph graph{{}, main_input_descriptors_};

            if (!entries_.empty())
            {
                const auto &device = entries_.front().param.device();
                auto a_scalar = graph.createTensor({1}, pi::tensorlib::DataType::BFLOAT16, device,
                                                   compute_stream_descriptor_, false);
                auto b_scalar = graph.createTensor({1}, pi::tensorlib::DataType::BFLOAT16, device,
                                                   compute_stream_descriptor_, false);
                auto c_scalar = graph.createTensor({1}, pi::tensorlib::DataType::BFLOAT16, device,
                                                   compute_stream_descriptor_, false);
                pi::tensorlib::FillConstant(graph, a_scalar, 3.4445f, compute_stream_descriptor_);
                pi::tensorlib::FillConstant(graph, b_scalar, -4.7750f, compute_stream_descriptor_);
                pi::tensorlib::FillConstant(graph, c_scalar, 2.0315f, compute_stream_descriptor_);

                auto decay_scalar = graph.createTensor({1}, pi::tensorlib::DataType::FLOAT32, device,
                                                       compute_stream_descriptor_, false);
                auto neg_lr_scalar = graph.createTensor({1}, pi::tensorlib::DataType::FLOAT32, device,
                                                        compute_stream_descriptor_, false);
                const float decay = 1.0f - learning_rate_ * weight_decay_;
                const float neg_lr = -learning_rate_;
                pi::tensorlib::FillConstant(graph, decay_scalar, decay, compute_stream_descriptor_);
                pi::tensorlib::FillConstant(graph, neg_lr_scalar, neg_lr, compute_stream_descriptor_);

                for (auto &entry : entries_)
                {
                    auto update_view = entry.update_bf16;
                    if (entry.needs_flatten)
                    {
                        update_view = update_view.view(graph, std::vector<int64_t>{static_cast<int64_t>(entry.rows),
                                                                                   static_cast<int64_t>(entry.cols)});
                    }
                    auto X = update_view;
                    if (entry.transpose)
                    {
                        X = X.transpose(graph, {1, 0});
                        X = X.contiguous(graph, compute_stream_descriptor_);
                    }

                    for (int i = 0; i < NUM_NEWTON_SCHULZ_ITERS; ++i)
                    {
                        auto Xt = X.transpose(graph, {1, 0});
                        auto A = graph.createTensor({X.shape()[0], X.shape()[0]}, pi::tensorlib::DataType::BFLOAT16,
                                                    device, compute_stream_descriptor_, false);
                        graph.recordOperation(
                            pi::tensorlib::OperationEntry{.type = pi::tensorlib::OpType::MATMUL,
                                                          .inputs = {X, Xt},
                                                          .outputs = {A},
                                                          .attributes = {},
                                                          .gpu_stream_desc = compute_stream_descriptor_});
                        auto AA = graph.createTensor({X.shape()[0], X.shape()[0]}, pi::tensorlib::DataType::BFLOAT16,
                                                     device, compute_stream_descriptor_, false);
                        graph.recordOperation(
                            pi::tensorlib::OperationEntry{.type = pi::tensorlib::OpType::MATMUL,
                                                          .inputs = {A, A},
                                                          .outputs = {AA},
                                                          .attributes = {},
                                                          .gpu_stream_desc = compute_stream_descriptor_});
                        auto newton_schulz_buffer = A;
                        // newton_schulz_buffer currently holds A.
                        pi::tensorlib::InplaceMul(graph, newton_schulz_buffer, b_scalar, compute_stream_descriptor_);
                        // newton_schulz_buffer now holds B1.
                        auto squared_buffer = AA;
                        // squared_buffer currently holds AA.
                        pi::tensorlib::InplaceMul(graph, squared_buffer, c_scalar, compute_stream_descriptor_);
                        // squared_buffer now holds B2.
                        pi::tensorlib::InplaceAdd(graph, newton_schulz_buffer, squared_buffer,
                                                  compute_stream_descriptor_);
                        // newton_schulz_buffer now holds B.

                        auto BX = graph.createTensor({X.shape()[0], X.shape()[1]}, pi::tensorlib::DataType::BFLOAT16,
                                                     device, compute_stream_descriptor_, false);
                        graph.recordOperation(
                            pi::tensorlib::OperationEntry{.type = pi::tensorlib::OpType::MATMUL,
                                                          .inputs = {newton_schulz_buffer, X},
                                                          .outputs = {BX},
                                                          .attributes = {},
                                                          .gpu_stream_desc = compute_stream_descriptor_});

                        auto x_iteration_buffer = X;
                        // x_iteration_buffer currently holds X.
                        pi::tensorlib::InplaceMul(graph, x_iteration_buffer, a_scalar, compute_stream_descriptor_);
                        // x_iteration_buffer now holds aX.
                        pi::tensorlib::InplaceAdd(graph, x_iteration_buffer, BX, compute_stream_descriptor_);
                        // x_iteration_buffer now holds the next X.
                        X = x_iteration_buffer;
                    }

                    if (entry.transpose)
                    {
                        X = X.transpose(graph, {1, 0});
                        X = X.contiguous(graph, compute_stream_descriptor_);
                    }

                    auto scale_scalar = graph.createTensor({1}, pi::tensorlib::DataType::BFLOAT16, device,
                                                           compute_stream_descriptor_, false);
                    pi::tensorlib::FillConstant(graph, scale_scalar, entry.scale, compute_stream_descriptor_);
                    pi::tensorlib::InplaceMul(graph, X, scale_scalar, compute_stream_descriptor_);

                    if (entry.needs_flatten)
                    {
                        X = X.view(graph, entry.param.shape().dims());
                    }

                    auto parameter_update_buffer =
                        X.to(graph, pi::tensorlib::DataType::FLOAT32, compute_stream_descriptor_);
                    // parameter_update_buffer currently holds update_fp32.
                    auto &param_for_update = (entry.use_master && entry.master) ? entry.master.value() : entry.param;
                    pi::tensorlib::InplaceMul(graph, param_for_update, decay_scalar, compute_stream_descriptor_);

                    pi::tensorlib::InplaceMul(graph, parameter_update_buffer, neg_lr_scalar, compute_stream_descriptor_);
                    // parameter_update_buffer now holds scaled_update.
                    pi::tensorlib::InplaceAdd(graph, param_for_update, parameter_update_buffer,
                                              compute_stream_descriptor_);

                    if (entry.use_master && entry.master)
                    {
                        graph.recordOperation(
                            pi::tensorlib::OperationEntry{.type = pi::tensorlib::OpType::CAST,
                                                          .inputs = {param_for_update},
                                                          .outputs = {entry.param},
                                                          .attributes = {},
                                                          .gpu_stream_desc = compute_stream_descriptor_});
                    }
                }
            }

            graph.finalize();

            auto execution_plan = pi::tensorlib::ExecutionPlan::FromGraph(graph, {}, main_exec_parameters_);
            ApplyMuonPasses(execution_plan);

            auto &backend = pi::tensorlib::ExecutionBackend::getInstance();
            main_executor_ =
                std::make_unique<pi::tensorlib::Executor>(std::move(execution_plan), backend, device_.ordinal);
        }

        float learning_rate_{};
        float weight_decay_{};
        float momentum_{};
        pi::tensorlib::Device device_{};
        pi::tensorlib::GpuStreamDescriptor compute_stream_descriptor_;
        std::vector<ParameterGrad> parameters_{};
        std::vector<MuonEntry> entries_{};
        std::vector<MuonState> state_{};
        std::vector<pi::tensorlib::GraphInputDescriptor> init_input_descriptors_{};
        std::vector<pi::tensorlib::GraphInputDescriptor> pre_input_descriptors_{};
        std::vector<pi::tensorlib::GraphInputDescriptor> main_input_descriptors_{};
        std::vector<pi::tensorlib::GraphExecutionInputDescriptor> init_exec_parameters_{};
        std::vector<pi::tensorlib::GraphExecutionInputDescriptor> pre_exec_parameters_{};
        std::vector<pi::tensorlib::GraphExecutionInputDescriptor> main_exec_parameters_{};
        std::unique_ptr<pi::tensorlib::Executor> pre_executor_{};
        std::unique_ptr<pi::tensorlib::Executor> main_executor_{};
    };

    class MuonWithAuxAdamOptimizer final : public Optimizer
    {
      public:
        MuonWithAuxAdamOptimizer(const OptimizerConfiguration &config, const std::vector<ParameterGrad> &muon_params,
                                 const std::vector<ParameterGrad> &adam_params,
                                 const pi::tensorlib::Device &device,
                                 const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor)
        {
            if (!muon_params.empty())
            {
                muon_optimizer_ =
                    std::make_unique<MuonOptimizer>(config, muon_params, device, compute_stream_descriptor);
            }
            if (!adam_params.empty())
            {
                adam_optimizer_ =
                    std::make_unique<AdamwOptimizer>(config, adam_params, device, compute_stream_descriptor);
            }
        }

        void step() override
        {
            GPUTX_RANGE("fbamtrain::optimizer::muon_with_aux_adam_step");
            if (muon_optimizer_)
            {
                muon_optimizer_->step();
            }
            if (adam_optimizer_)
            {
                adam_optimizer_->step();
            }
        }

        void appendOptimState(std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &out,
                              const std::string &prefix) override
        {
            const std::string muon_prefix = prefix.empty() ? "muon" : prefix + "/muon";
            const std::string adam_prefix = prefix.empty() ? "adam" : prefix + "/adam";
            if (muon_optimizer_)
            {
                muon_optimizer_->appendOptimState(out, muon_prefix);
            }
            if (adam_optimizer_)
            {
                adam_optimizer_->appendOptimState(out, adam_prefix);
            }
        }

        void loadState(const std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &in,
                       const std::string &prefix) override
        {
            const std::string muon_prefix = prefix.empty() ? "muon" : prefix + "/muon";
            const std::string adam_prefix = prefix.empty() ? "adam" : prefix + "/adam";
            if (muon_optimizer_)
            {
                muon_optimizer_->loadState(in, muon_prefix);
            }
            if (adam_optimizer_)
            {
                adam_optimizer_->loadState(in, adam_prefix);
            }
        }

        void setStep(const uint64_t step) override
        {
            if (muon_optimizer_)
            {
                muon_optimizer_->setStep(step);
            }
            if (adam_optimizer_)
            {
                adam_optimizer_->setStep(step);
            }
        }

      private:
        std::unique_ptr<MuonOptimizer> muon_optimizer_{};
        std::unique_ptr<AdamwOptimizer> adam_optimizer_{};
    };

    std::unique_ptr<Optimizer> CreateOptimizer(const OptimizerConfiguration &config,
                                               const std::vector<ParameterGrad> &parameters,
                                               const pi::tensorlib::Device &device,
                                               const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor)
    {
        const auto type = config.type;
        if (type == "sgd" || type == "SGD")
        {
            return std::make_unique<SgdOptimizer>(config, parameters, device, compute_stream_descriptor);
        }
        if (type == "adamw" || type == "AdamW" || type == "ADAMW")
        {
            return std::make_unique<AdamwOptimizer>(config, parameters, device, compute_stream_descriptor);
        }
        if (type == "muon" || type == "Muon" || type == "MUON")
        {
            std::vector<ParameterGrad> muon_params{};
            std::vector<ParameterGrad> adam_params{};
            muon_params.reserve(parameters.size());
            adam_params.reserve(parameters.size());

            // group parameters based on their dimensionality
            // 2d -> muon, others -> adam
            for (const auto &parameter : parameters)
            {
                if (parameter.param->shape().ndims() >= 2)
                {
                    muon_params.push_back(parameter);
                }
                else
                {
                    adam_params.push_back(parameter);
                }
            }

            if (muon_params.empty() && adam_params.empty())
            {
                throw std::runtime_error("Muon optimizer requires at least one parameter.");
            }
            return std::make_unique<MuonWithAuxAdamOptimizer>(config, muon_params, adam_params, device,
                                                              compute_stream_descriptor);
        }
        throw std::runtime_error("Unsupported optimizer type: " + type);
    }
} // namespace fbamtrain::optim
