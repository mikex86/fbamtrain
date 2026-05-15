#include "optimizers.h"
#include "stream_descriptor.h"
#include "testing.h"

#include <safe_tensors.h>
#include <tensorlib.h>

#include <stdexcept>
#include <string>
#include <vector>

using namespace pi::tensorlib;

int main()
{
    const auto tensors = safetensors::Load("reference.safetensors");
    const auto fetch = [&tensors](const std::string &name)
    {
        const auto it = tensors.find(name);
        if (it == tensors.end())
        {
            throw std::runtime_error("Missing tensor in reference.safetensors: " + name);
        }
        return it->second;
    };

    const std::vector<std::string> names{
        "fc1_weight",
        "fc1_bias",
        "fc2_weight",
        "fc2_bias",
    };

    std::vector<fbamtrain::optim::ParameterGrad> parameters{};
    parameters.reserve(names.size());
    std::vector<std::shared_ptr<RealTensor>> expected{};
    expected.reserve(names.size());
    constexpr Device device_gpu{DeviceType::GPU, 0};
    const auto compute_stream_descriptor = GpuStreamDescriptors::Main;

    for (const auto &name : names)
    {
        auto param = fetch("param_" + name);
        auto grad = fetch("grad_" + name);
        auto updated = fetch("updated_" + name);
        parameters.push_back(fbamtrain::optim::ParameterGrad{
            .name = name,
            .param = param->to(device_gpu, compute_stream_descriptor),
            .grad = grad->to(device_gpu, compute_stream_descriptor)});
        expected.push_back(updated);
    }

    const fbamtrain::OptimizerConfiguration config{.type = "muon",
                                                   .learning_rate = 1.0e-3f,
                                                   .weight_decay = 1.0e-2f,
                                                   .beta1 = 0.9f,
                                                   .beta2 = 0.999f,
                                                   .eps = 1.0e-8f,
                                                   .momentum = 0.95f,
                                                   .nesterov = true};

    const auto optimizer = fbamtrain::optim::CreateOptimizer(config, parameters, device_gpu, compute_stream_descriptor);
    optimizer->step();

    for (size_t i = 0; i < parameters.size(); ++i)
    {
        testing::AssertSimilar(expected[i], parameters[i].param, 5e-3);
    }
    return 0;
}
