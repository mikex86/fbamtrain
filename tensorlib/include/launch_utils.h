#pragma once

#ifndef CUDA_WARP_SIZE
#define CUDA_WARP_SIZE 32
#endif

// TODO: Define based on target architecture
#ifndef TRITON_WARP_SIZE
#define TRITON_WARP_SIZE 32
#endif

#ifndef CEIL_DIV
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#endif

inline std::string_view DeviceTypeToString(const pi::tensorlib::DeviceType device_type)
{
    switch (device_type)
    {
        case pi::tensorlib::DeviceType::CPU:
            return "CPU";
        case pi::tensorlib::DeviceType::GPU:
            return "GPU";
    }
    throw std::runtime_error("Unknown device type");
}


inline int
ValidateSameDeviceOrdinal(const std::string_view op_name,
                          const std::initializer_list<std::shared_ptr<pi::tensorlib::RealTensor>> tensors,
                          const pi::tensorlib::DeviceType expected_device_type = pi::tensorlib::DeviceType::GPU)
{
    std::optional<int> device_ordinal;

    for (const auto &tensor : tensors)
    {
        if (!tensor)
        {
            continue;
        }

        const auto &device = tensor->device();
        if (device.device_type != expected_device_type)
        {
            throw std::runtime_error(std::string(op_name) + " requires tensors on " +
                                     std::string(DeviceTypeToString(expected_device_type)) + " devices");
        }

        if (!device_ordinal.has_value())
        {
            device_ordinal = device.ordinal;
            continue;
        }

        if (device.ordinal != device_ordinal.value())
        {
            throw std::runtime_error(std::string(op_name) + " requires all tensors to reside on the same device");
        }
    }

    if (!device_ordinal.has_value())
    {
        throw std::runtime_error(std::string(op_name) + " requires at least one tensor to determine the device");
    }

    return device_ordinal.value();
}