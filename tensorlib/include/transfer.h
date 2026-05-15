#pragma once

#include "tensorlib.h"

namespace pi::tensorlib
{
    enum class TransferType
    {
        H2D,
        D2H
    };

    inline TransferType TransferTypeFrom(const DeviceType source, const DeviceType destination)
    {
        if (source == DeviceType::CPU && destination == DeviceType::GPU)
        {
            return TransferType::H2D;
        }
        if (source == DeviceType::GPU && destination == DeviceType::CPU)
        {
            return TransferType::D2H;
        }
        throw std::invalid_argument("Invalid device types for transfer");
    }

} // namespace pi::tensorlib