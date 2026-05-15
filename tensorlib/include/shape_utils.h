#pragma once

#include <memory>
#include <optional>

namespace pi::tensorlib
{
    class Shape;
    class Strides;
    class RealTensor;
} // namespace pi::tensorlib

namespace pi::tensorlib::shape_utils
{
    std::optional<Strides> ComputeViewStrides(const Shape &oldShape, const Strides &oldStrides, const Shape &newShape);

    bool IsBroadcastable(const Shape &shapeA, const Shape &shapeB);

    bool IsRowMajorContiguous(const Shape &shape, const Strides &strides);

    bool IsRowMajorContiguous(const std::shared_ptr<RealTensor> &tensor);

};
