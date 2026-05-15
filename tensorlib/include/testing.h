#pragma once
#include <memory>

// Forward declare RealTensor
namespace pi::tensorlib
{
    struct RealTensor;
};

namespace pi::tensorlib::testing
{

    void AssertSimilar(const std::shared_ptr<RealTensor> &expected, const std::shared_ptr<RealTensor> &actual, double tolerance = 1e-5);
}