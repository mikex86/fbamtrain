#include "testing.h"

#include "tensorlib.h"
#include "utils.h"

#include <cmath>

void pi::tensorlib::testing::AssertSimilar(const std::shared_ptr<RealTensor> &expected,
                                           const std::shared_ptr<RealTensor> &actual, const double tolerance)
{
    const auto &expected_shape = expected->shape();
    const auto &actual_shape = actual->shape();
    const auto &expected_strides = expected->strides();
    const auto &actual_strides = actual->strides();
    if (expected_shape.ndims() != actual_shape.ndims())
    {
        throw std::runtime_error("Tensors have different number of dimensions");
    }
    for (int64_t i = 0; i < expected_shape.ndims(); ++i)
    {
        if (expected_shape[i] != actual_shape[i])
        {
            throw std::runtime_error("Tensors have different shapes");
        }
    }
    if (expected->dtype() != actual->dtype())
    {
        throw std::runtime_error("Tensors have different data types");
    }

    auto expected_storage = expected->storage();
    auto actual_storage = actual->storage();
    if (expected_storage->device().device_type != DeviceType::CPU)
    {
        expected_storage = expected_storage->toCPU();
    }
    if (actual_storage->device().device_type != DeviceType::CPU)
    {
        actual_storage = actual_storage->toCPU();
    }
    if (expected->shape().numel() != actual->shape().numel())
    {
        throw std::runtime_error("Tensors have different number of elements");
    }

    const void *expected_data = static_cast<uint8_t *>(expected_storage->dataptr()) +
                                expected->storageOffset() * GetDataTypeSize(expected->dtype());
    const void *actual_data =
        static_cast<uint8_t *>(actual_storage->dataptr()) + actual->storageOffset() * GetDataTypeSize(actual->dtype());

    const size_t num_elements = expected->shape().numel();
    for (size_t i = 0; i < num_elements; ++i)
    {
        size_t lin = i;
        size_t off_expected = 0;
        size_t off_actual = 0;

        for (auto d = static_cast<int64_t>(expected_shape.ndims()); d-- > 0;)
        {
            const size_t idx_d = lin % expected_shape[d];
            lin /= expected_shape[d];

            off_expected += idx_d * expected_strides[d];
            off_actual += idx_d * actual_strides[d];
        }

        double expected_value{};
        double actual_value{};

        switch (expected->dtype())
        {
            case DataType::BFLOAT16:
            {
                const auto *e = static_cast<const uint16_t *>(expected_data);
                const auto *a = static_cast<const uint16_t *>(actual_data);
                const uint16_t expected_raw = e[off_expected];
                const uint16_t actual_raw = a[off_actual];
                expected_value = static_cast<double>(utils::Fp32FromBf16(expected_raw));
                actual_value = static_cast<double>(utils::Fp32FromBf16(actual_raw));
                if (expected_raw == actual_raw)
                {
                    continue;
                }
                const auto ulp_diff =
                    static_cast<uint16_t>(expected_raw > actual_raw ? expected_raw - actual_raw : actual_raw - expected_raw);
                if (ulp_diff <= 1)
                {
                    continue;
                }
                break;
            }
            case DataType::FLOAT16:
            {
                const auto *e = static_cast<const uint16_t *>(expected_data);
                const auto *a = static_cast<const uint16_t *>(actual_data);
                const uint16_t expected_raw = e[off_expected];
                const uint16_t actual_raw = a[off_actual];
                expected_value = static_cast<double>(utils::Fp32FromFp16(expected_raw));
                actual_value = static_cast<double>(utils::Fp32FromFp16(actual_raw));
                if (expected_raw == actual_raw)
                {
                    continue;
                }
                const auto ulp_diff =
                    static_cast<uint16_t>(expected_raw > actual_raw ? expected_raw - actual_raw : actual_raw - expected_raw);
                if (ulp_diff <= 1)
                {
                    continue;
                }
                break;
            }
            case DataType::FLOAT32:
            {
                const auto *e = static_cast<const float *>(expected_data);
                const auto *a = static_cast<const float *>(actual_data);
                expected_value = static_cast<double>(e[off_expected]);
                actual_value = static_cast<double>(a[off_actual]);
                break;
            }
            default:
                throw std::runtime_error("Unsupported data type in AssertSimilar");
        }

        if (!std::isfinite(expected_value) || !std::isfinite(actual_value))
        {
            throw std::runtime_error("Tensors contain non-finite value at logical element " + std::to_string(i) +
                                     ": expected " + std::to_string(expected_value) + ", got " +
                                     std::to_string(actual_value));
        }

        if (std::abs(expected_value - actual_value) > tolerance)
        {
            throw std::runtime_error("Tensors differ at logical element " + std::to_string(i) + ": expected " +
                                     std::to_string(expected_value) + ", got " + std::to_string(actual_value));
        }
    }
}
