#include <cmath>
#include <iostream>
#include <vector>

#include <mfu.h>

using namespace pi::tensorlib;

namespace
{
    struct Expected
    {
        const char *name;
        float bf16_fp32;
        float fp16_fp32;
        float fp16_acc16;
        float tf32;
    };

    bool NearlyEqual(const float a, const float b, const float tol = 1e-1f) { return std::fabs(a - b) <= tol; }

    bool Check(const Expected &entry)
    {
        const auto bf16 = GetPromisedTFlops(entry.name, PrecisionMode::BF16);
        const auto fp16 = GetPromisedTFlops(entry.name, PrecisionMode::FP16);
        const auto fp16_acc16 = GetPromisedTFlops(entry.name, PrecisionMode::FP16_ACC16);
        const auto tf32 = GetPromisedTFlops(entry.name, PrecisionMode::FP32);

        if (!bf16 || !fp16 || !fp16_acc16 || !tf32)
        {
            std::cerr << "Missing promised TFLOPs entry for " << entry.name << "\n";
            return false;
        }

        const bool ok = NearlyEqual(*bf16, entry.bf16_fp32) && NearlyEqual(*fp16, entry.fp16_fp32) &&
                        NearlyEqual(*fp16_acc16, entry.fp16_acc16) && NearlyEqual(*tf32, entry.tf32);
        if (!ok)
        {
            std::cerr << "Mismatch for " << entry.name << ": got (bf16=" << *bf16 << ", fp16=" << *fp16
                      << ", fp16_acc16=" << *fp16_acc16 << ", tf32=" << *tf32 << ") expected (" << entry.bf16_fp32
                      << ", " << entry.fp16_fp32 << ", " << entry.fp16_acc16 << ", " << entry.tf32 << ")\n";
        }
        return ok;
    }
} // namespace

int main()
{
    const std::vector<Expected> cases = {
        {"NVIDIA GeForce RTX 4090", 165.2f, 165.2f, 330.3f, 82.6f},
        {"NVIDIA GeForce RTX 4080", 97.5f, 97.5f, 194.9f, 48.8f},
        {"NVIDIA GeForce RTX 4070 Ti", 80.2f, 80.2f, 160.4f, 40.1f},
        {"NVIDIA GeForce RTX 4070", 58.3f, 58.3f, 116.6f, 29.2f},
        {"NVIDIA GeForce RTX 3090 Ti", 80.0f, 80.0f, 160.0f, 40.0f},
        {"NVIDIA GeForce RTX 3090", 71.2f, 71.2f, 142.4f, 35.6f},
        {"NVIDIA GeForce RTX 3080 Ti", 68.2f, 68.2f, 136.4f, 34.1f},
        {"NVIDIA GeForce RTX 3080", 59.5f, 59.5f, 119.0f, 29.8f},
        {"NVIDIA GeForce RTX 3070 Ti", 43.5f, 43.5f, 87.0f, 21.8f},
        {"NVIDIA GeForce RTX 3070", 40.6f, 40.6f, 81.2f, 20.3f},
        {"NVIDIA RTX A6000", 77.4f, 77.4f, 154.8f, 38.7f},
        {"NVIDIA A100-PCIE-80GB", 312.0f, 312.0f, 312.0f, 156.0f},
        {"NVIDIA H100 80GB HBM3", 988.8f, 988.8f, 988.8f, 494.4f},
        {"NVIDIA H100 PCIe", 756.0f, 756.0f, 756.0f, 378.0f},
    };

    bool all_ok = true;
    for (const auto &c : cases)
    {
        all_ok = Check(c) && all_ok;
    }
    return all_ok ? 0 : 1;
}
