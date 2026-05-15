#pragma once

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string_view>

#include <tensorlib.h>

namespace bench_utils
{
    struct BenchmarkConfig
    {
        std::size_t warmup_runs;
        std::size_t measure_runs;
        pi::tensorlib::DataType dtype;
    };

    namespace detail
    {
        inline std::size_t ParseSizeEnv(const char *env_name, const std::size_t default_value)
        {
            if (const char *env = std::getenv(env_name))
            {
                char *end = nullptr;
                const unsigned long long parsed = std::strtoull(env, &end, 10);
                if (end != env && end && *end == '\0')
                {
                    return parsed;
                }
                std::cerr << "[bench] Warning: invalid value for " << env_name << "='" << env << "', using default "
                          << default_value << '\n';
            }
            return default_value;
        }

        inline pi::tensorlib::DataType ParseBenchDtype()
        {
            if (const char *env = std::getenv("BENCH_DTYPE"))
            {
                if (std::strcmp(env, "fp16") == 0)
                {
                    return pi::tensorlib::DataType::FLOAT16;
                }
                if (std::strcmp(env, "bf16") == 0)
                {
                    return pi::tensorlib::DataType::BFLOAT16;
                }
                if (std::strcmp(env, "fp32") == 0)
                {
                    return pi::tensorlib::DataType::FLOAT32;
                }
                std::cerr << "[bench] Warning: unsupported BENCH_DTYPE='" << env << "', defaulting to bf16\n";
            }
            return pi::tensorlib::DataType::BFLOAT16;
        }
    } // namespace detail

    inline BenchmarkConfig LoadBenchmarkConfig()
    {
        BenchmarkConfig cfg{
            .warmup_runs = detail::ParseSizeEnv("BENCH_WARMUP", 16),
            .measure_runs = detail::ParseSizeEnv("BENCH_ITERS", 1024),
            .dtype = detail::ParseBenchDtype(),
        };
        if (cfg.measure_runs == 0)
        {
            std::cerr << "[bench] Warning: BENCH_ITERS resolved to zero, forcing measure_runs=1\n";
            cfg.measure_runs = 1;
        }
        return cfg;
    }

    inline void PrintResult(const std::string_view name, const BenchmarkConfig &cfg, const double avg_ms,
                            const double tflops, const double flops_per_iter)
    {
        std::cout << "benchmark=" << name << ' '
                  << "dtype=" << pi::tensorlib::GetDataTypeName(cfg.dtype) << ' '
                  << "warmup=" << cfg.warmup_runs << ' '
                  << "iters=" << cfg.measure_runs << ' '
                  << "avg_ms=" << avg_ms << ' '
                  << "tflops=" << tflops << ' '
                  << "flops_per_iter=" << flops_per_iter << std::endl;
    }
} // namespace bench_utils
