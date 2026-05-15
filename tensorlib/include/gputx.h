#pragma once

#include <cstdint>
#include <string_view>

namespace pi::tensorlib::gputx
{

    enum class RangeColor : std::uint32_t
    {
        Default = 0xFF4CAF50,
        Blue = 0xFF2196F3,
        Red = 0xFFF44336,
        Orange = 0xFFFF9800,
        Purple = 0xFF9C27B0,
        Teal = 0xFF009688,
        Gray = 0xFF607D8B
    };

    /// Returns true when a GPU tracing backend is available at runtime.
    bool IsAvailable() noexcept;

    /// Pushes a tracing range; returns true if it was emitted to an active backend.
    bool PushRange(std::string_view name, RangeColor color = RangeColor::Default);

    /// Pops the most recent tracing range if a backend is active.
    void PopRange() noexcept;

    class ScopedRange
    {
      public:
        explicit ScopedRange(std::string_view name, RangeColor color = RangeColor::Default);
        ScopedRange(const ScopedRange &) = delete;
        ScopedRange &operator=(const ScopedRange &) = delete;
        ScopedRange(ScopedRange &&other) noexcept;
        ScopedRange &operator=(ScopedRange &&other) noexcept;
        ~ScopedRange();

      private:
        bool active_;
    };

} // namespace fbamtrain::gputx

#define GPUTX_RANGE_DETAIL_JOIN(lhs, rhs) lhs##rhs
#define GPUTX_RANGE_DETAIL_MAKE_NAME(id) GPUTX_RANGE_DETAIL_JOIN(gputx_scope_, id)

/// Helper macro for concise scoped tracing ranges.
#define GPUTX_RANGE(name, ...)                                                                                         \
    pi::tensorlib::gputx::ScopedRange GPUTX_RANGE_DETAIL_MAKE_NAME(__COUNTER__)(name, ##__VA_ARGS__)

/// Helper macro to create a scoped range for tracing ranges
#define MAKE_GPUTX_RANGE(name, ...) new pi::tensorlib::gputx::ScopedRange(name, ##__VA_ARGS__)