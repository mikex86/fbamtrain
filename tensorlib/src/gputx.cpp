#include "gputx.h"

#include <string>
#include <utility>

#if defined(GPUTX_HAS_NVTX) && GPUTX_HAS_NVTX
#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#elif __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#else
#undef GPUTX_HAS_NVTX
#define GPUTX_HAS_NVTX 0
#endif
#else
#include <nvToolsExt.h>
#endif
#endif

#if defined(GPUTX_HAS_ROCTX) && GPUTX_HAS_ROCTX
#if defined(__has_include)
#if __has_include(<roctracer/roctx.h>)
#include <roctracer/roctx.h>
#elif __has_include(<roctx/roctx.h>)
#include <roctx/roctx.h>
#elif __has_include(<roctx.h>)
#include <roctx.h>
#else
#undef GPUTX_HAS_ROCTX
#define GPUTX_HAS_ROCTX 0
#endif
#endif
#endif

#ifndef GPUTX_HAS_NVTX
#define GPUTX_HAS_NVTX 0
#endif

#ifndef GPUTX_HAS_ROCTX
#define GPUTX_HAS_ROCTX 0
#endif

namespace pi::tensorlib::gputx
{

namespace
{
constexpr bool kBackendAvailable = GPUTX_HAS_NVTX || GPUTX_HAS_ROCTX;

#if GPUTX_HAS_NVTX
constexpr std::uint32_t ToNvtxColor(const RangeColor color)
{
    return static_cast<std::uint32_t>(color);
}
#endif
} // namespace

bool IsAvailable() noexcept
{
    return kBackendAvailable;
}

bool PushRange(const std::string_view name, const RangeColor color)
{
#if GPUTX_HAS_NVTX
    const std::string message{name};
    nvtxEventAttributes_t event{};
    event.version = NVTX_VERSION;
    event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    event.colorType = NVTX_COLOR_ARGB;
    event.color = ToNvtxColor(color);
    event.messageType = NVTX_MESSAGE_TYPE_ASCII;
    event.message.ascii = message.c_str();
    nvtxRangePushEx(&event);
    return true;
#elif GPUTX_HAS_ROCTX
    std::string message{name};
    (void)color;
    roctxRangePushA(message.c_str());
    return true;
#else
    (void)name;
    (void)color;
    return false;
#endif
}

void PopRange() noexcept
{
#if GPUTX_HAS_NVTX
    nvtxRangePop();
#elif GPUTX_HAS_ROCTX
    roctxRangePop();
#endif
}

ScopedRange::ScopedRange(const std::string_view name, const RangeColor color)
    : active_(PushRange(name, color))
{
}

ScopedRange::ScopedRange(ScopedRange &&other) noexcept
    : active_(std::exchange(other.active_, false))
{
}

ScopedRange &ScopedRange::operator=(ScopedRange &&other) noexcept
{
    if (this != &other)
    {
        if (active_)
        {
            PopRange();
        }
        active_ = std::exchange(other.active_, false);
    }
    return *this;
}

ScopedRange::~ScopedRange()
{
    if (active_)
    {
        PopRange();
    }
}

} // namespace fbamtrain::gputx
