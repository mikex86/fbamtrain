#include "cpudbg.h"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace cpudbg
{
    void DebugBreak()
    {
#if defined(_MSC_VER)
        __debugbreak();
#elif defined(__clang__) || defined(__GNUC__)
        __builtin_trap();
#else
        while (true)
        {
        }
#endif
    }
} // namespace cpudbg
