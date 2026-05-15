#pragma once

#include <new>

namespace pi::tensorlib::utils
{
    template <typename T>
    class NeverDeleted
    {
      public:
        NeverDeleted() { new (storage_) T(); }

        NeverDeleted(const NeverDeleted &) = delete;
        NeverDeleted &operator=(const NeverDeleted &) = delete;
        NeverDeleted(NeverDeleted &&) = delete;
        NeverDeleted &operator=(NeverDeleted &&) = delete;

        T &get() { return *std::launder(reinterpret_cast<T *>(storage_)); }
        const T &get() const { return *std::launder(reinterpret_cast<const T *>(storage_)); }

      private:
        alignas(T) unsigned char storage_[sizeof(T)]{};
    };
} // namespace pi::tensorlib::utils
