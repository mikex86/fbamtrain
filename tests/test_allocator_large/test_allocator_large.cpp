#include "testing.h"

#include <allocator.h>
#include <execution_backend.h>

#include <cuda.h>
#include <iostream>

using namespace pi::tensorlib;

int main()
{
    // Pinned host allocation requires an active CUDA context.
    (void)ExecutionBackend::GetStreamBundle(Device{DeviceType::GPU, 0});

    // Match the streaming LSTM host buffers to sanity-check pinned paths at scale:
    // x_host: 512 * 1024 * 1024 * sizeof(half) ~= 1.07GB
    // gate caches: 512 * 1024 * 4096 * sizeof(float) ~= 8.6GB each
    const size_t size_bytes_x = 512ull * 1024ull * 1024ull * sizeof(uint16_t);
    const size_t size_bytes_gate = 512ull * 1024ull * 4096ull * sizeof(float);
    auto &registry = allocator::DefaultAllocatorRegistry::instance();
    auto &cpu_alloc = registry.getAllocator(DeviceType::CPU);

    // Allocate input buffer (pinned)
    void *x_ptr = cpu_alloc.allocate(size_bytes_x, /*device_ordinal=*/0, /*pinned=*/true, 0, /*zero_initialize=*/false);
    if (x_ptr == nullptr)
    {
        throw std::runtime_error("x allocation returned nullptr");
    }

    // Allocate gate caches (this will stress the chunked registration)
    void *gate_h_ptr =
        cpu_alloc.allocate(size_bytes_gate, /*device_ordinal=*/0, /*pinned=*/true, 0, /*zero_initialize=*/false);
    void *gate_c_ptr =
        cpu_alloc.allocate(size_bytes_gate, /*device_ordinal=*/0, /*pinned=*/true, 0, /*zero_initialize=*/false);

    cpu_alloc.deallocate(x_ptr, /*device_ordinal=*/0, 0);
    cpu_alloc.deallocate(gate_h_ptr, /*device_ordinal=*/0, 0);
    cpu_alloc.deallocate(gate_c_ptr, /*device_ordinal=*/0, 0);
    return 0;
}
