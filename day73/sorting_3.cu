#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>

// input, output are device pointers
extern "C" void solve(float* data, int N) {
    size_t size = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, size, data, data, N);
    void* buffer = nullptr;
    cudaMalloc(&buffer, size);
    cub::DeviceRadixSort::SortKeys(buffer, size, data, data, N);  
    cudaFree(buffer);
}