#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

// input, output are device pointers
extern "C" void solve(float* data, int N) {
    thrust::device_ptr<float> arr(data);
    thrust::sort(arr, arr+N);
}