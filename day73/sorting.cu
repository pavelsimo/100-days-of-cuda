#include <cuda_runtime.h>

__global__ void bitonic_step(float* a, int mask, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int p = i ^ mask;
    if (i >= N || p <= i || p >= N) {
        return;
    }
    float x = a[i];
    float y = a[p];
    if (x > y) {
        a[i] = y;
        a[p] = x;
    }
}

// input, output are device pointers
extern "C" void solve(float* data, int N) {
    int M = 1;
    while (M < N) {
        M <<= 1;
    }

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    for (int size = 2; size <= M; size <<= 1) {
        bitonic_step<<<blocks, threads>>>(data, size - 1, N);
        for (int j = size >> 2; j > 0; j >>= 1) {
            bitonic_step<<<blocks, threads>>>(data, j, N);
        }
    }
    cudaDeviceSynchronize();
}
