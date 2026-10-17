#include <cuda_runtime.h>

__global__ void scatter_add(const float* src, const int* index, float* output, int N, int D, int M) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N * D) {
        return;
    }

    int i = idx / D;
    int d = idx % D;
    int m = index[i];
    atomicAdd(&output[m * D + d], src[i * D + d]);
}

// src, index, output are device pointers
extern "C" void solve(const float* src, const int* index, float* output, int N, int D, int M) {
    int threads = 256;
    int blocks = (N * D + threads - 1) / threads;
    scatter_add<<<blocks, threads>>>(src, index, output, N, D, M); 
    cudaDeviceSynchronize();
}
