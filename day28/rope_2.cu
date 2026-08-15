#include <cuda_runtime.h>

__global__ void rope(float* Q, float* cos, float* sin, float* output, int M, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * D) {
        int j = idx % D;
        int i = (idx - j) / D;
        float R = j < D/2 ? -Q[i * D + (j + D/2)]: Q[i * D + (j - D/2)];
        output[idx] = Q[idx] * cos[idx] + R * sin[idx];
    }
}

// Q, cos, sin, output are device pointers
extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {
    int threads = 256;
    int blocks = (M * D + threads - 1) / threads;
    rope<<<blocks, threads>>>(Q, cos, sin, output, M, D);
    cudaDeviceSynchronize();
}
