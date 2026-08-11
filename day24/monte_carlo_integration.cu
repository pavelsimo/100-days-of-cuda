#include <cuda_runtime.h>


__global__ void monte_carlo_kernel(const float* y_samples, float* result, float a, float b, int n_samples) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    __shared__ float t[256];
    t[tid] = (i < n_samples) ? y_samples[i]: 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] += t[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float c = 1.0 * (b - a) / n_samples; 
        atomicAdd(result, c * t[0]);
    }
}

// y_samples, result are device pointers
extern "C" void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
    int threads = 256;
    int blocks = (n_samples + threads - 1) / threads;;
    monte_carlo_kernel<<<blocks, threads>>>(y_samples, result, a, b, n_samples);
    cudaDeviceSynchronize();
}
