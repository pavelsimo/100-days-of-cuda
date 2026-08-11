#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce(float value) {
    unsigned mask = 0xFFFFFFFFU;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

__global__ void monte_carlo_integration(const float* y_samples, float* result, float a, float b, int n_samples, float c) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    __shared__ float t[256];
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    int stride = gridDim.x * blockDim.x;
    float sum = 0.0f;

    while (i < n_samples - 7 * stride) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            sum += y_samples[i + k * stride];
        }
        i += 8 * stride;
    }

    if (i < n_samples) {
        sum += y_samples[i];
        i += stride;
    }

    sum = warp_reduce(sum);
    if (lane == 0) {
        t[warpId] = sum;
    }
    __syncthreads();

    if (warpId == 0) {
        int numWarps = (blockDim.x / warpSize);
        sum = (lane < numWarps) ? t[lane]: 0.0f;
        sum = warp_reduce(sum);
        if (tid == 0) {
            atomicAdd(result, c * sum);
        }
    }
}

// y_samples, result are device pointers
extern "C" void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
    int threads = 256;
    int blocks = (n_samples + threads - 1) / threads;
    float c = 1.0 * (b - a) / n_samples;
    monte_carlo_integration<<<blocks, threads>>>(y_samples, result, a, b, n_samples, c);
    cudaDeviceSynchronize();
}
