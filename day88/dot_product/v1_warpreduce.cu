#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void dot_product(const float* A, const float* B, float* result, int N) {
    __shared__ float t[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    unsigned mask = 0xFFFFFFFFU;
    float val = 0.0f;
    int stride = gridDim.x * blockDim.x;

    while (idx < N) {
        val += A[idx] * B[idx];
        idx += stride;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }

    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    if (lane == 0) {
        t[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x / warpSize;
        val = (tid < num_warps) ? t[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }

        if (tid == 0) {
            atomicAdd(result, val);
        }
    }
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threads = 256;
    int blocks = min((N + threads - 1) / threads, 1024);
    cudaMemset(result, 0, sizeof(float));
    dot_product<<<blocks, threads>>>(A, B, result, N);
    cudaDeviceSynchronize();
}
