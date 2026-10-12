#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_BLOCKS 1024

// static device memory, so no cudaMalloc per call
__device__ float g_part_m[MAX_BLOCKS];
__device__ float g_part_s[MAX_BLOCKS];

__device__ __forceinline__ void combine(float &m, float &s, float m2, float s2) {
    float mn = fmaxf(m, m2);
    if (mn == -INFINITY) {
        return;
    }
    s = s * expf(m - mn) + s2 * expf(m2 - mn);
    m = mn;
}

__device__ __forceinline__ void warp_reduce(float &m, float &s) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float m2 = __shfl_xor_sync(0xffffffff, m, offset);
        float s2 = __shfl_xor_sync(0xffffffff, s, offset);
        combine(m, s, m2, s2);
    }
}

__device__ __forceinline__ void block_reduce(float &m, float &s) {
    __shared__ float sm[32], ss[32];

    warp_reduce(m, s);

    const int lane = threadIdx.x % warpSize;
    const int warp = threadIdx.x / warpSize;
    if (lane == 0) {
        sm[warp] = m;
        ss[warp] = s;
    }
    __syncthreads();

    if (warp == 0) {
        const int num_warps = blockDim.x / warpSize;
        m = (lane < num_warps) ? sm[lane] : -INFINITY;
        s = (lane < num_warps) ? ss[lane] : 0.0f;

        warp_reduce(m, s);
        if (lane == 0) {
            sm[0] = m;
            ss[0] = s;
        }
    }
    __syncthreads();

    m = sm[0];
    s = ss[0];
}

__global__ void softmax_partials(const float* __restrict__ input, int N) {
    float m = -INFINITY;
    float s = 0.0f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        combine(m, s, input[i], 1.0f);
    }

    block_reduce(m, s);
    if (threadIdx.x == 0) {
        g_part_m[blockIdx.x] = m;
        g_part_s[blockIdx.x] = s;
    }
}

__global__ void softmax_normalize(const float* __restrict__ input, float* __restrict__ output, int N, int num_parts) {
    float m = -INFINITY;
    float s = 0.0f;

    for (int i = threadIdx.x; i < num_parts; i += blockDim.x) {
        combine(m, s, g_part_m[i], g_part_s[i]);
    }

    block_reduce(m, s);

    float inv = 1.0f / s;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        output[i] = expf(input[i] - m) * inv;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > MAX_BLOCKS) {
        blocks = MAX_BLOCKS;
    }
    softmax_partials<<<blocks, threads>>>(input, N);
    softmax_normalize<<<blocks, threads>>>(input, output, N, blocks);
    cudaDeviceSynchronize();
}
