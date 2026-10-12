#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

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

__global__ void softmax_online(const float* __restrict__ input, float* __restrict__ output, int M, int N, int H) {
    float m = -INFINITY;
    float s = 0.0f;
    __shared__ float sm[32], ss[32];

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        combine(m, s, input[i], 1.0f);
    }

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

    float inv = 1.0f / s;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - m) * inv;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threads = 1024;
    int blocks = 1;
    softmax_online<<<blocks, threads>>>(input, output, 1, N, 1);
    cudaDeviceSynchronize();
}
