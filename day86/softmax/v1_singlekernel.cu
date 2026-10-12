#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 32


__global__ void softmax(const float* input, float* output, int M, int N, int H) {
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int tid = threadIdx.x;
    __shared__ float t[TILE_SIZE*TILE_SIZE];

    if (row >= M || head >= H) {
        return;
    }

    const float* row_input = input + (head * M + row) * N;
    float* row_scores = output + (head * M + row) * N;
    float thread_max = -FLT_MAX;
    for (int col = tid; col < N; col += blockDim.x) {
        thread_max = fmaxf(thread_max, row_input[col]);
    }
    t[tid] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] = fmaxf(t[tid], t[tid + stride]);
        }
        __syncthreads();
    }

    const float row_max = t[0];
    // Every thread must read the maximum before t is reused for the sum.
    __syncthreads();
    float thread_sum = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        const float e = expf(row_input[col] - row_max);
        row_scores[col] = e;
        thread_sum += e;
    }
    t[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] += t[tid + stride];
        }
        __syncthreads();
    }

    const float inv_sum = 1.0f / t[0];
    for (int col = tid; col < N; col += blockDim.x) {
        row_scores[col] *= inv_sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threads = 1024;
    int blocks = 1;
    softmax<<<blocks, threads>>>(input, output, 1, N, 1);
    cudaDeviceSynchronize();
}
