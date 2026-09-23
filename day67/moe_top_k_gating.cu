#include <cuda_runtime.h>
#include <float.h>

#define TILE_SIZE 16

__global__ void softmax(float* scores, int M, int N) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float t[TILE_SIZE*TILE_SIZE];

    if (row >= M) {
        return;
    }

    float threadMax = -FLT_MAX;
    for (int col = tid; col < N; col += blockDim.x) {
        threadMax = fmaxf(threadMax, scores[row * N + col]);
    }
    t[tid] = threadMax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] = fmaxf(t[tid], t[tid + stride]);
        }
        __syncthreads();
    }

    const float rowMax = t[0];
    float threadSum = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        const float e = expf(scores[row * N + col] - rowMax);
        scores[row * N + col] = e;
        threadSum += e;
    }
    t[tid] = threadSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] += t[tid + stride];
        }
        __syncthreads();
    }

    const float inv = 1.0f / t[0];
    for (int col = tid; col < N; col += blockDim.x) {
        scores[row * N + col] *= inv;
    }
}

__global__ void top_k(const float* logits, float* topk_weights, int* topk_indices,
                      int M, int E, int K) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) {
        return;
    }

    const float* p_logits = logits + m * E;
    float* p_weights = topk_weights + m * K;
    int* p_indices = topk_indices + m * K;
    for (int k = 0; k < K; k++) {
        float best_val = -FLT_MAX;
        int   best_idx = -1;
        for (int e = 0; e < E; e++) {
            bool used = false;
            for (int j = 0; j < k; j++) {
                if (p_indices[j] == e) {
                    used = true;
                    break;
                }
            }

            if (!used && p_logits[e] > best_val) {
                best_val = p_logits[e];
                best_idx = e;
            }
        }
        p_weights[k] = best_val;
        p_indices[k] = best_idx;
    }
}

// logits, topk_weights, topk_indices are device pointers
extern "C" void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k) {
    // logits       - [M, E]
    // topk_weights - [M, K]
    // topk_indices - [M, K]
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    top_k<<<blocks, threads>>>(logits, topk_weights, topk_indices, M, E, k);
    softmax<<<M, threads>>>(topk_weights, M, k);
    cudaDeviceSynchronize();
}
