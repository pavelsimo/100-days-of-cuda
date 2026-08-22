#include <cuda_runtime.h>
#include <float.h>

#define TILE 32

__device__ __forceinline__ float warp_reduce_sum(float value) {
    unsigned int mask = 0xFFFFFFFFU;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    unsigned int mask = 0xFFFFFFFFU;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(mask, value, offset));
    }
    return value;
}

__global__ void softmax(float* scores, int M, int N, int num_sinks, int window_size) {
    const int i = blockIdx.x;
    const int tid = threadIdx.x;

    if (i >= M) {
        return;
    }

    float threadMax = -FLT_MAX;
    for (int j = tid; j < N; j += blockDim.x) {
        if (j <= i && (j < num_sinks || j >= i - window_size + 1)) {
            threadMax = fmaxf(threadMax, scores[i * N + j]);
        }
    }
    const float rowMax = __shfl_sync(0xFFFFFFFFU, warp_reduce_max(threadMax), 0);
    float threadSum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        if (j <= i && (j < num_sinks || j >= i - window_size + 1)) {
            const float e = expf(scores[i * N + j] - rowMax);
            scores[i * N + j] = e;
            threadSum += e;
        } else {
            scores[i * N + j] = 0.0f;
        }
    }
    threadSum = __shfl_sync(0xFFFFFFFFU, warp_reduce_sum(threadSum), 0);
    const float inv = 1.0f / threadSum;
    for (int col = tid; col < N; col += blockDim.x) {
        scores[i * N + col] *= inv;
    }
}

__global__ void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_t(const float *A, const float *B, float *C, int M, int N, int K, float alpha) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = alpha * sum;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d,
                      int num_sinks, int window_size) {
    // Q: M x d
    // K: M x d
    // V: M x d
    // K^T: d x M
    // Q * K^T: M x M
    dim3 threads(TILE, TILE);
    dim3 blocks(
        (M + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );
    float *scores;
    float alpha = 1.0f / fsqrt(d);
    cudaMalloc(&scores, M * M * sizeof(float));
    matmul_t<<<blocks, threads>>>(Q, K, scores, M, M, d, alpha);
    softmax<<<M, TILE>>>(scores, M, M, num_sinks, window_size);
    matmul<<<blocks, threads>>>(scores, V, output, M, d, M);
    cudaDeviceSynchronize();
    cudaFree(scores);
}
