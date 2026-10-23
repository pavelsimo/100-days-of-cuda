#include <cuda_runtime.h>
#include <cstdlib>

#define BLOCKSIZE 16

__device__ __forceinline__ void matmul(const float* A, const float* B, float* C, int M, int N, int K, float alpha) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = alpha * sum;
        }
    }
}

__device__ __forceinline__ void matadd(const float* A, const float* B, float* C, int M, int N, float alpha) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            C[m * N + n] = alpha * A[m * N + n] + B[m * N + n];
        }
    }
}

__global__ void lora_linear(const float* x, const float* W, const float* A, const float* B,
                            const int* adapter_ids, float* y, float* bufA, float* bufB,
                            int T, int K, int N, int R, int L, float scale) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) {
        return;
    }

    float *tempA = bufA + (t * R);
    float *tempB = bufB + (t * N);
    const float *xt = x + (t * K);
    const int a = adapter_ids[t];
    float *yt = y + (t * N);
    matmul(xt, W, yt, 1, N, K, 1.0f);
    if (a >= 0) {
        const float *Aa = A + (a * K * R);
        const float *Ba = B + (a * R * N);
        matmul(xt, Aa, tempA, 1, R, K, scale);
        matmul(tempA, Ba, tempB, 1, N, R, 1.0f);
        matadd(yt, tempB, yt, 1, N, 1.0f);
    }
}

// x, W, A, B, adapter_ids, y are device pointers
extern "C" void solve(const float* x, const float* W, const float* A, const float* B,
                      const int* adapter_ids, float* y, int T, int K, int N, int R, int L,
                      float scale) {

    // T - tokens
    // N - output features
    // K - input features
    // R - rank
    // L - number of adapters
    // 
    // W - [K, N]
    // y - [T, N]
    // x - [T, K]
    // A - [L, K, R]
    // B - [L, R, N]
    float *bufA, *bufB;
    cudaMalloc(&bufA, T * R * sizeof(float));
    cudaMalloc(&bufB, T * N * sizeof(float));
    
    int threads = 256;
    int blocks = (T + threads - 1) / threads;
    lora_linear<<<blocks, threads>>>(x, W, A, B, adapter_ids, y, bufA, bufB, T, K, N, R, L, scale);
    cudaDeviceSynchronize();

    cudaFree(bufA);
    cudaFree(bufB);
}
