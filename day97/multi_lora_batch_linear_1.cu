#include <cuda_runtime.h>
#include <cstdlib>

#define BLOCKSIZE 16

__global__ void matmul(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int M, int N, int K, float alpha) {
    __shared__ float As[BLOCKSIZE][BLOCKSIZE + 1];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCKSIZE + ty;
    const int col = blockIdx.x * BLOCKSIZE + tx;

    float sum = 0.0f;
    for (int t = 0; t < K; t += BLOCKSIZE) {
        As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.0f;
        Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCKSIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * sum;
    }
}

__global__ void matadd(const float* A, const float* B, float* C, int M, int N , float alpha) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCKSIZE + ty;
    const int col = blockIdx.x * BLOCKSIZE + tx;

    if (row < M && col < N) {
        C[row * N + col] = alpha * A[row * N + col] + B[row * N + col];
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
    dim3 threads(BLOCKSIZE, BLOCKSIZE);
    dim3 blocks_1(
        (N + threads.x - 1) / threads.x, 
        (T + threads.y - 1) / threads.y
    );
    dim3 blocks_2(
        (R + threads.x - 1) / threads.x,
        1
    );

    float *bufA, *bufB;
    cudaMalloc(&bufA, R * sizeof(float));
    cudaMalloc(&bufB, N * sizeof(float));

    int* h_adapter_ids = (int*)malloc(T * sizeof(int));
    cudaMemcpy(h_adapter_ids, adapter_ids, T * sizeof(int), cudaMemcpyDeviceToHost);
    for (int t = 0; t < T; ++t) {
        const float *xt = x + (t * K);
        const int a = h_adapter_ids[t];
        float *yt = y + (t * N);
        matmul<<<blocks_1, threads>>>(xt, W, yt, 1, N, K, 1.0f);
        if (a >= 0) {
            const float *Aa = A + (a * K * R);
            const float *Ba = B + (a * R * N);
            matmul<<<blocks_2, threads>>>(xt, Aa, bufA, 1, R, K, scale);
            matmul<<<blocks_1, threads>>>(bufA, Ba, bufB, 1, N, R, 1.0f);
            matadd<<<blocks_1, threads>>>(yt, bufB, yt, 1, N, 1.0f);
        }
    }
    cudaDeviceSynchronize();
    free(h_adapter_ids);
    cudaFree(bufA);
    cudaFree(bufB);
}
