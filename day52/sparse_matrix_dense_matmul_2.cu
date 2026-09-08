#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int rowB = blockIdx.y * TILE_SIZE;
    const int colB = blockIdx.x * TILE_SIZE;
    const int row = rowB + ty;
    const int col = colB + tx;

    float sum = 0;
    for (int t = 0; t < K; t += TILE_SIZE) {
        const int kA = t + tx;
        const int kB = t + ty;
        As[ty][tx] = (row < M && kA < K) ? (A[row * K + kA]) : 0;
        Bs[ty][tx] = (kB < K && col < N) ? (B[kB * N + col]) : 0;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K, int nnz) {
    dim3 threads(16, 16);
    dim3 grid(
        (K + threads.x - 1) / threads.x, 
        (M + threads.y - 1) / threads.y);
    matmul<<<grid, threads>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
