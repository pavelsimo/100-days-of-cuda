#include <cuda_runtime.h>

__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
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
