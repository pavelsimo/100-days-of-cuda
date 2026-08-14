#include <cuda_runtime.h>


__global__ void matmul(const float *A, const float *B, float *C, int BATCH, int M, int N, int K) {
    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch < BATCH && row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[batch * (M * K) + row * K + k] * B[batch * (K * N) + k * N + col];
        }
        C[batch * (M * N) + row * N + col] = sum;
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {

    dim3 blockDim(8, 8, 8);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y,
                 (BATCH + blockDim.z - 1) / blockDim.z);
    matmul<<<gridDim, blockDim>>>(A, B, C, BATCH, M, N, K);
    cudaDeviceSynchronize();
}


