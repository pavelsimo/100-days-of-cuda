#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 16

__global__ void matmul(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    const int batch = blockIdx.z;
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int aBatchOffset = batch * M * K;
    const int bBatchOffset = batch * K * N;
    const int cBatchOffset = batch * M * N;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (int tile = 0; tile < K; tile += TILE_SIZE) {
        const int aCol = tile + threadIdx.x;
        const int bRow = tile + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? __half2float(A[aBatchOffset + row * K + aCol]) : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? __half2float(B[bBatchOffset + bRow * N + col]) : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[cBatchOffset + row * N + col] = __float2half(sum);
    }
}

// A, B, C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE,
                 BATCH);
    matmul<<<gridDim, blockDim>>>(A, B, C, BATCH, M, N, K);
    cudaDeviceSynchronize();
}
