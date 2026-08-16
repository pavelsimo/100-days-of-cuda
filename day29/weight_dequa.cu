#include <cuda_runtime.h>

__global__ void weight_dequantization(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE, int S_COLS) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        int row = i / TILE_SIZE;
        int col = j / TILE_SIZE;
        Y[i * N + j] = X[i * N + j] * S[row * S_COLS + col];
    }
}

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
    dim3 threads(16, 16);
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );
    int S_COLS = (N + TILE_SIZE - 1) / TILE_SIZE;
    weight_dequantization<<<blocks, threads>>>(X, S, Y, M, N, TILE_SIZE, S_COLS);
    cudaDeviceSynchronize();
}
