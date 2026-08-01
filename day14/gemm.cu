#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void gemm_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    if (row >= M || col >= N) {
        return;
    }
    float c = __half2float(C[row * N + col]);
    float sum = 0.0f;
    for (int n = 0; n < K; ++n) {
        sum += __half2float(A[row * K + n]) * __half2float(B[n * N + col]);
    }
    C[row * N + col] = __float2half(alpha * sum + beta * c);
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 threadsPerBlock(8, 8);
    dim3 blocksPerGrid(
        (M + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}
