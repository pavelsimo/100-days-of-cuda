#include <cuda_runtime.h>

__global__ void sparse_matrix_vec_mult(const float* A, const float* x, float* y, int M, int N, int nnz) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < M) {
        float sum =  0;
        for (int j = 0; j < N; ++j) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    sparse_matrix_vec_mult<<<blocks, threads>>>(A, x, y, M, N, nnz);
}
