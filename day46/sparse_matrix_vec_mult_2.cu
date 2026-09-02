#include <cuda_runtime.h>

__global__ void init_indices(int* row_count, float* A_pack, int* x_idx, const float* A, int M, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < M * N) {
        int i = idx / N;
        int j = idx % N;
        if (A[idx] != 0) {
            int pos = atomicAdd(&row_count[i], 1);
            A_pack[i * N + pos] = A[idx];
            x_idx[i * N + pos] = j;
        }
    }
}

__global__ void sparse_matrix_vec_mult(const int* row_count, const float* A_pack, const int* x_idx,
                                       const float* x, float* y, int M, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < M) {
        float sum = 0;
        int cnt = row_count[i];
        for (int k = 0; k < cnt; ++k) {
            sum += A_pack[i * N + k] * x[x_idx[i * N + k]];
        }
        y[i] = sum;
    }
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    int *row_count, *x_idx;
    float *A_pack;
    cudaMalloc(&row_count, M * sizeof(int));
    cudaMemset(row_count, 0, M * sizeof(int));

    cudaMalloc(&A_pack, M * N * sizeof(float));
    cudaMalloc(&x_idx, M * N * sizeof(int));
    
    init_indices<<<(M * N + threads - 1) / threads, threads>>>(row_count, A_pack, x_idx, A, M, N);
    sparse_matrix_vec_mult<<<blocks, threads>>>(row_count, A_pack, x_idx, x, y, M, N);
    
    cudaFree(row_count);
    cudaFree(A_pack);
    cudaFree(x_idx);
}
