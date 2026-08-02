#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCKSIZE 32

__global__ void gemm_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    int col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
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
    dim3 threadsPerBlock(BLOCKSIZE * BLOCKSIZE);
    dim3 blocksPerGrid(
        (M + BLOCKSIZE - 1) / BLOCKSIZE,
        (N + BLOCKSIZE - 1) / BLOCKSIZE
    );
    gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}

static void print_matrix(const char *title, const half *a, int rows, int cols)
{
    printf("%s (%dx%d):\n", title, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.2f ", __half2float(a[i * cols + j]));
        }
        printf("\n");
    }
    printf("\n");
}

int main(void)
{
    half *h_A, *h_B, *h_C;
    half *d_A, *d_B, *d_C;
    float alpha = 1.0f, beta = 0.5f;
    int szA = M_DIM * K_DIM * sizeof(half);
    int szB = K_DIM * N_DIM * sizeof(half);
    int szC = M_DIM * N_DIM * sizeof(half);

    h_A = (half *)malloc(szA);
    h_B = (half *)malloc(szB);
    h_C = (half *)malloc(szC);

    cudaMalloc((void **)&d_A, szA);
    cudaMalloc((void **)&d_B, szB);
    cudaMalloc((void **)&d_C, szC);

    for (int i = 0; i < M_DIM * K_DIM; i++) {
        h_A[i] = __float2half((float)(i % 5));
    }
    for (int i = 0; i < K_DIM * N_DIM; i++) {
        h_B[i] = __float2half((float)(i % 3));
    }
    for (int i = 0; i < M_DIM * N_DIM; i++) {
        h_C[i] = __float2half(1.0f);
    }

    cudaMemcpy(d_A, h_A, szA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, szB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, szC, cudaMemcpyHostToDevice);

    print_matrix("A", h_A, M_DIM, K_DIM);
    print_matrix("B", h_B, K_DIM, N_DIM);
    print_matrix("C", h_C, M_DIM, N_DIM);
    printf("alpha = %.2f, beta = %.2f\n\n", alpha, beta);

    solve(d_A, d_B, d_C, M_DIM, N_DIM, K_DIM, alpha, beta);

    cudaMemcpy(h_C, d_C, szC, cudaMemcpyDeviceToHost);

    print_matrix("C = alpha * A x B + beta * C", h_C, M_DIM, N_DIM);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
