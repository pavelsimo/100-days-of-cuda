#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define M_DIM 4
#define N_DIM 3
#define K_DIM 5

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= M || j >= K) return;
    C[i * K + j] = 0;
    for (int n = 0; n < N; ++n) {
        C[i * K + j] += A[i * N + n] * B[n * K + j];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

static void print_matrix(const char *title, const float *m, int rows, int cols)
{
    printf("%s (%dx%d):\n", title, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.1f ", m[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void)
{
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    int sz_A = M_DIM * N_DIM * sizeof(float);
    int sz_B = N_DIM * K_DIM * sizeof(float);
    int sz_C = M_DIM * K_DIM * sizeof(float);

    /* Step 1: Allocate host memory */
    h_A = (float *)malloc(sz_A);
    h_B = (float *)malloc(sz_B);
    h_C = (float *)malloc(sz_C);

    /* Step 2: Allocate device memory */
    cudaMalloc((void **)&d_A, sz_A);
    cudaMalloc((void **)&d_B, sz_B);
    cudaMalloc((void **)&d_C, sz_C);

    /* Step 3: Initialize host input matrices */
    for (int i = 0; i < M_DIM * N_DIM; i++) {
        h_A[i] = (float)i;
    }
    for (int i = 0; i < N_DIM * K_DIM; i++) {
        h_B[i] = (float)(i % 3);
    }

    /* Step 4: Copy input matrices to device */
    cudaMemcpy(d_A, h_A, sz_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sz_B, cudaMemcpyHostToDevice);

    /* Step 5: Launch kernel via solve() */
    solve(d_A, d_B, d_C, M_DIM, N_DIM, K_DIM);

    /* Step 6: Copy result back to host */
    cudaMemcpy(h_C, d_C, sz_C, cudaMemcpyDeviceToHost);

    /* Step 7: Print result */
    print_matrix("A", h_A, M_DIM, N_DIM);
    print_matrix("B", h_B, N_DIM, K_DIM);
    print_matrix("C = A x B", h_C, M_DIM, K_DIM);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
