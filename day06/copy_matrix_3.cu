#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N_DIM 5

__global__ void copy_matrix_kernel(const float* A, float* B, int total) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int base = idx * 4;
    
    if (base + 3 < total) {
        reinterpret_cast<float4*>(B)[idx] = reinterpret_cast<const float4*>(A)[idx];
    } else if (base < total) {
        // fewer than 4 elements left
        for (int k = base; k < total; ++k) {
            B[k] = A[k];
        }
    }
}

extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int vecCount = (total + 3) / 4;  // number of float4-sized chunks (incl. partial tail)
    int blocksPerGrid = (vecCount + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, total);
    cudaDeviceSynchronize();
}

static void print_matrix(const char *title, const float *m, int n)
{
    printf("%s (%dx%d):\n", title, n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%6.1f ", m[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void)
{
    float *h_A, *h_B;
    float *d_A, *d_B;
    int total = N_DIM * N_DIM;
    int sz = total * sizeof(float);

    /* Step 1: Allocate host memory */
    h_A = (float *)malloc(sz);
    h_B = (float *)malloc(sz);

    /* Step 2: Allocate device memory */
    cudaMalloc((void **)&d_A, sz);
    cudaMalloc((void **)&d_B, sz);

    /* Step 3: Initialize host input matrix */
    for (int i = 0; i < total; i++) {
        h_A[i] = (float)i;
    }

    /* Step 4: Copy input matrix to device */
    cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice);

    /* Step 5: Print input matrix */
    print_matrix("Input", h_A, N_DIM);

    /* Step 6: Launch kernel via solve() */
    solve(d_A, d_B, N_DIM);

    /* Step 7: Copy result back to host */
    cudaMemcpy(h_B, d_B, sz, cudaMemcpyDeviceToHost);

    /* Step 8: Print result */
    print_matrix("Output", h_B, N_DIM);

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}