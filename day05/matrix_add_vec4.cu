
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N_DIM 5

__global__ void matrix_add(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
   
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    // all four elements exist
    if (i * 4 + 3 < N * N) {
        float4 a = A4[i];
        float4 b = B4[i];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        reinterpret_cast<float4*>(C)[i] = c;
    }
    
    // handle remaining elements
    int remaining = N * N - i * 4;
    if (remaining > 0 && remaining < 4) {
        for (int j = 0; j < remaining; ++j) {
            C[i * 4 + j] = A[i * 4 + j] + B[i * 4 + j];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
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
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    int total = N_DIM * N_DIM;
    int sz = total * sizeof(float);

    h_A = (float *)malloc(sz);
    h_B = (float *)malloc(sz);
    h_C = (float *)malloc(sz);

    cudaMalloc((void **)&d_A, sz);
    cudaMalloc((void **)&d_B, sz);
    cudaMalloc((void **)&d_C, sz);

    for (int i = 0; i < total; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(total - i);
    }

    cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice);

    print_matrix("A", h_A, N_DIM);
    print_matrix("B", h_B, N_DIM);

    solve(d_A, d_B, d_C, N_DIM);

    cudaMemcpy(h_C, d_C, sz, cudaMemcpyDeviceToHost);

    print_matrix("C = A + B", h_C, N_DIM);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
