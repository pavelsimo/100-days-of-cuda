#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N_DIM 5

__global__ void copy_matrix_kernel(const float* A, float* B, int total) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    if (idx * 4 + 3  < total) {
        float4 a = A4[idx];
        float4 b = B4[idx];
        b.x = a.x;
        b.y = a.y;
        b.z = a.z;
        b.w = a.w;
        reinterpret_cast<float4*>(B)[idx] = b;
    }

    int rem = total - idx * 4;
    if (rem > 0 && rem < 4) {
        for (int k = 0; k < rem; ++k) {
            B[idx * 4 + k] = A[idx * 4 + k];
        }
    }  
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
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

    h_A = (float *)malloc(sz);
    h_B = (float *)malloc(sz);

    cudaMalloc((void **)&d_A, sz);
    cudaMalloc((void **)&d_B, sz);

    for (int i = 0; i < total; i++) {
        h_A[i] = (float)i;
    }

    cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice);

    print_matrix("Input", h_A, N_DIM);

    solve(d_A, d_B, N_DIM);

    cudaMemcpy(h_B, d_B, sz, cudaMemcpyDeviceToHost);

    print_matrix("Output", h_B, N_DIM);

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}
