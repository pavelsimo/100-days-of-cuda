#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N_DIM 5

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int base = idx * 2;
    if (idx < N) {
        output[base] = A[idx];
        output[base + 1] = B[idx];
    }
}

// A, B, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, N);
    cudaDeviceSynchronize();
}

static void print_array(const char *title, const float *a, int n)
{
    printf("%s (%d):\n", title, n);
    for (int i = 0; i < n; i++) {
        printf("%6.1f ", a[i]);
    }
    printf("\n\n");
}

int main(void)
{
    float *h_A, *h_B, *h_output;
    float *d_A, *d_B, *d_output;
    int sz = N_DIM * sizeof(float);
    int sz_out = 2 * N_DIM * sizeof(float);

    h_A = (float *)malloc(sz);
    h_B = (float *)malloc(sz);
    h_output = (float *)malloc(sz_out);

    cudaMalloc((void **)&d_A, sz);
    cudaMalloc((void **)&d_B, sz);
    cudaMalloc((void **)&d_output, sz_out);

    for (int i = 0; i < N_DIM; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i + 100);
    }

    cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice);

    print_array("A", h_A, N_DIM);
    print_array("B", h_B, N_DIM);

    solve(d_A, d_B, d_output, N_DIM);

    cudaMemcpy(h_output, d_output, sz_out, cudaMemcpyDeviceToHost);

    print_array("Output", h_output, 2 * N_DIM);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_output);
    free(h_A);
    free(h_B);
    free(h_output);

    return 0;
}
