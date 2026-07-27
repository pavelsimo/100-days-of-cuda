#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N_DIM 10

__global__ void dot_product(const float* A, const float* B, float* result, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    __shared__ float t[256];
    t[tid] = (i < N) ? A[i] * B[i]: 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] += t[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, t[0]);
    }
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dot_product<<<blocksPerGrid, threadsPerBlock>>>(A, B, result, N);
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
    float *h_A, *h_B, h_result;
    float *d_A, *d_B, *d_result;
    int sz = N_DIM * sizeof(float);

    h_A = (float *)malloc(sz);
    h_B = (float *)malloc(sz);

    cudaMalloc((void **)&d_A, sz);
    cudaMalloc((void **)&d_B, sz);
    cudaMalloc((void **)&d_result, sizeof(float));

    for (int i = 0; i < N_DIM; i++) {
        h_A[i] = (float)(i + 1);
        h_B[i] = (float)(N_DIM - i);
    }

    cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    print_array("A", h_A, N_DIM);
    print_array("B", h_B, N_DIM);

    solve(d_A, d_B, d_result, N_DIM);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Dot product: %6.1f\n\n", h_result);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);
    free(h_A);
    free(h_B);

    return 0;
}
