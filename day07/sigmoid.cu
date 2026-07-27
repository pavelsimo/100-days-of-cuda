#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N_DIM 10

__forceinline__ __device__ float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

__global__ void sigmoid_kernel(const float* X, float* Y, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        Y[idx] = sigmoid(X[idx]);
    }
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}

static void print_array(const char *title, const float *a, int n)
{
    printf("%s (%d):\n", title, n);
    for (int i = 0; i < n; i++) {
        printf("%8.4f ", a[i]);
    }
    printf("\n\n");
}

int main(void)
{
    float *h_X, *h_Y;
    float *d_X, *d_Y;
    int sz = N_DIM * sizeof(float);

    h_X = (float *)malloc(sz);
    h_Y = (float *)malloc(sz);

    cudaMalloc((void **)&d_X, sz);
    cudaMalloc((void **)&d_Y, sz);

    for (int i = 0; i < N_DIM; i++) {
        h_X[i] = (float)(i - N_DIM / 2);
    }

    cudaMemcpy(d_X, h_X, sz, cudaMemcpyHostToDevice);

    print_array("Input", h_X, N_DIM);

    solve(d_X, d_Y, N_DIM);

    cudaMemcpy(h_Y, d_Y, sz, cudaMemcpyDeviceToHost);

    print_array("Output", h_Y, N_DIM);

    cudaFree(d_X);
    cudaFree(d_Y);
    free(h_X);
    free(h_Y);

    return 0;
}
