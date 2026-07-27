#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N_DIM 10
#define MAX(a, b) ((a) > (b) ? (a) : (b))

__global__ void relu_kernel(const float* input, float* output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        output[idx] = MAX(0, input[idx]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
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
    float *h_input, *h_output;
    float *d_input, *d_output;
    int sz = N_DIM * sizeof(float);

    h_input = (float *)malloc(sz);
    h_output = (float *)malloc(sz);

    cudaMalloc((void **)&d_input, sz);
    cudaMalloc((void **)&d_output, sz);

    for (int i = 0; i < N_DIM; i++) {
        h_input[i] = (float)(i - N_DIM / 2);
    }

    cudaMemcpy(d_input, h_input, sz, cudaMemcpyHostToDevice);

    print_array("Input", h_input, N_DIM);

    solve(d_input, d_output, N_DIM);

    cudaMemcpy(h_output, d_output, sz, cudaMemcpyDeviceToHost);

    print_array("Output", h_output, N_DIM);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
