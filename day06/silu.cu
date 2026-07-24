#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N_DIM 10

__global__ void silu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sigma = 1.0 / (1.0 + exp(-input[idx]));
        float silu = input[idx] * sigma;
        output[idx] = silu;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
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
    float *h_input, *h_output;
    float *d_input, *d_output;
    int sz = N_DIM * sizeof(float);

    /* Step 1: Allocate host memory */
    h_input = (float *)malloc(sz);
    h_output = (float *)malloc(sz);

    /* Step 2: Allocate device memory */
    cudaMalloc((void **)&d_input, sz);
    cudaMalloc((void **)&d_output, sz);

    /* Step 3: Initialize host input array */
    for (int i = 0; i < N_DIM; i++) {
        h_input[i] = (float)(i - N_DIM / 2);
    }

    /* Step 4: Copy input array to device */
    cudaMemcpy(d_input, h_input, sz, cudaMemcpyHostToDevice);

    /* Step 5: Print input array */
    print_array("Input", h_input, N_DIM);

    /* Step 6: Launch kernel via solve() */
    solve(d_input, d_output, N_DIM);

    /* Step 7: Copy result back to host */
    cudaMemcpy(h_output, d_output, sz, cudaMemcpyDeviceToHost);

    /* Step 8: Print result */
    print_array("Output", h_output, N_DIM);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
