#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N_DIM 10

__forceinline__ __device__ float gelu(float x) {
    return 0.5 * x * (1.0 + erff(x * 0.70710678118f));
}

__forceinline__ __device__ float geglu(float x1, float x2) {
    return x1 * gelu(x2);
}

__global__ void geglu_kernel(const float* input, float* output, int halfN) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < halfN) {
        float x1 = input[idx];
        float x2 = input[idx + halfN];
        output[idx] = geglu(x1, x2);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
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
    int halfN = N_DIM / 2;
    int sz_in = N_DIM * sizeof(float);
    int sz_out = halfN * sizeof(float);

    /* Step 1: Allocate host memory */
    h_input = (float *)malloc(sz_in);
    h_output = (float *)malloc(sz_out);

    /* Step 2: Allocate device memory */
    cudaMalloc((void **)&d_input, sz_in);
    cudaMalloc((void **)&d_output, sz_out);

    /* Step 3: Initialize host input array (first half x1, second half x2) */
    for (int i = 0; i < N_DIM; i++) {
        h_input[i] = (float)(i - N_DIM / 2);
    }

    /* Step 4: Copy input array to device */
    cudaMemcpy(d_input, h_input, sz_in, cudaMemcpyHostToDevice);

    /* Step 5: Print input array */
    print_array("Input", h_input, N_DIM);

    /* Step 6: Launch kernel via solve() */
    solve(d_input, d_output, N_DIM);

    /* Step 7: Copy result back to host */
    cudaMemcpy(h_output, d_output, sz_out, cudaMemcpyDeviceToHost);

    /* Step 8: Print result */
    print_array("Output", h_output, halfN);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
