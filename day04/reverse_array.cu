#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N_DIM 10

__global__ void reverse_array(float* input, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i * 2 < N) {
        float t = input[i];
        input[i] = input[N - i - 1];
        input[N - i - 1] = t;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
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
    float *h_input;
    float *d_input;
    int sz_input = N_DIM * sizeof(float);

    /* Step 1: Allocate host memory */
    h_input = (float *)malloc(sz_input);

    /* Step 2: Allocate device memory */
    cudaMalloc((void **)&d_input, sz_input);

    /* Step 3: Initialize host input array */
    for (int i = 0; i < N_DIM; i++) {
        h_input[i] = (float)i;
    }

    /* Step 4: Copy input array to device */
    cudaMemcpy(d_input, h_input, sz_input, cudaMemcpyHostToDevice);

    /* Step 5: Print input array */
    print_array("Input", h_input, N_DIM);

    /* Step 6: Launch kernel via solve() */
    solve(d_input, N_DIM);

    /* Step 7: Copy result back to host */
    cudaMemcpy(h_input, d_input, sz_input, cudaMemcpyDeviceToHost);

    /* Step 8: Print result */
    print_array("Reversed", h_input, N_DIM);

    cudaFree(d_input);
    free(h_input);

    return 0;
}
