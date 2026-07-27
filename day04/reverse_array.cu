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

    h_input = (float *)malloc(sz_input);

    cudaMalloc((void **)&d_input, sz_input);

    for (int i = 0; i < N_DIM; i++) {
        h_input[i] = (float)i;
    }

    cudaMemcpy(d_input, h_input, sz_input, cudaMemcpyHostToDevice);

    print_array("Input", h_input, N_DIM);

    solve(d_input, N_DIM);

    cudaMemcpy(h_input, d_input, sz_input, cudaMemcpyDeviceToHost);

    print_array("Reversed", h_input, N_DIM);

    cudaFree(d_input);
    free(h_input);

    return 0;
}
