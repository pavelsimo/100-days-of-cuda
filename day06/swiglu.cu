#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N_DIM 10

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < halfN) {
        float x1 = input[idx];
        float x2 = input[idx + halfN];
        float sigma = 1.0 / (1.0 + exp(-x1));
        float silu = x1 * sigma;
        float swiglu = silu * x2;
        output[idx] = swiglu;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
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

    h_input = (float *)malloc(sz_in);
    h_output = (float *)malloc(sz_out);

    cudaMalloc((void **)&d_input, sz_in);
    cudaMalloc((void **)&d_output, sz_out);

    for (int i = 0; i < N_DIM; i++) {
        h_input[i] = (float)(i - N_DIM / 2);
    }

    cudaMemcpy(d_input, h_input, sz_in, cudaMemcpyHostToDevice);

    print_array("Input", h_input, N_DIM);

    solve(d_input, d_output, N_DIM);

    cudaMemcpy(h_output, d_output, sz_out, cudaMemcpyDeviceToHost);

    print_array("Output", h_output, halfN);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
