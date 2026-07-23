#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define INPUT_SIZE 10
#define KERNEL_SIZE 3

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx + kernel_size - 1 < input_size) {
        float sum = 0;
        for (int k = 0; k < kernel_size; ++k) {
            sum += input[idx + k] * kernel[k];
        }
        output[idx] = sum;
    }

}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                              kernel_size);
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
    float *h_input, *h_kernel, *h_output;
    float *d_input, *d_kernel, *d_output;
    int output_size = INPUT_SIZE - KERNEL_SIZE + 1;
    int sz_input = INPUT_SIZE * sizeof(float);
    int sz_kernel = KERNEL_SIZE * sizeof(float);
    int sz_output = output_size * sizeof(float);

    /* Step 1: Allocate host memory */
    h_input = (float *)malloc(sz_input);
    h_kernel = (float *)malloc(sz_kernel);
    h_output = (float *)malloc(sz_output);

    /* Step 2: Allocate device memory */
    cudaMalloc((void **)&d_input, sz_input);
    cudaMalloc((void **)&d_kernel, sz_kernel);
    cudaMalloc((void **)&d_output, sz_output);

    /* Step 3: Initialize host arrays */
    for (int i = 0; i < INPUT_SIZE; i++) {
        h_input[i] = (float)i;
    }
    for (int i = 0; i < KERNEL_SIZE; i++) {
        h_kernel[i] = 1.0f / KERNEL_SIZE;
    }

    /* Step 4: Copy input and kernel to device */
    cudaMemcpy(d_input, h_input, sz_input, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sz_kernel, cudaMemcpyHostToDevice);

    /* Step 5: Print input arrays */
    print_array("Input", h_input, INPUT_SIZE);
    print_array("Kernel", h_kernel, KERNEL_SIZE);

    /* Step 6: Launch kernel via solve() */
    solve(d_input, d_kernel, d_output, INPUT_SIZE, KERNEL_SIZE);

    /* Step 7: Copy result back to host */
    cudaMemcpy(h_output, d_output, sz_output, cudaMemcpyDeviceToHost);

    /* Step 8: Print result */
    print_array("Output", h_output, output_size);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_input);
    free(h_kernel);
    free(h_output);

    return 0;
}
