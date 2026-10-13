#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void conv_kernel(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    if (i < output_rows && j < output_cols) {
        float sum = 0.0f;
        for (int m = 0; m < kernel_rows; ++m) {
            for (int n = 0; n < kernel_cols; ++n) {
                sum += input[((i + m) * input_cols + j + n)] * kernel[m * kernel_cols + n];
            }
        }
        output[i * output_cols + j] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((input_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (input_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    conv_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
