#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define INPUT_ROWS 5
#define INPUT_COLS 5
#define KERNEL_ROWS 3
#define KERNEL_COLS 3

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

static void print_matrix(const char *title, const float *a, int rows, int cols)
{
    printf("%s (%dx%d):\n", title, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.1f ", a[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void)
{
    float *h_input, *h_kernel, *h_output;
    float *d_input, *d_kernel, *d_output;
    int output_rows = INPUT_ROWS - KERNEL_ROWS + 1;
    int output_cols = INPUT_COLS - KERNEL_COLS + 1;
    int input_sz = INPUT_ROWS * INPUT_COLS * sizeof(float);
    int kernel_sz = KERNEL_ROWS * KERNEL_COLS * sizeof(float);
    int output_sz = output_rows * output_cols * sizeof(float);

    h_input = (float *)malloc(input_sz);
    h_kernel = (float *)malloc(kernel_sz);
    h_output = (float *)malloc(output_sz);

    cudaMalloc((void **)&d_input, input_sz);
    cudaMalloc((void **)&d_kernel, kernel_sz);
    cudaMalloc((void **)&d_output, output_sz);

    for (int i = 0; i < INPUT_ROWS * INPUT_COLS; i++) {
        h_input[i] = (float)(i + 1);
    }
    for (int i = 0; i < KERNEL_ROWS * KERNEL_COLS; i++) {
        h_kernel[i] = 1.0f;
    }

    cudaMemcpy(d_input, h_input, input_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_sz, cudaMemcpyHostToDevice);

    print_matrix("Input", h_input, INPUT_ROWS, INPUT_COLS);
    print_matrix("Kernel", h_kernel, KERNEL_ROWS, KERNEL_COLS);

    solve(d_input, d_kernel, d_output, INPUT_ROWS, INPUT_COLS, KERNEL_ROWS, KERNEL_COLS);

    cudaMemcpy(h_output, d_output, output_sz, cudaMemcpyDeviceToHost);

    print_matrix("Output", h_output, output_rows, output_cols);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_input);
    free(h_kernel);
    free(h_output);

    return 0;
}
