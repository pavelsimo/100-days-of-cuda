#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define INPUT_ROWS 5
#define INPUT_COLS 5
#define KERNEL_ROWS 3
#define KERNEL_COLS 3

__global__ void gaussian_blur_kernel(const float* input, const float* kernel, float* output, int input_rows, 
                                     int input_cols, int kernel_rows, int kernel_cols) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int kh = kernel_rows;
    int kw = kernel_cols;
    int half_kh = kh / 2;
    int half_kw = kw / 2;
    if (i < input_rows && j < input_cols) {
        float sum = 0.0f;
        for (int m = -half_kh; m <= half_kh; ++m) {
            for (int n = -half_kw; n <= half_kw; ++n) {
               int r = i + m;
               int c = j + n;
               if (r < 0 || r >= input_rows || c < 0 || c >= input_cols) {
                   continue;
               }
               sum += input[r * input_cols + c] * kernel[(m + half_kh) * kernel_cols + n + half_kw];
            }
        }
        output[i * input_cols + j] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows, 
                      int input_cols, int kernel_rows, int kernel_cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((input_cols + threadsPerBlock.x - 1 ) / threadsPerBlock.x,
                       (input_rows + threadsPerBlock.y - 1 ) / threadsPerBlock.y);
    gaussian_blur_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, 
                           input_rows, input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}

static void print_matrix(const char *title, const float *a, int rows, int cols)
{
    printf("%s (%dx%d):\n", title, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", a[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void)
{
    float *h_input, *h_output;
    float *d_input, *d_kernel, *d_output;
    int input_sz = INPUT_ROWS * INPUT_COLS * sizeof(float);
    int kernel_sz = KERNEL_ROWS * KERNEL_COLS * sizeof(float);

    // 3x3 gaussian kernel
    float h_kernel[KERNEL_ROWS * KERNEL_COLS] = {
        1.0f / 16, 2.0f / 16, 1.0f / 16,
        2.0f / 16, 4.0f / 16, 2.0f / 16,
        1.0f / 16, 2.0f / 16, 1.0f / 16
    };

    h_input = (float *)malloc(input_sz);
    h_output = (float *)malloc(input_sz);

    cudaMalloc((void **)&d_input, input_sz);
    cudaMalloc((void **)&d_kernel, kernel_sz);
    cudaMalloc((void **)&d_output, input_sz);

    for (int i = 0; i < INPUT_ROWS * INPUT_COLS; i++) {
        h_input[i] = (float)(i + 1);
    }

    cudaMemcpy(d_input, h_input, input_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_sz, cudaMemcpyHostToDevice);

    print_matrix("Input", h_input, INPUT_ROWS, INPUT_COLS);
    print_matrix("Kernel", h_kernel, KERNEL_ROWS, KERNEL_COLS);

    solve(d_input, d_kernel, d_output, INPUT_ROWS, INPUT_COLS, KERNEL_ROWS, KERNEL_COLS);

    cudaMemcpy(h_output, d_output, input_sz, cudaMemcpyDeviceToHost);

    print_matrix("Output", h_output, INPUT_ROWS, INPUT_COLS);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
