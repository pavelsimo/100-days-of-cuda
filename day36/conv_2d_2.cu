#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 32

__global__ void conv_kernel(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    
    __shared__ float inputS[TILE_DIM][TILE_DIM];
    if (i < input_rows && j < input_cols) {
        inputS[threadIdx.y][threadIdx.x] = input[i * input_cols + j];
    } else {
        inputS[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if (i < output_rows && j < output_cols) {
        float sum = 0.0f;
        for (int m = 0; m < kernel_rows; ++m) {
            for (int n = 0; n < kernel_cols; ++n) {
                const int sRow = threadIdx.y + m;
                const int sCol = threadIdx.x + n;
                const float kernel_value = kernel[m * kernel_cols + n];
                if (sRow < TILE_DIM && sCol < TILE_DIM) {
                    sum += inputS[sRow][sCol] * kernel_value;
                } else {
                    sum += input[(i + m) * input_cols + j + n] * kernel_value;
                }
            }
        }
        output[i * output_cols + j] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 blocksPerGrid((output_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (output_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    conv_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
