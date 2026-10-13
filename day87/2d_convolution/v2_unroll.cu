#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE 16
#define MAX_KERNEL 31

// every thread in a warp reads the same weight at the same time -> constant cache broadcast
__constant__ float c_k[MAX_KERNEL * MAX_KERNEL];

template <int KERNEL_ROWS, int KERNEL_COLS>
__global__ void conv_kernel(const float* input, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
                        
    const int kr = KERNEL_ROWS > 0 ? KERNEL_ROWS : kernel_rows;
    const int kc = KERNEL_COLS > 0 ? KERNEL_COLS : kernel_cols;
    int i = blockIdx.y * TILE + threadIdx.y;
    int j = blockIdx.x * TILE + threadIdx.x;

    extern __shared__ float s_in[];
    int tile_rows = TILE + kr - 1;
    int tile_cols = TILE + kc - 1;
    int output_rows = input_rows - kr + 1;
    int output_cols = input_cols - kc + 1;

    #pragma unroll
    for (int m = threadIdx.y; m < tile_rows; m += TILE) {
        for (int n = threadIdx.x; n < tile_cols; n += TILE) {
            int r = blockIdx.y * TILE + m;
            int c = blockIdx.x * TILE + n;
            s_in[m * tile_cols + n] = (r < input_rows && c < input_cols) ? input[r * input_cols + c] : 0.0f;
        }
    }
    __syncthreads();

    if (i < output_rows && j < output_cols) {
        float sum = 0.0f;
        #pragma unroll
        for (int m = 0; m < kr; ++m) {
            #pragma unroll
            for (int n = 0; n < kc; ++n) {
                sum += s_in[(threadIdx.y + m) * tile_cols + threadIdx.x + n] * c_k[m * kc + n];
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
    cudaMemcpyToSymbol(c_k, kernel, sizeof(float) * kernel_rows * kernel_cols, 0, cudaMemcpyDeviceToDevice);
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((output_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (output_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    size_t smem = sizeof(float) * (TILE + kernel_rows - 1) * (TILE + kernel_cols - 1);
    
    if (kernel_rows == 15 && kernel_cols == 15)
        conv_kernel<15, 15><<<blocksPerGrid, threadsPerBlock, smem>>>(input, output, input_rows, input_cols, kernel_rows, kernel_cols);
    else if (kernel_rows == 3 && kernel_cols == 3)
        conv_kernel<3, 3><<<blocksPerGrid, threadsPerBlock, smem>>>(input, output, input_rows, input_cols, kernel_rows, kernel_cols);
    else if (kernel_rows == 7 && kernel_cols == 7)
        conv_kernel<7, 7><<<blocksPerGrid, threadsPerBlock, smem>>>(input, output, input_rows, input_cols, kernel_rows, kernel_cols);
    else
        conv_kernel<0, 0><<<blocksPerGrid, threadsPerBlock, smem>>>(input, output, input_rows, input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
