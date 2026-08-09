#include <cuda_runtime.h>


__global__ void conv_3d(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols) {
    int depth = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int output_depth = input_depth - kernel_depth + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    if (depth < output_depth && row < output_rows && col < output_cols) {
        float sum = 0.0f;
        for(int d = 0; d < kernel_depth; ++d) {
            for (int r = 0; r < kernel_rows; ++r) {
                for (int c = 0; c < kernel_cols; ++c) {
                    int i = (depth + d) * input_rows * input_cols + (row + r) * input_cols + (col + c);
                    int j = d * kernel_rows * kernel_cols + r * kernel_cols + c;
                    sum += input[i] * kernel[j];
                }
            }
        }
        output[depth * output_rows * output_cols + row * output_cols + col] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols) {
    const int output_depth = input_depth - kernel_depth + 1;
    const int output_rows = input_rows - kernel_rows + 1;
    const int output_cols = input_cols - kernel_cols + 1;
    dim3 threads(8, 8, 8);
    dim3 blocks((output_cols + threads.x - 1) / threads.x,
                       (output_rows + threads.y - 1) / threads.y,
                       (output_depth + threads.z - 1) / threads.z);
    conv_3d<<<blocks, threads>>>(input, kernel, output, input_depth, input_rows,
                                         input_cols, kernel_depth, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
