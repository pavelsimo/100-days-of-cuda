#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define ROWS 5
#define COLS 5

__global__ void jacobi_stencil_kernel(const float* input, float* output, int rows, int cols) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= rows || j >= cols) {
        return;
    }

    if (i < rows - 1 && j < cols - 1 && i >= 1 && j >= 1) {
        int t = (i - 1) * cols + j;
        int l = i * cols + j - 1;
        int r = i * cols + j + 1;
        int b = (i + 1) * cols + j;
        output[i * cols + j] = 0.25f * (input[t] + input[l] + input[r] + input[b]);
    } else {
        output[i * cols + j] = input[i * cols + j];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1 ) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1 ) / threadsPerBlock.y);
    jacobi_stencil_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
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
    float *d_input, *d_output;
    int sz = ROWS * COLS * sizeof(float);

    h_input = (float *)malloc(sz);
    h_output = (float *)malloc(sz);

    cudaMalloc((void **)&d_input, sz);
    cudaMalloc((void **)&d_output, sz);
    
    for (int i = 0; i < ROWS * COLS; i++) {
        h_input[i] = 0.0f;
    }
    h_input[(ROWS / 2) * COLS + COLS / 2] = 100.0f;

    cudaMemcpy(d_input, h_input, sz, cudaMemcpyHostToDevice);

    print_matrix("Input", h_input, ROWS, COLS);

    solve(d_input, d_output, ROWS, COLS);

    cudaMemcpy(h_output, d_output, sz, cudaMemcpyDeviceToHost);

    print_matrix("Output", h_output, ROWS, COLS);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
