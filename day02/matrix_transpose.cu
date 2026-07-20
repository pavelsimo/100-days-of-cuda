#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define ROWS 4
#define COLS 6

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols)
    {
        output[j * rows + i] = input[i * cols + j];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

static void print_matrix(const char *title, const float *m, int rows, int cols)
{
    printf("%s (%dx%d):\n", title, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.1f ", m[i * cols + j]);
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

    /* Step 1: Allocate host memory */
    h_input = (float *)malloc(sz);
    h_output = (float *)malloc(sz);

    /* Step 2: Allocate device memory */
    cudaMalloc((void **)&d_input, sz);
    cudaMalloc((void **)&d_output, sz);

    /* Step 3: Initialize host input matrix */
    for (int i = 0; i < ROWS * COLS; i++) {
        h_input[i] = (float)i;
    }

    /* Step 4: Copy input matrix to device */
    cudaMemcpy(d_input, h_input, sz, cudaMemcpyHostToDevice);

    /* Step 5: Launch kernel via solve() */
    solve(d_input, d_output, ROWS, COLS);

    /* Step 6: Copy result back to host */
    cudaMemcpy(h_output, d_output, sz, cudaMemcpyDeviceToHost);

    /* Step 7: Print result */
    print_matrix("Input", h_input, ROWS, COLS);
    print_matrix("Transposed", h_output, COLS, ROWS);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
