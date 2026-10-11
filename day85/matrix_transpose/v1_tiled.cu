#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {

    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int k = 0; k < TILE_DIM; k += BLOCK_ROWS) {
        if (x < cols && y + k < rows) {
            tile[threadIdx.y + k][threadIdx.x] = input[(y + k) * cols + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int k = 0; k < TILE_DIM; k += BLOCK_ROWS) {
        if (x < rows && y + k < cols) {
            output[(y + k) * rows + x] = tile[threadIdx.x][threadIdx.y + k];
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threads(TILE_DIM, BLOCK_ROWS);
    dim3 blocks(
        (cols + TILE_DIM - 1) / TILE_DIM,
        (rows + TILE_DIM - 1) / TILE_DIM
    );
    matrix_transpose_kernel<<<blocks, threads>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
