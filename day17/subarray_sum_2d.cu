#include <cuda_runtime.h>

__global__ void subarray_sum(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL, int E_COL) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    extern __shared__ int t[];
    int row = j;
    int col = i;
    t[tid] = (row >= S_ROW && row <= E_ROW && col >= S_COL && col <= E_COL) ? input[row * M + col] : 0;
    __syncthreads();

    for (unsigned int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] += t[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, t[0]);
    }
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL,
                      int E_COL) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (M + threadsPerBlock.x - 1) / threadsPerBlock.x, 
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    int sharedBytes = threadsPerBlock.x * threadsPerBlock.y * sizeof(int);
    subarray_sum<<<blocksPerGrid, threadsPerBlock, sharedBytes>>>(input, output, N, M, S_ROW, E_ROW, S_COL, E_COL);
    cudaDeviceSynchronize();
}

