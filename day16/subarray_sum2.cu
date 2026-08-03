#include <cuda_runtime.h>

__global__ void subarray_sum(const int* input, int* output, int N, int S, int E) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    extern __shared__ int t[];
    t[tid] = (i >= S && i <= E) ? input[i] : 0;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
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
extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int sharedBytes = threadsPerBlock * sizeof(int);
    subarray_sum<<<blocksPerGrid, threadsPerBlock, sharedBytes>>>(input, output, N, S, E);
    cudaDeviceSynchronize();
}
