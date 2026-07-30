#include <cuda_runtime.h>

__global__ void count_element(const int* input, int* output, int N, int M, int K, int P) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    extern __shared__ int t[];
    t[tid] = (i < N*M*K) ? (input[i] == P) ? 1: 0: 0;
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

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int M, int K, int P) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * M * K + threadsPerBlock - 1) / threadsPerBlock;
    int sharedBytes = threadsPerBlock * sizeof(int);
    count_element<<<blocksPerGrid, threadsPerBlock, sharedBytes>>>(input, output, N, M, K, P);
    cudaDeviceSynchronize();
}