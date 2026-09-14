#include <cuda_runtime.h>
#include <limits.h>

__global__ void max_subarray_sum(const int* input, int* output, int N, int window_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && N - i >= window_size) {
        int sum = 0;
        for (int j = 0; j < window_size; ++j) {
            sum += input[i + j];
        }
        atomicMax(output, sum);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int window_size) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int min_int = INT_MIN;
    cudaMemcpy(output, &min_int, sizeof(int), cudaMemcpyHostToDevice);
    max_subarray_sum<<<blocks, threads>>>(input, output, N, window_size);
    cudaDeviceSynchronize();
}
