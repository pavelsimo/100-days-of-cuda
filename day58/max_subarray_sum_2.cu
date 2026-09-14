#include <cuda_runtime.h>
#include <limits.h>

__global__ void prefix_sum(const int* input, int* prefix_sums, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        prefix_sums[0] = input[0];
        for (int i = 1; i < N; ++i) {
            prefix_sums[i] = prefix_sums[i - 1] + input[i];
        }
    }
}

__global__ void max_subarray_sum(const int* input, const int* prefix_sums, int* output, int N, int window_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && N - i >= window_size) {
        int sum = prefix_sums[i + window_size - 1];
        if (i > 0) {
            sum -= prefix_sums[i - 1];
        }
        atomicMax(output, sum);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int window_size) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int min_int = INT_MIN;
    int *output_sums;
    cudaMemcpy(output, &min_int, sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&output_sums, N * sizeof(int));
    prefix_sum<<<1, 1>>>(input, output_sums, N);
    max_subarray_sum<<<blocks, threads>>>(input, output_sums, output, N, window_size);
    cudaDeviceSynchronize();
    cudaFree(output_sums);
}
