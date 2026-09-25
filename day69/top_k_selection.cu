#include <cuda_runtime.h>

__global__ void rank_sort(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    
    float v = in[i];
    int pos = 0;
    for (int j = 0; j < N; ++j) {
        float w = in[j];
        pos += (w > v) || (w == v && j > i);
    }
    out[pos] = v;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    float *input_sorted;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaMalloc(&input_sorted, N * sizeof(float));
    rank_sort<<<blocks, threads>>>(input, input_sorted, N);
    cudaMemcpy(output, input_sorted, k * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    cudaFree(input_sorted);
}
