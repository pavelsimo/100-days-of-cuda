#include <cuda_runtime.h>
#include <float.h>

__global__ void init_array(const float* in, float* out, int N, int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        out[i] = (i < N) ? in[i] : -FLT_MAX;
    }
}

__global__ void bitonic_step(float* a, int j, int k, int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int p = i ^ j;
    if (i >= M || p <= i) {
        return;
    }
    float x = a[i];
    float y = a[p];
    bool desc = (i & k) == 0;
    if (desc ? (x < y) : (x > y)) {
        a[i] = y;
        a[p] = x;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    int M = 1;
    while (M < N) M <<= 1;

    float *input_sorted;
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    cudaMalloc(&input_sorted, M * sizeof(float));
    init_array<<<blocks, threads>>>(input, input_sorted, N, M);
    for (int size = 2; size <= M; size <<= 1) {
        for (int j = size >> 1; j > 0; j >>= 1) {
            bitonic_step<<<blocks, threads>>>(input_sorted, j, size, M);
        }
    }
    cudaMemcpy(output, input_sorted, k * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    cudaFree(input_sorted);
}
