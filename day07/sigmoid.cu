#include <cuda_runtime.h>
#include <math.h>

__forceinline__ __device__ float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

__global__ void sigmoid_kernel(const float* X, float* Y, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        Y[idx] = sigmoid(X[idx]);
    }
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
