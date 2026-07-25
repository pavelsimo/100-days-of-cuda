#include <cuda_runtime.h>
#include <math.h>

__forceinline__ __device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void sigmoid_kernel(const float* X, float* Y, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const float4* x = reinterpret_cast<const float4*>(X);
    float4* y = reinterpret_cast<float4*>(Y);
    int base = idx * 4;
    if (base + 3 < N) {
        float4 t = x[idx];
        float4 res;
        res.x = sigmoid(t.x);
        res.y = sigmoid(t.y);
        res.z = sigmoid(t.z);
        res.w = sigmoid(t.w);
        y[idx] = res;
    } else {
        for (int k = base; k < N; ++k) {
            Y[k] = sigmoid(X[k]);
        }
    }
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
