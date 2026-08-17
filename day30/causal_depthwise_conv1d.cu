#include <cuda_runtime.h>

__global__ void causal_depthwise_conv1d(const float* x, const float* weight, const float* bias, float* output, int B,
                      int L, int D, int K) {
    int b = blockDim.z * blockIdx.z + threadIdx.z;
    int l = blockDim.x * blockIdx.x + threadIdx.x;
    int d = blockDim.y * blockIdx.y + threadIdx.y;
    if (b < B && l < L && d < D) {
        float sum = bias[d];
        for (int k = 0; k < K && l - k >= 0; ++k) {
            sum += weight[d * K + k] * x[b * (L * D) + (l - k) * D + d];
        }
        output[b * (L * D) + l * D + d] = sum;
    }
}

// x, weight, bias, output are device pointers
extern "C" void solve(const float* x, const float* weight, const float* bias, float* output, int B,
                      int L, int D, int K) {
    dim3 threads(8, 8, 8);
    dim3 blocks (
        (L + threads.x - 1) / threads.x,
        (D + threads.y - 1) / threads.y,
        (B + threads.z - 1) / threads.z
    );
    causal_depthwise_conv1d<<<blocks, threads>>>(x, weight, bias, output, B, L, D, K);
    cudaDeviceSynchronize();
}

