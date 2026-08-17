#include <cuda_runtime.h>

template <int K>
__global__ void causal_depthwise_conv1d(const float* x, const float* weight, const float* bias, float* output, int B,
                      int L, int D) {
    int b = blockDim.z * blockIdx.z + threadIdx.z;
    int l = blockDim.x * blockIdx.x + threadIdx.x;
    int d = blockDim.y * blockIdx.y + threadIdx.y;
    if (b < B && l < L && d < D) {
        float sum = bias[d];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float x_val = l - k >= 0 ? x[b * (L * D) + (l - k) * D + d]: 0.0f;
            sum += weight[d * K + k] * x_val;
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
    switch (K) {
        case 1:
            causal_depthwise_conv1d<1><<<blocks, threads>>>(x, weight, bias, output, B, L, D);
            break;
        case 2:
            causal_depthwise_conv1d<2><<<blocks, threads>>>(x, weight, bias, output, B, L, D);
            break;
        case 3:
            causal_depthwise_conv1d<3><<<blocks, threads>>>(x, weight, bias, output, B, L, D);
            break;
        case 4:
            causal_depthwise_conv1d<4><<<blocks, threads>>>(x, weight, bias, output, B, L, D);
            break;
        case 5:
            causal_depthwise_conv1d<5><<<blocks, threads>>>(x, weight, bias, output, B, L, D);
            break;
        case 6:
            causal_depthwise_conv1d<6><<<blocks, threads>>>(x, weight, bias, output, B, L, D);
            break;
        case 7:
            causal_depthwise_conv1d<7><<<blocks, threads>>>(x, weight, bias, output, B, L, D);
            break;
        case 8:
            causal_depthwise_conv1d<8><<<blocks, threads>>>(x, weight, bias, output, B, L, D);
            break;
        default:
            break;
    }
    cudaDeviceSynchronize();
}