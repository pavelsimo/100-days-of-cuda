#include <cuda_runtime.h>

__device__ __forceinline__ float rope_elem(const float* Q, float q, float c, float s, int idx, int D) {
    const int i = idx / D;
    const int j = idx % D;
    const int halfD = D / 2;
    const float r = j < halfD ? -Q[i * D + j + halfD] : Q[i * D + j - halfD];
    return q * c + r * s;
}

__global__ void rope(float* Q, float* cos, float* sin, float* output, int M, int D) {
    float4* Q4 = reinterpret_cast<float4*>(Q);
    float4* cos4 = reinterpret_cast<float4*>(cos);
    float4* sin4 = reinterpret_cast<float4*>(sin);
    float4* output4 = reinterpret_cast<float4*>(output);
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int K = M * D;
    if (idx < K) {
        if (idx * 4 + 3 < K) {
            const float4 q = Q4[idx];
            const float4 c = cos4[idx];
            const float4 s = sin4[idx];
            output4[idx] = make_float4(
                rope_elem(Q, q.x, c.x, s.x, idx * 4, D),
                rope_elem(Q, q.y, c.y, s.y, idx * 4 + 1, D),
                rope_elem(Q, q.z, c.z, s.z, idx * 4 + 2, D),
                rope_elem(Q, q.w, c.w, s.w, idx * 4 + 3, D)
            );
        } else {
            for (int k = idx * 4; k < K; ++k) {
                output[k] = rope_elem(Q, Q[k], cos[k], sin[k], k, D);
            }
        }
    }
}

// Q, cos, sin, output are device pointers
extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {
    constexpr int threads = 256;
    const int K = M * D;
    const int blocks = (K + threads - 1) / threads;
    rope<<<blocks, threads>>>(Q, cos, sin, output, M, D);
    cudaDeviceSynchronize();
}
