#include <cuda_runtime.h>

__global__ void compute_temporal_diff(const float* r, const float* V, float* D, float gamma, int B, int S) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * S) return;

    int b = idx / S;
    int s = idx % S;
    float v1 = V[b * S + s];
    float v2 = s + 1 < S ? V[b * S + s + 1]: 0;
    float d = r[b * S + s] + gamma * v2 - v1;
    D[b * S + s] = d;
}

__global__ void reverse_gae_scan(const float* D, float* A, float gamma, float lambda, int B, int S) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) return;

    int b = idx;
    float a = 0.0f;
    for (int t = S - 1; t >= 0; --t) {
        float d = D[b * S + t];
        a = d + gamma * lambda * a;
        A[b * S + t] = a;
    }
}

// rewards, values, advantages are device pointers
extern "C" void solve(const float* rewards, const float* values, float* advantages, float gamma,
                      float lam, int B, int S) {
    // rewards - [B, S]
    // values - [B, S]
    // advantages - [B, S]
    float *D;
    cudaMalloc(&D, B * S * sizeof(float));
    int threads = 256;
    int blocks = (B * S + threads - 1) / threads;
    compute_temporal_diff<<<blocks, threads>>>(rewards, values, D, gamma, B, S);
    reverse_gae_scan<<<(B + threads - 1) / threads, threads>>>(D, advantages, gamma, lam, B, S);
    cudaDeviceSynchronize();

    cudaFree(D);
}
