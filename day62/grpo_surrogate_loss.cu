#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float t[32];
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    val = warp_reduce_sum(val);
    if (lane == 0) {
        t[warpId] = val;
    }
    __syncthreads();
    if (warpId == 0) {
        int numWarps = (blockDim.x + warpSize - 1) / warpSize;
        val = (threadIdx.x < numWarps) ? t[lane] : 0.0f;
        val = warp_reduce_sum(val); 
    }
    return val;
}

__global__ void b_mean(const float* rewards, float* mean, int B, int G) {
    float sum = 0.0f;
    int b = blockIdx.x;
    for (int g = threadIdx.x; g < G; g += blockDim.x) {
        sum += rewards[b * G + g];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        mean[b] = sum / G;
    }
}

__global__ void b_var(const float* rewards, const float* mean, float* var, int B, int G) {
    float sum = 0.0f;
    int b = blockIdx.x;
    for (int g = threadIdx.x; g < G; g += blockDim.x) {
        float diff = rewards[b * G + g] - mean[b];
        sum += diff * diff;
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        var[b] = sum / G;
    }
}

__device__ __forceinline__ float clip(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__global__ void grpo_loss(const float* rewards, const float* log_pi, const float* log_pi_old,
                      const float* log_ref, float* mean, float *var, float* output, 
                      float clip_eps, float beta, int B, int G, int S) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = B * G * S;
    const float eps = 1e-8;
    float sum = 0.0f;
    for (int bgs = idx; bgs < B * G * S; bgs += gridDim.x * blockDim.x) {
        int b = bgs / (G * S);
        int g = (bgs / S) % G;
        int s = bgs % S;
        int bg = bgs / S;
        float A = (rewards[bg] - mean[b]) / (sqrtf(var[b]) + eps);
        float r = expf(log_pi[bgs] - log_pi_old[bgs]);
        float r_hat = clip(r, 1 - clip_eps, 1 + clip_eps);
        float L = min(r * A, r_hat * A);
        float d = log_ref[bgs] - log_pi[bgs];
        float K = expf(d) - d - 1.0f;
        sum += (L - beta * K);
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        atomicAdd(output, -sum / total);
    }
}

// rewards, log_pi, log_pi_old, log_ref, output are device pointers
extern "C" void solve(const float* rewards, const float* log_pi, const float* log_pi_old,
                      const float* log_ref, float* output, float clip_eps, float beta, int B, int G,
                      int S) {
        // rewards - [B, G]
        // log_pi - [B, G, S]
        // log_pi_old - [B, G, S]
        // log_ref - [B, G, S]
        // output - [B, G, S]
        // mean - [B]
        // var - [B]
        // A - [B, G]
        
        float *mean, *var;
        cudaMalloc(&mean, B * sizeof(float));
        cudaMalloc(&var, B * sizeof(float));
        cudaMemset(output, 0, sizeof(float));
        cudaMemset(mean, 0, B * sizeof(float));
        cudaMemset(var, 0, B * sizeof(float));
        int threads = 256;
        b_mean<<<B, threads>>>(rewards, mean, B, G);
        b_var<<<B, threads>>>(rewards, mean, var, B, G);
        grpo_loss<<<(B * G * S + threads - 1) / threads, threads>>>(rewards, log_pi, log_pi_old, log_ref, mean, var, output, clip_eps, beta, B, G, S);
        cudaDeviceSynchronize();
        cudaFree(mean);
        cudaFree(var);
}
