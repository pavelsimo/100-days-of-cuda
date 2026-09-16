#include <cuda_runtime.h>

__device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float block_reduce_sum(float val) {
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

__device__ __forceinline__ float clip(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__global__ void ppo_loss(const float* advantages, const float* log_pi, const float* log_pi_old,
                      float* output, float clip_eps, int B, int S) {
    const int N = B * S;
    const int stride = gridDim.x * blockDim.x;
    const float lo = 1.0f - clip_eps;
    const float hi = 1.0f + clip_eps;
    const float inv_N = 1.0f / N;
    float sum = 0.0f;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = idx; i < N; i += stride) {
        float r = expf(log_pi[i] - log_pi_old[i]);
        float r_hat = clip(r, lo, hi);
        float A = advantages[i];
        sum += fminf(r * A, r_hat * A);
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        atomicAdd(output, -sum * inv_N);
    }
}

// advantages, log_pi, log_pi_old, output are device pointers
extern "C" void solve(const float* advantages, const float* log_pi, const float* log_pi_old,
                      float* output, float clip_eps, int B, int S) {
    const int N = B * S;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaMemset(output, 0, sizeof(float));
    ppo_loss<<<blocks, threads>>>(advantages, log_pi, log_pi_old, output, clip_eps, B, S);
    cudaDeviceSynchronize();
}
