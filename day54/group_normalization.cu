#include <cuda_runtime.h>


__device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
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
        val = (threadIdx.x < numWarps) ? t[lane]: 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void group_mean(const float* X, float* mean, int group_size) {
    float sum = 0.0f;
    int ng = blockIdx.x;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        sum += X[ng * group_size + i];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        mean[ng] = sum / group_size;
    }
}

__global__ void group_var(const float* X, const float* mean, float *var, int group_size) {
    float sum = 0.0f;
    int ng = blockIdx.x;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float d = X[ng * group_size + i] - mean[ng];
        sum += d * d;
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        var[ng] = sum / group_size;
    }
}

__global__ void group_norm(const float* X, const float* gamma, const float* mean, const float *var, const float* beta, float* Y, int total, int C,
                      int H, int W, int G, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
         int c = (idx / (H * W)) % C;
         int g = c / (C / G);
         int n = idx / (C * H * W);
         int ng = n * G + g;
         float x = X[idx];
         float x_hat = (x - mean[ng]) / sqrtf(var[ng] + eps);
         float output = gamma[c] * x_hat + beta[c];
         Y[idx] = output;
    }
}

// X, gamma, beta, Y are device pointers
extern "C" void solve(const float* X, const float* gamma, const float* beta, float* Y, int N, int C,
                      int H, int W, int G, float eps) {
    float *mean, *var;
    int threads = 256;
    int group_size = (C / G) * H * W;
    int total = N * C * H * W;
    cudaMalloc(&mean, N * G * sizeof(float));
    cudaMalloc(&var, N * G * sizeof(float));
    group_mean<<<N * G, threads>>>(X, mean, group_size);
    group_var<<<N * G, threads>>>(X, mean, var, group_size);
    int blocks = (total + threads - 1) / threads;
    group_norm<<<blocks, threads>>>(X, gamma, mean, var, beta, Y, total, C, H, W, G, eps);
    cudaFree(mean);
    cudaFree(var);
    cudaDeviceSynchronize();
}
