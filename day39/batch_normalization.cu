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

__global__ void column_mean(const float * input, float *output, int N, int C, float alpha) {
    int col = blockIdx.x;
    float sum = 0.0f;
    for (int row = threadIdx.x; row < N; row += blockDim.x) {
        sum += input[col + row * C];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        output[col] = sum * alpha;
    }
}

__global__ void column_var(const float * input, const float* mean, float *output, int N, int C, float alpha) {
    int col = blockIdx.x;
    float sum = 0.0f;
    for (int row = threadIdx.x; row < N; row += blockDim.x) {
        float x = input[col + row * C];
        float m = mean[col];
        float d = x - m;
        sum += d * d;
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        output[col] = sum * alpha;
    }
}

__global__ void batch_norm(const float* input, const float* gamma, const float* beta, const float *mean, const float *var, float* output,
                      int N, int C, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C) {
        int j = idx % C;
        float a = input[idx] - mean[j];
        float b = sqrtf(var[j] + eps);
        float xp = a / b;
        output[idx] = gamma[j] * xp + beta[j];
    }
}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output,
                      int N, int C, float eps) {
    int threads = 256;
    int blocks = (N * C + threads - 1) / threads;
    float *mean, *var, *x;
    const float alpha = 1.0f / N;
    cudaMalloc(&mean, C * sizeof(float));
    cudaMemset(mean, 0, C * sizeof(float));
    cudaMalloc(&var, C * sizeof(float));
    cudaMemset(var, 0, C * sizeof(float));

    column_mean<<<C, threads>>>(input, mean, N, C, alpha);
    column_var<<<C, threads>>>(input, mean, var, N, C, alpha);
    batch_norm<<<blocks, threads>>>(input, gamma, beta, mean, var, output, N, C, eps);
    
    cudaFree(mean);
    cudaFree(var);
}
