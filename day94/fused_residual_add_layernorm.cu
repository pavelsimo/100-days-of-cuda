#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
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
        val = (threadIdx.x < numWarps) ? t[lane]: 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void row_mean(const float* __restrict__ input, float* __restrict__ mean, int N, int C) {
    int row = blockIdx.x;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < C; col += blockDim.x) {
        sum += input[row * C + col];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        mean[row] = sum / C;
    }
}

__global__ void row_rstd(const float* __restrict__ input, const float* __restrict__ mean, float* __restrict__ rstd, int N, int C, float eps) {
    int row = blockIdx.x;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < C; col += blockDim.x) {
        float d = input[row * C + col] - mean[row];
        sum += d * d;
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        rstd[row] = 1.0f / sqrtf(sum / C + eps);
    }
}

__global__ void fuse_residual(const float* __restrict__ x, const float* __restrict__ residual, float* __restrict__ residual_out, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) {
        return;
    }

    residual_out[idx] = x[idx] + residual[idx];
}

__global__ void normalize(const float* __restrict__ residual_out, const float* __restrict__ mean, const float* __restrict__ rstd, const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ y, 
                          int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) {
        return;
    }
    int i = idx / C;
    int j = idx % C;
    y[idx] = (residual_out[idx] - mean[i]) * rstd[i] * gamma[j] + beta[j];
}

// x, residual, gamma, beta, y, residual_out, mean, rstd are device pointers
extern "C" void solve(const float* x, const float* residual, const float* gamma, const float* beta,
                      float* y, float* residual_out, float* mean, float* rstd, int N, int C,
                      float eps) {
    int threads = 256;
    int blocks = (N * C + threads - 1) / threads;
    fuse_residual<<<blocks, threads>>>(x, residual, residual_out, N, C);
    row_mean<<<N, threads>>>(residual_out, mean, N, C);
    row_rstd<<<N, threads>>>(residual_out, mean, rstd, N, C, eps);
    normalize<<<blocks, threads>>>(residual_out, mean, rstd, gamma, beta, y, N, C);
    cudaDeviceSynchronize();
}
