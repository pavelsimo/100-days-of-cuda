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

__global__ void sum_row_dgamma(const float* __restrict__ dy, const float* __restrict__ x_hat, float* __restrict__ output, int N, int C) {
    int j = blockIdx.x;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum += dy[i * C + j] * x_hat[i * C + j];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        output[j] = sum;
    }
}

__global__ void sum_row_dbeta(const float* __restrict__ dy, float* __restrict__ output, int N, int C) {
    int j = blockIdx.x;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum += dy[i * C + j];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        output[j] = sum;
    }
}

__global__ void mean_col_g(const float* __restrict__ g, const float* __restrict__ x_hat, float* __restrict__ output, int N, int C) {
    int i = blockIdx.x;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        sum += g[i * C + j];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        output[i] = sum / C;
    }
}

__global__ void mean_col_g_xhat(const float* __restrict__ g, const float* __restrict__ x_hat, float* __restrict__ output, int N, int C) {
    int i = blockIdx.x;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        sum += g[i * C + j] * x_hat[i * C + j];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        output[i] = sum / C;
    }
}

__global__ void compute_xhat(const float* __restrict__ x, const float* __restrict__ mean, const float* __restrict__ rstd, float* __restrict__ x_hat, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) {
        return;
    }

    int i = idx / C;
    x_hat[idx] = (x[idx] - mean[i]) * rstd[i];
}

__global__ void compute_g(const float* __restrict__ dy, const float* __restrict__ gamma, float* __restrict__ g, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) {
        return;
    }

    int j = idx % C;
    g[idx] = dy[idx] * gamma[j];
}

__global__ void compute_dx(const float* __restrict__ dy, const float* __restrict__ x_hat, const float* __restrict__ rstd, const float* __restrict__ gamma, const float* __restrict__ g, const float* __restrict__ mean_g, const float* __restrict__ mean_g_xhat, float* __restrict__ dx, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) {
        return;
    }

    int i = idx / C;
    int j = idx % C;
    dx[idx] = rstd[i] * (g[idx] - mean_g[i] - x_hat[idx] * mean_g_xhat[i]);
}

// x, gamma, mean, rstd, dy, dx, dgamma, dbeta are device pointers
extern "C" void solve(const float* x, const float* gamma, const float* mean, const float* rstd,
                      const float* dy, float* dx, float* dgamma, float* dbeta, int N, int C) {

    float *r, *x_hat, *g, *mean_g, *mean_g_xhat;
    cudaMalloc(&r, N * sizeof(float));
    cudaMalloc(&x_hat, N * C * sizeof(float));
    cudaMalloc(&g, N * C * sizeof(float));
    cudaMalloc(&mean_g, N * sizeof(float));
    cudaMalloc(&mean_g_xhat, N * sizeof(float));
    
    int threads = 256;
    int blocks = (N * C + threads - 1) / threads; 
    compute_xhat<<<blocks, threads>>>(x, mean, rstd, x_hat, N, C);
    sum_row_dgamma<<<C, threads>>>(dy, x_hat, dgamma, N, C);
    sum_row_dbeta<<<C, threads>>>(dy, dbeta, N, C);
    compute_g<<<blocks, threads>>>(dy, gamma, g, N, C);
    mean_col_g<<<N, threads>>>(g, x_hat, mean_g, N, C);
    mean_col_g_xhat<<<N, threads>>>(g, x_hat, mean_g_xhat, N, C);
    compute_dx<<<blocks, threads>>>(dy, x_hat, rstd, gamma, g, mean_g, mean_g_xhat, dx, N, C);
    cudaDeviceSynchronize();
    
    cudaFree(r);
    cudaFree(x_hat);
    cudaFree(g);
    cudaFree(mean_g);
    cudaFree(mean_g_xhat);
}
