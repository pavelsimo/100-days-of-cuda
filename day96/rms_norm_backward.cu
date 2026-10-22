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

__global__ void row_mean(const float* __restrict__ g, const float* __restrict__ x_hat, float* __restrict__ output, int N, int C) {
    int row = blockIdx.x;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < C; col += blockDim.x) {
        sum += g[row * C + col] * x_hat[row * C + col];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        output[row] = sum / C;
    }
}

__global__ void col_sum(const float* __restrict__ dy, const float* __restrict__ x_hat, float* __restrict__ output, int N, int C) {
    int col = blockIdx.x;
    float sum = 0.0f;
    for (int row = threadIdx.x; row < N; row += blockDim.x) {
        sum += dy[row * C + col] * x_hat[row * C + col];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        output[col] = sum;
    }
}

__global__ void row_rstd(const float* __restrict__ input, float* __restrict__ output, int N, int C, float eps) {
    int row = blockIdx.x;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < C; col += blockDim.x) {
        float d = input[row * C + col];
        sum += d * d;
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        output[row] = 1.0f / sqrtf(sum / C + eps);
    }
}

__global__ void normalize(const float* x, const float *w, const float *r, const float *dy, float *x_hat, float *g, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) {
        return;
    }
    int i = idx / C;
    int j = idx % C;
    x_hat[idx] = x[i * C + j] * r[i];
    g[idx] = dy[idx] * w[j];
}

__global__ void compute_dx(const float* __restrict__ g, const float* __restrict__ x_hat, const float* __restrict__ r, const float* __restrict__ mean, float* __restrict__ dx, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) {
        return;
    }
    int i = idx / C;
    int j = idx % C;
    dx[idx] = r[i] * (g[idx] - x_hat[idx] * mean[i]);
}

// x, weight, dy, dx, dweight are device pointers
extern "C" void solve(const float* x, const float* weight, const float* dy, float* dx,
                      float* dweight, int N, int C, float eps) {

    float *r, *x_hat, *g, *mean;
    cudaMalloc(&r, N * sizeof(float));
    cudaMalloc(&x_hat, N * C * sizeof(float));
    cudaMalloc(&g, N * C * sizeof(float));
    cudaMalloc(&mean, N * sizeof(float));
    
    int threads = 256;
    int blocks = (N * C + threads - 1) / threads; 
    row_rstd<<<N, threads>>>(x, r, N, C, eps);
    normalize<<<blocks, threads>>>(x, weight, r, dy, x_hat, g, N, C);
    row_mean<<<N, threads>>>(g, x_hat, mean, N, C);
    compute_dx<<<blocks, threads>>>(g, x_hat, r, mean, dx, N, C);
    col_sum<<<C, threads>>>(dy, x_hat, dweight, N, C);
    
    cudaFree(r);
    cudaFree(x_hat);
    cudaFree(g);
    cudaFree(mean);
}
