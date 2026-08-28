
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

__global__ void row_mean(const float* input, float* mean, int N, int C) {
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

__global__ void row_var(const float* input, const float* mean, float* var, int N, int C) {
    int row = blockIdx.x;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < C; col += blockDim.x) {
        float d = input[row * C + col] - mean[row];
        sum += d * d;
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        var[row] = sum / C;
    }
}

__global__ void layer_norm(const float* input, const float* weight, const float* bias, float* output,
                      const float* mean, const float* var, int N, int C, float eps) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < N && j < C) {
        output[i * C + j] = weight[j] * (input[i * C + j] - mean[i]) / sqrtf(var[i] + eps) + bias[j];
    } 
}

// input, weight, bias, output are device pointers
extern "C" void solve(const float* input, const float* weight, const float* bias, float* output,
                      int N, int C, float eps) {
    float *mean, *var;
    cudaMalloc(&mean, N * sizeof(float));
    cudaMalloc(&var, N * sizeof(float));

    dim3 threads(16, 16);
    dim3 grid(
        (C + threads.x - 1) / threads.x,
        (N + threads.y - 1) / threads.y
    );

    int nthreads = threads.x * threads.y;
    row_mean<<<N, nthreads>>>(input, mean, N, C);
    row_var<<<N, nthreads>>>(input, mean, var, N, C);
    layer_norm<<<grid, threads>>>(input, weight, bias, output, mean, var, N, C, eps);

    cudaFree(mean);
    cudaFree(var);
}


