#include <cuda_runtime.h>
#include <float.h>

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float t[32];
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    val = warp_reduce_max(val);
    if (lane == 0) {
        t[warpId] = val;
    }
    __syncthreads();
    if (warpId == 0) {
        int numWarps = (blockDim.x + warpSize - 1) / warpSize;
        val = (threadIdx.x < numWarps) ? t[lane]: -FLT_MAX;
        val = warp_reduce_max(val);
    }
    return val;
}

__global__ void col_max(const float* __restrict__ input, float* __restrict__ mh, int num_splits, int num_heads) {
    int col = blockIdx.x;
    float max_value = -FLT_MAX;
    for (int row = threadIdx.x; row < num_splits; row += blockDim.x) {
        max_value = fmaxf(max_value, input[row * num_heads + col]);
    }
    max_value = block_reduce_max(max_value);
    if (threadIdx.x == 0) {
        mh[col] = max_value;
    }
}

__global__ void compute_weights(const float* __restrict__ partial_lse, const float* __restrict__ mh, float* __restrict__ w, int num_splits, int num_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_heads * num_splits) {
        return;
    }
    const int h = idx / num_splits;
    const int s = idx % num_splits;
    const int sh = s * num_heads + h;
    w[sh] = expf(partial_lse[sh] - mh[h]);
}

__global__ void reduce(const float* __restrict__ partial_out, const float* __restrict__ w, float* __restrict__ output, int num_splits, int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_heads * head_dim) {
        return;
    }
    const int h = idx / head_dim;
    const int d = idx % head_dim;
    float w_sum = 0.0f;
    float sum = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        const int sh = s * num_heads + h;
        const int shd = s * (num_heads * head_dim) + h * (head_dim) + d;
        const float wsh = w[sh];
        w_sum += wsh * partial_out[shd];
        sum += wsh;
    }
    output[idx] = w_sum / sum;
}

// partial_out, partial_lse, output are device pointers
extern "C" void solve(const float* partial_out, const float* partial_lse, float* output,
                      int num_splits, int num_heads, int head_dim) {
    
    // partial_out - [num_splits, num_heads, head_dim]
    // partial_lse - [num_splits, num_heads]
    // output - [num_heads, head_dim]
    int threads = 256;
    float *mh;
    float *w;
    cudaMalloc(&mh, num_heads * sizeof(float));
    cudaMalloc(&w, num_heads * num_splits * sizeof(float));
    col_max<<<num_heads, threads>>>(partial_lse, mh, num_splits, num_heads);
    int blocks = (num_splits * num_heads + threads - 1) / threads;
    compute_weights<<<blocks, threads>>>(partial_lse, mh, w, num_splits, num_heads);
    blocks = (num_heads * head_dim + threads - 1) / threads;
    reduce<<<blocks, threads>>>(partial_out, w, output, num_splits, num_heads, head_dim);
    cudaFree(mh);
    cudaFree(w);
    cudaDeviceSynchronize();
}