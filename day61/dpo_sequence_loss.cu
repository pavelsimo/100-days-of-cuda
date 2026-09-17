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

__device__ __forceinline__ float softplus(float x) {
    // logf(1.0f + expf(z));
    return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
}

__global__ void dpo_loss(const float* chosen_logps, const float* rejected_logps,
                         const float* chosen_ref_logps, const float* rejected_ref_logps, float* output,
                         float beta, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    const int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < B; i += stride) {
        float d1 = chosen_logps[i] - rejected_logps[i];
        float d2 = chosen_ref_logps[i] - rejected_ref_logps[i];
        float z = beta * (d1 - d2);
        sum += softplus(-z);
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        atomicAdd(output, sum / B);
    }
}

// chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps, output are device pointers
extern "C" void solve(const float* chosen_logps, const float* rejected_logps,
                      const float* chosen_ref_logps, const float* rejected_ref_logps, float* output,
                      float beta, int B) {

    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    cudaMemset(output, 0, sizeof(float));
    dpo_loss<<<blocks, threads>>>(chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps, output, beta, B);
    cudaDeviceSynchronize();
}
