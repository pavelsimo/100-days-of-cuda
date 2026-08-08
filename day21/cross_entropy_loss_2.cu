#include <cuda_runtime.h>
#include <math_constants.h>

__device__ __forceinline__ float warp_reduce_max(float value) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void cross_entropy(const float* logits, const int* true_labels, float* loss, int N, int C) {
    const int idx = blockIdx.x;
    const int lane = threadIdx.x;
    const int offset = idx * C;
    const int stride = warpSize;

    float max_logit = -CUDART_INF_F;
    for (int c = lane; c < C; c += stride) {
        max_logit = fmaxf(max_logit, logits[offset + c]);
    }
    max_logit = warp_reduce_max(max_logit);
    max_logit = __shfl_sync(0xffffffffu, max_logit, 0);

    float sum_logit = 0.0f;
    for (int c = lane; c < C; c += stride) {
        sum_logit += __expf(logits[offset + c] - max_logit);
    }
    sum_logit = warp_reduce_sum(sum_logit);

    if (lane == 0) {
        const int y = true_labels[idx];
        const float loss_value = max_logit + __logf(sum_logit) - logits[offset + y];
        atomicAdd(loss, loss_value / static_cast<float>(N));
    }
}

extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    cudaMemset(loss, 0, sizeof(float));
    cross_entropy<<<N, 32>>>(logits, true_labels, loss, N, C);
}