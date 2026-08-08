#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce(float val) {
    unsigned mask = 0xFFFFFFFFU;
    float res = val;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        res += __shfl_down_sync(mask, res, offset);
    }
    return res;
}

__global__ void reduce_loss(float* loss_values, float* loss, int N) {
    __shared__ float sums[32];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    float sum = 0;
    int lane = threadIdx.x & (warpSize - 1);
    int warpId = threadIdx.x >> 5;
    int stride = gridDim.x * blockDim.x;

    while (i < N - 7 * stride) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            sum += loss_values[i + k * stride];
        }
        i += 8 * stride;
    }

    while (i < N) {
        sum += loss_values[i];
        i += stride;
    }

    sum = warp_reduce(sum);
    if (lane == 0) {
        sums[warpId] = sum;
    }
    __syncthreads();

    if (warpId == 0) {
        int numWarps = (blockDim.x / warpSize);
        sum = (lane < numWarps) ? sums[lane] : 0.0f;
        sum = warp_reduce(sum);
        if (tid == 0) {
            atomicAdd(loss, sum);
        }
    }
}

__global__ void cross_entropy_loss_kernel(const float* logits, const int* true_labels, float* loss, int N, int C) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float total_loss = 0.0f;
    if (j < N) {
        float sum = 0.0f;
        for (int k = 0; k < C; ++k) {
            float z_jk = logits[j * C + k];
            sum += expf(z_jk);
        }
        float true_label = logits[j * C + true_labels[j]];
        float loss_val = (logf(sum) - true_label);
        loss[j] = loss_val;
    }
}

__global__ void finalize_loss(float* loss, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *loss /= N;
    }
}

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, 
    int N, int C) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    float *loss_values;
    cudaMalloc(&loss_values, N * sizeof(float));
    cross_entropy_loss_kernel<<<blocks, threads>>>(logits, true_labels, loss_values, N, C);
    reduce_loss<<<blocks, threads>>>(loss_values, loss, N);
    finalize_loss<<<1, 1>>>(loss, N);
    cudaDeviceSynchronize();
    cudaFree(loss_values);
}
