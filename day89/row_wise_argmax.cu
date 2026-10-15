#include <cuda_runtime.h>
#include <float.h>

__device__ __forceinline__ void warp_reduce_max(float &val, int &val_idx) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, val_idx, offset);
        if (other_val > val) {
            val = other_val;
            val_idx = other_idx;
        }
        else if (other_val == val && other_idx < val_idx) {
            val_idx = other_idx;
        }
    }
}

__device__ __forceinline__ void block_reduce_max(float &val, int &val_idx) {
    __shared__ float t[32];
    __shared__ int t_idx[32];
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    warp_reduce_max(val, val_idx);
    if (lane == 0) {
        t[warpId] = val;
        t_idx[warpId] = val_idx;
    }
    __syncthreads();
    if (warpId == 0) {
        int numWarps = (blockDim.x + warpSize - 1) / warpSize;
        val = (threadIdx.x < numWarps) ? t[lane]: -FLT_MAX;
        val_idx = (threadIdx.x < numWarps) ? t_idx[lane] : -1;
        warp_reduce_max(val, val_idx);
    }
}

__global__ void argmax(const float* logits, int* output, int B, int V) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    float max_val = -FLT_MAX;
    int max_idx = -1;
    for (int i = tid; i < V; i += blockDim.x) {
        if (logits[bid * V + i] > max_val || max_idx == -1) {
            max_val = logits[bid * V + i];
            max_idx = i;
        }
    }
    block_reduce_max(max_val, max_idx);
    if (tid == 0) {
        output[bid] = max_idx;
    }
}

// logits, output are device pointers
extern "C" void solve(const float* logits, int* output, int B, int V) {
    int threads = 256;
    int blocks = B;
    argmax<<<blocks, threads>>>(logits, output, B, V);
    cudaDeviceSynchronize();
}
