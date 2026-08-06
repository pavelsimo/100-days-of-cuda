#include <cuda_runtime.h>

__device__ __forceinline__ int warp_reduce(int val) {
    unsigned mask = 0xFFFFFFFFU;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__device__ __forceinline__ int input_index(int k, int W, int H, int M, int K, int S_DEP, int S_ROW, int S_COL) {
    int plane_size = W * H;
    int depth = S_DEP + k / plane_size;
    int offset_in_plane = k % plane_size;
    int row = S_ROW + offset_in_plane / W;
    int col = S_COL + offset_in_plane % W;

    return (depth * M + row) * K + col;
}

__global__ void subarray_sum(const int* input, int* output, int N, int M, int K, int S_DEP, int E_DEP,
                      int S_ROW, int E_ROW, int S_COL, int E_COL) {
    __shared__ int t[32];
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int sum = 0;
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    int W = E_COL - S_COL + 1;
    int H = E_ROW - S_ROW + 1;
    int D = E_DEP - S_DEP + 1;
    int len = W * H * D;

    int stride = gridDim.x * blockDim.x;
    while (idx + 7 * stride < len) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            int k = idx + u * stride;
            int input_idx = input_index(k, W, H, M, K, S_DEP, S_ROW, S_COL);
            sum += input[input_idx];
        }
        idx += 8 * stride;
    }

    while (idx < len) {
        int input_idx = input_index(idx, W, H, M, K, S_DEP, S_ROW, S_COL);
        sum += input[input_idx];
        idx += stride;
    }

    sum = warp_reduce(sum);
    if (lane == 0) {
        t[warpId] = sum;
    }
    __syncthreads();

    if (warpId == 0) {
        int numWarps = (blockDim.x / warpSize);
        sum = (lane < numWarps) ? t[lane] : 0;
        sum = warp_reduce(sum);
        if (tid == 0) {
            atomicAdd(output, sum);
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K, int S_DEP, int E_DEP,
                      int S_ROW, int E_ROW, int S_COL, int E_COL) {
    int len = (E_DEP - S_DEP + 1) * (E_ROW - S_ROW + 1) * (E_COL - S_COL + 1);
    int threadsPerBlock  = 256;
    int elementsPerBlock = 256 * 8;   // each thread handles 8 elements = 2048 per block
    int blocksPerGrid    = (len + elementsPerBlock - 1) / elementsPerBlock;
    subarray_sum<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K, S_DEP, E_DEP, S_ROW, E_ROW, S_COL, E_COL);
    cudaDeviceSynchronize();
}
