#include <cuda_runtime.h>

__global__ void subarray_sum(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL,
                      int E_COL) {
    __shared__ int t[32];
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int val = 0;
    unsigned mask = 0xFFFFFFFFU;
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    int W = E_COL - S_COL + 1;
    int H = E_ROW - S_ROW + 1;
    int len = W * H;

    int stride = gridDim.x * blockDim.x;
    while (idx + 7 * stride < len) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            int k = idx + u * stride;
            val += input[(S_ROW + k / W) * M + (S_COL + k % W)];
        }
        idx += 8 * stride;
    }

    while (idx < len) {
        val += input[(S_ROW + idx / W) * M + (S_COL + idx % W)];
        idx += stride;
    }

    // 1st warp-shuffle reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }

    if (lane == 0) {
        t[warpId] = val;
    }
    __syncthreads(); // put warp results into shared memory

    if (warpId == 0) {
        val = (tid < blockDim.x / warpSize) ? t[lane] : 0;
        // 2nd warp-shuffle reduction
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }

        if (tid == 0) {
            atomicAdd(output, val);
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL,
                      int E_COL) {
    int len = (E_ROW - S_ROW + 1) * (E_COL - S_COL + 1);
    int threadsPerBlock  = 256;
    int elementsPerBlock = 256 * 8;   // each thread handles 8 elements = 2048 per block
    int blocksPerGrid    = (len + elementsPerBlock - 1) / elementsPerBlock;
    subarray_sum<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, S_ROW, E_ROW, S_COL, E_COL);
    cudaDeviceSynchronize();
}
