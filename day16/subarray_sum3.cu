#include <cuda_runtime.h>

__global__ void subarray_sum(const int* input, int* output, int N, int S, int E) {
    __shared__ int t[32];
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int val = 0;
    unsigned mask = 0xFFFFFFFFU;
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    const int* range = input + S;
    int len = E - S + 1;

    int stride = gridDim.x * blockDim.x;
    while (idx + 7 * stride < len) {
        val += range[idx];
        val += range[idx + 1 * stride];
        val += range[idx + 2 * stride];
        val += range[idx + 3 * stride];
        val += range[idx + 4 * stride];
        val += range[idx + 5 * stride];
        val += range[idx + 6 * stride];
        val += range[idx + 7 * stride];
        idx += 8 * stride;
    }

    while (idx < len) {
        val += range[idx];
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
extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    int len = E - S + 1;
    int threadsPerBlock  = 256;
    int elementsPerBlock = 256 * 8;   // each thread handles 8 elements = 2048 per block
    int blocksPerGrid    = (len + elementsPerBlock - 1) / elementsPerBlock;
    subarray_sum<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, S, E);
    cudaDeviceSynchronize();
}
