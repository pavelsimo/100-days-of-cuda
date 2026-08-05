#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce(float val) {
    unsigned mask = 0xFFFFFFFFU;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__global__ void dot_product(const half* A, const half* B, half* result, int N) {
    __shared__ float t[32];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    float val = 0;
    unsigned mask = 0xFFFFFFFFU;
    int lane = threadIdx.x & (warpSize - 1);
    int warpId = threadIdx.x >> 5;
    int stride = gridDim.x * blockDim.x;
    while (i + 7 * stride < N) {
        #pragma unroll 
        for (int k = 0; k < 8; ++k) {
            val += __half2float(A[i + k * stride]) * __half2float(B[i + k * stride]);
        }
        i += 8 * stride;
    }

    while (i < N) {
        val += __half2float(A[i]) * __half2float(B[i]);
        i += stride;
    }

    val = warp_reduce(val);
    if (lane == 0) {
        t[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0) {
        val = (tid < blockDim.x / warpSize) ? t[lane]: 0.0f;
        val = warp_reduce(val);

        if (tid == 0) {
            atomicAdd(result, __float2half(val));
        }
    }
}

// A, B, result are device pointers
extern "C" void solve(const half* A, const half* B, half* result, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 1024) blocksPerGrid = 1024;
    dot_product <<<blocksPerGrid, threadsPerBlock>>>(A, B, result, N);
    cudaDeviceSynchronize();
}
