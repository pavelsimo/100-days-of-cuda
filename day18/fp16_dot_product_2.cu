#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce(float val) {
    unsigned mask = 0xFFFFFFFFU;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__global__ void dot_product(const half* A, const half* B, float* result, int N) {
    __shared__ float t[32];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    unsigned mask = 0xFFFFFFFFU;
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);

    int stride = gridDim.x * blockDim.x;
    int N8 = N / 8;
    for (int v = i; v < N8; v += stride) {
        float4 a = A4[v];
        float4 b = B4[v];
        const half2* ah = reinterpret_cast<const half2*>(&a);
        const half2* bh = reinterpret_cast<const half2*>(&b);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 af = __half22float2(ah[k]);
            float2 bf = __half22float2(bh[k]);
            sum += af.x * bf.x + af.y * bf.y;
        }
    }

    for (int j = N8 * 8 + i; j < N; j += stride) {
        sum += __half2float(A[j]) * __half2float(B[j]);
    }

    sum = warp_reduce(sum);
    if (lane == 0) {
        t[warpId] = sum;
    }
    __syncthreads();

    if (warpId == 0) {
        int numWarps = (blockDim.x / warpSize);
        sum = (lane < numWarps) ? t[lane] : 0.0f;
        sum = warp_reduce(sum); 
        if (tid == 0) {
            atomicAdd(result, sum);
        }
    }
}

__global__ void set_value(const float* value, half* result) {
    *result = __float2half(*value);
}

// A, B, result are device pointers
extern "C" void solve(const half* A, const half* B, half* result, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N / 8 + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 2048) blocksPerGrid = 2048;
    if (blocksPerGrid < 1) blocksPerGrid = 1;
    float* sum;
    cudaMalloc(&sum, sizeof(float));
    cudaMemset(sum, 0, sizeof(float));
    dot_product<<<blocksPerGrid, threadsPerBlock>>>(A, B, sum, N);
    set_value<<<1, 1>>>(sum, result);
    cudaDeviceSynchronize();
    cudaFree(sum);
}
