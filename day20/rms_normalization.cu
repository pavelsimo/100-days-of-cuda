#include <cuda_runtime.h>
#include <float.h>

__device__ __forceinline__ float warp_reduce(float val) {
    unsigned mask = 0xFFFFFFFFU;
    float res = val;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        res += __shfl_down_sync(mask, res, offset);
    }
    return res;
}

__global__ void finalize_rms(float* sum_squared, int N, float eps) {
    *sum_squared = rsqrtf(*sum_squared / N + eps);
}

__global__ void normalize(const float* input, float gamma, float beta, float* output, int N, const float *inv_rms) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        output[i] = gamma * input[i] * (*inv_rms) + beta;
    }
}

__global__ void rms(const float* input, float* sum_squared, int N) {
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
            sum += input[i + k * stride] * input[i + k * stride];
        }
        i += 8 * stride;
    }

    while (i < N) {
        sum += input[i] * input[i];
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
            atomicAdd(sum_squared, sum);
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N, float eps) {
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    float *sum_squared;
    cudaMalloc(&sum_squared, sizeof(float));
    cudaMemset(sum_squared, 0, sizeof(float));
    rms<<<blocks, threads>>>(input, sum_squared, N);
    finalize_rms<<<1, 1>>>(sum_squared, N, eps);
    normalize<<<blocks, threads>>>(input, gamma, beta, output, N, sum_squared);
    cudaDeviceSynchronize();
    cudaFree(sum_squared);
}
