#include <cuda_runtime.h>
#include <limits.h>

__device__ int scan_warp(int val) {
    unsigned mask = 0xffffffff;
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(mask, val, offset);
        if (lane >= offset) {
            val += n;
        }
    }
    return val;
}

__global__ void scan_block(int* output, const int* block_sums, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && tid < N) {
        output[tid] += block_sums[blockIdx.x - 1];
    }
}

__global__ void scan(const int* input, int* output, int* block_sums, int N) {
    __shared__ int warp_sums[32];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    int val = (tid < N) ? input[tid] : 0;
    val = scan_warp(val);

    if (lane == 31) {
        warp_sums[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        int s = (lane < blockDim.x / 32) ? warp_sums[lane] : 0;
        s = scan_warp(s);
        warp_sums[lane] = s;
    }
    __syncthreads();

    if (wid > 0) {
        val += warp_sums[wid - 1];
    }

    if (tid < N) {
        output[tid] = val;
    }

    if (block_sums != nullptr && threadIdx.x == blockDim.x - 1) {
        block_sums[blockIdx.x] = val;
    }
}

static void prefix_sum(const int* input, int* output_sums, int N) {
    const int threads = 256;
    int blocks = (N + threads - 1) / threads;

    if (blocks == 1) {
        scan<<<1, threads>>>(input, output_sums, nullptr, N);
        return;
    }

    int* block_sums;
    cudaMalloc(&block_sums, blocks * sizeof(int));
    scan<<<blocks, threads>>>(input, output_sums, block_sums, N);
    prefix_sum(block_sums, block_sums, blocks);
    scan_block<<<blocks, threads>>>(output_sums, block_sums, N);
    cudaFree(block_sums);
}

__global__ void max_subarray_sum(const int* input, const int* output_sums, int* output, int N, int window_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && N - i >= window_size) {
        int sum = output_sums[i + window_size - 1];
        if (i > 0) {
            sum -= output_sums[i - 1];
        }
        atomicMax(output, sum);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int window_size) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int min_int = INT_MIN;
    int *output_sums;
    cudaMemcpy(output, &min_int, sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&output_sums, N * sizeof(int));
    prefix_sum(input, output_sums, N);
    max_subarray_sum<<<blocks, threads>>>(input, output_sums, output, N, window_size);
    cudaDeviceSynchronize();
    cudaFree(output_sums);
}
