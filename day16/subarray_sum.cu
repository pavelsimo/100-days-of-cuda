#include <cuda_runtime.h>


__device__ int scan_warp(int val) {
    int res = val;
    unsigned mask = 0xffffffff;
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(mask, res, offset);
        if (lane >= offset) {
            res += n;
        }
    }
    return res;
}

__global__ void scan_block(int *output, const int *block_sums, int N) {
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

    int val = (tid < N) ? input[tid]: 0;
    val = scan_warp(val);

    if (lane == 31) {
        warp_sums[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        int s = (lane < blockDim.x / 32) ? warp_sums[lane]: 0.0f;
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

__global__ void subarray_sum(const int *output_sums, int* output, int N, int S, int E) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        int lo = lo = (S >= 1) ? output_sums[S-1]: 0;
        int hi = output_sums[E];
        output[0] = hi - lo;
    }
}

static void prefix_sum(const int* input, int* output_sums, int N) {
    const int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    if (blocksPerGrid == 1) {
        scan<<<1, threadsPerBlock>>>(input, output_sums, nullptr, N);
        return;
    }

    int* block_sums;
    cudaMalloc(&block_sums, blocksPerGrid * sizeof(int));
    scan<<<blocksPerGrid, threadsPerBlock>>>(input, output_sums, block_sums, N);
    prefix_sum(block_sums, block_sums, blocksPerGrid);
    scan_block<<<blocksPerGrid, threadsPerBlock>>>(output_sums, block_sums, N);
    
    cudaFree(block_sums);
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    int* output_sums;
    cudaMalloc(&output_sums, N * sizeof(int));
    prefix_sum(input, output_sums, N);
    subarray_sum<<<1, 1>>>(output_sums, output, N, S, E);
    cudaFree(output_sums);
    cudaDeviceSynchronize();
}
