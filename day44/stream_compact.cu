#include <cuda_runtime.h>
#define BLOCK 1024

__device__ int warp_scan_inclusive(int val) {
    int lane = threadIdx.x % warpSize;
    for (int offset = 1; offset < warpSize; offset <<= 1) {
        int n = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) {
            val += n;
        }
    }
    return val;
}

__device__ int block_scan_inclusive(int val) {
    __shared__ int t[32];
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    val = warp_scan_inclusive(val);
    if (lane == warpSize - 1) {
        t[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0) {
        int numWarps = (blockDim.x + warpSize - 1) / warpSize;
        int w = (lane < numWarps) ? t[lane] : 0;
        t[lane] = warp_scan_inclusive(w);
    }
    __syncthreads();

    if (warpId > 0) {
        val += t[warpId - 1];
    }
    return val;
}

__global__ void scan_blocks(const int* in, int* out, int* block_sums, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (idx < N) ? in[idx] : 0;
    val = block_scan_inclusive(val);
    if (idx < N) {
        out[idx] = val;
    }
    if (threadIdx.x == blockDim.x - 1) {
        block_sums[blockIdx.x] = val;
    }
}

__global__ void scan_sums(int* sums, int n) {
    __shared__ int carry_s;
    if (threadIdx.x == 0) {
        carry_s = 0;
    }
    __syncthreads();
    for (int start = 0; start < n; start += blockDim.x) {
        int i = start + threadIdx.x;
        int val = (i < n) ? sums[i] : 0;
        val = block_scan_inclusive(val);
        int c = carry_s;
        if (i < n) {
            sums[i] = val + c;
        }
        __syncthreads();
        if (threadIdx.x == blockDim.x - 1) {
            carry_s = c + val;
        }
        __syncthreads();
    }
}

__global__ void add_offsets(int* out, const int* sums, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && idx < N) {
        out[idx] += sums[blockIdx.x - 1];
    }
}

static void prefix_sum(const int* in, int* out, int N) {
    int blocks = (N + BLOCK - 1) / BLOCK;
    int* block_sums;
    cudaMalloc(&block_sums, blocks * sizeof(int));
    scan_blocks<<<blocks, BLOCK>>>(in, out, block_sums, N);
    scan_sums<<<1, BLOCK>>>(block_sums, blocks);
    add_offsets<<<blocks, BLOCK>>>(out, block_sums, N);
    cudaFree(block_sums);
}

__global__ void prepare_arr(const float* A, int N, int* out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        out[idx] = (A[idx] > 0) ? 1: 0;
    }
}

__global__ void compact(const float* A, const int* P, int N, float* out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N && A[idx] > 0) {
        int pos = P[idx] - 1;
        out[pos] = A[idx];
    }
}

// A, out are device pointers
extern "C" void solve(const float* A, int N, float* out) {
    int *B, *C;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    cudaMalloc(&B, N * sizeof(int));
    cudaMalloc(&C, N * sizeof(int));
    prepare_arr<<<blocks, threads>>>(A, N, B);
    prefix_sum(B, C, N);
    compact<<<blocks, threads>>>(A, C, N, out);
    cudaFree(B);
    cudaFree(C);
    cudaDeviceSynchronize();
}
