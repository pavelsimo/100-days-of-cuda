#include <cuda_runtime.h>

#define BLOCKSIZE 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void sparse_matrix_vec_mult(const float* A, const float* x, float* y, int M, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float sum =  0;
    for (int j = threadIdx.x; j < N; j += BLOCKSIZE) {
        sum += A[blockIdx.x * N + j] * x[j];
    }
    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) {
        y[blockIdx.x] = sum;
    }
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    sparse_matrix_vec_mult<<<M, BLOCKSIZE>>>(A, x, y, M, N);
}
