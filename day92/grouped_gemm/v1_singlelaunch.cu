#include <cuda_runtime.h>

#define BLOCKSIZE 16

__global__ void group_matmul(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int N, int K, const int* group_offsets) {
    __shared__ float As[BLOCKSIZE][BLOCKSIZE + 1];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE + 1];

    const int g = blockIdx.z;
    const int group_start = group_offsets[g];
    const int group_size = group_offsets[g + 1] - group_start;
    if (blockIdx.y * BLOCKSIZE >= group_size) {
        return;
    }

    const float *Ag = A + group_start * K;
    const float *Bg = B + g * K * N;
    float *Cg = C + group_start * N;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCKSIZE + ty;
    const int col = blockIdx.x * BLOCKSIZE + tx;
    float sum = 0.0f;
    for (int t = 0; t < K; t += BLOCKSIZE) {
        As[ty][tx] = (row < group_size && t + tx < K) ? Ag[row * K + t + tx] : 0.0f;
        Bs[ty][tx] = (t + ty < K && col < N) ? Bg[(t + ty) * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCKSIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < group_size && col < N) {
        Cg[row * N + col] = sum;
    }
}

// A, B, group_offsets, C are device pointers
extern "C" void solve(const float* A, const float* B, const int* group_offsets, float* C, int G,
                      int M_total, int K, int N) {
    dim3 threads(BLOCKSIZE, BLOCKSIZE);
    dim3 blocks(
        (N + BLOCKSIZE - 1) / BLOCKSIZE,
        (M_total + BLOCKSIZE - 1) / BLOCKSIZE,
        G
    );
    group_matmul<<<blocks, threads>>>(A, B, C, N, K, group_offsets);
    cudaDeviceSynchronize();
}
