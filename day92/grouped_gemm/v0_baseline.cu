#include <cuda_runtime.h>

#define BLOCKSIZE 16

__global__ void matmul(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int M, int N, int K) {
    __shared__ float As[BLOCKSIZE][BLOCKSIZE + 1];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCKSIZE + ty;
    const int col = blockIdx.x * BLOCKSIZE + tx;

    float sum = 0.0f;
    for (int t = 0; t < K; t += BLOCKSIZE) {
        As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.0f;
        Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCKSIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// A, B, group_offsets, C are device pointers
extern "C" void solve(const float* A, const float* B, const int* group_offsets, float* C, int G,
                      int M_total, int K, int N) {
    
    dim3 threads(BLOCKSIZE, BLOCKSIZE);
    int *h_group_offsets = new int[G + 1];
    cudaMemcpy(h_group_offsets, group_offsets, (G + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < G; ++i) {
        int group_start = h_group_offsets[i];
        int group_end = h_group_offsets[i + 1];
        if (group_end == group_start) {
            continue;
        }

        const int group_size = group_end - group_start;
        const float *a = A + group_start * K;
        const float *b = B + i * (K * N);
        float *c = C + group_start * N;
        dim3 blocks(
            (N + BLOCKSIZE - 1) / threads.x, 
            (group_size + BLOCKSIZE - 1) / threads.y
        );

        // 1 2  -> 2x2 -> (group_size, G)
        // 3 4 
        //
        // 1 0 1 -> 2x3 -> (K, N)
        // 0 1 0
        //
        // --> (group_size, N)
        matmul<<<blocks, threads>>>(a, b, c, group_size, N, K);
    }

    delete[] h_group_offsets;
}
