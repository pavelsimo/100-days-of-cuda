#include <cuda_runtime.h>
#include <utility>

#define BLOCKSIZE 16

__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;
    int row = blockIdx.y * BLOCKSIZE + threadRow;
    int col = blockIdx.x * BLOCKSIZE + threadCol;
    
    __shared__ float As[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

    float sum = 0.0f;
    for (int tile = 0; tile < K; tile += BLOCKSIZE) {
        const int aCol = tile + threadCol;
        const int bRow = tile + threadRow;

        As[threadRow][threadCol] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadRow][threadCol] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        for (int d = 0; d < BLOCKSIZE; ++d) {
            sum += As[threadRow][d] * Bs[d][threadCol];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int P) {
    size_t sz = static_cast<size_t>(N) * N * sizeof(float);
    if (P == 1) {
        cudaMemcpy(output, input, sz, cudaMemcpyDeviceToDevice);
        return;
    }

    dim3 threads(BLOCKSIZE, BLOCKSIZE);
    dim3 blocks((N + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);
    float *temp;
    cudaMalloc(&temp, sz);
    cudaMemcpy(output, input, sz, cudaMemcpyDeviceToDevice);

    float *cur = output;
    float *nxt = temp;
    int hi_bit = 1 << (31 - __builtin_clz(P));
    for (int bit_mask = hi_bit >> 1; bit_mask > 0; bit_mask >>= 1) {
        matmul<<<blocks, threads>>>(cur, cur, nxt, N, N, N);
        std::swap(cur, nxt);

        if (P & bit_mask) {
            matmul<<<blocks, threads>>>(cur, input, nxt, N, N, N);
            std::swap(cur, nxt);
        }
    }

    if (cur != output) {
        cudaMemcpy(output, cur, sz, cudaMemcpyDeviceToDevice);
    }

    cudaDeviceSynchronize();
    cudaFree(temp);
}
