#include <cuda_runtime.h>

#define BLOCKSIZE 32

__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;
    int row = blockIdx.y * BLOCKSIZE + threadRow;
    int col = blockIdx.x * BLOCKSIZE + threadCol;
    
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    float sum = 0.0f;
    for (int tile = 0; tile < K; tile += BLOCKSIZE) {
        const int aCol = tile + threadCol;
        const int bRow = tile + threadRow;

        As[threadRow * BLOCKSIZE + threadCol] = 
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadRow * BLOCKSIZE + threadCol] = 
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        for (int d = 0; d < BLOCKSIZE; ++d) {
            sum += As[threadRow * BLOCKSIZE + d] * Bs[d * BLOCKSIZE + threadCol];
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
    matmul<<<blocks, threads>>>(input, input, output, N, N, N);
    for (int p = 2; p < P; ++p) {
        matmul<<<blocks, threads>>>(output, input, temp, N, N, N);
        cudaMemcpy(output, temp, sz, cudaMemcpyDeviceToDevice);
    }

    cudaDeviceSynchronize();
    cudaFree(temp);
}
