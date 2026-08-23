#include <cuda_runtime.h>
#include <cfloat>
#include <stdio.h>
#include <stdlib.h>

#define BLOCKSIZE 16

__global__ void softmax(float* scores, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float max = -FLT_MAX;
        for (int col = 0; col < N; ++col) {
            max = fmaxf(max, scores[row * N + col]);
        }
        float sum = 0.0f;
        for (int col = 0; col < N; ++col) {
            float e = expf(scores[row * N + col] - max);
            scores[row * N + col] = e;
            sum += e;
        }
        for (int col = 0; col < N; ++col) {
            scores[row * N + col] /= sum;
        }
    }
}

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

__global__ void softmax_attention(const float* Q,
                      const float* K, 
                      const float* V, 
                      float* output, 
                      int M, 
                      int N,
                      int d,
                      float alpha) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float score = 0.0f;
        for (int k = 0; k < d; ++k) {
            score += Q[row * d + k] * K[col * d + k];
        }
        output[row * N + col] = score * rsqrtf((float)d) + alpha * (row - col);
    }
}

extern "C" void solve(const float* Q, 
                      const float* K, 
                      const float* V, 
                      float* output, 
                      int M, 
                      int N,
                      int d,
                      float alpha) {
    float* scores;
    cudaMalloc(&scores, M * N * sizeof(float));

    dim3 threadsPerBlockGrid(BLOCKSIZE, BLOCKSIZE);
    int threadsPerBlock = threadsPerBlockGrid.x * threadsPerBlockGrid.y; 
    dim3 scoresGrid((M + threadsPerBlockGrid.x - 1) / threadsPerBlockGrid.x,
                    (N + threadsPerBlockGrid.y - 1) / threadsPerBlockGrid.y);
    softmax_attention<<<scoresGrid, threadsPerBlockGrid>>>(Q, K, V, scores, M, N, d, alpha);

    softmax<<<(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(scores, M, N);

    dim3 outputGrid((d + threadsPerBlockGrid.x - 1) / threadsPerBlockGrid.x,
                    (M + threadsPerBlockGrid.y - 1) / threadsPerBlockGrid.y);
    matmul<<<outputGrid, threadsPerBlockGrid>>>(scores, V, output, M, d, N);

    cudaDeviceSynchronize();
    cudaFree(scores);
}
