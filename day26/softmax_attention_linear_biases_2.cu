#include <cuda_runtime.h>
#include <cfloat>
#include <stdio.h>
#include <stdlib.h>

#define BLOCKSIZE 16

__global__ void softmax(float* scores, int M, int N) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float t[BLOCKSIZE*BLOCKSIZE];

    if (row >= M) {
        return;
    }

    float threadMax = -FLT_MAX;
    for (int col = tid; col < N; col += blockDim.x) {
        threadMax = fmaxf(threadMax, scores[row * N + col]);
    }
    t[tid] = threadMax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] = fmaxf(t[tid], t[tid + stride]);
        }
        __syncthreads();
    }

    const float rowMax = t[0];
    float threadSum = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        const float e = expf(scores[row * N + col] - rowMax);
        scores[row * N + col] = e;
        threadSum += e;
    }
    t[tid] = threadSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] += t[tid + stride];
        }
        __syncthreads();
    }

    const float invSum = 1.0f / t[0];
    for (int col = tid; col < N; col += blockDim.x) {
        scores[row * N + col] *= invSum;
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
    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;
    const int row = blockIdx.y * BLOCKSIZE + threadRow;
    const int col = blockIdx.x * BLOCKSIZE + threadCol;

    __shared__ float qTile[BLOCKSIZE][BLOCKSIZE];
    __shared__ float kTile[BLOCKSIZE][BLOCKSIZE + 1];

    float score = 0.0f;
    for (int tile = 0; tile < d; tile += BLOCKSIZE) {
        const int dimension = tile + threadCol;
        const int keyRow = blockIdx.x * BLOCKSIZE + threadRow;

        qTile[threadRow][threadCol] =
            (row < M && dimension < d) ? Q[row * d + dimension] : 0.0f;
        kTile[threadCol][threadRow] =
            (keyRow < N && dimension < d) ? K[keyRow * d + dimension] : 0.0f;
        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; ++k) {
            score += qTile[threadRow][k] * kTile[k][threadCol];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
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
    dim3 scoresGrid((N + threadsPerBlockGrid.x - 1) / threadsPerBlockGrid.x,
                    (M + threadsPerBlockGrid.y - 1) / threadsPerBlockGrid.y);
    softmax_attention<<<scoresGrid, threadsPerBlockGrid>>>(Q, K, V, scores, M, N, d, alpha);

    softmax<<<M, BLOCKSIZE*BLOCKSIZE>>>(scores, M, N);

    dim3 outputGrid((d + threadsPerBlockGrid.x - 1) / threadsPerBlockGrid.x,
                    (M + threadsPerBlockGrid.y - 1) / threadsPerBlockGrid.y);
    matmul<<<outputGrid, threadsPerBlockGrid>>>(scores, V, output, M, d, N);

    cudaDeviceSynchronize();
    cudaFree(scores);
}
