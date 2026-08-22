#include <cuda_runtime.h>
#include <float.h>

#define TILE_SIZE 16

__global__ void softmax(float* scores, int M, int N, int num_sinks, int window_size) {
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float t[TILE_SIZE*TILE_SIZE];

    if (i >= M) {
        return;
    }

    float threadMax = -FLT_MAX;
    for (int j = tid; j < N; j += blockDim.x) {
        if (j <= i && (j < num_sinks || j >= i - window_size + 1)) {
            threadMax = fmaxf(threadMax, scores[i * N + j]);
        }
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
    for (int j = tid; j < N; j += blockDim.x) {
        if (j <= i && (j < num_sinks || j >= i - window_size + 1)) {
            const float e = expf(scores[i * N + j] - rowMax);
            scores[i * N + j] = e;
            threadSum += e;
        } else {
            scores[i * N + j] = 0.0f;
        }
    }
    t[tid] = threadSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] += t[tid + stride];
        }
        __syncthreads();
    }

    const float inv = 1.0f / t[0];
    for (int col = tid; col < N; col += blockDim.x) {
        scores[i * N + col] *= inv;
    }
}

__global__ void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadRow;
    int col = blockIdx.x * TILE_SIZE + threadCol;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (int tile = 0; tile < K; tile += TILE_SIZE) {
        const int aCol = tile + threadCol;
        const int bRow = tile + threadRow;

        As[threadRow][threadCol] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadRow][threadCol] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        for (int d = 0; d < TILE_SIZE; ++d) {
            sum += As[threadRow][d] * Bs[d][threadCol];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void matmul_t(const float *A, const float *B, float *C, int M, int N, int K, float alpha = 1.0f) {
    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadRow;
    int col = blockIdx.x * TILE_SIZE + threadCol;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (int tile = 0; tile < K; tile += TILE_SIZE) {
        const int aCol = tile + threadCol;
        const int bRow = blockIdx.x * TILE_SIZE + threadRow;
        const int bCol = tile + threadCol;

        As[threadRow][threadCol] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadRow][threadCol] = (bRow < N && bCol < K) ? B[bRow * K + bCol] : 0.0f;
        __syncthreads();

        for (int d = 0; d < TILE_SIZE; ++d) {
            sum += As[threadRow][d] * Bs[threadCol][d];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d,
                      int num_sinks, int window_size) {
    // Q: M x d
    // K: M x d
    // V: M x d
    // K^T: d x M
    // Q * K^T: M x M
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (M + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );
    float *scores;
    float alpha = 1.0f / fsqrt(d);
    cudaMalloc(&scores, M * M * sizeof(float));
    matmul_t<<<blocks, threads>>>(Q, K, scores, M, M, d, alpha);
    softmax<<<M, 256>>>(scores, M, M, num_sinks, window_size);
    matmul<<<blocks, threads>>>(scores, V, output, M, d, M);
    cudaDeviceSynchronize();
    cudaFree(scores);
}
