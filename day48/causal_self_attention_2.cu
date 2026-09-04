#include <cuda_runtime.h>
#include <float.h>


#define BLOCKSIZE 16

template <bool TRANS_A = false, bool TRANS_B = false, bool MASK = false>

__global__ void matmul(const float *A, const float *B, float *C, int M, int N, int K, float alpha) {
    __shared__ float As[BLOCKSIZE][BLOCKSIZE + 1];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int rowB = blockIdx.y * BLOCKSIZE;
    const int colB = blockIdx.x * BLOCKSIZE;
    const int row = rowB + ty;
    const int col = colB + tx;

    float sum = 0.0f;
    for (int t = 0; t < K; t += BLOCKSIZE) {
        if (TRANS_A) {
            const int k = t + ty;
            const int r = rowB + tx;
            As[tx][ty] = (k < K && r < M) ? A[k * M + r] : 0.0f;
        } else {
            const int k = t + tx;
            As[ty][tx] = (row < M && k < K) ? A[row * K + k] : 0.0f;
        }

        if (TRANS_B) {
            const int k = t + tx;
            const int c = colB + ty;
            Bs[tx][ty] = (c < N && k < K) ? B[c * K + k] : 0.0f;
        } else {
            const int k = t + ty;
            Bs[ty][tx] = (k < K && col < N) ? B[k * N + col] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCKSIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = MASK && col > row ? -FLT_MAX : alpha * sum;
    }
}

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

    const float inv = 1.0f / t[0];
    for (int col = tid; col < N; col += blockDim.x) {
        scores[row * N + col] *= inv;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {
    dim3 threads(BLOCKSIZE, BLOCKSIZE);

    // Q - M x d
    // K - M x d 
    // K^T - d x M
    // V - M x d
    // S = Q @ K^T / sqrt(d) - M x M
    // S_M = masked(S_M) - M x M
    // P = softmax(S_M) - M x M
    // P @ V - M x d 
    dim3 grid1(
        (M + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );
    dim3 grid2(
        (d + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    float *S;
    float alpha = 1.0f / sqrt((float)d);
    cudaMalloc(&S, M * M * sizeof(float));
    // matmul + mask
    matmul<false, true, true><<<grid1, threads>>>(Q, K, S, M, M, d, alpha);
    softmax<<<M, BLOCKSIZE * BLOCKSIZE>>>(S, M, M);
    matmul<false, false, false><<<grid2, threads>>>(S, V, output, M, d, M, 1.0f);
    cudaFree(S);
}




