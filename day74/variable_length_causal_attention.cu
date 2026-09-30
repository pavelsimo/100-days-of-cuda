#include <cuda_runtime.h>
#include <float.h>

#define TILE_SIZE 16

template <bool TRANS_A = false, bool TRANS_B = false, bool MASK = false>
__global__ void matmul(const float *A, const float *B, float *C, int M, int N, int K, float alpha, 
    const int *cu_seqlens, int S) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int iB = blockIdx.y * TILE_SIZE;
    const int jB = blockIdx.x * TILE_SIZE;
    const int i = iB + ty;
    const int j = jB + tx;

    float sum = 0.0f;
    for (int t = 0; t < K; t += TILE_SIZE) {
        if (TRANS_A) {
            const int k = t + ty;
            const int r = iB + tx;
            As[tx][ty] = (k < K && r < M) ? A[k * M + r] : 0.0f;
        } else {
            const int k = t + tx;
            As[ty][tx] = (i < M && k < K) ? A[i * K + k] : 0.0f;
        }

        if (TRANS_B) {
            const int k = t + tx;
            const int c = jB + ty;
            Bs[tx][ty] = (c < N && k < K) ? B[c * K + k] : 0.0f;
        } else {
            const int k = t + ty;
            Bs[ty][tx] = (k < K && j < N) ? B[k * N + j] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (i >= M || j >= N) {
        return;
    }

    float value = alpha * sum;
    if constexpr (MASK) {
        int seq = 0;
        for (int s = 0; s < S; ++s) {
            if (i >= cu_seqlens[s] &&
                i <  cu_seqlens[s + 1]) {
                seq = s;
                break;
            }
        }
        int seq_start = cu_seqlens[seq];
        int seq_end   = cu_seqlens[seq + 1];
        C[i * N + j] = j >= seq_start && j <  seq_end && j <= i ? value : -INFINITY;
    } else {
        C[i * N + j] = value;
    }
}

__global__ void softmax(float* scores, int M, int N) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float t[TILE_SIZE*TILE_SIZE];

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

// Q, K, V, cu_seqlens, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const int* cu_seqlens,
                      float* output, int T, int d, int S) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid_1(
        (T + threads.x - 1) / threads.x, 
        (T + threads.y - 1) / threads.y
    );
    dim3 grid_2(
        (d + threads.x - 1) / threads.x, 
        (T + threads.y - 1) / threads.y
    );

    float alpha = 1.0f / sqrtf(d);
    float *scores;
    cudaMalloc(&scores, T * T * sizeof(float));
    matmul<false, true, true><<<grid_1, threads>>>(Q, K, scores, T, T, d, alpha, cu_seqlens, S);
    softmax<<<T, TILE_SIZE * TILE_SIZE>>>(scores, T, T);
    matmul<false, false, false><<<grid_2, threads>>>(scores, V, output, T, d, T, 1.0f, cu_seqlens, S);
    
    cudaFree(scores);
}
