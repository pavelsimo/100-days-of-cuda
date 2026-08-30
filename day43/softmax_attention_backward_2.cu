#include <cuda_runtime.h>
#include <float.h>

#define BLOCKSIZE 16

template <bool TRANS_A = false, bool TRANS_B = false>
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
        C[row * N + col] = alpha * sum;
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

__global__ void softmax_backward(const float *P, const float *dP, float *dS, const float *D, int M, int N, int d, float alpha) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < M && j < N) {
        dS[i * N + j] = P[i * N + j] * (dP[i * N + j] - D[i]) * alpha;
    }
}

__global__ void softmax_backward_rowsum(const float* P, const float *dP, float *D, int M, int N) {
    float sum = 0.0f;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < M) {
        for (int k = 0; k < N; ++k) {
            sum += P[i * N + k] * dP[i * N + k];
        }
        D[i] = sum;
    }
}

// Q, K, V, dO, dQ, dK, dV are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* dO, float* dQ,
                      float* dK, float* dV, int M, int N, int d) {
    dim3 threads(BLOCKSIZE, BLOCKSIZE);
    dim3 grid1(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );
    dim3 grid2(
        (d + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );
    dim3 grid3(
        (d + threads.x - 1) / threads.x,
        (N + threads.y - 1) / threads.y
    );

    float *P, *dP, *dS, *D;
    cudaMalloc(&P, M * N * sizeof(float));
    cudaMalloc(&dP, M * N * sizeof(float));
    cudaMalloc(&dS, M * N * sizeof(float));
    cudaMalloc(&D, M * sizeof(float));
    float alpha = 1.0f / sqrtf((float)d);

    // P  = softmax(Q @ K^T / sqrt(d))
    matmul<false, true><<<grid1, threads>>>(Q, K, P, M, N, d, alpha);
    softmax<<<M, BLOCKSIZE>>>(P, M, N);

    // dV = P^T @ dO
    matmul<true, false><<<grid3, threads>>>(P, dO, dV, N, d, M, 1.0f);

    // dP = dO @ V^T
    matmul<false, true><<<grid1, threads>>>(dO, V, dP, M, N, d, 1.0f);

    // dS = P * (dP - rowsum(P * dP)) / sqrt(d)
    softmax_backward_rowsum<<<(M + 256 - 1) / 256, 256>>>(P, dP, D, M, N);
    softmax_backward<<<grid1, threads>>>(P, dP, dS, D, M, N, d, alpha); 
    
    // dQ = dS @ K
    matmul<false, false><<<grid2, threads>>>(dS, K, dQ, M, d, N, 1.0f);

    // dK = dS^T @ Q
    matmul<true, false><<<grid3, threads>>>(dS, Q, dK, N, d, M, 1.0f);

    cudaDeviceSynchronize();
    cudaFree(P);
    cudaFree(dP);
    cudaFree(dS);
    cudaFree(D);
}
