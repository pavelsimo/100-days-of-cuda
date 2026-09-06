#include <cuda_runtime.h>
#include <float.h>

#define BLOCKSIZE 16

template <bool TRANS_A = false, bool TRANS_B = false, bool MASK = false>
__global__ void matmul(const float *A, const float *B, float *C, int M, int N, int K,
                       float alpha, int lda, int ldb, int ldc) {
    __shared__ float As[BLOCKSIZE][BLOCKSIZE + 1];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int iB = blockIdx.y * BLOCKSIZE;
    const int jB = blockIdx.x * BLOCKSIZE;
    const int i = iB + ty;
    const int j = jB + tx;

    float sum = 0.0f;
    for (int t = 0; t < K; t += BLOCKSIZE) {
        if (TRANS_A) {
            const int k = t + ty;
            const int r = iB + tx;
            As[tx][ty] = (k < K && r < M) ? A[k * lda + r] : 0.0f;
        } else {
            const int k = t + tx;
            As[ty][tx] = (i < M && k < K) ? A[i * lda + k] : 0.0f;
        }

        if (TRANS_B) {
            const int k = t + tx;
            const int c = jB + ty;
            Bs[tx][ty] = (c < N && k < K) ? B[c * ldb + k] : 0.0f;
        } else {
            const int k = t + ty;
            Bs[ty][tx] = (k < K && j < N) ? B[k * ldb + j] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCKSIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (i < M && j < N) {
        C[i * ldc + j] = alpha * sum;
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

// Q and output: (M, H, D); K and V: (N, H, D), all device pointers.
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int H, int D) {
    dim3 threads(BLOCKSIZE, BLOCKSIZE);
    dim3 scoreGrid(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );
    dim3 outputGrid(
        (D + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    float *S;
    float alpha = 1.0f / sqrt((float)D);
    cudaMalloc(&S, M * N * sizeof(float));
    const int stride = H * D;
    for (int h = 0; h < H; ++h) {
        const float* q = Q + h * D;
        const float* k = K + h * D;
        const float* v = V + h * D;
        float *o = output + h * D;
        // Q @ K^T / sqrt(d)
        matmul<false, true, true><<<scoreGrid, threads>>>(
            q, k, S, M, N, D, alpha, stride, stride, N);
        // softmax the scores
        softmax<<<M, BLOCKSIZE>>>(S, M, N);
        // S @ V
        matmul<false, false, false><<<outputGrid, threads>>>(
            S, v, o, M, D, N, 1.0f, N, stride, stride);
    }

    cudaFree(S);
}
