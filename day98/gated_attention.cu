#include <cuda_runtime.h>
#include <float.h>

#define TILE_SIZE 16

__global__ void sigmoid(float *A, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M || j >= N) {
        return; 
    }

    float x = A[i * N + j];
    A[i * N + j] = 1.0f / (1.0f + expf(-x));
}

__global__ void softmax(float* scores, int M, int N) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float t[TILE_SIZE*TILE_SIZE];

    if (row >= M) {
        return;
    }

    float t_max = -FLT_MAX;
    for (int col = tid; col < N; col += blockDim.x) {
        t_max = fmaxf(t_max, scores[row * N + col]);
    }
    t[tid] = t_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] = fmaxf(t[tid], t[tid + stride]);
        }
        __syncthreads();
    }

    const float row_max = t[0];
    float t_sum = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        const float e = expf(scores[row * N + col] - row_max);
        scores[row * N + col] = e;
        t_sum += e;
    }
    t[tid] = t_sum;
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

template <bool TRANS_A = false, bool TRANS_B = false, bool CAUSAL_MASK = false>
__global__ void matmul(const float * __restrict__ A, const float * __restrict__ B,
                       float * __restrict__ C, int M, int N, int K,
                       int lda, int ldb, int ldc, float alpha) {
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
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (i < M && j < N) {
        if (CAUSAL_MASK) {
            // causal mask
            C[i * ldc + j] = j > i ? -INFINITY: alpha * sum;
        } else {
            C[i * ldc + j] = alpha * sum;
        }
    }
}

__global__ void elementwise_matmul(const float *A, const float* B, float *C, int M, int N,
                             int lda, int ldb, int ldc) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M || j >= N) {
        return;
    }
    C[i * ldc + j] = A[i * lda + j] * B[i * ldb + j];
}

// Q, K, V, X, W_g, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* X,
                      const float* W_g, float* output, int M, int d_model, int h) {

    // Q - [M, d_model] 
    // K - [M, d_model] 
    // V - [M, d_model] 
    // X - [M, d_model] 
    // W_g - [d_model, d_model] 
    // output - [M, d_model] 
    // M - number of tokens
    // d_model - model dimension
    // h - number of attention heads
    // head dimension - d_model / h
    
    float *g, *S;
    int d_k = d_model / h;
    float alpha = 1.0f / sqrtf(d_k);
    cudaMalloc(&g, M * d_model * sizeof(float));
    cudaMalloc(&S, M * M * sizeof(float));
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks_1(
        (d_model + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    dim3 blocks_2(
        (M + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    dim3 blocks_3(
        (d_k + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    // g = sigmoid(X @ Wg) -> [M, d_model]
    matmul<<<blocks_1, threads>>>(X, W_g, g, M, d_model, d_model, d_model, d_model, d_model, 1.0f);
    sigmoid<<<blocks_1, threads>>>(g, M, d_model);
    for (int j = 0; j < h; ++j) {
        // Q_j - [M, d_k]
        // K_t - [d_k, M]
        // V_j - [M, d_k]
        const int h_offset = j * d_k;
        const float *Q_j = Q + h_offset;
        const float *K_j = K + h_offset;
        const float *V_j = V + h_offset;
        const float *g_j = g + h_offset;
        float *o_j = output + h_offset;
        
        // Q_j @ K_j^T -> [M, M]
        // S - [M, M]
        matmul<false, true, true><<<blocks_2, threads>>>(Q_j, K_j, S, M, M, d_k, d_model, d_model, M, alpha);
        softmax<<<M, TILE_SIZE * TILE_SIZE>>>(S, M, M);

        // S @ V_j -> [M, d_k]
        matmul<<<blocks_3, threads>>>(S, V_j, o_j, M, d_k, M, M, d_model, d_model, 1.0f);
        elementwise_matmul<<<blocks_3, threads>>>(o_j, g_j, o_j, M, d_k, d_model, d_model, d_model);
    }

    cudaFree(g);
    cudaFree(S);
}
