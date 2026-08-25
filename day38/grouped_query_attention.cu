#include <cuda_runtime.h>
#include <cfloat>

#define BLOCKSIZE 16

__global__ void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_t(const float *A, const float *B, float *C, int M, int N, int K, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = scale * sum;
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
extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {    
    float *weights;
    cudaMalloc(&weights, seq_len * seq_len * sizeof(float));
    dim3 threads(BLOCKSIZE, BLOCKSIZE);
    dim3 grid1(
        (seq_len + threads.x - 1) / threads.x,
        (seq_len + threads.y - 1) / threads.y
    );
    dim3 grid2(
        (head_dim + threads.x - 1) / threads.x,
        (seq_len + threads.y - 1) / threads.y
    );

    float scale = 1.0 / sqrt(head_dim);
    int group_size = num_q_heads / num_kv_heads;
    for (int head_idx = 0; head_idx < num_q_heads; ++head_idx) {
        int kv_head_idx = head_idx / group_size;
        const float *q = Q + head_idx * seq_len * head_dim;
        const float *k = K + kv_head_idx * seq_len * head_dim;
        const float *v = V + kv_head_idx * seq_len * head_dim;
        float *o = output + head_idx * seq_len * head_dim;
        matmul_t<<<grid1, threads>>>(q, k, weights, seq_len, seq_len, head_dim, scale);
        softmax<<<seq_len, BLOCKSIZE*BLOCKSIZE>>>(weights, seq_len, seq_len);
        matmul<<<grid2, threads>>>(weights, v, o, seq_len, head_dim, seq_len);
    }

    cudaFree(weights);
    cudaDeviceSynchronize();
}
