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

__global__ void scale(const int8_t* A, const float* scale, float *output, int M, int N) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < M && j < N) {
        output[i * N + j] = static_cast<float>(A[i * N + j]) * scale[i];
    }
}

// Q, K_int8, V_int8, k_scale, v_scale, output are device pointers
extern "C" void solve(const float* Q, const int8_t* K_int8, const int8_t* V_int8,
                      const float* k_scale, const float* v_scale, float* output, int num_heads,
                      int seq_len, int head_dim) {

    dim3 threads(BLOCKSIZE, BLOCKSIZE);
    dim3 grid1((head_dim + BLOCKSIZE - 1) / BLOCKSIZE, (seq_len + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 grid2((seq_len + BLOCKSIZE - 1) / BLOCKSIZE, 1);
    dim3 grid3((head_dim + BLOCKSIZE - 1) / BLOCKSIZE, 1);

    float alpha = 1.0f / sqrtf(head_dim);
    float *K, *V, *score;
    cudaMalloc(&K, seq_len * head_dim * sizeof(float));
    cudaMalloc(&V, seq_len * head_dim * sizeof(float));    
    cudaMalloc(&score, seq_len * sizeof(float));
    for (int h = 0; h < num_heads; ++h) {
        const int8_t* k_int8 = K_int8 + h * seq_len * head_dim;
        const int8_t* v_int8 = V_int8 + h * seq_len * head_dim;
        const float* k_s = k_scale + h * seq_len;
        const float* v_s = v_scale + h * seq_len;
        const float* q = Q + h * head_dim;
        float* o = output + h * head_dim;

        scale<<<grid1, threads>>>(k_int8, k_s, K, seq_len, head_dim);
        scale<<<grid1, threads>>>(v_int8, v_s, V, seq_len, head_dim);

        // q - 1 x head_dim
        // K^T - head_dim x seq_len
        // scores = q @ K^T - 1 x seq_len
        matmul<false, true><<<grid2, threads>>>(q, K, score, 1, seq_len, head_dim, alpha);
        softmax<<<1, 256>>>(score, 1, seq_len);
        
        // scores - 1 x seq_len
        // V - seq_len x head_dim
        // output - 1 x head_dim
        matmul<false, false><<<grid3, threads>>>(score, V, o, 1, head_dim, seq_len, 1.0f);
    }
    
    cudaFree(K);
    cudaFree(V);
    cudaFree(score);
}
