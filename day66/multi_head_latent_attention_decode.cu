#include <cuda_runtime.h>
#include <float.h>

#define TILE_SIZE 16

template <bool TRANS_A = false, bool TRANS_B = false>
__global__ void matmul(const float *A, const float *B, float *C, int M, int N, int K, float alpha) {
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

    if (i < M && j < N) {
        C[i * N + j] = alpha * sum;
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

__global__ void compute_s(const float* q_hat, const float* q_pe, const float* kv_cache, float* s,
                          int seq_len, int kv_lora_rank, int rope_dim, float scale) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < seq_len) {
        const float* kv_row = kv_cache + t * (kv_lora_rank + rope_dim);
        float sum = 0.0f;
        for (int d = 0; d < kv_lora_rank; ++d) {
            sum += q_hat[d] * kv_row[d];
        }
        for (int d = 0; d < rope_dim; ++d) {
            sum += q_pe[d] * kv_row[kv_lora_rank + d];
        }
        s[t] = sum * scale;
    }
}

__global__ void compute_l(const float* a, const float* kv_cache, float* l,
                          int seq_len, int kv_lora_rank, int rope_dim) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < kv_lora_rank) {
        int stride = kv_lora_rank + rope_dim;
        float sum = 0.0f;
        for (int t = 0; t < seq_len; ++t) {
            sum += a[t] * kv_cache[t * stride + d];
        }
        l[d] = sum;
    }
}

// q, kv_cache, W_UK, W_UV, output are device pointers
extern "C" void solve(const float* q, 
                      const float* kv_cache, 
                      const float* W_UK, 
                      const float* W_UV,
                      float* output, 
                      int num_heads, 
                      int seq_len, 
                      int kv_lora_rank, 
                      int head_dim,
                      int rope_dim) {
    
    // q        - (num_heads, head_dim + rope_dim)
    // kv_cache - (seq_len, kv_lora_rank + rope_dim)
    // W_UK     - (num_heads, head_dim, kv_lora_rank)
    // W_UV     - (num_heads, kv_lora_rank, head_dim)
    // output   - (num_heads, head_dim)
    float *q_hat, *s, *l;
    cudaMalloc(&s, seq_len * sizeof(float));
    cudaMalloc(&q_hat, kv_lora_rank * sizeof(float));
    cudaMalloc(&l, kv_lora_rank * sizeof(float));

    const float scale = 1.0f / sqrtf((float)(head_dim + rope_dim));
    const int threads = 256;
    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid1((kv_lora_rank + TILE_SIZE - 1) / TILE_SIZE, 1);
    const dim3 grid2((head_dim + TILE_SIZE - 1) / TILE_SIZE, 1);
    for (int h = 0; h < num_heads; ++h) {
        const float *q_nope = q + h * (head_dim + rope_dim);
        const float *q_pe = q_nope + head_dim;
        const float *w_uk = W_UK + h * head_dim * kv_lora_rank;
        const float *w_uv = W_UV + h * kv_lora_rank * head_dim;

        // q~ = q_nope · W_UK[h]  -> (1, kv_lora_rank)
        matmul<false, false><<<grid1, block>>>(q_nope, w_uk, q_hat, 1, kv_lora_rank, head_dim, 1.0f);
        
        // s = (q~·c + q_pe·k_pe) * scale
        compute_s<<<(seq_len + threads - 1) / threads, threads>>>(q_hat, q_pe, kv_cache, s, seq_len, kv_lora_rank, rope_dim, scale);

        // a = softmax(s)
        softmax<<<1, threads>>>(s, 1, seq_len);
        
        // l = sum_t a[t] * c[t]  -> (kv_lora_rank)
        compute_l<<<(kv_lora_rank + threads - 1) / threads, threads>>>(s, kv_cache, l, seq_len, kv_lora_rank, rope_dim);
        
        // out[h] = l · W_UV[h]  -> (1, head_dim)
        matmul<false, false><<<grid2, block>>>(l, w_uv, output + h * head_dim, 1, head_dim, kv_lora_rank, 1.0f);
    }
    
    cudaFree(q_hat);
    cudaFree(s);
    cudaFree(l);
}
