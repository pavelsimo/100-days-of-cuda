#include <cuda_runtime.h>
#include <float.h>

#define BLOCKSIZE 16
#define MAX_CONTENT_LEN 8192

__device__ __forceinline__ float dot_product(const float* A, const float* B, int N) {
    float result = 0.0f;
    for (int i = 0; i < N; ++i) {
        result += A[i] * B[i];
    }
    return result;
}

__device__ __forceinline__ void softmax(float* scores, int N) {
    float row_max = -FLT_MAX;
    for (int i = 0; i < N; ++i) {
        row_max = fmaxf(row_max, scores[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        scores[i] = expf(scores[i] - row_max);
        sum += scores[i];
    }

    const float inv = 1.0f / sum;
    for (int i = 0; i < N; ++i) {
        scores[i] *= inv;
    }
}

__global__ void paged_attention(const float* Q, const float* K_cache, const float* V_cache,
                                const int* block_table, const int* context_lens, float* output,
                                int batch_size, int num_heads, int head_dim, int block_size,
                                int max_blocks_per_seq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_heads) {
        return;
    }

    int s = idx / num_heads;
    int h = idx % num_heads;
    int T = context_lens[s];
    float scores[MAX_CONTENT_LEN];

    const float* q = Q + s * (num_heads * head_dim) + h * head_dim;
    float scale = 1.0f / sqrtf(head_dim);
    for (int t = 0; t < T; ++t) {
        int blk_logical = t / block_size;
        int blk_slot = t % block_size;
        int blk_physical = block_table[s * max_blocks_per_seq + blk_logical];
        int blk_offset = blk_physical * (block_size * num_heads * head_dim) 
                       + blk_slot * (num_heads * head_dim) 
                       + h * head_dim;
        
        const float* k = K_cache + blk_offset;
        const float* v = V_cache + blk_offset;
        scores[t] = dot_product(q, k, head_dim) * scale;
    }

    softmax(scores, T);

    float* out = output + s * (num_heads * head_dim) + h * head_dim;
    for (int t = 0; t < T; ++t) {
        int blk_logical = t / block_size;
        int blk_slot = t % block_size;
        int blk_physical = block_table[s * max_blocks_per_seq + blk_logical];
        int blk_offset = blk_physical * (block_size * num_heads * head_dim)
                       + blk_slot * (num_heads * head_dim)
                       + h * head_dim;
        
        const float* v = V_cache + blk_offset;
        float w = scores[t];
        for (int d = 0; d < head_dim; ++d) {
            out[d] += w * v[d];
        }
    }
}

// Q, K_cache, V_cache, block_table, context_lens, output are device pointers
extern "C" void solve(const float* Q, const float* K_cache, const float* V_cache,
                      const int* block_table, const int* context_lens, float* output,
                      int batch_size, int num_heads, int head_dim, int block_size,
                      int max_blocks_per_seq) {
    // Q - [batch_size, num_heads, head_dim]
    // K_cache - [num_blocks, block_size, num_heads, head_dim]
    // V_cache - [num_blocks, block_size, num_heads, head_dim]
    // block_table - [batch_size, max_blocks_per_seq]
    // context_lens - [batch_size]
    // output - [batch_size, num_heads, head_dim]
    int threads = 256;
    int blocks = (batch_size * num_heads + threads - 1) / threads;
    paged_attention<<<blocks, threads>>>(Q, K_cache, V_cache, block_table, context_lens, output,
                                         batch_size, num_heads, head_dim, block_size, max_blocks_per_seq);
    cudaDeviceSynchronize();
}
