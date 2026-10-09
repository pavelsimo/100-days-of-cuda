#include <cuda_runtime.h>
#include <float.h>

#define TILE_SIZE 16

__device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float block_reduce_sum(float val) {
    __shared__ float t[32];
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    val = warp_reduce_sum(val);
    if (lane == 0) {
        t[warpId] = val;
    }
    __syncthreads();
    if (warpId == 0) {
        int numWarps = (blockDim.x + warpSize - 1) / warpSize;
        val = (threadIdx.x < numWarps) ? t[lane]: 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void softmax(float* scores, int M, int N) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float t[];

    if (row >= M) {
        return;
    }

    float thread_max = -FLT_MAX;
    for (int col = tid; col < N; col += blockDim.x) {
        thread_max = fmaxf(thread_max, scores[row * N + col]);
    }
    t[tid] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] = fmaxf(t[tid], t[tid + stride]);
        }
        __syncthreads();
    }

    const float row_max = t[0];
    float thread_sum = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        const float e = expf(scores[row * N + col] - row_max);
        scores[row * N + col] = e;
        thread_sum += e;
    }
    t[tid] = thread_sum;
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

__device__ __forceinline__ int select_max(const float *scores, int *selected, int N, int K) {
    float best_score = -INFINITY;
    int best_idx = -1;
    for (int i = 0; i < N; ++i) {        
        bool ok = true;
        for (int k = 0; k < K; ++k) {
            if (selected[k] == i) {
                ok = false;
                break;
            }
        }

        if (!ok) {
            continue;
        }

        if (scores[i] > best_score) {
            best_score = scores[i];
            best_idx = i;
        }
    }
    return best_idx;
}

__global__ void compute_scores(const float* K, const float *Q, float *scores, int num_heads, int seq_len, int head_dim, 
    int block_size, int num_selected, int num_blocks) {
    
    int b = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;
    int start = b * block_size;
    int end = min((b + 1) * block_size, seq_len);
    float block_score = 0.0f;
    if (d < head_dim) {
        float sum = 0.0f;
        for (int t = start; t < end; ++t) {
            int idx = h * (seq_len * head_dim) + t * head_dim + d;
            sum += K[idx];
        }
        float k = sum / (end - start);
        float q = Q[h * head_dim + d];
        block_score = k * q;
    }

    float score = block_reduce_sum(block_score);
    if (d == 0) {
        scores[h * num_blocks + b] = score;
    }
}

__global__ void select_top_blocks(const float *scores, int *top_indices, int num_heads, int num_blocks, int num_selected) {
    int N = min(num_selected, num_blocks);
    const float *head_scores = scores + blockIdx.x * num_blocks;
    int *head_top = top_indices + blockIdx.x * N;
    for (int i = 0; i < N; ++i) {
        head_top[i] = select_max(head_scores, head_top, num_blocks, i);
    }
}

__global__ void mask_skipped_blocks(float *S, const int *top_indices, int seq_len, int block_size, int num_selected) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    const int *head_top = top_indices + h * num_selected;
    for (int i = 0; i < num_selected; ++i) {
        if (head_top[i] == b) {
            return;
        }
    }
    
    int start = b * block_size;
    int end = min((b + 1) * block_size, seq_len);
    for (int t = start + threadIdx.x; t < end; t += blockDim.x) {
        S[h * seq_len + t] = -INFINITY;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int num_heads,
                      int seq_len, int head_dim, int block_size, int num_selected) {

    // Q - [num_heads, head_dim] - single query per head
    // K - [num_heads, seq_len, head_dim] 
    // V - [num_heads, seq_len, head_dim]
    // scale factor - 1 / sqrt(head_dim)
    // num_blocks = ceil(seq_len / block_size)
    // block b cover positions - [b * block_size, min((b + 1) * block_size, seq_len))]
    // 
    // score block of head:
    // Q[h] * mean(K[h, t] for block in b)
    // keep -> min(num_selected, num_blocks) highest scoring blocks
    // smallest block index in case of ties
    //
    
    float *scores, *S;
    int *top_indices;
    int num_blocks = (seq_len + block_size - 1) / block_size;
    int num_selected_blocks = min(num_selected, num_blocks);
    cudaMalloc(&scores, num_heads * num_blocks * sizeof(float));
    cudaMalloc(&top_indices, num_heads * num_selected_blocks * sizeof(int));
    cudaMalloc(&S, num_heads * seq_len * sizeof(float));
    
    int threads = 256;
    float scale_factor = 1.0f / sqrtf(head_dim);
    dim3 grid_1(num_blocks, num_heads);
    compute_scores<<<grid_1, threads>>>(K, Q, scores, num_heads, seq_len, head_dim, block_size, num_selected, num_blocks);
    select_top_blocks<<<num_heads, 1>>>(scores, top_indices, num_heads, num_blocks, num_selected);

    dim3 tile(TILE_SIZE, TILE_SIZE);
    dim3 grid_2((seq_len + TILE_SIZE - 1) / TILE_SIZE, 1);
    for (int h = 0; h < num_heads; ++h) {
        const float *Q_h = Q + h * head_dim;
        const float *K_h = K + h * seq_len * head_dim;
        float *S_h = S + h * seq_len;
        // S_h = scale * Q_h @ K_h^T
        matmul<false, true><<<grid_2, tile>>>(Q_h, K_h, S_h, 1, seq_len, head_dim, scale_factor);
    }

    mask_skipped_blocks<<<grid_1, threads>>>(S, top_indices, seq_len, block_size, num_selected_blocks);
    softmax<<<num_heads, threads, threads * sizeof(float)>>>(S, num_heads, seq_len);
    
    dim3 grid_3((head_dim + TILE_SIZE - 1) / TILE_SIZE, 1);
    for (int h = 0; h < num_heads; ++h) {
        const float *P_h = S + h * seq_len;
        const float *V_h = V + h * seq_len * head_dim;
        float *O_h = output + h * head_dim;
        // O_h = P_h @ V_h
        matmul<false, false><<<grid_3, tile>>>(P_h, V_h, O_h, 1, head_dim, seq_len, 1.0f);
    }

    cudaDeviceSynchronize();
    cudaFree(S);
    cudaFree(scores);
    cudaFree(top_indices);
}
