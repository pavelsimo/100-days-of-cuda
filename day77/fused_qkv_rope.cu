#include <cuda_runtime.h>

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

__device__ __forceinline__ void rope(const float* cos_sin_cache, float* t, int pos, int j, float a, float b, int D) {
    float cj = cos_sin_cache[pos * D + j];
    float sj = cos_sin_cache[pos * D + D/2 + j];
    t[j] = a * cj - b * sj;
    t[j + D/2] = b * cj + a * sj;
}

__global__ void fused_qkv_rope(const float* x, const float* W_qkv, const float* cos_sin_cache,
                               const int* positions, float* K_cache, float* V_cache, float* Q_out,
                               int B, int d_model, int H_q, int H_kv, int D, int S_max, const float* qkv) { 
    int b = blockIdx.x;
    int h = blockIdx.y;
    if (b >= B || h >= H_q + H_kv) {
        return;
    }

    int qkv_dim = (H_q + 2 * H_kv) * D;

    // step 1: get new token
    const float* qkv_b = qkv + b * qkv_dim;

    // step 2: split Q, K, V
    const float* Q = qkv_b;
    const float* K = qkv_b + H_q * D;
    const float* V = qkv_b + (H_q + H_kv) * D;

    // step 3: get current token position
    int pos = positions[b];

    if (h < H_q) {
        // Q head
        const float* q_head = Q + h * D;
        float *q_out = Q_out + b * (H_q * D) + h * D;
        for (int j = threadIdx.x; j < D/2; j += blockDim.x) {
            float q0 = q_head[j];
            float q1 = q_head[j + D/2];
            rope(cos_sin_cache, q_out , pos, j, q0, q1, D);
        }
    } else {
        // K/V head
        int kv_h = h - H_q;
        const float *k_head = K + kv_h * D;
        const float *v_head = V + kv_h * D;
        float* k_cache = K_cache + b * (H_kv * S_max * D) + kv_h * (S_max * D) + pos * D;
        float* v_cache = V_cache + b * (H_kv * S_max * D) + kv_h * (S_max * D) + pos * D;
        for (int j = threadIdx.x; j < D/2; j += blockDim.x) {
            float k0 = k_head[j];
            float k1 = k_head[j + D/2];
            rope(cos_sin_cache, k_cache , pos, j, k0, k1, D);
            v_cache[j]       = v_head[j];
            v_cache[j + D/2] = v_head[j + D/2];
        }
    }
}

// x, W_qkv, cos_sin_cache, positions, K_cache, V_cache, Q_out are device pointers
extern "C" void solve(const float* x, const float* W_qkv, const float* cos_sin_cache,
                      const int* positions, float* K_cache, float* V_cache, float* Q_out, int B,
                      int d_model, int H_q, int H_kv, int D, int S_max) {

    // x               - [B, d_model]
    // W_qkv           - [d_model (H_q + 2 * H_kv) * D]
    // cos_sin_cache   - [S_max, D]
    // positions       - [B]
    // K_cache         - [B, H_kv, S_max, D]
    // V_cache         - [B, H_kv, S_max, D]
    // Q_out           - [B, H_q, D]
    // B               - batch size
    // d_model         - hidden state dimension
    // H_q             - number of query heads
    // H_kv            - number of key/value heads
    // D               - head dimension
    // S_max           - maximum sequence length
    
    float *qkv;
    int qkv_dim = (H_q + 2 * H_kv) * D;
    int total = B * qkv_dim;
    cudaMalloc(&qkv, total * sizeof(float));
    
    dim3 threads_1(TILE_SIZE, TILE_SIZE);
    dim3 blocks_1(
        (qkv_dim + TILE_SIZE - 1) / TILE_SIZE, 
        (B + TILE_SIZE - 1) / TILE_SIZE
    );
    matmul<<<blocks_1, threads_1>>>(x, W_qkv, qkv, B, qkv_dim, d_model, 1.0f);
    
    dim3 threads_2(128);
    dim3 blocks_2(
        B, 
        H_q + H_kv
    );
    fused_qkv_rope<<<blocks_2, threads_2>>>(x, W_qkv, cos_sin_cache, positions, K_cache, V_cache, Q_out,
                                        B, d_model, H_q, H_kv, D, S_max, qkv);
    
    cudaDeviceSynchronize();
    cudaFree(qkv);
}
