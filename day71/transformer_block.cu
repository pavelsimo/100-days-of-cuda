#include <cuda_runtime.h>
#include <float.h>

#define TILE_SIZE 16

// utilities 

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

__global__ void row_mean(const float* input, float* mean, int N, int C) {
    int row = blockIdx.x;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < C; col += blockDim.x) {
        sum += input[row * C + col];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        mean[row] = sum / C;
    }
}

__global__ void row_var(const float* input, const float* mean, float* var, int N, int C) {
    int row = blockIdx.x;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < C; col += blockDim.x) {
        float d = input[row * C + col] - mean[row];
        sum += d * d;
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        var[row] = sum / C;
    }
}

__global__ void gelu(float* x, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        float val = x[idx];
        x[idx] = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
    }
}

// transformer block functions

__global__ void layer_norm(const float* input, const float* gamma, const float* beta, float* output,
                      const float* mean, const float* var, int N, int C, float eps) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N * C) {
        int j = idx % C;
        int i = idx / C;
        output[i * C + j] = (input[i * C + j] - mean[i]) / sqrtf(var[i] + eps) * gamma[j] + beta[j];
    } 
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

template <bool TRANS_A = false, bool TRANS_B = false, bool ADD_C = false>
__global__ void gemm(const float *A, const float *B, float *AB, const float *C, int M, int N, int K, float alpha, float beta, int lda = 0, int ldb = 0, int ldc = 0) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    if (lda <= 0) lda = TRANS_A ? M : K;
    if (ldb <= 0) ldb = TRANS_B ? K : N;
    if (ldc <= 0) ldc = N;

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
        if constexpr (ADD_C) {
            AB[i * ldc + j] = alpha * sum + beta * C[j];
        } else {
            AB[i * ldc + j] = alpha * sum;
        }
    }
}

__global__ void matadd(const float *A, const float *B, float *C, int total) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx + 3 < total) {
        float4 a = *reinterpret_cast<const float4*>(A + idx);
        float4 b = *reinterpret_cast<const float4*>(B + idx);
        *reinterpret_cast<float4*>(C + idx) = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    } else {
        for (int k = idx; k < total; ++k) {
            C[k] = A[k] + B[k];
        }
    }
}

// x, output, weights are device pointers
extern "C" void solve(const float* x, float* output, const float* weights, int seq_len) {
    // (768,)
    const float *gamma = weights;
    // (768,)
    const float *beta  = weights + 768;
    // (768, 2304)
    const float *Wqkv  = weights + 1536;
    // (2304,)
    const float *Bqkv  = weights + 1771008;
    // (768, 768)
    const float *Wattn = weights + 1773312;
    // (768,)
    const float *Battn = weights + 2363136;
    // (768,)
    const float *gamma2 = weights + 2363904;
    // (768,)
    const float *beta2 = weights + 2364672;
    // (768, 3072)
    const float *Wfc = weights + 2365440;
    // (3072,)
    const float *Bfc = weights + 4724736;
    // (3072, 768)
    const float *Wproj = weights + 4727808;
    // (768,)
    const float *Bproj = weights + 7087104;

    int threads = 256;
    int N = seq_len;
    int C = 768;
    float eps = 1e-5;
    int dk = 64;
    float alpha = 1.0f / sqrtf(dk);
    int H = C / dk;
    int D = C / H;

    float *x_mean, *x_var, *h_mean, *h_var;
    float *x_norm, *h_norm;
    float *qkv;
    float *att;
    float *concat_att;
    float *P, *x_hat;
    float *F;
    float *F_proj;

    cudaMalloc(&x_mean, N * sizeof(float));
    cudaMalloc(&x_var, N * sizeof(float));
    cudaMalloc(&h_mean, N * sizeof(float));
    cudaMalloc(&h_var, N * sizeof(float));
    cudaMalloc(&x_norm, N * C * sizeof(float));
    cudaMalloc(&h_norm, N * C * sizeof(float));
    cudaMalloc(&qkv, N * 3 * C * sizeof(float));
    cudaMalloc(&att, N * N * sizeof(float));
    cudaMalloc(&concat_att, N * C * sizeof(float));
    cudaMalloc(&P, N * C * sizeof(float));
    cudaMalloc(&x_hat, N * C * sizeof(float));
    cudaMalloc(&F, N * 4 * C * sizeof(float));
    cudaMalloc(&F_proj, N * C * sizeof(float));

    const float *Q = qkv;
    const float *K = qkv + C;
    const float *V = qkv + 2 * C;

    // [N, C]: x_norm
    row_mean<<<N, threads>>>(x, x_mean, N, C);
    row_var<<<N, threads>>>(x, x_mean, x_var, N, C);
    layer_norm<<<(N * C + threads - 1) / threads, threads>>>(x, gamma, beta, x_norm, x_mean, x_var, N, C, eps);
    
    // [N, 3 * C]: x_norm @ Wqkv + Bqkv = qkv
    dim3 threads_1(16, 16);
    dim3 grid_1(
        (3 * C + threads_1.x - 1) / threads_1.x,
        (N + threads_1.y - 1) / threads_1.y
    );
    gemm<false, false, true><<<grid_1, threads_1>>>(x_norm, Wqkv, qkv, Bqkv, N, 3 * C, C, 1.0f, 1.0f);

    // [N, N]: Q_head @ K_head^T
    dim3 threads_2(16, 16);
    dim3 grid_2(
        (N + threads_2.x - 1) / threads_2.x,
        (N + threads_2.y - 1) / threads_2.y
    );

    // [N, D]: softmax(att) @ V_head
    dim3 threads_3(16, 16);
    dim3 grid_3(
        (D + threads_3.x - 1) / threads_3.x,
        (N + threads_3.y - 1) / threads_3.y
    );

    for (int head = 0; head < H; ++head) {
        const float *Q_head = Q + head * D;
        const float *K_head = K + head * D;
        const float *V_head = V + head * D;
        float *score_head = concat_att + (head * D);
        
        // Q @ K^T -> att
        gemm<false, true, false><<<grid_2, threads_2>>>(Q_head, K_head, att, nullptr, N, N, D, alpha, 0.0f, 3 * C, 3 * C);
        
        // softmax(att)
        softmax<<<N, threads, threads * sizeof(float)>>>(att, N, N);
        
        // att @ V_head
        gemm<false, false, false><<<grid_3, threads_3>>>(att, V_head, score_head, nullptr, N, D, N, 1.0f, 0.0f, N, 3 * C, C);
    }

    // P = concat_att @ Wattn + Battn
    dim3 threads_4(16, 16);
    dim3 grid_4(
        (C + threads_4.x - 1) / threads_4.x,
        (N + threads_4.y - 1) / threads_4.y
    );
    gemm<false, false, true><<<grid_4, threads_4>>>(concat_att, Wattn, P, Battn, N, C, C, 1.0f, 1.0f);

    // x_hat = x + P
    matadd<<<(N * C + threads - 1) / threads, threads>>>(x, P, x_hat, N * C);
    
    // [N, C]: h_norm
    row_mean<<<N, threads>>>(x_hat, h_mean, N, C);
    row_var<<<N, threads>>>(x_hat, h_mean, h_var, N, C);
    layer_norm<<<(N * C + threads - 1) / threads, threads>>>(x_hat, gamma2, beta2, h_norm, h_mean, h_var, N, C, eps);

    // [N, 4 * C]: F = GELU(h_norm @ Wfc + Bfc)
    dim3 threads_5(16, 16);
    dim3 grid_5(
        (4 * C + threads_5.x - 1) / threads_5.x,
        (N + threads_5.y - 1) / threads_5.y
    );
    gemm<false, false, true><<<grid_5, threads_5>>>(h_norm, Wfc, F, Bfc, N, 4 * C, C, 1.0f, 1.0f);
    gelu<<<(N * 4 * C + threads - 1) / threads, threads>>>(F, N * 4 * C);

    // [N, C]: F_proj = F @ Wproj + Bproj
    dim3 threads_6(16, 16);
    dim3 grid_6(
        (C + threads_6.x - 1) / threads_6.x,
        (N + threads_6.y - 1) / threads_6.y
    );
    gemm<false, false, true><<<grid_6, threads_6>>>(F, Wproj, F_proj, Bproj, N, C, 4 * C, 1.0f, 1.0f);
    
    // output = x_hat + F_proj
    matadd<<<(N * C + threads - 1) / threads, threads>>>(x_hat, F_proj, output, N * C);

    cudaFree(x_mean);
    cudaFree(x_var);
    cudaFree(h_mean);
    cudaFree(h_var);
    cudaFree(x_norm);
    cudaFree(h_norm);
    cudaFree(qkv);
    cudaFree(att);
    cudaFree(concat_att);
    cudaFree(P);
    cudaFree(x_hat);
    cudaFree(F);
    cudaFree(F_proj);
}
