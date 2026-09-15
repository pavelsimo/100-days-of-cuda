#include <cuda_runtime.h>

__device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
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
        val = (threadIdx.x < numWarps) ? t[lane] : 0.0f;
        val = warp_reduce_sum(val); 
    }
    return val;
}

__global__ void sum_embedding(const int* token_ids, const int* position_ids, 
                              const float* token_embeddings, const float* position_embeddings, 
                              float *s, int B, int T, int D) {
    int b = blockDim.x * blockIdx.x + threadIdx.x;
    int t = blockDim.y * blockIdx.y + threadIdx.y;
    if (b < B && t < T) {
        for (int d = 0; d < D; ++d) {
            float et = token_embeddings[token_ids[b * T + t] * D + d];
            float ep = position_embeddings[position_ids[t] * D + d];
            s[b * (T * D) + t * D + d] = et + ep;
        }
    }
}

__global__ void bt_mean(float *s, float *mean, int B, int T, int D) {
    float sum = 0.0f;
    int bt = blockIdx.x;
    for(int d = threadIdx.x; d < D; d += blockDim.x) {
        sum += s[bt * D + d];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        mean[bt] = sum / D;
    }
}

__global__ void bt_var(float *s, float *mean, float *var, int B, int T, int D) {
    float sum = 0.0f;
    int bt = blockIdx.x;
    for(int d = threadIdx.x; d < D; d += blockDim.x) {
        float diff = s[bt * D + d] - mean[bt];
        sum += diff * diff;
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        var[bt] = sum / D;
    }
}

__global__ void token_emb_layer(float *s, float *mean, float *var, float *output, float eps, const float *gamma, const float *beta, int B, int T, int D) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < B*T*D) {
        int d = idx % D; 
        int b = idx / (T*D);
        int t = (idx / D) % T;
        float y = gamma[d] * (s[idx] - mean[b * T + t]) / sqrtf(var[b * T + t] + eps) + beta[d]; 
        output[idx] = y;
    }
}

// token_ids, position_ids, token_embeddings, position_embeddings, gamma, beta, output are device
// pointers
extern "C" void solve(const int* token_ids, const int* position_ids, const float* token_embeddings,
                      const float* position_embeddings, const float* gamma, const float* beta,
                      float* output, int B, int T, int V, int P, int D, float eps) {
    // token ids - B x T
    // position ids - T
    // token embeddings - V x D
    // position embeddings - P x D
    // gamma - D
    // beta - D
    // output - B x T x D
    float *s, *mean, *var;
    cudaMalloc(&s, B * T * D * sizeof(float));
    cudaMalloc(&mean, B * T * sizeof(float));
    cudaMalloc(&var, B * T * sizeof(float));
    dim3 threads(16, 16);
    dim3 grid(
        (B + threads.x - 1) / threads.x,
        (T + threads.y - 1) / threads.y
    );                    
    sum_embedding<<<grid, threads>>>(token_ids, position_ids, token_embeddings, position_embeddings, s, B, T, D);
    bt_mean<<<B * T, 256>>>(s, mean, B, T, D);
    bt_var<<<B * T, 256>>>(s, mean, var, B, T, D);
    token_emb_layer<<<(B*T*D + 255) / 256, 256>>>(s, mean, var, output, eps, gamma, beta, B, T, D);
    
    cudaDeviceSynchronize();
    cudaFree(s);
    cudaFree(mean);
    cudaFree(var);
}
