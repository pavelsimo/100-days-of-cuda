#include <cuda_runtime.h>

__global__ void top_k(const float* in, float* out, int* out_beam, int* out_tok, int N, int K, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    float value = in[i];
    int pos = 0;
    for (int j = 0; j < N && pos < K; ++j) {
        float w = in[j];
        pos += (w > value) || (w == value && j > i);
    }

    if (pos >= K) {
        return;
    }

    out[pos] = value;
    out_beam[pos] = i / V;
    out_tok[pos] = i % V;
}

__global__ void init_candidates(const float* beam_scores, const float* token_logprobs, float* cand, int B, int K, int V) {
    int bkv = blockIdx.x * blockDim.x + threadIdx.x;
    if (bkv >= B * K * V) {
        return;
    }
    int bk = bkv / V;
    cand[bkv] = beam_scores[bk] + token_logprobs[bkv];
}

// beam_scores, token_logprobs, new_beam_scores, parent_beam_indices, next_tokens are device
// pointers
extern "C" void solve(const float* beam_scores, const float* token_logprobs, float* new_beam_scores,
                      int* parent_beam_indices, int* next_tokens, int B, int K, int V) {
    int num_candidates = B * K * V;
    int threads = 256;
    int blocks = (num_candidates + threads - 1) / threads;
    float *cand;
    cudaMalloc(&cand, B * K * V * sizeof(float));

    init_candidates<<<blocks, threads>>>(beam_scores, token_logprobs, cand, B, K, V);
    for (int b = 0; b < B; ++b) {
        const float* cur_cand = cand + b * K * V;
        int N = K * V;
        top_k<<<(N + threads - 1) / threads, threads>>>(
            cur_cand,
            new_beam_scores + b * K,
            parent_beam_indices + b * K,
            next_tokens + b * K,
            N,
            K,
            V
        );
    }
    
    cudaDeviceSynchronize();
    cudaFree(cand);
}
