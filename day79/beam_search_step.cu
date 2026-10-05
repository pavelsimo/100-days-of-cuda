#include <cuda_runtime.h>

__global__ void rank_sort(const float* in, float* out, int* out_beam, int* out_tok, int N, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    float value = in[i];
    int pos = 0;
    for (int j = 0; j < N; ++j) {
        float w = in[j];
        pos += (w > value) || (w == value && j > i);
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
    float *buffer_new_beam_scores;
    int *buffer_parent_beam_indices;
    int *buffer_next_tokens;
    cudaMalloc(&cand, B * K * V * sizeof(float));
    cudaMalloc(&buffer_new_beam_scores, K * V * sizeof(float));
    cudaMalloc(&buffer_parent_beam_indices, K * V * sizeof(int));
    cudaMalloc(&buffer_next_tokens, K * V * sizeof(int));

    init_candidates<<<blocks, threads>>>(beam_scores, token_logprobs, cand, B, K, V);
    for (int b = 0; b < B; ++b) {
        const float* cur_cand = cand + b * K * V;
        float* batch_new_beam_scores = new_beam_scores + b * K;
        int* batch_parent_beam_indices = parent_beam_indices + b * K;
        int* batch_next_tokens = next_tokens + b * K;
        int N = K * V;
        rank_sort<<<(N + threads - 1) / threads, threads>>>(
            cur_cand,
            buffer_new_beam_scores, 
            buffer_parent_beam_indices,
            buffer_next_tokens,
            N,
            V
        );
        cudaMemcpy(batch_new_beam_scores, buffer_new_beam_scores, K * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(batch_parent_beam_indices, buffer_parent_beam_indices, K * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(batch_next_tokens, buffer_next_tokens, K * sizeof(int), cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize();
    cudaFree(cand);
    cudaFree(buffer_new_beam_scores);
    cudaFree(buffer_parent_beam_indices);
    cudaFree(buffer_next_tokens);
}
