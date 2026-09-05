#include <cuda_runtime.h>
#include <float.h>

#define MAX_WINDOW 32

__global__ void sliding_window_self_attention(const float* Q, const float* K, const float* V, float* output, int M, int d, int window_size, float alpha) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= M) {
        return;
    }
    
    const int min_j = max(0, i - window_size);
    const int max_j = min(M - 1, i + window_size);
    float scores[2 * MAX_WINDOW + 1];

    // Q @ K^T / sqrt(d)
    float max_score = -FLT_MAX;
    for (int j = min_j; j <= max_j; ++j) {
        float score = 0.0f;
        for (int k = 0; k < d; ++k) {
            score += Q[i * d + k] * K[j * d + k];
        }
        score *= alpha;
        scores[j - min_j] = score;
        max_score = fmaxf(max_score, score);
    }

    // scores = softmax(Q @ K^T / sqrt(d))
    float sum_exp = 0.0f;
    for (int j = min_j; j <= max_j; ++j) {
        float exp_score = expf(scores[j - min_j] - max_score);
        scores[j - min_j] = exp_score;
        sum_exp += exp_score;
    }
    
    // scores @ V
    const float inv_sum = 1.0f / sum_exp;
    for (int k = 0; k < d; ++k) {
        float sum = 0.0f;
        for (int j = min_j; j <= max_j; ++j) {
            sum += scores[j - min_j] * V[j * d + k];
        }
        output[i * d + k] = sum * inv_sum;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d,
                      int window_size) {
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    float alpha = 1.0f / sqrtf((float)d);
    sliding_window_self_attention<<<blocks, threads>>>(Q, K, V, output, M, d, window_size, alpha);
}
