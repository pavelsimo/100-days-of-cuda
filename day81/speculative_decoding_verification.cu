#include <cuda_runtime.h>

__global__ void acceptance_prob(const int* draft_tokens, const float* draft_probs, const float* target_probs,
                       const float* uniform_samples, int* first_rejection, int B, int T, int V) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) {
        return;   
    }

    int rejection = T;
    for (int i = 0; i < T; ++i) {
        int bi = b * T + i;
        int t = draft_tokens[bi];
        int bit = b * (T * V) + i * V + t;
        float alpha = fminf(1.0f, target_probs[bit] / draft_probs[bit]);
        float u = uniform_samples[b * (T + 1) + i];
        if (u >= alpha) {
            rejection = i;
            break;
        }
    }
    first_rejection[b] = rejection;
}

__global__ void resample(const int* first_rejection, const int* draft_tokens, const float* draft_probs, const float* target_probs, const float* uniform_samples, int* output_tokens, int B, int T, int V) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) {
        return;   
    }

    int k = first_rejection[b];

    // keep tokens before the first rejection
    for (int i = 0; i < k; ++i) {
        output_tokens[b * (T + 1) + i] = draft_tokens[b * T + i];
    }

    // initialize tokens after the first rejection to 0
    for (int i = k; i <= T; ++i) {
        output_tokens[b * (T + 1) + i] = 0;
    }

    if (k < T) {
        float sum_d = 0.0f;
        for (int v = 0; v < V; ++v) {
            int btv = b * (T * V) + k * V + v;
            float d = fmaxf(0.0f, target_probs[btv] - draft_probs[btv]);
            sum_d += d;
        }
        
        float cdf = 0.0f;
        int token = V - 1;
        float r = uniform_samples[b * (T + 1) + T];
        for (int v = 0; v < V; ++v) {
            int btv = b * (T * V) + k * V + v;
            float d = fmaxf(0.0f, target_probs[btv] - draft_probs[btv]);
            float p = (sum_d > 0.0f) ? d / sum_d : 1.0f / V;
            cdf += p;
            if (cdf >= r) {
                token = v;
                break;
            }
        }
        output_tokens[b * (T + 1) + k] = token;
    } else {
        // sample bonus token
        float cdf = 0.0f;
        int token = V - 1;
        float r = uniform_samples[b * (T + 1) + T];
        for (int v = 0; v < V; ++v) {
            int btv = b * (T * V) + (T - 1) * V + v;
            cdf += target_probs[btv];
            if (cdf >= r) {
                token = v;
                break;
            }
        }
        output_tokens[b * (T + 1) + T] = token;
    }
}

// draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens are device pointers
extern "C" void solve(const int* draft_tokens, const float* draft_probs, const float* target_probs,
                      const float* uniform_samples, int* output_tokens, int B, int T, int V) {
    // draft_tokens: B x T         -> ti
    // draft_probs: B x T x V      -> pi(v)
    // target_probs: B x T x V     -> qi(v)
    // uniform_samples: B x T      ->  ui
    // output_tokens: B x (T + 1)
    
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    int *first_rejection;
    cudaMalloc(&first_rejection, B * sizeof(int));
    acceptance_prob<<<blocks, threads>>>(draft_tokens, draft_probs, target_probs, uniform_samples, first_rejection, B, T, V);
    resample<<<blocks, threads>>>(first_rejection, draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens, B, T, V);
    cudaFree(first_rejection);
}
