#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float dot(const float* a, const float* b, int N) {
    float res = 0.0f;
    for (int i = 0; i < N; ++i) {
        res += a[i] * b[i];
    }
    return res;
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ void softmax(float* scores, int N) {
    float max_score = -1e20f;
    for (int i = 0; i < N; ++i) {
        max_score = fmaxf(max_score, scores[i]);
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < N; ++i) {
        scores[i] = expf(scores[i] - max_score);
        sum_exp += scores[i];
    }
    for (int i = 0; i < N; ++i) {
        scores[i] /= sum_exp;
    }
}

__device__ __forceinline__ void rmsnorm(float* x, int N, float eps) {
    float sum_sqr = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum_sqr += x[i] * x[i];
    }
    float rms = sqrtf(sum_sqr / N + eps);
    for (int i = 0; i < N; ++i) {
        x[i] = x[i] / rms;
    }
}

__device__ __forceinline__ void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

__device__ __forceinline__ void rope(float* x, int p, float ang_freq) {
    float x0 = x[0];
    float x1 = x[1];
    float angle = p * ang_freq;
    float t1 = x0 * cosf(angle) - x1 * sinf(angle);
    float t2 = x0 * sinf(angle) + x1 * cosf(angle);
    x[0] = t1;
    x[1] = t2;
}

__global__ void adder_kernel(const int* prompts, float* output, const float* weights, int batch_size) {
    const float w0 = weights[0];
    const float w1 = weights[1];
    const float q0 = weights[2];
    const float q1 = weights[3];
    const float v0 = weights[4];
    const float a = weights[5];
    const float c = weights[6];
    const float carry = weights[7];
    const float n0 = weights[8];
    const float n1 = weights[9];
    const int vocab_size = 10;
    const int hidden_dim = 2;
    const int head_dim = 2;
    const int num_heads = 1;
    const int prompt_len = 31;
    const int decode_steps = 11;
    const float eps = 1e-6f;
    const float ang_freq = 2 * M_PI  / 19.0f;
    const float s_sqr = sqrtf(logf(10) / (sqrtf(2.0f) * (cosf(0.3f * ang_freq) - cosf(0.7f * ang_freq))));
    const float att_scale = s_sqr / sqrtf(head_dim);

    const int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batch_size) {
        return;
    }
    
    const int max_len = prompt_len + decode_steps;
    int tokens[max_len];
    for (int p = 0; p < prompt_len; ++p) {
        tokens[p] = prompts[batch * prompt_len + p];
    }

    for (int i = 0; i < decode_steps; ++i) {
        const int seq_len = prompt_len + i;
        const int i_q = seq_len - 1;

        // step 1: encode the tokens
        float h[max_len * hidden_dim];
        for (int p = 0; p < seq_len; ++p) {
            int d = tokens[p];
            h[p * hidden_dim + 0] = w0 - w1 * d * d;
            h[p * hidden_dim + 1] = -d;
        }

        // step 2: x = RMSNorm(h), per token
        float x[max_len * hidden_dim];
        for (int p = 0; p < seq_len; ++p) {
            x[p * hidden_dim + 0] = h[p * hidden_dim + 0];
            x[p * hidden_dim + 1] = h[p * hidden_dim + 1];
            rmsnorm(&x[p * hidden_dim], hidden_dim, eps);
        }

        // step 3: construct Q, K, V from x
        float Q[max_len * hidden_dim];
        float K[max_len * hidden_dim];
        float V[max_len * hidden_dim];
        for (int p = 0; p < seq_len; ++p) {
            float x0 = x[p * hidden_dim + 0];
            float x1 = x[p * hidden_dim + 1];
            Q[p * hidden_dim + 0] = x0 * q0;
            Q[p * hidden_dim + 1] = x0 * q1;
            K[p * hidden_dim + 0] = x0;
            K[p * hidden_dim + 1] = 0;
            V[p * hidden_dim + 0] = x1 * v0;
            V[p * hidden_dim + 1] = 0;
        }

        // step 4: RMSNorm for Q and K, per token
        for (int p = 0; p < seq_len; ++p) {
            rmsnorm(&Q[p * hidden_dim], hidden_dim, eps);
            rmsnorm(&K[p * hidden_dim], hidden_dim, eps);
        }

        // step 5: RoPE for Q and K
        for (int p = 0; p < seq_len; ++p) {
            rope(&Q[p * hidden_dim], p, ang_freq);
            rope(&K[p * hidden_dim], p, ang_freq);
        }

        // step 6: dot product attention
        const float* q = &Q[i_q * hidden_dim];
        float scores[max_len];
        for (int j = 0; j < seq_len; ++j) {
            scores[j] = dot(q, &K[j * hidden_dim], head_dim) * att_scale;
        }

        // step 7: causal mask
        for (int j = 0; j < seq_len; ++j) {
            if (j > i_q) {
                scores[j] = -INFINITY;
            }
        }

        // step 8: softmax over the attention scores
        softmax(scores, seq_len);

        // step 9: score @ V
        float attn[head_dim];
        matmul(scores, V, attn, 1, seq_len, hidden_dim);

        // step 10: attn. projection
        attn[1] = attn[0];
        attn[0] = 0.0f;

        // step 11: residual conn.
        float* h_q = &h[i_q * hidden_dim];
        h_q[1] += attn[1];

        // step 12: RMSNorm after residual connection
        float x_q[hidden_dim] = { h_q[0], h_q[1] };
        rmsnorm(x_q, hidden_dim, eps);

        // step 13: MLP
        float g0 = x_q[0] * a + x_q[1] * c;
        float g1 = x_q[0] * (a - c / 1000.0f) + x_q[1] * c;
        float base = x_q[0];
        float mix0 = silu(g0) * base;
        float mix1 = silu(g1) * base;
        float mlp_out[hidden_dim] = { 0.0f, carry * (mix1 - mix0) };

        // step 14: MLP residual
        h_q[1] += mlp_out[1];

        // step 15: again RSNorm
        float out[hidden_dim] = { h_q[0], h_q[1] };
        rmsnorm(out, hidden_dim, eps);
        out[0] *= n0;
        out[1] *= n1;

        // step 16: logits
        float logits[vocab_size];
        for (int d = 0; d < vocab_size; ++d) {
            float e[hidden_dim] = { w0 - w1 * d * d, -d };
            logits[d] = dot(out, e, hidden_dim);
            output[(batch * decode_steps + i) * vocab_size + d] = logits[d];
        }

        // step 17: choose next digit
        int next_digit = 0;
        for (int d = 1; d < vocab_size; ++d) {
            if (logits[d] > logits[next_digit]) {
                next_digit = d;
            }
        }
        tokens[seq_len] = next_digit;
    }
}

// prompts, output, weights are device pointers
extern "C" void solve(const int* prompts, float* output, const float* weights, int batch_size) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    adder_kernel<<<blocks, threads>>>(prompts, output, weights, batch_size);
    cudaDeviceSynchronize();
}
