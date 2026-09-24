#include <cuda_runtime.h>
#include <float.h>

#define TILE_SIZE 32

__global__ void softmax(float* scores, int M, int N) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float t[TILE_SIZE*TILE_SIZE];

    if (row >= M) {
        return;
    }

    float threadMax = -FLT_MAX;
    for (int col = tid; col < N; col += blockDim.x) {
        threadMax = fmaxf(threadMax, scores[row * N + col]);
    }
    t[tid] = threadMax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] = fmaxf(t[tid], t[tid + stride]);
        }
        __syncthreads();
    }

    const float rowMax = t[0];
    float threadSum = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        const float e = expf(scores[row * N + col] - rowMax);
        scores[row * N + col] = e;
        threadSum += e;
    }
    t[tid] = threadSum;
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

__global__ void rank_sort(const float* in, float* out, int* out_idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float v = in[i];
    int pos = 0;
    for (int j = 0; j < n; ++j) {
        float w = in[j];
        pos += (w < v) || (w == v && j > i);
    }
    out[pos] = v;
    out_idx[pos] = i;
}

__device__ float philox_uniform(int seed, int subsequence, int offset) {
    const unsigned long long seed_bits = (unsigned long long)(long long)seed;
    uint2 key = make_uint2((unsigned int)seed_bits, (unsigned int)(seed_bits >> 32));
    uint4 counter = make_uint4(offset / 4, 0u, subsequence, 0u);
    const unsigned int m0 = 0xD2511F53u, m1 = 0xCD9E8D57u;
    for (int round = 0; round < 10; ++round) {
        counter = make_uint4(__umulhi(m1, counter.z) ^ counter.y ^ key.x,
                             m1 * counter.z,
                             __umulhi(m0, counter.x) ^ counter.w ^ key.y,
                             m0 * counter.x);
        key.x += 0x9E3779B9u;
        key.y += 0xBB67AE85u;
    }
    const unsigned int words[] = {counter.x, counter.y, counter.z, counter.w};
    const float scale = 1.0f / 4294967296.0f;
    return words[offset % 4] * scale + 0.5f * scale;
}

__device__ float random_exponential(int seed, int pos) {
    constexpr int stride = 40 * 4 * 256;
    const float u = philox_uniform(seed, pos % stride, pos / stride);
    const float half_eps = 0.5f * FLT_EPSILON;
    return (u >= 1.0f - half_eps) ? half_eps : -logf(u);
}

__global__ void top_p_sampling(float* sorted_weights, const int* sorted_idx, const float* p,
                               const int* seed, int* sampled_token, int vocab_size) {
    float sum = 0.0f;
    int first = 0;
    for (int i = vocab_size - 1; i >= 0; --i) {
        sum += sorted_weights[i];
        if (sum >= *p) {
            first = i;
            break;
        }
    }
    for (int i = first; i < vocab_size; ++i) {
        sorted_weights[i] /= sum;
    }

    const int N = vocab_size - first;
    int sample_idx = vocab_size - 1;
    float best = -1.0f;
    for (int i = 0; i < N; ++i) {
        const float q = random_exponential(*seed, i);
        const float score = sorted_weights[vocab_size - 1 - i] / q;
        if (score > best) {
            best = score;
            sample_idx = vocab_size - 1 - i;
        }
    }
    *sampled_token = sorted_idx[sample_idx];
}

extern "C" void solve(const float* logits, const float* p, const int* seed, int* sampled_token,
                      int vocab_size) {
    int threads = 256;
    int blocks = (vocab_size + threads - 1) / threads;
    float *weights, *sorted_weights;
    int *sorted_idx;
    cudaMalloc(&weights, vocab_size * sizeof(float));
    cudaMemcpy(weights, logits, vocab_size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMalloc(&sorted_weights, vocab_size * sizeof(float));
    cudaMalloc(&sorted_idx, vocab_size * sizeof(int));

    softmax<<<blocks, threads>>>(weights, 1, vocab_size);
    rank_sort<<<blocks, threads>>>(weights, sorted_weights, sorted_idx, vocab_size);
    top_p_sampling<<<1, 1>>>(sorted_weights, sorted_idx, p, seed, sampled_token, vocab_size);

    cudaFree(weights);
    cudaFree(sorted_weights);
    cudaFree(sorted_idx);
}
