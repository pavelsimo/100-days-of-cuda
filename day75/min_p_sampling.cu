#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void min_p_sampling(const float* logits, float* probs, int M, int N, float min_p) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float t[];

    if (row >= M) {
        return;
    }

    // softmax

    float t_max = -FLT_MAX;
    for (int col = tid; col < N; col += blockDim.x) {
        t_max = fmaxf(t_max, logits[row * N + col]);
    }
    t[tid] = t_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] = fmaxf(t[tid], t[tid + stride]);
        }
        __syncthreads();
    }

    const float row_max = t[0];
    __syncthreads();

    float t_sum = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        float e = expf(logits[row * N + col] - row_max);
        probs[row * N + col] = e;
        t_sum += e;
    }
    t[tid] = t_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] += t[tid + stride];
        }
        __syncthreads();
    }

    const float inv = 1.0f / t[0];
    __syncthreads();

    // normalize 

    float p_sum = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        float e = probs[row * N + col];
        float prob = (e < min_p) ? 0.0f : e * inv;
        probs[row * N + col] = prob;
        p_sum += prob;
    }
    t[tid] = p_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] += t[tid + stride];
        }
        __syncthreads();
    }

    const float inv_p = 1.0f / t[0];
    for (int col = tid; col < N; col += blockDim.x) {
        probs[row * N + col] *= inv_p;
    }
}

// logits, probs are device pointers
extern "C" void solve(const float* logits, float* probs, float min_p, int B, int V) {
    int threads = 256;
    min_p_sampling<<<B, threads, threads * sizeof(float)>>>(logits, probs, B, V, min_p);
    cudaDeviceSynchronize();
}
