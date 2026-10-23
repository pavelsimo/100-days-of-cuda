#include <cuda_runtime.h>
#include <cstdlib>

#define BLOCKSIZE 16

__global__ void matmul(const float* __restrict__ A, const float* __restrict__ B,
                       float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BLOCKSIZE][BLOCKSIZE + 1];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCKSIZE + ty;
    const int col = blockIdx.x * BLOCKSIZE + tx;

    float sum = 0.0f;
    for (int t = 0; t < K; t += BLOCKSIZE) {
        As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.0f;
        Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCKSIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void lora_down(const float* __restrict__ x, const float* __restrict__ A,
                          const int* __restrict__ adapter_ids, float* __restrict__ bufA,
                          int T, int K, int R, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * R) {
        return;
    }

    const int t = idx / R;
    const int r = idx % R;
    const int a = adapter_ids[t];
    if (a < 0) {
        return;
    }

    const float *xt = x + (t * K);
    const float *Aa = A + (a * K * R);
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += xt[k] * Aa[k * R + r];
    }
    bufA[idx] = scale * sum;
}

__global__ void lora_up(const float* __restrict__ bufA, const float* __restrict__ B,
                        const int* __restrict__ adapter_ids, float* __restrict__ y,
                        int T, int N, int R) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * N) {
        return;
    }

    const int t = idx / N;
    const int n = idx % N;
    const int a = adapter_ids[t];
    if (a < 0) {
        return;
    }

    const float *tmpA = bufA + (t * R);
    const float *Ba = B + (a * R * N);
    float sum = 0.0f;
    for (int r = 0; r < R; ++r) {
        sum += tmpA[r] * Ba[r * N + n];
    }
    y[idx] += sum;
}

// x, W, A, B, adapter_ids, y are device pointers
extern "C" void solve(const float* x, const float* W, const float* A, const float* B,
                      const int* adapter_ids, float* y, int T, int K, int N, int R, int L,
                      float scale) {

    // T - tokens
    // N - output features
    // K - input features
    // R - rank
    // L - number of adapters
    //
    // W - [K, N]
    // y - [T, N]
    // x - [T, K]
    // A - [L, K, R]
    // B - [L, R, N]
    float *buf;
    cudaMalloc(&buf, T * R * sizeof(float));
                        
    int threads = 256;
    int blocks = (T * R + threads - 1) / threads;
    lora_down<<<blocks, threads>>>(x, A, adapter_ids, buf, T, K, R, scale);

    dim3 threads_2(BLOCKSIZE, BLOCKSIZE);
    dim3 blocks_2(
        (N + threads_2.x - 1) / threads_2.x,
        (T + threads_2.y - 1) / threads_2.y
    );
    matmul<<<blocks_2, threads_2>>>(x, W, y, T, N, K);

    blocks = (T * N + threads - 1) / threads;
    lora_up<<<blocks, threads>>>(buf, B, adapter_ids, y, T, N, R);
    cudaDeviceSynchronize();

    cudaFree(buf);
}
