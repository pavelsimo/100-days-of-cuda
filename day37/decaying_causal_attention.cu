#include <cuda_runtime.h>
#include <float.h>

__global__ void matmul(const float *A, const float *B, float *C, int M, int N, int K, float gamma) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k <= row; ++k) {
            sum += powf(gamma, row - k) * A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_t(const float *A, const float *B, float *C, int M, int N, int K, float alpha) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = alpha * sum;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int seq_len,
                      int d_model, float gamma) {
    dim3 threads(16, 16);
    dim3 grid(
        (seq_len + threads.x - 1) / threads.x,
        (seq_len + threads.y - 1) / threads.y
    );
    float *scores;
    float alpha = 1.0f / sqrtf(static_cast<float>(d_model));
    cudaMalloc(&scores, seq_len * seq_len * sizeof(float));
    matmul_t<<<grid, threads>>>(Q, K, scores, seq_len, seq_len, d_model, alpha);
    matmul<<<grid, threads>>>(scores, V, output, seq_len, d_model, seq_len, gamma);
    cudaDeviceSynchronize();
    cudaFree(scores);
}
