#include <cuda_runtime.h>
#include <math.h>

__device__ int8_t clamp(float x) {
    x = fmaxf(-128.0f, x);
    x = fminf(x, 127.0f);
    return static_cast<int8_t>(x);
}

__global__ void matmul(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        int sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += (A[row * K + k] - zero_point_A) *
                   (B[k * N + col] - zero_point_B);
        }
        float t = sum * (scale_B / scale_C) * scale_A;
        C[row * N + col] = clamp(roundf(t) + zero_point_C);
    }
}

// A, B, C are device pointers
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C) {
    dim3 threads(16, 16);
    dim3 grid(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );
    matmul<<<grid, threads>>>(A, B, C, M, N, K, scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C);
    cudaDeviceSynchronize();
}
