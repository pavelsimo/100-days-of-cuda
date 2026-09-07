#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 16

__device__ int8_t clamp(float x) {
    x = fmaxf(-128.0f, x);
    x = fminf(x, 127.0f);
    return static_cast<int8_t>(x);
}

__global__ void matmul(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C, float alpha) {
    __shared__ int As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ int Bs[TILE_SIZE][TILE_SIZE + 1];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int rowB = blockIdx.y * TILE_SIZE;
    const int colB = blockIdx.x * TILE_SIZE;
    const int row = rowB + ty;
    const int col = colB + tx;

    int sum = 0;
    for (int t = 0; t < K; t += TILE_SIZE) {
        const int kA = t + tx;
        const int kB = t + ty;
        As[ty][tx] = (row < M && kA < K) ? (A[row * K + kA] - zero_point_A) : 0;
        Bs[ty][tx] = (kB < K && col < N) ? (B[kB * N + col] - zero_point_B) : 0;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float t = sum * alpha;
        C[row * N + col] = clamp(roundf(t) + zero_point_C);
    }
}

// A, B, C are device pointers
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );
    float alpha = (scale_B / scale_C) * scale_A;
    matmul<<<grid, threads>>>(A, B, C, M, N, K, scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C, alpha);
    cudaDeviceSynchronize();
}
