#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void matmul(const __half* x, const uint8_t* w_q, const __half* scales, __half* y, int M,
                      int N, int K, int group_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        int half_k = K / 2;
        int groups_per_row = K / group_size;
        for (int k = 0; k < half_k; ++k) {
            float X[2] = {x[row * K + k*2], x[row * K + k*2+1]};
            uint8_t byte = w_q[col * half_k + k]; // W^T
            uint8_t lo = byte & 0x0F;
            uint8_t hi = (byte >> 4) & 0x0F;
            int group_idx = 2 * k / group_size;
            float scale = __half2float(scales[col * groups_per_row + group_idx]);
            float w_lo = static_cast<float>(lo - 8) * scale;
            float w_hi = static_cast<float>(hi - 8) * scale;
            sum += X[0] * w_hi;
            sum += X[1] * w_lo;
        }
        y[row * N + col] = sum;
    }
}

// x, w_q, scales, y are device pointers
extern "C" void solve(const __half* x, const uint8_t* w_q, const __half* scales, __half* y, int M,
                      int N, int K, int group_size) {
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (N + blockDim.x - 1) / blockDim.x, 
        (M + blockDim.y - 1) / blockDim.y);
    matmul<<<gridDim, blockDim>>>(x, w_q, scales, y, M, N, K, group_size);
    cudaDeviceSynchronize();
}
