#include <cuda_runtime.h>

__global__ void scan_rows(const float* __restrict__ input, float* __restrict__ output, int H, int W) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= H) {
        return;
    }
    for (int col = 0; col < W; ++col) {
        output[row * W + col] = input[row * W + col];
        output[row * W + col] += (col - 1 >= 0) ? output[row * W + col - 1]: 0.0f;
    }
}

__global__ void scan_cols(const float* input, float* __restrict__ output, int H, int W) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= W) {
        return;
    }
    for (int row = 1; row < H; ++row) {
        output[row * W + col] += output[(row - 1) * W + col];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int H, int W) {
    // 1 2 3
    // 4 5 6
    // 7 8 9

    // horizontal cumulative sum
    // 1 3 6
    // 4 9 15
    // 7 15 24

    // vertical cumulative sum
    // 1 3 6
    // 5 12 21
    // 12 27 45
    int threads = 256;
    scan_rows<<<(H + threads - 1) / threads, threads>>>(input, output, H, W);
    scan_cols<<<(W + threads - 1) / threads, threads>>>(input, output, H, W);
    cudaDeviceSynchronize();
}
