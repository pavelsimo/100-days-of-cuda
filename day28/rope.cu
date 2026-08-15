#include <cuda_runtime.h>


__global__ void rope(float* Q, float* cos, float* sin, float* output, int M, int D) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < D) {
        int halfD = D / 2;
        float R = j < halfD ? -Q[i * D + (j + halfD)]: Q[i * D + (j - halfD)];
        output[i * D + j] = Q[i * D + j] * cos[i * D + j] + R * sin[i * D + j];
    }
}

// Q, cos, sin, output are device pointers
extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {
    dim3 threads(16, 16);
    dim3 grid(
        (D + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );
    rope<<<grid, threads>>>(Q, cos, sin, output, M, D);
    cudaDeviceSynchronize();
}
