#include <cuda_runtime.h>

__global__ void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int P) {
    size_t sz = static_cast<size_t>(N) * N * sizeof(float);
    if (P == 1) {
        cudaMemcpy(output, input, sz, cudaMemcpyDeviceToDevice);
        return;
    }

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    float *temp;
    cudaMalloc(&temp, sz);
    matmul<<<blocks, threads>>>(input, input, output, N, N, N);
    for (int p = 2; p < P; ++p) {
        matmul<<<blocks, threads>>>(output, input, temp, N, N, N);
        cudaMemcpy(output, temp, sz, cudaMemcpyDeviceToDevice);
    }

    cudaDeviceSynchronize();
    cudaFree(temp);
}
