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

__global__ void matmul_t(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matscale(const float *A, float *C, int M, int N, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        C[row * N + col] = A[row * N + col] * scale;
    }
}

__global__ void matadd(const float *A, const float *B, float *C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
}

// x, W, A, B, output are device pointers
extern "C" void solve(const float* x, const float* W, const float* A, const float* B, float* output,
                      int batch, int d_in, int d_out, int rank, float lora_scale) {
    
    dim3 threads(16, 16);
    dim3 grid((8192 + threads.x - 1) / threads.x,
              (8192 + threads.y - 1) / threads.y);    
    
    // output = x × WT + lora_scale × (x × AT) × BT
    float *xWT, *xAT;
    cudaMalloc(&xWT, batch * d_out * sizeof(float));
    cudaMalloc(&xAT, batch * rank * sizeof(float));
    matmul_t<<<grid, threads>>>(x, W, xWT, batch, d_out, d_in);
    matmul_t<<<grid, threads>>>(x, A, xAT, batch, rank, d_in);
    matscale<<<grid, threads>>>(xAT, xAT, batch, rank, lora_scale);
    matmul_t<<<grid, threads>>>(xAT, B, output, batch, d_out, rank);
    matadd<<<grid, threads>>>(xWT, output, output, batch, d_out);
    
    cudaDeviceSynchronize();
    cudaFree(xWT);
    cudaFree(xAT);
}
