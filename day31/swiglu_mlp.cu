#include <cuda_runtime.h>

__global__ void mul(const float *A, const float *B, float *C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        C[row * N + col] = A[row * N + col] * B[row * N + col];
    }
}

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

__global__ void silu(float *z, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        z[i] *= 1.0f / (1.0f + expf(-z[i]));
    }
}

// x, W_gate, W_up, W_down, output are device pointers
extern "C" void solve(const float* x, const float* W_gate, const float* W_up, const float* W_down,
                      float* output, int M, int d_model, int d_ffn) {

    // output = (SiLU(x × W_gate) ⊙ (x × W_up)) × W_down
    // SiLU(z) = z × sigmoid(z)
    // W_gate -> [d_model, d_fnn]
    // W_up   -> [d_model, d_fnn]
    // x      -> [M, d_model]
    float *gate_proj;
    cudaMalloc(&gate_proj, M * d_ffn * sizeof(float));
    float *up_proj;
    cudaMalloc(&up_proj, M * d_ffn * sizeof(float));
    dim3 threads(16, 16);        
    dim3 blocks(
        (d_ffn + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );
    matmul<<<blocks, threads>>>(x, W_gate, gate_proj, M, d_ffn, d_model);
    matmul<<<blocks, threads>>>(x, W_up, up_proj, M, d_ffn, d_model);
    silu<<<(M * d_ffn + 256 - 1) / 256, 256>>>(gate_proj, M * d_ffn);
    
    float *hidden;
    cudaMalloc(&hidden, M * d_ffn * sizeof(float));
    dim3 blocksHidden(
        (d_ffn + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );
    mul<<<blocksHidden, threads>>>(gate_proj, up_proj, hidden, M, d_ffn);
    
    dim3 blocksOutput(
        (d_model + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    ); 
    matmul<<<blocksOutput, threads>>>(hidden, W_down, output, M, d_model, d_ffn);

    cudaDeviceSynchronize();
    cudaFree(gate_proj);
    cudaFree(up_proj);
    cudaFree(hidden);
}
