#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadRow;
    int col = blockIdx.x * TILE_SIZE + threadCol;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (int tile = 0; tile < K; tile += TILE_SIZE) {
        const int aCol = tile + threadCol;
        const int bRow = tile + threadRow;

        As[threadRow][threadCol] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadRow][threadCol] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        for (int d = 0; d < TILE_SIZE; ++d) {
            sum += As[threadRow][d] * Bs[d][threadCol];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void mul(const float *A, const float *B, float *C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        C[row * N + col] = A[row * N + col] * B[row * N + col];
    }
}

__global__ void silu(float *z, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        z[i] *= __fdividef(1.0f, 1.0f + __expf(-z[i]));
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
    dim3 threads(TILE_SIZE, TILE_SIZE);
    
    float *gate_proj, *up_proj, *hidden;
    cudaMalloc(&gate_proj, M * d_ffn * sizeof(float));
    cudaMalloc(&up_proj, M * d_ffn * sizeof(float));
    cudaMalloc(&hidden, M * d_ffn * sizeof(float));
    
    dim3 grid1((d_ffn + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    matmul<<<grid1, threads>>>(x, W_gate, gate_proj, M, d_ffn, d_model);
    matmul<<<grid1, threads>>>(x, W_up, up_proj, M, d_ffn, d_model);
    silu<<<(M * d_ffn + 256 - 1) / 256, 256>>>(gate_proj, M * d_ffn);
    mul<<<grid1, threads>>>(gate_proj, up_proj, hidden, M, d_ffn);
    
    dim3 grid2((d_model + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y); 
    matmul<<<grid2, threads>>>(hidden, W_down, output, M, d_model, d_ffn);

    cudaDeviceSynchronize();
    cudaFree(gate_proj);
    cudaFree(up_proj);
    cudaFree(hidden);
}
