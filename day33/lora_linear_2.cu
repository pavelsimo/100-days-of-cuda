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

__global__ void matmul_t(const float *A, const float *B, float *C, int M, int N, int K) {
    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadRow;
    int col = blockIdx.x * TILE_SIZE + threadCol;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (int tile = 0; tile < K; tile += TILE_SIZE) {
        const int aCol = tile + threadCol;
        const int bRow = blockIdx.x * TILE_SIZE + threadRow;
        const int bCol = tile + threadCol;

        As[threadRow][threadCol] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadRow][threadCol] = (bRow < N && bCol < K) ? B[bRow * K + bCol] : 0.0f;
        __syncthreads();

        for (int d = 0; d < TILE_SIZE; ++d) {
            sum += As[threadRow][d] * Bs[threadCol][d];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
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

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid1((d_out + threads.x - 1) / threads.x, (batch + threads.y - 1) / threads.y);
    dim3 grid2((rank + threads.x - 1) / threads.x, (batch + threads.y - 1) / threads.y);

    // output = x × WT + lora_scale × (x × AT) × BT
    float *xWT, *xAT;
    cudaMalloc(&xWT, batch * d_out * sizeof(float));
    cudaMalloc(&xAT, batch * rank * sizeof(float));
    matmul_t<<<grid1, threads>>>(x, W, xWT, batch, d_out, d_in);
    matmul_t<<<grid2, threads>>>(x, A, xAT, batch, rank, d_in);
    matscale<<<grid2, threads>>>(xAT, xAT, batch, rank, lora_scale);
    matmul_t<<<grid1, threads>>>(xAT, B, output, batch, d_out, rank);
    matadd<<<grid1, threads>>>(xWT, output, output, batch, d_out);
    
    cudaDeviceSynchronize();
    cudaFree(xWT);
    cudaFree(xAT);
}
