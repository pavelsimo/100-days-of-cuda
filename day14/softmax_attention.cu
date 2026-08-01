#include <cuda_runtime.h>
#include <cfloat>

__global__ void row_softmax(float* scores, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float max = -FLT_MAX;
        for (int col = 0; col < N; ++col) {
            max = fmaxf(max, scores[row * N + col]);
        }
        float sum = 0.0f;
        for (int col = 0; col < N; ++col) {
            float e = expf(scores[row * N + col] - max);
            scores[row * N + col] = e;
            sum += e;
        }
        for (int col = 0; col < N; ++col) {
            scores[row * N + col] /= sum;
        }
    }
}

__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
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

__global__ void softmax_attention(const float* Q,
                      const float* K, 
                      const float* V, 
                      float* output, 
                      int M, 
                      int N,
                      int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float score = 0.0f;
        for (int k = 0; k < d; ++k) {
            score += Q[row * d + k] * K[col * d + k];
        }
        output[row * N + col] = score * rsqrtf((float)d);
    }
}

extern "C" void solve(const float* Q, 
                      const float* K, 
                      const float* V, 
                      float* output, 
                      int M, 
                      int N,
                      int d) {
    float* scores;
    cudaMalloc(&scores, M * N * sizeof(float));

    dim3 threadsPerBlockGrid(16, 16);
    int threadsPerBlock = threadsPerBlockGrid.x * threadsPerBlockGrid.y; 
    dim3 scoresGrid((M + threadsPerBlockGrid.x - 1) / threadsPerBlockGrid.x,
                    (N + threadsPerBlockGrid.y - 1) / threadsPerBlockGrid.y);
    softmax_attention<<<scoresGrid, threadsPerBlockGrid>>>(Q, K, V, scores, M, N, d);

    row_softmax<<<(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(scores, M, N);

    dim3 outputGrid((d + threadsPerBlockGrid.x - 1) / threadsPerBlockGrid.x,
                    (M + threadsPerBlockGrid.y - 1) / threadsPerBlockGrid.y);
    matmul<<<outputGrid, threadsPerBlockGrid>>>(scores, V, output, M, d, N);

    cudaDeviceSynchronize();
    cudaFree(scores);
}
