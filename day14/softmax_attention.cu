#include <cuda_runtime.h>
#include <cfloat>
#include <stdio.h>
#include <stdlib.h>

#define M_DIM 4
#define N_DIM 4
#define D_DIM 8

__global__ void softmax(float* scores, int M, int N) {
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

    softmax<<<(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(scores, M, N);

    dim3 outputGrid((d + threadsPerBlockGrid.x - 1) / threadsPerBlockGrid.x,
                    (M + threadsPerBlockGrid.y - 1) / threadsPerBlockGrid.y);
    matmul<<<outputGrid, threadsPerBlockGrid>>>(scores, V, output, M, d, N);

    cudaDeviceSynchronize();
    cudaFree(scores);
}

static void print_matrix(const char *title, const float *a, int rows, int cols)
{
    printf("%s (%dx%d):\n", title, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.4f ", a[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void)
{
    float *h_Q, *h_K, *h_V, *h_output;
    float *d_Q, *d_K, *d_V, *d_output;
    int szQ = M_DIM * D_DIM * sizeof(float);
    int szK = N_DIM * D_DIM * sizeof(float);
    int szV = N_DIM * D_DIM * sizeof(float);
    int szO = M_DIM * D_DIM * sizeof(float);

    h_Q      = (float *)malloc(szQ);
    h_K      = (float *)malloc(szK);
    h_V      = (float *)malloc(szV);
    h_output = (float *)malloc(szO);

    cudaMalloc((void **)&d_Q, szQ);
    cudaMalloc((void **)&d_K, szK);
    cudaMalloc((void **)&d_V, szV);
    cudaMalloc((void **)&d_output, szO);

    for (int i = 0; i < M_DIM * D_DIM; i++) {
        h_Q[i] = (float)(i % 7) * 0.1f;
    }
    for (int i = 0; i < N_DIM * D_DIM; i++) {
        h_K[i] = (float)(i % 5) * 0.1f;
        h_V[i] = (float)(i % 3) * 0.5f;
    }

    cudaMemcpy(d_Q, h_Q, szQ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, szK, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, szV, cudaMemcpyHostToDevice);

    print_matrix("Q", h_Q, M_DIM, D_DIM);
    print_matrix("K", h_K, N_DIM, D_DIM);
    print_matrix("V", h_V, N_DIM, D_DIM);

    solve(d_Q, d_K, d_V, d_output, M_DIM, N_DIM, D_DIM);

    cudaMemcpy(h_output, d_output, szO, cudaMemcpyDeviceToHost);

    print_matrix("Attention Output", h_output, M_DIM, D_DIM);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_output);

    return 0;
}
