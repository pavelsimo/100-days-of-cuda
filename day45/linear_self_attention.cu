#include <cuda_runtime.h>

#define TILE_SIZE 16

__device__ __forceinline__ float elu(float x) {
    return x > 0 ? x + 1 : expf(x);
}

__global__ void col_sum(const float *A, float *B, int M, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += elu(A[i * N + j]);
        }
        B[j] = sum;
    }
}

template <bool TRANS_A = false, bool TRANS_B = false, bool ELU_A = false, bool ELU_B = false>
__global__ void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = TRANS_A ? A[k * M + i] : A[i * K + k];
            float b = TRANS_B ? B[j * K + k] : B[k * N + j];
            a = ELU_A ? elu(a) : a;
            b = ELU_B ? elu(b) : b;
            sum += a * b;
        }
        C[i * N + j] = sum;
    }
}

__global__ void linear_attn(const float *qktv, const float *qkj, float *output, int M, int d) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < d) {
        output[i * d + j] = qktv[i * d + j] / qkj[i];
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid1(
        (d + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );
    dim3 grid2(
        (d + threads.x - 1) / threads.x,
        (d + threads.y - 1) / threads.y
    );

    float *ktv, *qktv, *kj, *qkj;
    cudaMalloc(&ktv, d * d * sizeof(float));
    cudaMalloc(&qktv, M * d * sizeof(float));
    cudaMalloc(&kj, d * sizeof(float));
    cudaMalloc(&qkj, M * sizeof(float));

    // Q - M x d
    // K - M x d
    // V - M x d

    // K^T @ V - d x d matrix stored in ktv
    // ktv - d x d  

    // Q @ (K^T @ V) - M x d matrix stored in qktv
    // qktv - M x d

    // Sum of columns of K stored in kj
    // Q @ kj - M x 1 vector stored in output
    // qkj - M x 1

    // qktv / qkj - M x d matrix stored in output
    // output = qktv / qkj element-wise

    matmul<true, false, true, false><<<grid2, threads>>>(K, V, ktv, d, d, M);
    matmul<false, false, true, false><<<grid1, threads>>>(Q, ktv, qktv, M, d, d);
    col_sum<<<(d + TILE_SIZE*TILE_SIZE - 1) / (TILE_SIZE*TILE_SIZE), TILE_SIZE*TILE_SIZE>>>(K, kj, M, d);
    matmul<false, false, true, false><<<grid1, threads>>>(Q, kj, qkj, M, 1, d);
    linear_attn<<<grid1, threads>>>(qktv, qkj, output, M, d);

    cudaDeviceSynchronize();
    cudaFree(ktv);
    cudaFree(qktv);
    cudaFree(kj);
    cudaFree(qkj);
}
