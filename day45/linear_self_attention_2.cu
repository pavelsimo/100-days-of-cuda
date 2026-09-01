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
__global__ void matmul(const float *A, const float *B, float *C, int M, int N, int K, float alpha) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int rowB = blockIdx.y * TILE_SIZE;
    const int colB = blockIdx.x * TILE_SIZE;
    const int row = rowB + ty;
    const int col = colB + tx;

    float sum = 0.0f;
    for (int t = 0; t < K; t += TILE_SIZE) {
        if (TRANS_A) {
            const int k = t + ty;
            const int r = rowB + tx;
            As[tx][ty] = (k < K && r < M) ? A[k * M + r] : 0.0f;
            As[tx][ty] = ELU_A ? elu(As[tx][ty]) : As[tx][ty];
        } else {
            const int k = t + tx;
            As[ty][tx] = (row < M && k < K) ? A[row * K + k] : 0.0f;
            As[ty][tx] = ELU_A ? elu(As[ty][tx]) : As[ty][tx];
        }

        if (TRANS_B) {
            const int k = t + tx;
            const int c = colB + ty;
            Bs[tx][ty] = (c < N && k < K) ? B[c * K + k] : 0.0f;
            Bs[tx][ty] = ELU_B ? elu(Bs[tx][ty]) : Bs[tx][ty];
        } else {
            const int k = t + ty;
            Bs[ty][tx] = (k < K && col < N) ? B[k * N + col] : 0.0f;
            Bs[ty][tx] = ELU_B ? elu(Bs[ty][tx]) : Bs[ty][tx];
        }
        
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];;
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * sum;
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

    matmul<true, false, true, false><<<grid2, threads>>>(K, V, ktv, d, d, M, 1.0f);
    matmul<false, false, true, false><<<grid1, threads>>>(Q, ktv, qktv, M, d, d, 1.0f);
    col_sum<<<(d + TILE_SIZE*TILE_SIZE - 1) / (TILE_SIZE*TILE_SIZE), TILE_SIZE*TILE_SIZE>>>(K, kj, M, d);
    matmul<false, false, true, false><<<grid1, threads>>>(Q, kj, qkj, M, 1, d, 1.0f);
    linear_attn<<<grid1, threads>>>(qktv, qkj, output, M, d);

    cudaDeviceSynchronize();
    cudaFree(ktv);
    cudaFree(qktv);
    cudaFree(kj);
    cudaFree(qkj);
}
