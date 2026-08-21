#include <cuda_runtime.h>
#include <float.h>

#define TILE_SIZE 8

__global__ void matmul(const float *A, const float *B, float *C, int M, int N, int K, int H) {
    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;
    int h = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadRow;
    int col = blockIdx.x * TILE_SIZE + threadCol;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int d_model = N * H;
    float sum = 0.0f;
    for (int tile = 0; tile < K; tile += TILE_SIZE) {
        const int aCol = tile + threadCol;
        const int bRow = tile + threadRow;

        As[threadRow][threadCol] = (row < M && aCol < K) ? A[h * (M * K) + row * K + aCol] : 0.0f;
        Bs[threadRow][threadCol] = (bRow < K && col < N) ? B[bRow * d_model + h * N + col] : 0.0f;
        __syncthreads();

        for (int d = 0; d < TILE_SIZE; ++d) {
            sum += As[threadRow][d] * Bs[d][threadCol];
        }
        __syncthreads();
    }
    if (h < H && row < M && col < N) {
        C[row * (d_model) + h * N + col] = sum;
    }
}

__global__ void matmul_t(const float *A, const float *B, float *C, int M, int N, int K, int H, float alpha) {
    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;
    int h = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadRow;
    int col = blockIdx.x * TILE_SIZE + threadCol;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    int d_model = K * H;

    float sum = 0.0f;
    for (int tile = 0; tile < K; tile += TILE_SIZE) {
        const int k = tile + threadCol;
        const int bRow = blockIdx.x * TILE_SIZE + threadRow;

        As[threadRow][threadCol] = (row < M && k < K) ? A[row * d_model + h * K + k] : 0.0f;
        Bs[threadRow][threadCol] = (bRow < N && k < K) ? B[bRow * d_model + h * K + k] : 0.0f;
        __syncthreads();

        for (int d = 0; d < TILE_SIZE; ++d) {
            sum += As[threadRow][d] * Bs[threadCol][d];
        }
        __syncthreads();
    }
    if (h < H && row < M && col < N) {
        C[h * (M * N) + row * N + col] = alpha * sum;
    }
}

__global__ void softmax(float* scores, int M, int N, int H) {
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int tid = threadIdx.x;
    __shared__ float t[TILE_SIZE*TILE_SIZE];

    if (row >= M || head >= H) {
        return;
    }

    float* rowScores = scores + (head * M + row) * N;
    float threadMax = -FLT_MAX;
    for (int col = tid; col < N; col += blockDim.x) {
        threadMax = fmaxf(threadMax, rowScores[col]);
    }
    t[tid] = threadMax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] = fmaxf(t[tid], t[tid + stride]);
        }
        __syncthreads();
    }

    const float rowMax = t[0];
    float threadSum = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        const float e = expf(rowScores[col] - rowMax);
        rowScores[col] = e;
        threadSum += e;
    }
    t[tid] = threadSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] += t[tid + stride];
        }
        __syncthreads();
    }

    const float invSum = 1.0f / t[0];
    for (int col = tid; col < N; col += blockDim.x) {
        rowScores[col] *= invSum;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N,
                      int d_model, int h) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid1(
        (N + threads.x - 1) / threads.x,
        (N + threads.y - 1) / threads.y,
        h
    );
    int head_size = d_model / h;
    float alpha = rsqrtf(head_size);
    float *scores;

    // Q, K, V -> N x d_model
    // scores -> h x N x N
    cudaMalloc(&scores, h * N * N * sizeof(float));

    // scores_h = (Q_h @ K_h^T) / sqrt(head_size)
    // (N x head_size) @ (head_size x N) -> (N x N), per head
    matmul_t<<<grid1, threads>>>(Q, K, scores, N, N, head_size, h, alpha);

    // scores_h = softmax(scores_h), row-wise: each row sums to 1
    dim3 grid2(N, h);
    softmax<<<grid2, TILE_SIZE>>>(scores, N, N, h);

    // output_h = scores_h @ V_h
    // (N x N) @ (N x head_size) -> (N x head_size), per head
    dim3 grid3(
        (head_size + threads.x - 1) / threads.x,
        (N + threads.y - 1) / threads.y,
        h
    );

    matmul<<<grid3, threads>>>(scores, V, output, N, head_size, N, h);
    cudaDeviceSynchronize();
    cudaFree(scores);
}
