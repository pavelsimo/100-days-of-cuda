#include <cuda_runtime.h>

// step 1: build M = [X^T X | X^T y]
__global__ void build_matrix(const float* X, const float* y, float* M, int n_samples, int n_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n_features || j >= n_features) {
        return;
    }

    int W = n_features + 1;

    // step 1.1: compute X^T X
    float sum = 0.0f;
    for (int k = 0; k < n_samples; k++) {
        sum += X[k * n_features + i] * X[k * n_features + j];
    }
    M[i * W + j] = sum;

    // step 1.2: compute X^T y
    sum = 0.0f;
    for (int k = 0; k < n_samples; k++) {
        sum += X[k * n_features + i] * y[k];
    }
    M[i * W + n_features] = sum;
}

// step 2: forward elimination
__global__ void forward_elimination(float* M, int n_features) {
    int tid = threadIdx.x;
    int W = n_features + 1;
    __shared__ int s_pivot;

    for (int col = 0; col < n_features; col++) {
        // step 2.1: find the pivot row
        if (tid == 0) {
            int pivot = col;
            float best = fabsf(M[col * W + col]);
            for (int r = col + 1; r < n_features; r++) {
                float v = fabsf(M[r * W + col]);
                if (v > best) { 
                    best = v; 
                    pivot = r; 
                }
            }
            s_pivot = pivot;
        }
        __syncthreads();

        // step 2.2: swap the pivot row into place
        int pivot = s_pivot;
        if (pivot != col) {
            for (int c = tid; c < W; c += blockDim.x) {
                float tmp = M[col * W + c];
                M[col * W + c] = M[pivot * W + c];
                M[pivot * W + c] = tmp;
            }
        }
        __syncthreads();

        // step 2.3: zero out below the pivot
        for (int r = col + 1 + tid; r < n_features; r += blockDim.x) {
            float factor = M[r * W + col] / M[col * W + col];
            for (int c = col; c < W; c++)
                M[r * W + c] -= factor * M[col * W + c];
        }
        __syncthreads();
    }
}

// step 3: back-substitution
__global__ void back_substitution(float* M, float* beta, int n_features) {
    int W = n_features + 1;

    // step 3.1: walk rows bottom-up
    for (int i = n_features - 1; i >= 0; i--) {

        // step 3.2: subtract solved unknowns
        float sum = M[i * W + n_features];
        for (int k = i + 1; k < n_features; k++) {
            sum -= M[i * W + k] * beta[k];
        }

        // step 3.3: divide by the diagonal
        beta[i] = sum / M[i * W + i];
    }
}

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    dim3 threads(16, 16);
    dim3 blocks((n_features + threads.x - 1) / threads.x,
                (n_features + threads.y - 1) / threads.y);
    float *M;
    cudaMalloc(&M, sizeof(float) * n_features * (n_features + 1));

    build_matrix<<<blocks, threads>>>(X, y, M, n_samples, n_features);
    forward_elimination<<<1, 256>>>(M, n_features);
    back_substitution<<<1, 1>>>(M, beta, n_features);
    
    cudaDeviceSynchronize();
    cudaFree(M);
}
