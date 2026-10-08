#include <cuda_runtime.h>

__global__ void calc_residual(const float* X, const float* y, float* beta, float* residual, int n_samples, int n_features) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_samples) {
        float z = 0.0f;
        for (int j = 0; j < n_features; ++j) {
            z += X[i * n_features + j] * beta[j];
        }
        float p = 1.0f / (1.0f + expf(-z));
        residual[i] = y[i] - p;
    }
}

__global__ void update_coefficients(const float* X, const float* y, float* beta, float *residual, int n_samples, int n_features, float lr, float lambda) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < n_features) {
        float sum = 0.0f;
        for (int i = 0; i < n_samples; ++i) {
            sum += X[i * n_features + j] * residual[i];
        }
        float g = (sum - lambda * beta[j]) / n_samples;
        beta[j] += lr * g;
    }
}

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta,  int n_samples, int n_features) {
    // X - n_samples x n_features
    // y - n_samples
    // beta - n_features
    cudaMemset(beta, 0, n_features * sizeof(float));
    float *residual;
    float lr = 1.0f;
    float lambda = 1e-6f;
    cudaMalloc(&residual, n_samples * sizeof(float));
    int epochs = 1000000;
    int threads = 256;
    int blocks_1 = (n_samples + threads - 1) / threads;
    int blocks_2 = (n_features + threads - 1) / threads;
    for (int i = 0; i < epochs; ++i) {
        calc_residual<<<blocks_1, threads>>>(X, y, beta, residual, n_samples, n_features);
        update_coefficients<<<blocks_2, threads>>>(X, y, beta, residual, n_samples, n_features, lr, lambda);
    }
    cudaFree(residual);
}
