#include <cuda_runtime.h>


__global__ void build_matrix(const float* X, const float* y, float* M, int n_samples, int n_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n_features || j >= n_features) {
        return;
    }

    int W = n_features + 1;

    float sum = 0.0f;
    for (int k = 0; k < n_samples; k++) {
        sum += X[k * n_features + i] * X[k * n_features + j];
    }
    M[i * W + j] = sum;

    sum = 0.0f;
    for (int k = 0; k < n_samples; k++) {
        sum += X[k * n_features + i] * y[k];
    }
    M[i * W + n_features] = sum;
}

__global__ void forward_elimination(float* M, int n_features) {
    int tid = threadIdx.x;
    int W = n_features + 1;
    __shared__ int s_pivot;

    for (int col = 0; col < n_features; col++) {
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

        int pivot = s_pivot;
        if (pivot != col) {
            for (int c = tid; c < W; c += blockDim.x) {
                float tmp = M[col * W + c];
                M[col * W + c] = M[pivot * W + c];
                M[pivot * W + c] = tmp;
            }
        }
        __syncthreads();

        for (int r = col + 1 + tid; r < n_features; r += blockDim.x) {
            float factor = M[r * W + col] / M[col * W + col];
            for (int c = col; c < W; c++)
                M[r * W + c] -= factor * M[col * W + c];
        }
        __syncthreads();
    }
}

__global__ void back_substitution(float* M, float* beta, int n_features) {
    int W = n_features + 1;

    for (int i = n_features - 1; i >= 0; i--) {

        float sum = M[i * W + n_features];
        for (int k = i + 1; k < n_features; k++) {
            sum -= M[i * W + k] * beta[k];
        }

        beta[i] = sum / M[i * W + i];
    }
}

void gaussian_elimimation(float* M, float* beta, int n_features) {
    int threads = 256;
    int blocks = 1;
    forward_elimination<<<blocks, threads>>>(M, n_features);
    back_substitution<<<blocks, threads>>>(M, beta, n_features);
}

__device__ __forceinline__ float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

__global__ void newton_system(const float* X, const float* y, const float* beta, float* M, int n_samples, int n_features, float eps) {
    int W = n_features + 1;
    
    for (int i = 0; i < n_samples; i++) {
        const float* X_i = X + i * n_features;

        float z = 0.0f;
        for (int j = 0; j < n_features; j++) {
            z += X_i[j] * beta[j];
        }
        float p = sigmoid(z);
        float w = p * (1.0f - p);

        for (int j = 0; j < n_features; j++) {
            M[j * W + n_features] += X_i[j] * (y[i] - p);

            for (int k = 0; k < n_features; k++) {
                M[j * W + k] += X_i[j] * w * X_i[k];
            }
        }
    }

    for (int j = 0; j < n_features; j++) {
        M[j * W + j] += eps;
        M[j * W + n_features] -= eps * beta[j];
    }
}

__global__ void update_beta(float* beta, const float* delta, int* done, int n_features, float eps) {
    float max_delta = 0.0f;
    float max_beta = 0.0f;
    for (int j = 0; j < n_features; j++) {
        beta[j] += delta[j];
        max_delta = fmaxf(max_delta, fabsf(delta[j]));
        max_beta = fmaxf(max_beta, fabsf(beta[j]));
    }
    *done = max_delta <= eps * (1.0f + max_beta);
}


extern "C" void solve(const float* X, const float* y, float* beta,  int n_samples, int n_features) {
    const int max_iterations = 100;
    const float eps = 1e-6f;

    int W = n_features + 1;
    float* M;
    float* delta;
    int* d_done;
    cudaMalloc(&M, n_features * W * sizeof(float));
    cudaMalloc(&delta, n_features * sizeof(float));
    cudaMalloc(&d_done, sizeof(int));
    cudaMemset(beta, 0, n_features * sizeof(float));

    for (int step = 0; step < max_iterations; ++step) {
        cudaMemset(M, 0, n_features * W * sizeof(float));
        newton_system<<<1, 1>>>(X, y, beta, M, n_samples, n_features, eps);
        gaussian_elimimation(M, delta, n_features);
        update_beta<<<1, 1>>>(beta, delta, d_done, n_features, eps);

        int done = 0;
        cudaMemcpy(&done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
        if (done) {
            break;
        }
    }

    cudaFree(M);
    cudaFree(delta);
    cudaFree(d_done);
}
