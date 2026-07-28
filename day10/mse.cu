#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N_DIM 10

__global__ void squared_error_kernel(const float* predictions, const float* targets, float* mse, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    __shared__ float t[256];
    float diff = (predictions[i] - targets[i]);
    t[tid] = (i < N) ? diff * diff: 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] += t[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(mse, t[0]);
    }
}

__global__ void mean_kernel(float *mse, int N) {
    *mse /= N;
}

// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    squared_error_kernel<<<blocksPerGrid, threadsPerBlock>>>(predictions, targets, mse, N);
    mean_kernel<<<1, 1>>>(mse, N);
    cudaDeviceSynchronize();
}

static void print_array(const char *title, const float *a, int n)
{
    printf("%s (%d):\n", title, n);
    for (int i = 0; i < n; i++) {
        printf("%6.1f ", a[i]);
    }
    printf("\n\n");
}

int main(void)
{
    float *h_predictions, *h_targets, h_mse;
    float *d_predictions, *d_targets, *d_mse;
    int sz = N_DIM * sizeof(float);

    h_predictions = (float *)malloc(sz);
    h_targets = (float *)malloc(sz);

    cudaMalloc((void **)&d_predictions, sz);
    cudaMalloc((void **)&d_targets, sz);
    cudaMalloc((void **)&d_mse, sizeof(float));

    for (int i = 0; i < N_DIM; i++) {
        h_predictions[i] = (float)(i + 1);
        h_targets[i] = (float)(i + 2);
    }

    cudaMemcpy(d_predictions, h_predictions, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, sz, cudaMemcpyHostToDevice);
    cudaMemset(d_mse, 0, sizeof(float));

    print_array("Predictions", h_predictions, N_DIM);
    print_array("Targets", h_targets, N_DIM);

    solve(d_predictions, d_targets, d_mse, N_DIM);

    cudaMemcpy(&h_mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost);

    printf("MSE: %6.1f\n\n", h_mse);

    cudaFree(d_predictions);
    cudaFree(d_targets);
    cudaFree(d_mse);
    free(h_predictions);
    free(h_targets);

    return 0;
}
