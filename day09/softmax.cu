#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define N_DIM 10

__device__ void atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    while (__int_as_float(old) < value) {
        int assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
}

__global__ void max_kernel(const float* input, float* output, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    __shared__ float t[256];
    t[tid] = (i < N) ? input[i] : -FLT_MAX;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] = fmaxf(t[tid], t[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxFloat(output, t[0]);
    }
}

__global__ void exp_sum_kernel(const float* input, float* output, const float* max, int N) {
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float t[256];
    t[tid] = (i < N) ? expf(input[i] - *max) : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            t[tid] += t[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, t[0]);
    }
}

__global__ void softmax_kernel(const float* input, float* output, const float* sum, const float* max, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        output[i] = expf(input[i] - *max) / *sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* d_max;
    float* d_sum;
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));

    float init_max = -FLT_MAX;
    float init_sum = 0.0f;
    cudaMemcpy(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &init_sum, sizeof(float), cudaMemcpyHostToDevice);

    max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, d_max, N);
    exp_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, d_sum, d_max, N);
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, d_sum, d_max, N);
    cudaDeviceSynchronize();

    cudaFree(d_max);
    cudaFree(d_sum);
}

static void print_array(const char *title, const float *a, int n)
{
    printf("%s (%d):\n", title, n);
    for (int i = 0; i < n; i++) {
        printf("%8.4f ", a[i]);
    }
    printf("\n\n");
}

int main(void)
{
    float *h_input, *h_output;
    float *d_input, *d_output;
    int sz = N_DIM * sizeof(float);

    h_input = (float *)malloc(sz);
    h_output = (float *)malloc(sz);

    cudaMalloc((void **)&d_input, sz);
    cudaMalloc((void **)&d_output, sz);

    for (int i = 0; i < N_DIM; i++) {
        h_input[i] = (float)(i - N_DIM / 2);
    }

    cudaMemcpy(d_input, h_input, sz, cudaMemcpyHostToDevice);

    print_array("Input", h_input, N_DIM);

    solve(d_input, d_output, N_DIM);

    cudaMemcpy(h_output, d_output, sz, cudaMemcpyDeviceToHost);

    print_array("Output", h_output, N_DIM);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
