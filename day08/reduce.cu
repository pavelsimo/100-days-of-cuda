 #include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N_DIM 10

__global__ void reduce(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float a[256];
    a[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            a[tid] += a[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, a[0]); 
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    reduce<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
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
    float *h_input, h_output;
    float *d_input, *d_output;
    int sz = N_DIM * sizeof(float);

    h_input = (float *)malloc(sz);

    cudaMalloc((void **)&d_input, sz);
    cudaMalloc((void **)&d_output, sizeof(float));

    for (int i = 0; i < N_DIM; i++) {
        h_input[i] = (float)(i + 1);
    }

    cudaMemcpy(d_input, h_input, sz, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float));

    print_array("Input", h_input, N_DIM);

    solve(d_input, d_output, N_DIM);

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sum: %6.1f\n\n", h_output);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);

    return 0;
}
