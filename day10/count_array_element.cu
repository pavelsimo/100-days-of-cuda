#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N_DIM 10
#define K_VALUE 3

__global__ void count_element_kernel(const int* input, int* output, int N, int K) {
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ int t[];
    t[tid] = (i < N) ? ((input[i] == K) ? 1: 0) : 0;
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

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedBytes = threadsPerBlock * sizeof(int);
    count_element_kernel<<<blocksPerGrid, threadsPerBlock, sharedBytes>>>(input, output, N, K);
    cudaDeviceSynchronize();
}

static void print_array(const char *title, const int *a, int n)
{
    printf("%s (%d):\n", title, n);
    for (int i = 0; i < n; i++) {
        printf("%4d ", a[i]);
    }
    printf("\n\n");
}

int main(void)
{
    int *h_input, h_output;
    int *d_input, *d_output;
    int sz = N_DIM * sizeof(int);

    h_input = (int *)malloc(sz);

    cudaMalloc((void **)&d_input, sz);
    cudaMalloc((void **)&d_output, sizeof(int));

    for (int i = 0; i < N_DIM; i++) {
        h_input[i] = i % 4;
    }

    cudaMemcpy(d_input, h_input, sz, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(int));

    print_array("Input", h_input, N_DIM);

    solve(d_input, d_output, N_DIM, K_VALUE);

    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Count of %d: %d\n\n", K_VALUE, h_output);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);

    return 0;
}
