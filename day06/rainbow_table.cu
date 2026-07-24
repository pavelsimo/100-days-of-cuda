#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ unsigned int fnv1a_hash(unsigned int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;

    unsigned int hash = OFFSET_BASIS;

    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFFu;
        hash = (hash ^ byte) * FNV_PRIME;
    }

    return hash;
}

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        int value = input[idx];
        for(int r = 0; r < R; ++r) {
            value = fnv1a_hash(value);
        }
        output[idx] = value;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, R);
    cudaDeviceSynchronize();
}

#define N_DIM 8
#define ROUNDS 3

int main(void)
{
    int *h_input;
    unsigned int *h_output;
    int *d_input;
    unsigned int *d_output;
    int sz_in = N_DIM * sizeof(int);
    int sz_out = N_DIM * sizeof(unsigned int);

    /* Step 1: Allocate host memory */
    h_input = (int *)malloc(sz_in);
    h_output = (unsigned int *)malloc(sz_out);

    /* Step 2: Allocate device memory */
    cudaMalloc((void **)&d_input, sz_in);
    cudaMalloc((void **)&d_output, sz_out);

    /* Step 3: Initialize host input array */
    for (int i = 0; i < N_DIM; i++) {
        h_input[i] = i;
    }

    /* Step 4: Copy input array to device */
    cudaMemcpy(d_input, h_input, sz_in, cudaMemcpyHostToDevice);

    /* Step 5: Launch kernel via solve() */
    solve(d_input, d_output, N_DIM, ROUNDS);

    /* Step 6: Copy result back to host */
    cudaMemcpy(h_output, d_output, sz_out, cudaMemcpyDeviceToHost);

    /* Step 7: Print input and hashed output */
    printf("FNV-1a hash, %d rounds:\n", ROUNDS);
    for (int i = 0; i < N_DIM; i++) {
        printf("%4d -> %10u\n", h_input[i], h_output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
