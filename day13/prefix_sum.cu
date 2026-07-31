#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N_DIM 10

__device__ float warp_inclusive_scan(float val) {
    unsigned mask = 0xffffffff;
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(mask, val, offset);
        if (lane >= offset) val += n;
    }
    return val;
}

__global__ void block_prefix_sum(const float* input, float* output, float* block_sums, int N) {
    __shared__ float warp_sums[32];
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    float val = (tid < N) ? input[tid] : 0.0f;
    val = warp_inclusive_scan(val);

    if (lane == 31) {
        warp_sums[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        float s = (lane * 32 < blockDim.x) ? warp_sums[lane] : 0.0f;
        s = warp_inclusive_scan(s);
        warp_sums[lane] = s;
    }
    __syncthreads();

    if (wid > 0) {
        val += warp_sums[wid - 1];
    }

    if (tid < N) {
        output[tid] = val;
    }

    if (block_sums != nullptr && threadIdx.x == blockDim.x - 1) {
        block_sums[blockIdx.x] = val;
    }
}

__global__ void add_block_offsets(float* output,
                                  const float* scanned_block_sums, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && tid < N) {
        output[tid] += scanned_block_sums[blockIdx.x - 1];
    }
}

static void prefix_sum(const float* input, float* output, int N) {
    const int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    if (blocksPerGrid == 1) {
        block_prefix_sum<<<1, threadsPerBlock>>>(input, output, nullptr, N);
        return;
    }

    float* block_sums;
    cudaMalloc(&block_sums, blocksPerGrid * sizeof(float));
    block_prefix_sum<<<blocksPerGrid, threadsPerBlock>>>(input, output, block_sums, N);
    prefix_sum(block_sums, block_sums, blocksPerGrid);
    add_block_offsets<<<blocksPerGrid, threadsPerBlock>>>(output, block_sums, N);

    cudaFree(block_sums);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    prefix_sum(input, output, N);
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
    float *h_input, *h_output;
    float *d_input, *d_output;
    int sz = N_DIM * sizeof(float);

    h_input  = (float *)malloc(sz);
    h_output = (float *)malloc(sz);

    cudaMalloc((void **)&d_input, sz);
    cudaMalloc((void **)&d_output, sz);

    for (int i = 0; i < N_DIM; i++) {
        h_input[i] = (float)(i + 1);
    }

    cudaMemcpy(d_input, h_input, sz, cudaMemcpyHostToDevice);

    print_array("Input", h_input, N_DIM);

    solve(d_input, d_output, N_DIM);

    cudaMemcpy(h_output, d_output, sz, cudaMemcpyDeviceToHost);

    print_array("Prefix Sum", h_output, N_DIM);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}