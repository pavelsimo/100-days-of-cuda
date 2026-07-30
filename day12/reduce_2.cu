#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N_DIM 10

__global__ void reduce(const float* input, float* output, int N) {
    __shared__ float t[32];
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float val = 0.0f;;
    unsigned mask = 0xFFFFFFFFU;
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    int stride = gridDim.x * blockDim.x;
    while (idx + 7 * stride < N) {
        val += input[idx];
        val += input[idx + 1 * stride];
        val += input[idx + 2 * stride];
        val += input[idx + 3 * stride];
        val += input[idx + 4 * stride];
        val += input[idx + 5 * stride];
        val += input[idx + 6 * stride];
        val += input[idx + 7 * stride];
        idx += 8 * stride;
    }
    
    while (idx < N) {
        val += input[idx];
        idx += stride;
    }

    // 1st warp-shuffle reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }

    if (lane == 0) {
        t[warpId] = val;
    }
    __syncthreads(); // put warp results into shared memory

    if (warpId == 0) {
        val = (tid < blockDim.x / warpSize) ? t[lane] : 0.0f;
        // 2nd warp-shuffle reduction
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }

        if (tid == 0) {
            atomicAdd(output, val);
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock  = 256;
    int elementsPerBlock = 256 * 8;   // each thread handles 8 elements = 2048 per block
    int blocksPerGrid    = (N + elementsPerBlock - 1) / elementsPerBlock;
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
