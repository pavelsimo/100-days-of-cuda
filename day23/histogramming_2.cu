#include <cuda_runtime.h>
#include <stdio.h>

__global__ void histogram_kernel(const int* input, int* histogram, int N, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int freq[1024];
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        freq[i] = 0;
    }
    __syncthreads();

    if (idx < N) {
        atomicAdd(&freq[input[idx]], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&histogram[i], freq[i]);
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaMemset(histogram, 0, num_bins * sizeof(int));
    histogram_kernel<<<blocks, threads>>>(input, histogram, N, num_bins);
    cudaDeviceSynchronize();
}

static bool run_test(const int* input, int N, int num_bins, const int* expected) {
    int* d_input;
    int* d_histogram;
    int* histogram = new int[num_bins];

    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_histogram, num_bins * sizeof(int));
    cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, num_bins * sizeof(int));

    solve(d_input, d_histogram, N, num_bins);
    cudaMemcpy(histogram, d_histogram, num_bins * sizeof(int), cudaMemcpyDeviceToHost);

    bool passed = true;
    printf("Output:   [");
    for (int i = 0; i < num_bins; ++i) {
        printf("%d%s", histogram[i], i + 1 == num_bins ? "" : ", ");
        if (histogram[i] != expected[i]) {
            passed = false;
        }
    }
    printf("]\nExpected: [");
    for (int i = 0; i < num_bins; ++i) {
        printf("%d%s", expected[i], i + 1 == num_bins ? "" : ", ");
    }
    printf("]\n%s\n\n", passed ? "PASS" : "FAIL");

    cudaFree(d_input);
    cudaFree(d_histogram);
    delete[] histogram;

    return passed;
}

int main() {
    const int input_1[] = {0, 1, 2, 1, 0};
    const int expected_1[] = {2, 2, 1};

    const int input_2[] = {3, 3, 3, 3};
    const int expected_2[] = {0, 0, 0, 4, 0};

    bool passed = true;
    passed &= run_test(input_1, 5, 3, expected_1);
    passed &= run_test(input_2, 4, 5, expected_2);

    return passed ? 0 : 1;
}
