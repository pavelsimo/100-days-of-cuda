#include <cuda_runtime.h>

__global__ void linear_recurrence(const float* __restrict__ a, const float* __restrict__ x, float* __restrict__ h, int B, int L) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) {
        return;
    }

    float sum = x[b * L];
    h[b * L] = sum;
    for (int t = 1; t < L; ++t) {
        sum = sum * a[b * L + t] + x[b * L + t];
        h[b * L + t] = sum;
    }
}

// a, x, h are device pointers
extern "C" void solve(const float* a, const float* x, float* h, int B, int L) {
    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    // h[t] = a[t] · h[t-1] + x[t]
    //
    // h[0] = x[0]
    // h[1] = a[1] · (x[0]) + x[1]
    // h[2] = a[2] · (a[1] · x[0] + x[1]) + x[2]
    // h[3] = a[3] · (a[2] · (a[1] · x[0] + x[1]) + x[2]) + x[3]
    //
    // ...
    //
    // h[0] = x[0]
    //
    // h[1] = x[1]
    //      + a[1] * x[0]
    //
    // h[2] = x[2]
    //      + a[2] * x[1]
    //      + a[2] * a[1] * x[0]
    //
    // h[3] = x[3]
    //      + a[3] * x[2]
    //      + a[3] * a[2] * x[1]
    //      + a[3] * a[2] * a[1] * x[0]

    linear_recurrence<<<blocks, threads>>>(a, x, h, B, L);
    cudaDeviceSynchronize();
}

