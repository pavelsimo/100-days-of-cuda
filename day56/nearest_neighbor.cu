#include <cuda_runtime.h>

__global__ void nearest_neighbor(const float* points, int* indices, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int nearest = -1;
    float best = 1e30f;
    for (int i = 0; i < N; ++i) {
        if (i == idx) continue;
        float sqr_dist = 0.0f;
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            float diff = points[3 * idx + j] - points[3 * i + j];
            sqr_dist += diff * diff;
        }
        if (sqr_dist < best) {
            best = sqr_dist;
            nearest = i;
        }
    }
    indices[idx] = nearest;
}

// points and indices are device pointers
extern "C" void solve(const float* points, int* indices, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    nearest_neighbor<<<blocks, threads>>>(points, indices, N);
    cudaDeviceSynchronize();
}
