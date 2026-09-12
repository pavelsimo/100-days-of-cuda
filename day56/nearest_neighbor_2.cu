#include <cuda_runtime.h>
#include <float.h>

__global__ void nearest_neighbor(const float* points, int* indices, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    int nearest = -1;
    float best = FLT_MAX;
    const float3* p3 = reinterpret_cast<const float3*>(points);
    float3 pi = p3[idx];
    for (int i = 0; i < N; ++i) {
        float3 pj = p3[i];
        float sqr_dist = (pi.x - pj.x) * (pi.x - pj.x) 
            + (pi.y - pj.y) * (pi.y - pj.y) 
            + (pi.z - pj.z) * (pi.z - pj.z);
        if (sqr_dist < best && i != idx) {
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
