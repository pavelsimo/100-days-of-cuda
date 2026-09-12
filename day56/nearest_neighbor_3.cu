#include <cuda_runtime.h>
#include <float.h>

#define TILE_SIZE 256

__global__ void nearest_neighbor(const float* __restrict__ points,
                                 int* __restrict__ indices,
                                 int N) {
    __shared__ float3 tile[TILE_SIZE];
    const float3* __restrict__ p3 = reinterpret_cast<const float3*>(points);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool ok = idx < N;
    float3 pi = ok ? p3[idx] : make_float3(0.0f, 0.0f, 0.0f);
    int nearest = -1;
    float best = FLT_MAX;
    for (int base = 0; base < N; base += TILE_SIZE) {
        int p_idx = base + threadIdx.x;
        if (p_idx < N) {
            tile[threadIdx.x] = p3[p_idx];
        }
        __syncthreads();
        
        int count = min(TILE_SIZE, N - base);
        for (int t = 0; t < count; ++t) {
            float3 pj = tile[t];
            float dx = pi.x - pj.x;
            float dy = pi.y - pj.y;
            float dz = pi.z - pj.z;
            float sqr_dist = dx * dx + dy * dy + dz * dz;
            int j = base + t;
            if (sqr_dist < best && j != idx) {
                best = sqr_dist;
                nearest = j;
            }
        }
        __syncthreads();
    }

    if (ok) {
        indices[idx] = nearest;
    }
}

// points and indices are device pointers
extern "C" void solve(const float* points, int* indices, int N) {
    int threads = TILE_SIZE;
    int blocks = (N + threads - 1) / threads;
    nearest_neighbor<<<blocks, threads>>>(points, indices, N);
    cudaDeviceSynchronize();
}
