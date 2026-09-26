#include <cuda_runtime.h>

#define TILE_SIZE 256

__global__ void simulate(const float* __restrict__ agents, float* __restrict__ agents_next, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool ok = idx < N;
    const float alpha = 0.05f;
    const float neighbor_sqr_dist = 25.0f;
    __shared__ float4 tile[TILE_SIZE];
    float4* __restrict__  agents4 = reinterpret_cast<float4*>(const_cast<float*>(agents));
    float4* __restrict__ agents4_next = reinterpret_cast<float4*>(agents_next);
    float4 a = ok ? agents4[idx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float x = a.x;
    float y = a.y;
    float vx = a.z;
    float vy = a.w;
    float vx_avg = 0.0f;
    float vy_avg = 0.0f;
    int neighbor_count = 0;
    for (int base = 0; base < N; base += TILE_SIZE) {
        int k = base + threadIdx.x;
        if (k < N) {
            tile[threadIdx.x] = agents4[k];
        }
        __syncthreads();
        
        int count = min(TILE_SIZE, N - base);
        for (int j = 0; j < count; ++j) {
            float4 b = tile[j];
            float nx = b.x;
            float ny = b.y;
            float nvx = b.z;
            float nvy = b.w;
            float sqrt_dist = (nx - x) * (nx - x) + (ny - y) * (ny - y);
            if (base + j != idx && sqrt_dist < neighbor_sqr_dist) {
                vx_avg += nvx;
                vy_avg += nvy;
                neighbor_count++;
            }
        
        }
        __syncthreads();
    }

    if (neighbor_count > 0) {
        vx_avg /= neighbor_count;
        vy_avg /= neighbor_count;
    } else {
        vx_avg = vx;
        vy_avg = vy;
    }

    if (ok) {
        float vx_new = vx + alpha * (vx_avg - vx);
        float vy_new = vy + alpha * (vy_avg - vy);
        agents4_next[idx] = make_float4(x + vx_new, y + vy_new, vx_new, vy_new);
    }
}

// agents, agents_next are device pointers
extern "C" void solve(const float* agents, float* agents_next, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    simulate<<<blocks, threads>>>(agents, agents_next, N);
    cudaDeviceSynchronize();
}
