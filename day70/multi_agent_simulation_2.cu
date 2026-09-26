#include <cuda_runtime.h>

__global__ void simulate(const float* agents, float* agents_next, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float alpha = 0.05f;
    const float neighbor_sqr_dist = 25.0f;
    float4* agents4 = reinterpret_cast<float4*>(const_cast<float*>(agents));
    float4* agents4_next = reinterpret_cast<float4*>(agents_next);
    if (idx < N) {
        float4 a = agents4[idx];
        float x = a.x;
        float y = a.y;
        float vx = a.z;
        float vy = a.w;
        float vx_avg = 0.0f;
        float vy_avg = 0.0f;
        int neighbor_count = 0;
        for (int j = 0; j < N; ++j) {
            if (j != idx) {
                float4 b = agents4[j];
                float nx = b.x;
                float ny = b.y;
                float nvx = b.z;
                float nvy = b.w;
                float sqrt_dist = (nx - x) * (nx - x) + (ny - y) * (ny - y);
                if (sqrt_dist < neighbor_sqr_dist) {
                    vx_avg += nvx;
                    vy_avg += nvy;
                    neighbor_count++;
                }
            }
        }
        if (neighbor_count > 0) {
            vx_avg /= neighbor_count;
            vy_avg /= neighbor_count;
        } else {
            vx_avg = vx;
            vy_avg = vy;
        }
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
