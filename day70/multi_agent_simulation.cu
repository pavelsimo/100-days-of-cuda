#include <cuda_runtime.h>

__global__ void simulate(const float* agents, float* agents_next, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float alpha = 0.05f;
    const float neighbor_sqr_dist = 25.0f;
    if (idx < N) {
        float x = agents[idx * 4];
        float y = agents[idx * 4 + 1];
        float vx = agents[idx * 4 + 2];
        float vy = agents[idx * 4 + 3];
        float vx_avg = 0.0f;
        float vy_avg = 0.0f;
        int neighbor_count = 0;
        for (int j = 0; j < N; ++j) {
            if (j != idx) {
                float nx = agents[j * 4];
                float ny = agents[j * 4 + 1];
                float nvx = agents[j * 4 + 2];
                float nvy = agents[j * 4 + 3];
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
        agents_next[idx * 4]     = x + vx_new;
        agents_next[idx * 4 + 1] = y + vy_new;
        agents_next[idx * 4 + 2] = vx_new;
        agents_next[idx * 4 + 3] = vy_new;
    }
}

// agents, agents_next are device pointers
extern "C" void solve(const float* agents, float* agents_next, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    simulate<<<blocks, threads>>>(agents, agents_next, N);
    cudaDeviceSynchronize();
}
