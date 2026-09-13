#include <cuda_runtime.h>

__global__ void floyd_warshall(float* dist, int N, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) 
        return;
    float dist_ik = dist[i*N + k];
    float dist_kj = dist[k*N + j];
    if(dist_ik != INFINITY && dist_kj != INFINITY) {
        dist[i*N + j] = fmin(dist[i*N + j], dist_ik + dist_kj);
    }
}

// dist, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* dist, float* output, int N) {
    cudaMemcpy(output, dist, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);
    for (int k = 0; k < N; k++) {
        floyd_warshall<<<blocks, threads>>>(output, N, k);
    }
    cudaDeviceSynchronize();
}
