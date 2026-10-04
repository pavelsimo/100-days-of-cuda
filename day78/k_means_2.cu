#include <cuda_runtime.h>


__global__ void assign_labels(const float* data_x, const float* data_y, int* labels,
                      float* initial_centroid_x, float* initial_centroid_y, float* final_centroid_x,
                      float* final_centroid_y, int sample_size, int k, int* done) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sample_size) {
        return;
    }

    float best_distance = INFINITY;
    int best_cluster = 0;
    const float x = data_x[i];
    const float y = data_y[i];
    for (int c = 0; c < k; ++c) {
        const float dx = x - final_centroid_x[c];
        const float dy = y - final_centroid_y[c];
        float sqr_dist = dx*dx + dy*dy;
        if (sqr_dist < best_distance) {
            best_distance = sqr_dist;
            best_cluster = c;
            *done = 0;
        }
    }
    labels[i] = best_cluster;
}

__global__ void accumulate(const float* data_x, const float* data_y, int* labels, float* sum_data_x, float* sum_data_y, int* counts, float* final_centroid_x, float* final_centroid_y, int sample_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sample_size) {
        return;
    }

    int c = labels[i];
    atomicAdd(&sum_data_x[c], data_x[i]);
    atomicAdd(&sum_data_y[c], data_y[i]);
    atomicAdd(&counts[c], 1);
}

__global__ void update_centroids(float* sum_data_x, float* sum_data_y, int* counts, float* final_centroid_x, float* final_centroid_y, int k) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k || counts[c] <= 0) {
        return;
    }

    final_centroid_x[c] = sum_data_x[c] / counts[c];
    final_centroid_y[c] = sum_data_y[c] / counts[c];
}


// data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
// final_centroid_x, final_centroid_y are device pointers
extern "C" void solve(const float* data_x, const float* data_y, int* labels,
                      float* initial_centroid_x, float* initial_centroid_y, float* final_centroid_x,
                      float* final_centroid_y, int sample_size, int k, int max_iterations) {
    int threads = 256;
    int blocks_1 = (sample_size + threads - 1) / threads;
    int blocks_2 = (k + threads - 1) / threads;

    float* sum_data_x;
    float* sum_data_y;
    int* counts;
    int* d_done;
    int h_done;
    cudaMalloc(&d_done, sizeof(int));
    cudaMalloc(&sum_data_x, k * sizeof(float));
    cudaMalloc(&sum_data_y, k * sizeof(float));
    cudaMalloc(&counts, k * sizeof(int));
    cudaMemcpy(final_centroid_x, initial_centroid_x, k * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(final_centroid_y, initial_centroid_y, k * sizeof(float), cudaMemcpyDeviceToDevice);
    for (int step = 0; step < max_iterations; ++step) {
        cudaMemset(sum_data_x, 0, k * sizeof(float));
        cudaMemset(sum_data_y, 0, k * sizeof(float));
        cudaMemset(counts, 0, k * sizeof(int));
        cudaMemset(d_done, 1, sizeof(int));

        assign_labels<<<blocks_1, threads>>>(data_x, data_y, labels, initial_centroid_x, initial_centroid_y, final_centroid_x, final_centroid_y, sample_size, k, d_done);
        accumulate<<<blocks_1, threads>>>(data_x, data_y, labels, sum_data_x, sum_data_y, counts, final_centroid_x, final_centroid_y, sample_size);
        update_centroids<<<blocks_2, threads>>>(sum_data_x, sum_data_y, counts, final_centroid_x, final_centroid_y, k);
        
        cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_done) {
            break;
        }
    }

    cudaDeviceSynchronize();
    cudaFree(sum_data_x);
    cudaFree(sum_data_y);
    cudaFree(counts);
    cudaFree(d_done);
}
