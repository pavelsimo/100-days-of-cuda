#include <cuda_runtime.h>

#define CELL_FREE 0
#define CELL_BLOCK 1

__global__ void init_vertices(int* dist, int start_row, int start_col, int rows, int cols) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < rows * cols) {
        int row = v / cols;
        int col = v % cols;
        if (row == start_row && col == start_col) {
            dist[row * cols + col] = 0;
        } else {
            dist[row * cols + col] = -1;
        }
    }
}

__global__ void bfs(const int* grid, int *dist, int* result, int rows, int cols, 
                    int start_row, int start_col, int end_row, int end_col, int current_depth, int *changed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) {
        return;
    }
    
    int row = idx / cols;
    int col = idx % cols;
    if (grid[row * cols + col] == CELL_BLOCK) {
        return;
    }

    if (dist[idx] != current_depth) {
        return;
    }

    int dr[] = {-1, 1, 0, 0};
    int dc[] = {0, 0, -1, 1};
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int nrow = row + dr[i];
        int ncol = col + dc[i];
        if (nrow < 0 || nrow >= rows || ncol < 0 || ncol >= cols) {
            continue;
        }
        int nxt_idx = nrow * cols + ncol;
        if (grid[nxt_idx] == CELL_FREE && dist[nxt_idx] == -1) {
            dist[nxt_idx] = current_depth + 1;
            *changed = 1;
        }
    }

    if (row == end_row && col == end_col) {
        *result = dist[row * cols + col];
    }
}

// grid, result are device pointers
extern "C" void solve(const int* grid, int* result, int rows, int cols, int start_row,
                      int start_col, int end_row, int end_col) {
    int *dist;
    int *changed;
    int threads = 256;
    int blocks = (rows * cols + threads - 1) / threads;
    cudaMalloc(&dist, rows * cols * sizeof(int));
    cudaMalloc(&changed, sizeof(int));
    init_vertices<<<blocks, threads>>>(dist, start_row, start_col, rows, cols);
    cudaMemset(result, -1, sizeof(int));

    int h_changed = 1;
    for (int current_depth = 0; h_changed; current_depth++) {
        cudaMemset(changed, 0, sizeof(int));
        bfs<<<blocks, threads>>>(grid, dist, result, rows, cols, start_row, start_col, end_row, end_col, current_depth, changed);
        cudaMemcpy(&h_changed, changed, sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaFree(dist);
    cudaFree(changed);
    cudaDeviceSynchronize();
}
