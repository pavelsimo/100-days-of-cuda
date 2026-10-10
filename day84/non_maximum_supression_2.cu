#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

__global__ void fill(int* keep, int value, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        keep[i] = value;
    }
}

__device__ __forceinline__ float intersect(const float4 a, const float4 b) {
    const float a_x1 = a.x;
    const float a_y1 = a.y;
    const float a_x2 = a.z;
    const float a_y2 = a.w;
    const float b_x1 = b.x;
    const float b_y1 = b.y;
    const float b_x2 = b.z;
    const float b_y2 = b.w;
    float w = fmaxf(0, fminf(a_x2, b_x2) - fmaxf(a_x1, b_x1));
    float h = fmaxf(0, fminf(a_y2, b_y2) - fmaxf(a_y1, b_y1));
    return w * h;
}

__device__ __forceinline__ float length(const float4 a) {
    const float a_x1 = a.x;
    const float a_y1 = a.y;
    const float a_x2 = a.z;
    const float a_y2 = a.w;
    return (a_x2 - a_x1) * (a_y2 - a_y1);
}

__device__ __forceinline__ float iou(const float4 a, const float4 b) {
    return intersect(a, b) / (length(a) + length(b) - intersect(a, b));
}

__global__ void supress_boxes(const float* boxes, const float* scores_sorted, const int* scores_sorted_idx, int* keep, float iou_threshold, int i, int N) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    const float4 a = reinterpret_cast<const float4*>(boxes)[scores_sorted_idx[i]];
    if (j >= N || i >= j || keep[scores_sorted_idx[i]] == 0) {
        return;
    }
    
    const float4 b = reinterpret_cast<const float4*>(boxes)[scores_sorted_idx[j]];
    float iou_value = iou(a, b);
    if (iou_value > iou_threshold) {
        keep[scores_sorted_idx[j]] = 0; 
    }
}

// boxes, scores, keep are device pointers
extern "C" void solve(const float* boxes, const float* scores, int* keep, int N,
                      float iou_threshold) {
               
    float *scores_sorted;
    cudaMalloc(&scores_sorted, N * sizeof(float));
    int *scores_sorted_idx;
    cudaMalloc(&scores_sorted_idx, N * sizeof(int));
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    thrust::copy(thrust::device, scores, scores + N, scores_sorted);
    thrust::sequence(thrust::device, scores_sorted_idx, scores_sorted_idx + N);
    thrust::stable_sort_by_key(thrust::device, scores_sorted, scores_sorted + N, scores_sorted_idx, thrust::greater<float>());
    fill<<<blocks, threads>>>(keep, 1, N);
    for (int i = 0; i < N; ++i) {
        int idx, keep_value;
        cudaMemcpy(&idx, scores_sorted_idx + i, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&keep_value, keep + idx, sizeof(int), cudaMemcpyDeviceToHost);
        if (keep_value == 0) {
            continue;
        }
        supress_boxes<<<blocks, threads>>>(boxes, scores_sorted, scores_sorted_idx, keep, iou_threshold, i, N);
    }
    cudaDeviceSynchronize();
    cudaFree(scores_sorted);
    cudaFree(scores_sorted_idx);
}
