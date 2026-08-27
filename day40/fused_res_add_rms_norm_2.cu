#include <cuda_runtime.h>


__global__ void matadd(const float *A, const float *B, float *C, int total) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx + 3 < total) {
        float4 a = *reinterpret_cast<const float4*>(A + idx);
        float4 b = *reinterpret_cast<const float4*>(B + idx);
        *reinterpret_cast<float4*>(C + idx) = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    } else {
        for (int k = idx; k < total; ++k) {
            C[k] = A[k] + B[k];
        }
    }
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float block_reduce_sum(float val) {
    __shared__ float t[32];
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    val = warp_reduce_sum(val);
    if (lane == 0) {
        t[warpId] = val;
    }
    __syncthreads();
    if (warpId == 0) {
        int numWarps = (blockDim.x + warpSize - 1) / warpSize;
        val = (threadIdx.x < numWarps) ? t[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void row_rms(const float *Z, int N, int C, float *rms, float eps) {
    int row = blockIdx.x;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < C; col += blockDim.x) {
        float z = Z[row * C + col];
        sum += z * z;
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        rms[row] = sqrtf(sum / static_cast<float>(C) + eps);
    }
}

__global__ void norm(const float* Z, const float* rms, const float* weight, float *out, int N, int C) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N && j < C) {
        out[i * C + j] = Z[i * C + j] * weight[j] / rms[i];
    }
}

// x, residual, weight, out are device pointers
extern "C" void solve(const float* x, const float* residual, const float* weight, float* out, int N,
                      int C, float eps) {
    
    // x - (N, C)
    // residual - (N, C)
    // N - num tokens
    // C - hidden dimension
    // weight - (C,)
    
    dim3 threads(16, 16);
    int totalThreads = threads.x * threads.y;
    dim3 grid(
        (C + threads.x - 1) / threads.x,
        (N + threads.y - 1) / threads.y
    );

    float *Z;
    float *rms;
    cudaMalloc(&rms, N * sizeof(float));
    cudaMalloc(&Z, N * C * sizeof(float));

    int total = N * C;
    int blocks = (total + totalThreads * 4 - 1) / (totalThreads * 4);
    matadd<<<blocks, threads.x * threads.y>>>(x, residual, Z, total);
    row_rms<<<N, threads.x * threads.y>>>(Z, N, C, rms, eps);
    norm<<<grid, threads>>>(Z, rms, weight, out, N, C);

    cudaFree(Z);
    cudaFree(rms);
}
