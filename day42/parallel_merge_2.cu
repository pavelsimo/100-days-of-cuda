#include <cuda_runtime.h>
#include <cstdio>

__device__ int lower_bound(const float *arr, float target, int N) {
    int lo = 0;
    int hi = N - 1;
    int res = N;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] >= target) {
            res = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    } 
    return res;
}

__device__ int upper_bound(const float *arr, float target, int N) {
    int lo = 0;
    int hi = N - 1;
    int res = N;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] > target) {
            res = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    } 
    return res;
}

__global__ void merge(const float *A, const float *B, float *C, int M, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < M) {
        int idx = i + lower_bound(B, A[i], N);
        C[idx] = A[i];
    } else if (i < M + N) {
        int j = i - M;
        int idx = j + upper_bound(A, B[j], M);
        C[idx] = B[j];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    int threads = 256;
    merge<<<(M + N + threads - 1) / threads, threads>>>(A, B, C, M, N);
    cudaDeviceSynchronize();
}

static void run_test(const float *A, int M, const float *B, int N, const float *expected) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, (M + N) * sizeof(float));
    cudaMemcpy(d_A, A, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_A, d_B, d_C, M, N);

    float *C = new float[M + N];
    cudaMemcpy(C, d_C, (M + N) * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    printf("C = [");
    for (int i = 0; i < M + N; ++i) {
        printf("%s%.1f", i ? ", " : "", C[i]);
        if (C[i] != expected[i]) ok = false;
    }
    printf("] -> %s\n", ok ? "PASS" : "FAIL");

    delete[] C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    const float A1[] = {1.0f, 3.0f, 5.0f, 7.0f};
    const float B1[] = {2.0f, 4.0f, 6.0f, 8.0f};
    const float E1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    run_test(A1, 4, B1, 4, E1);

    const float A2[] = {-1.0f, 1.0f, 3.0f};
    const float B2[] = {2.0f};
    const float E2[] = {-1.0f, 1.0f, 2.0f, 3.0f};
    run_test(A2, 3, B2, 1, E2);

    return 0;
}
