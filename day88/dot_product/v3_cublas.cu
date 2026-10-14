#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>


// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    }
    cublasSdot(handle, N, A, 1, B, 1, result);
    cudaDeviceSynchronize();
}
