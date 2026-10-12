#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    static cudnnHandle_t handle = nullptr;
    static cudnnTensorDescriptor_t desc = nullptr;
    static int desc_n = -1;
    if (!handle) {
        cudnnCreate(&handle);
        cudnnCreateTensorDescriptor(&desc);
    }
    if (desc_n != N) {
        cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, N, 1, 1);
        desc_n = N;
    }
    const float alpha = 1.0f, beta = 0.0f;
    cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                        &alpha, desc, input,
                        &beta, desc, output);
    cudaDeviceSynchronize();
}
