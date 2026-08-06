#include <cuda_runtime.h>
#include <float.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))


__global__ void max_pooling(const float* input, float* output, int N, int C, int H, int W,
                            int H_out, int W_out, int kernel_size, int stride, int padding) {
    int w_out = blockDim.x * blockIdx.x + threadIdx.x;
    int h_out = blockDim.y * blockIdx.y + threadIdx.y;

    if (h_out < H_out && w_out < W_out) {
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                float max_value = -FLT_MAX;
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int h = h_out * stride - padding + kh;
                        int w = w_out * stride - padding + kw;
                        if (h >= 0  && h < H && w >= 0 && w < W) {
                            max_value = MAX(max_value, input[((n * C + c) * H + h) * W + w]);
                        }
                    }
                }
                output[((n * C + c) * H_out + h_out) * W_out + w_out] = max_value;
            }
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    dim3 threadsPerBlock(16, 16);

    dim3 blocksPerGrid(
        (W_out + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (H_out + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    max_pooling<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, N, C, H, W, H_out, W_out, kernel_size, stride, padding
    );
    cudaDeviceSynchronize();
}
