#include <cuda_runtime.h>

__global__ void patch_embedding(const float* images, const float* patch_weight, const float* patch_bias,
                                const float* cls_token, const float* pos_embed, float* output,
                                int B, int C, int H, int W, int P, int D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N * D) {
        return;
    }

    int b = idx / (N*D);
    int d = idx % D;
    int n = (idx / D) % N;
    int gw = W / P;
    int py = n / gw * P;
    int px = n % gw * P;
    float token = patch_bias[d];
    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < P; ++i) {
            for (int j = 0; j < P; ++j) {
                int k1 = b * (C*H*W) + c * (H*W) + (py + i) * W + (px + j);
                int k2 = d * (C*P*P) + c * (P*P) + i * P + j;
                token += images[k1] * patch_weight[k2];
            }
        }
    }
    output[b * ((N + 1) * D) + (n + 1) * D + d] = token + pos_embed[(n + 1) * D + d];
    if (n == 0) {
        output[b * ((N + 1) * D) + d] = cls_token[d] + pos_embed[d];
    }
}

// images, patch_weight, patch_bias, cls_token, pos_embed, output are device pointers
extern "C" void solve(const float* images, const float* patch_weight, const float* patch_bias,
                      const float* cls_token, const float* pos_embed, float* output, int B, int C,
                      int H, int W, int P, int D) {
    // images  - [C, H, W]
    // patches - [N, C, P, P], N = (H / P) * (W / P)
    // output  - [B, N, D]
    int N = (H / P) * (W / P);
    int threads = 256;
    int blocks = (B * (N + 1) * D + threads - 1) / threads;
    patch_embedding<<<blocks, threads>>>(images, patch_weight, patch_bias, cls_token, pos_embed, output, B, C, H, W, P, D, N);
    cudaDeviceSynchronize();
}
