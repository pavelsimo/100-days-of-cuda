#include <cuda_runtime.h>
#include <math.h>

__global__ void ssm_selective_scan(const float* u, const float* delta, const float* A, const float* B,
                          const float* C, const float* skip, float* y, int batch, int seq_len,
                          int d_model, int d_state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * d_model * d_state;
    if (idx >= total) {
        return;
    }
    int b = idx / (d_state * d_model);
    int d = (idx / d_state) % d_model;
    int n = idx % d_state;
    float h = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
        int btd = b * (seq_len*d_model) + t*(d_model) + d;
        int btn = b * (seq_len*d_state) + t*(d_state) + n;
        int btdn = b * (seq_len*d_model*d_state) + t * (d_model*d_state) + d * (d_state) + n;
        int dn = d * d_state + n;
        float A_hat = expf(delta[btd] * A[dn]);
        float B_hat = delta[btd] * B[btn];
        h = A_hat * h + B_hat * u[btd];
        float skip_val = (n == 0) ? skip[d] * u[btd] : 0.0f;
        atomicAdd(&y[btd], C[btn] * h + skip_val);
    }
}

// u, delta, A, B, C, skip, y are device pointers
extern "C" void solve(const float* u, const float* delta, const float* A, const float* B,
                      const float* C, const float* skip, float* y, int batch, int seq_len,
                      int d_model, int d_state) {
    int threads = 256;
    int blocks = (batch * d_model * d_state + threads - 1) / threads;
    cudaMemset(y, 0, batch * seq_len * d_model * sizeof(float));
    ssm_selective_scan<<<blocks, threads>>>(u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state);
    cudaDeviceSynchronize();
}
