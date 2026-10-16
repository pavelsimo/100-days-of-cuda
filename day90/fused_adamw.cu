#include <cuda_runtime.h>

__global__ void adamw_kernel(float* __restrict__ params, const float* __restrict__ grad,
                             float* __restrict__ m, float* __restrict__ v, int N, float lr,
                             float beta1, float beta2, float eps, float weight_decay, int t) {
    // θ is params
    // g is grad
    // η is the learning rate lr
    // λ is weight_decay
    // β1 is beta1
    // β2 is beta2 
    // ε is eps
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N) {
        const float g = grad[n];
        const float g_sqr = g * g;
        const float alpha1 = (1 - beta1) * g;
        const float alpha2 = (1 - beta2) * g_sqr;
        const float alpha3 = 1.0f / (1.0f - powf(beta1, t));
        const float alpha4 = 1.0f / (1.0f - powf(beta2, t));
        params[n] = params[n] * (1 - lr * weight_decay);
        m[n] = beta1 * m[n] + alpha1;
        v[n] = beta2 * v[n] + alpha2;
        float m_hat = m[n] * alpha3;
        float v_hat = v[n] * alpha4;
        params[n] = params[n] - lr * m_hat / (sqrtf(v_hat) + eps);
    }                        
}
// params, grad, m, v are device pointers
extern "C" void solve(float* params, const float* grad, float* m, float* v, int N, float lr,
                      float beta1, float beta2, float eps, float weight_decay, int t) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    adamw_kernel<<<blocks, threads>>>(params, grad, m, v, N, lr, beta1, beta2, eps, weight_decay, t);
    cudaDeviceSynchronize();
}
