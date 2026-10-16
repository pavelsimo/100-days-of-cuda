#include <cuda_runtime.h>

__device__ __forceinline__ void adamw_step(float& p, const float g, float& m, float& v, float lr,
                                           float beta1, float beta2, float eps, float weight_decay,
                                           float alpha3, float alpha4) {
    const float g_sqr = g * g;
    const float alpha1 = (1 - beta1) * g;
    const float alpha2 = (1 - beta2) * g_sqr;
    p = p * (1 - lr * weight_decay);
    m = beta1 * m + alpha1;
    v = beta2 * v + alpha2;
    float m_hat = m * alpha3;
    float v_hat = v * alpha4;
    p = p - lr * m_hat / (sqrtf(v_hat) + eps);
}

__global__ void adamw_kernel(float* params, const float* grad, float* m, float* v, int N, float lr,
                             float beta1, float beta2, float eps, float weight_decay, int t) {
    // θ is params
    // g is grad
    // η is the learning rate lr
    // λ is weight_decay
    // β1 is beta1
    // β2 is beta2
    // ε is eps
    int n = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    const float alpha3 = 1.0f / (1.0f - powf(beta1, t));
    const float alpha4 = 1.0f / (1.0f - powf(beta2, t));
    if (n + 3 < N) {
        float4 params4 = *reinterpret_cast<float4*>(&params[n]);
        const float4 grad4 = *reinterpret_cast<const float4*>(&grad[n]);
        float4 m4 = *reinterpret_cast<float4*>(&m[n]);
        float4 v4 = *reinterpret_cast<float4*>(&v[n]);
        adamw_step(params4.x, grad4.x, m4.x, v4.x, lr, beta1, beta2, eps, weight_decay, alpha3, alpha4);
        adamw_step(params4.y, grad4.y, m4.y, v4.y, lr, beta1, beta2, eps, weight_decay, alpha3, alpha4);
        adamw_step(params4.z, grad4.z, m4.z, v4.z, lr, beta1, beta2, eps, weight_decay, alpha3, alpha4);
        adamw_step(params4.w, grad4.w, m4.w, v4.w, lr, beta1, beta2, eps, weight_decay, alpha3, alpha4);
        *reinterpret_cast<float4*>(&params[n]) = params4;
        *reinterpret_cast<float4*>(&m[n]) = m4;
        *reinterpret_cast<float4*>(&v[n]) = v4;
    } else {
        for (int i = n; i < N; ++i) {
            adamw_step(params[i], grad[i], m[i], v[i], lr, beta1, beta2, eps, weight_decay, alpha3, alpha4);
        }
    }
}
// params, grad, m, v are device pointers
extern "C" void solve(float* params, const float* grad, float* m, float* v, int N, float lr,
                      float beta1, float beta2, float eps, float weight_decay, int t) {
    int threads = 256;
    int blocks = ((N + 3) / 4 + threads - 1) / threads;
    adamw_kernel<<<blocks, threads>>>(params, grad, m, v, N, lr, beta1, beta2, eps, weight_decay, t);
    cudaDeviceSynchronize();
}
