#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 4
#define HEIGHT 3

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) {
    int pixel = blockDim.x * blockIdx.x + threadIdx.x;
    if (pixel < width * height) {
        int idx = pixel * 3;
        float R = input[idx];
        float G = input[idx + 1];
        float B = input[idx + 2];
        output[pixel] = 0.299 * R + 0.587 * G + 0.114 * B;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}

static void print_rgb_image(const char *title, const float *image, int width, int height)
{
    printf("%s (%dx%d):\n", title, width, height);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = (i * width + j) * 3;
            printf("(%6.1f,%6.1f,%6.1f) ", image[idx + 0], image[idx + 1], image[idx + 2]);
        }
        printf("\n");
    }
    printf("\n");
}

static void print_grayscale_image(const char *title, const float *image, int width, int height)
{
    printf("%s (%dx%d):\n", title, width, height);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%8.3f ", image[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void)
{
    float *h_input, *h_output;
    float *d_input, *d_output;
    int sz_input = WIDTH * HEIGHT * 3 * sizeof(float);
    int sz_output = WIDTH * HEIGHT * sizeof(float);

    /* Step 1: Allocate host memory */
    h_input = (float *)malloc(sz_input);
    h_output = (float *)malloc(sz_output);

    /* Step 2: Allocate device memory */
    cudaMalloc((void **)&d_input, sz_input);
    cudaMalloc((void **)&d_output, sz_output);

    /* Step 3: Initialize host input image (RGB) */
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_input[i * 3 + 0] = (float)(i * 10 % 256); // R
        h_input[i * 3 + 1] = (float)(i * 20 % 256); // G
        h_input[i * 3 + 2] = (float)(i * 30 % 256); // B
    }

    /* Step 4: Copy input image to device */
    cudaMemcpy(d_input, h_input, sz_input, cudaMemcpyHostToDevice);

    /* Step 5: Print input image */
    print_rgb_image("Input", h_input, WIDTH, HEIGHT);

    /* Step 6: Launch kernel via solve() */
    solve(d_input, d_output, WIDTH, HEIGHT);

    /* Step 7: Copy result back to host */
    cudaMemcpy(h_output, d_output, sz_output, cudaMemcpyDeviceToHost);

    /* Step 8: Print result */
    print_grayscale_image("Grayscale", h_output, WIDTH, HEIGHT);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
