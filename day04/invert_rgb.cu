#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 4
#define HEIGHT 3

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int pixel = blockDim.x * blockIdx.x + threadIdx.x;
    if (pixel < width * height) {
        int idx = pixel * 4;
        image[idx + 0] = 255 - image[idx + 0]; // R
        image[idx + 1] = 255 - image[idx + 1]; // G
        image[idx + 2] = 255 - image[idx + 2]; // B
    }
}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}

static void print_image(const char *title, const unsigned char *image, int width, int height)
{
    printf("%s (%dx%d):\n", title, width, height);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = (i * width + j) * 4;
            printf("(%3d,%3d,%3d,%3d) ", image[idx + 0], image[idx + 1], image[idx + 2],
                   image[idx + 3]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void)
{
    unsigned char *h_image;
    unsigned char *d_image;
    int sz_image = WIDTH * HEIGHT * 4 * sizeof(unsigned char);

    /* Step 1: Allocate host memory */
    h_image = (unsigned char *)malloc(sz_image);

    /* Step 2: Allocate device memory */
    cudaMalloc((void **)&d_image, sz_image);

    /* Step 3: Initialize host input image (RGBA) */
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_image[i * 4 + 0] = (unsigned char)(i * 10 % 256);       // R
        h_image[i * 4 + 1] = (unsigned char)(i * 20 % 256);       // G
        h_image[i * 4 + 2] = (unsigned char)(i * 30 % 256);       // B
        h_image[i * 4 + 3] = 255;                                 // A
    }

    /* Step 4: Copy input image to device */
    cudaMemcpy(d_image, h_image, sz_image, cudaMemcpyHostToDevice);

    /* Step 5: Print input image */
    print_image("Input", h_image, WIDTH, HEIGHT);

    /* Step 6: Launch kernel via solve() */
    solve(d_image, WIDTH, HEIGHT);

    /* Step 7: Copy result back to host */
    cudaMemcpy(h_image, d_image, sz_image, cudaMemcpyDeviceToHost);

    /* Step 8: Print result */
    print_image("Inverted", h_image, WIDTH, HEIGHT);

    cudaFree(d_image);
    free(h_image);

    return 0;
}
