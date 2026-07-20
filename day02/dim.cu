#include <cuda_runtime.h>
#include <stdio.h>

__global__ void print_thread_ids() {
  //printf("threadIdx.x = %d, threadIdx.y = %d, threadIdx.z = %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
  printf("blockIdx.x = %d --- blockIdx.y = %d --- blockIdx.z = %d --- blockDim.x = %d --- blockDim.y = %d --- gridDim.x = %d --- gridDim.y = %d\n", 
    blockIdx.x, 
    blockIdx.y, 
    blockIdx.z, 
    blockDim.x, 
    blockDim.y,
    gridDim.x, 
    gridDim.y);
}

int main(void) {
  int nx = 4, ny = 4, nz = 4;
  dim3 block(2, 2, 2);
  dim3 grid(nx / block.x, ny / block.y, nz / block.z);

  print_thread_ids<<<grid, block>>>();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}