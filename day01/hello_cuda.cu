#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello_from_gpu() {
  int warpId = threadIdx.x / warpSize;
  printf("Block ID is %d --- Thread ID is %d --- Warp ID is %d.\n", 
    blockIdx.x, threadIdx.x, warpId);
}

int main(void) {
  
  // kernel name <<<number of blocks, number of threads per block>>>();
  hello_from_gpu <<<2, 64>>>();
  cudaDeviceSynchronize();
  return 0;
}
