# 100 days of CUDA challenge

This is a 100-day challenge to master CUDA:

- https://github.com/hkproj/100-days-of-gpu/blob/main/CUDA.md

## Progress 

### Day 1

- Learned that CUDA stands for **C**ompute **U**nified **D**evice **A**rchitecture.

- The CUDA compiler is called NVCC (**N**VIDIA **C**UDA **C**ompiler).

- CUDA is a platform with different levels of abstraction. NVIDIA has put a ton of work into developing libraries to make developers' lives easier, for example:
  - **cuBLAS** linear algebra
  - **cuFFT** fast fourier transform
  - **cuDNN** neural networks
  - **cuRAND** random numbers

- Found the NVIDIA [accelerated-computing-hub](https://github.com/NVIDIA/accelerated-computing-hub/tree/main) resource, plenty of courses to choose from, thinking of doing the Python ones.

- Watched the video [What's CUDA All About Anyway?](https://www.nvidia.com/en-us/on-demand/session/gtc25-S72571/), a really great introduction to CUDA.

- Created my first two CUDA kernels:
  - [hello_cuda.cu](https://github.com/pavelsimo/100-days-of-cuda/blob/main/day01/hello_cuda.cu)
  - [vector_addition.cu](https://github.com/pavelsimo/100-days-of-cuda/blob/main/day01/vector_addition.cu)
