# Programming Massively Parallel Processors: Chapter 2, Heterogeneous Data-Parallel Computing

1. [Data Parallelism vs. Task Parallelism](#1-data-parallelism-vs-task-parallelism)
   - [Data Parallelism](#data-parallelism)
   - [Task Parallelism](#task-parallelism)
   - [Comparison](#comparison)
2. [CUDA Program Structure: Host and Device](#2-cuda-program-structure-host-and-device)
3. [Device Global Memory and Data Transfer](#3-device-global-memory-and-data-transfer)
   - [`cudaMalloc()`](#cudamalloc)
   - [`cudaMemcpy()`](#cudamemcpy)
   - [`cudaFree()`](#cudafree)
   - [Device Pointer Rules](#device-pointer-rules)
4. [Error Checking](#4-error-checking)
5. [Kernel Functions and the Thread Hierarchy](#5-kernel-functions-and-the-thread-hierarchy)
   - [Built-in Variables](#built-in-variables)
   - [Global Thread Index](#global-thread-index)
   - [The Telephone Analogy](#the-telephone-analogy)
6. [Function Qualifiers](#6-function-qualifiers)
7. [Launching a Kernel](#7-launching-a-kernel)
   - [Why No `for` Loop in a Kernel?](#why-no-for-loop-in-a-kernel)
   - [Choosing the Number of Blocks](#choosing-the-number-of-blocks)
8. [CUDA Compilation](#8-cuda-compilation)

---

## 1. Data Parallelism vs. Task Parallelism

The key idea of this chapter is **data parallelism**: when computations on different data elements are independent, we can perform them all at the same time. This is exactly the kind of work GPUs love.

### Data Parallelism

- The **same operation** is applied to many data elements in parallel.
- Computations on different elements are **independent**, so nobody has to wait for anybody else.
- We reorganize the computation **around the data** instead of around a sequence of steps.
- Common in image processing, simulations, and scientific computing.
- This is where the massive scalability of GPUs comes from. More data usually means more parallelism to exploit.

### Task Parallelism

- **Different, independent tasks** run in parallel, and each task does a different computation.
- You get it by decomposing an application into separate tasks, for example vector addition in one task while another handles I/O or a data transfer.
- It complements data parallelism, but data parallelism is usually the **main source of performance improvement** because it scales with the size of the data.

### Comparison

| Data Parallelism                                   | Task Parallelism                                     |
| -------------------------------------------------- | ---------------------------------------------------- |
| Same operation on many data elements.              | Different operations running as separate tasks.      |
| Scales with the amount of data.                    | Scales with the number of independent tasks.         |
| Main source of speedup on GPUs.                    | Nice complement, but limited scalability.            |
| Example: adding two vectors element by element.    | Example: overlapping computation with I/O.           |

**Summary:** Data parallelism means doing the same thing to lots of independent data elements at once. Task parallelism means doing different things at the same time. GPUs are built for the first one, and that is where almost all the performance comes from.

---

## 2. CUDA Program Structure: Host and Device

A CUDA program is really two programs living in one source file: code that runs on the CPU (the **host**) and code that runs on the GPU (the **device**).

- A **thread** executes a sequential program. It contains the program code, the current execution point, and the values of its variables and data structures.
- Each CPU or GPU thread executes independently.
- CUDA launches **many threads** to process different data elements in parallel. The threads are organized into a **grid** and they all execute the same **kernel function** on the GPU.
- Don't be shy about the numbers here: large applications routinely launch **millions of threads**. GPU threads are cheap to create and schedule, unlike CPU threads.

**Summary:** A CUDA application runs on both the CPU and the GPU. The CPU part drives the show, and the GPU part does the heavy data-parallel lifting by launching a grid with tons of lightweight threads.

---

## 3. Device Global Memory and Data Transfer

Here is something important to internalize early: **host memory (CPU) and device memory (GPU) are separate memory spaces**. The GPU has its own DRAM, called **device global memory**, and data must usually be moved there explicitly before a kernel can use it.

The typical flow looks like this:

1. **Allocate** memory on the device.
2. **Copy** input data from host to device.
3. **Launch** the kernel, which reads and writes device global memory.
4. **Copy** results back from device to host.
5. **Free** the device memory when you are done.

CUDA provides higher-level memory management too, but `cudaMalloc()`, `cudaMemcpy()`, and `cudaFree()` are the fundamental APIs, so it is worth knowing them well.

### `cudaMalloc()`

- Allocates memory in **device global memory**. Called from host code.
- Takes two parameters:
  - The **address of a pointer** (`void**`), where the allocated device address gets written.
  - The **size** of the allocation in **bytes**.
- It is very similar to C's `malloc()`, with one twist: instead of returning the pointer, it writes the allocated address into the pointer you pass in, and **returns an error code** instead.

```cpp
float *A_d;
cudaMalloc((void**)&A_d, n * sizeof(float));
```

### `cudaMemcpy()`

- Copies data between host and device.
- Parameters, in order:
  - Destination
  - Source
  - Size in bytes
  - Copy direction (`cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, etc.)

```cpp
cudaMemcpy(A_d, A_h, n * sizeof(float), cudaMemcpyHostToDevice);
```

### `cudaFree()`

- Frees previously allocated device memory, just like C's `free()`.
- Takes one parameter: the pointer to the allocated device memory.
- Always do this when you are done, otherwise you are leaking GPU memory.

```cpp
cudaFree(A_d);
```

### Device Pointer Rules

- Device pointers store **GPU addresses**.
- Pass them to kernels and CUDA API functions. That is what they are for.
- **Never dereference a device pointer on the host.** It points to memory the CPU cannot directly access, so at best you get garbage and at worst a crash.
- If you need the data on the CPU, copy it back with `cudaMemcpy()` first.

A quick note on **Unified Virtual Addressing (UVA)**: it allows the GPU to access host memory directly. Convenient, but accessing host memory is much slower than device global memory, so for performance you still want your data living on the device.

**Summary:** The CPU and GPU have separate memories. Allocate on the device with `cudaMalloc()`, move data with `cudaMemcpy()`, and clean up with `cudaFree()`. Device pointers only make sense on the device, so never dereference them on the host.

---

## 4. Error Checking

CUDA API functions do not throw exceptions or crash loudly when something goes wrong. They quietly return a `cudaError_t` status, and it is on you to check it.

- **Always check the return value** of CUDA API calls.
- Compare the result with `cudaSuccess`.
- Use `cudaGetErrorString()` to turn the error code into a readable message.
- Checking errors early makes debugging so much easier. A failed allocation that goes unnoticed will show up later as a mysterious kernel bug.

```cpp
cudaError_t err = cudaMalloc((void**)&A_d, size);
if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
}
```

**Summary:** CUDA fails silently unless you check. Wrap your API calls with error checks and thank yourself later.

---

## 5. Kernel Functions and the Thread Hierarchy

A **kernel** is a function that runs on the GPU. You launch it from the host, and the launch creates a whole new **grid** of GPU threads:

```cpp
kernel<<<numBlocks, threadsPerBlock>>>(args);
```

All of those threads execute the **same kernel code**. This style is called **SPMD** (Single Program, Multiple Data): one program, many data elements. What makes each thread do something different is its coordinates.

The threads are organized hierarchically as **Grid → Blocks → Threads**:

- A **grid** contains one or more blocks.
- A **block** contains one or more threads, up to **1024 threads** per block.
- CUDA supports **1D, 2D, and 3D** organizations. Use `x` for arrays and vectors, `x, y` for images and matrices, and `x, y, z` for volumes and 3D grids.

### Built-in Variables

Every thread can look up where it lives using these built-in variables:

| Variable    | Meaning                              |
| ----------- | ------------------------------------ |
| `threadIdx` | Thread index within its block.       |
| `blockIdx`  | Block index within the grid.         |
| `blockDim`  | Number of threads per block.         |
| `gridDim`   | Number of blocks in the grid.        |

### Global Thread Index

Combining those coordinates gives each thread a unique global index:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

This is the single most common line in CUDA code. Each thread computes its own `i` and uses it to grab exactly one data element. One thread, one element.

### The Telephone Analogy

A nice way to think about the hierarchy:

- **Grid** = the entire telephone system.
- **Block (`blockIdx`)** = the area code.
- **Thread (`threadIdx`)** = the local phone number.
- **Global thread ID** = the full phone number, area code plus local number.

Just like phone numbers, this scheme gives us a huge number of unique IDs while keeping related threads close together (locality).

**Summary:** A kernel launch creates a grid of blocks, and each block holds up to 1024 threads. All threads run the same code, but each one computes a unique global index from `blockIdx`, `blockDim`, and `threadIdx`, and uses it to work on its own data element.

---

## 6. Function Qualifiers

CUDA extends C++ with qualifiers that say where a function runs and who can call it:

| Qualifier             | Runs on | Callable from                                          |
| --------------------- | ------- | ------------------------------------------------------ |
| `__host__` (default)  | CPU     | Host functions.                                        |
| `__global__`          | GPU     | Host (or device, with Dynamic Parallelism). Launches a new grid of threads. |
| `__device__`          | GPU     | Device and kernel functions only.                      |
| `__host__ __device__` | Both    | Compiler generates a CPU version and a GPU version.    |

A couple of details worth knowing:

- **Why does `__host__` exist if it is the default?** Its real job is combining with `__device__`. Marking a function `__host__ __device__` gives you the same function on both the CPU and the GPU without writing it twice.
- GPUs with **Dynamic Parallelism** allow a kernel to launch another kernel, so `__global__` functions are not strictly host-launched anymore on modern hardware.

---

## 7. Launching a Kernel

### Why No `for` Loop in a Kernel?

In sequential code, a vector addition is a loop over `n` elements. In CUDA, the loop simply disappears: **the grid of threads replaces the loop**. Each thread performs what used to be one iteration, and all the iterations happen in parallel. This is often called **loop parallelism**.

### Choosing the Number of Blocks

When launching a kernel, you pick the block size, then compute how many blocks you need to cover all `n` elements:

```cpp
int threadsPerBlock = 256;
int numBlocks = ceil((float)n / threadsPerBlock);
kernel<<<numBlocks, threadsPerBlock>>>(args);
```

- `ceil()` rounds **up**, which guarantees we launch at least `n` threads.
- Rounding up means we may launch a few extra threads. For example, with `n = 1000` and 256 threads per block we launch 4 blocks, which is 1024 threads.
- Those extra threads are harmless because of the boundary check inside the kernel:

```cpp
if (i < n) {
    C[i] = A[i] + B[i];
}
```

**Summary:** Pick a block size, round up the number of blocks so every element gets a thread, and guard the kernel body with `if (i < n)` so the leftover threads do nothing.

---

## 8. CUDA Compilation

So what actually happens when you compile a `.cu` file? The **NVCC** compiler splits your source into two paths:

```text
CUDA C++ Source
        │
        ▼
   NVCC Compiler
      /      \
     /        \
Host Code   Device Code (PTX)
     │            │
C/C++ Compiler   JIT Compiler
     │            │
     └──────┬─────┘
            ▼
      CPU + GPU Program
```

- **NVCC** separates host code from device code.
- **Host code** goes through a standard C/C++ compiler, business as usual.
- **Device code** is compiled into **PTX**, a virtual GPU assembly language. Think of it as an intermediate representation that is not tied to any specific GPU.
- At runtime, PTX is **Just-In-Time (JIT) compiled** into machine code for whatever GPU is actually in the machine. This is what lets the same binary run on GPUs that did not even exist when you compiled it.

**Summary:** NVCC splits a CUDA program in two. The CPU part is compiled normally, and the GPU part becomes PTX, a virtual assembly that gets JIT-compiled for the target GPU right before execution.
