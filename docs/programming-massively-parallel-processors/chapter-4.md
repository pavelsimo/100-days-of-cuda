> Dear reader: These notes were created with the help of AI, with me cherry-picking the parts of the book I found most relevant. I also reviewed the content to make sure no AI hallucinations slipped through. I hope you find them useful. Happy reading! :)

# Programming Massively Parallel Processors: Chapter 4, Compute Architecture and Scheduling

1. [Modern GPU Architecture](#1-modern-gpu-architecture)
   - [Key Components](#key-components)
   - [Memory System](#memory-system)
   - [The Hardware Hierarchy](#the-hardware-hierarchy)
2. [Thread Block Scheduling](#2-thread-block-scheduling)
3. [Synchronization and Transparent Scalability](#3-synchronization-and-transparent-scalability)
   - [Barrier Synchronization](#barrier-synchronization)
   - [Synchronization Scopes](#synchronization-scopes)
   - [Barrier Synchronization Rules](#barrier-synchronization-rules)
   - [Transparent Scalability](#transparent-scalability)
   - [Waves](#waves)
4. [Warps and SIMD Hardware](#4-warps-and-simd-hardware)
   - [Warp Partitioning](#warp-partitioning)
   - [Linearizing 2D and 3D Blocks](#linearizing-2d-and-3d-blocks)
   - [SIMD Execution and the Von Neumann Model](#simd-execution-and-the-von-neumann-model)
   - [Flynn's Taxonomy](#flynns-taxonomy)
5. [Control Divergence](#5-control-divergence)
   - [How Divergence Happens](#how-divergence-happens)
   - [Independent Thread Scheduling (Volta+)](#independent-thread-scheduling-volta)
   - [Common Causes of Divergence](#common-causes-of-divergence)
   - [Boundary Divergence](#boundary-divergence)
6. [Warp Scheduling and Latency Tolerance](#6-warp-scheduling-and-latency-tolerance)
   - [Latency Hiding](#latency-hiding)
   - [Zero-Overhead Scheduling](#zero-overhead-scheduling)
7. [Resource Partitioning and Occupancy](#7-resource-partitioning-and-occupancy)
   - [What Limits Occupancy](#what-limits-occupancy)
   - [Block Slots vs. Thread Slots](#block-slots-vs-thread-slots)
   - [Registers and the Performance Cliff](#registers-and-the-performance-cliff)
8. [Querying Device Properties](#8-querying-device-properties)
   - [The Query API](#the-query-api)
   - [Important Device Properties](#important-device-properties)
   - [Why Query Device Properties?](#why-query-device-properties)

---

## 1. Modern GPU Architecture

The key idea of this chapter is that a GPU is a collection of **Streaming Multiprocessors (SMs)** that execute thousands of threads in parallel, and understanding how the hardware schedules those threads is the foundation of CUDA performance.

### Key Components

- **GPU**: A collection of **SMs** that execute thousands of threads in parallel.
- **SM (Streaming Multiprocessor)**: The main execution unit. It executes thread blocks and contains CUDA cores and on-chip memory.
- **Streaming Processor (CUDA Core)**: The arithmetic unit that executes instructions.
- **GPC (GPU Processing Cluster)**: A group of multiple SMs, used for scalability.

### Memory System

- **On-chip Memory**: Fast memory inside each SM (registers, shared memory, caches).
- **Global Memory**: Large off-chip DRAM accessible by all SMs, with higher latency than on-chip memory.
- **Memory Controller**: Connects SMs to global memory and manages memory accesses.
- **DRAM**: The general term for GPU global memory. It comes in two main forms:
  - **GDDR**: Traditional GPU DRAM used in many GPUs.
  - **HBM (High Bandwidth Memory)**: Stacked DRAM integrated into the GPU package, providing much higher bandwidth than GDDR.

### The Hardware Hierarchy

```
GPU
└── GPC
    └── SM
        └── Streaming Processors (CUDA Cores)
```

**Summary:** A GPU is built from clusters of SMs, each SM containing many CUDA cores plus fast on-chip memory, all backed by large but slower off-chip DRAM (GDDR or HBM).

---

## 2. Thread Block Scheduling

When a kernel is launched, CUDA creates a **grid of thread blocks**, and the hardware assigns whole blocks to SMs.

- **Thread blocks are assigned to SMs**, not individual threads.
- **All threads in a block execute on the same SM.**
- An SM can execute **multiple blocks simultaneously** if enough hardware resources are available.
- The number of concurrent blocks per SM is limited by available resources (registers, shared memory, and so on).
- If there are more blocks than available SM resources, the **remaining blocks wait** until an SM finishes a block.
- Scheduling is handled **automatically by the CUDA runtime**.

Because all threads in a block live on the same SM, they can:

- Synchronize using `__syncthreads()`
- Share data through **shared memory**

Threads in **different blocks cannot directly synchronize or share shared memory**.

**Note:** **Thread Block Clusters (Hopper and newer)** are an optional grouping of thread blocks that enables closer cooperation between blocks.

**Summary:** The unit of scheduling is the thread block: a block always lands on a single SM, which is what makes barrier synchronization and shared memory possible within a block and impossible across blocks.

---

## 3. Synchronization and Transparent Scalability

### Barrier Synchronization

- **`__syncthreads()`** synchronizes **all threads in the same block**.
- A thread that reaches the barrier **waits** until every thread in the block arrives.
- After the last thread arrives, **all threads continue together**.
- This ensures all threads complete one phase before starting the next.
- The `__` prefix indicates an **intrinsic function**, which the compiler translates directly into a hardware instruction with no normal function call overhead.

**Analogy:** Four friends go shopping at different stores. Each friend shops independently (parallel execution), but everyone must return to the car before leaving. Friends who finish early wait for the last one, and once everyone has arrived, they leave together. That is exactly a barrier: **no thread proceeds until every thread reaches the synchronization point**.

### Synchronization Scopes

The **scope** of a barrier is the set of threads that participate in it:

| Scope        | Mechanism                                        |
| ------------ | ------------------------------------------------ |
| Block-wide   | `__syncthreads()`                                |
| Cluster-wide | Cooperative Groups API (Thread Block Clusters)   |
| Grid-wide    | Cooperative Groups API, with extra restrictions  |

### Barrier Synchronization Rules

- **Every thread in a block must execute the same `__syncthreads()` barrier.**
- Using different barriers in different branches of an `if` statement is **incorrect** and results in **undefined behavior**.
- Incorrect barrier usage can cause:
  - Incorrect results
  - **Deadlock**, where threads wait forever
- CUDA assigns an **entire thread block** to the same SM precisely so that all its threads can reach the barrier.

### Transparent Scalability

- Synchronization is limited to the **thread block**, which allows **blocks to execute independently**.
- Since blocks are independent, the CUDA runtime can execute them **in any order**.
- As a result, the same kernel automatically scales from GPUs with **few SMs** to GPUs with **many SMs**.

### Waves

- A **wave** is the group of thread blocks executing simultaneously.
- If the grid contains more blocks than the GPU can run at once, execution happens in **multiple waves**.
- More waves improve load balancing when block workloads are uneven.
- The **last wave** may contain fewer blocks, reducing hardware utilization.

Example: a grid with **660 blocks** on a GPU that can execute **264 blocks simultaneously** runs in **2.5 waves**:

- Wave 1: 264 blocks
- Wave 2: 264 blocks
- Wave 3: 132 blocks (partial wave)

**Summary:** Barriers only work inside a block, and that restriction is a feature: independent blocks can run in any order, which gives CUDA transparent scalability across GPUs of any size, with grids executing in waves when they exceed the hardware's capacity.

---

## 4. Warps and SIMD Hardware

Threads within a block **should not be assumed to execute in any specific order**; thread scheduling is **hardware dependent** and may vary between GPU architectures. Use `__syncthreads()` whenever threads must complete one phase before starting the next.

### Warp Partitioning

- A **warp** is the basic unit of thread scheduling on an SM.
- A warp contains **32 consecutive threads**, and this holds on all current CUDA GPUs (future architectures may differ).
- A thread block is automatically divided into warps before execution:
  - **Number of warps =** `ceil(threads per block / 32)`
  - Example: 256 threads per block gives **8 warps**; 3 such blocks on an SM give **24 warps**.
- Warps are formed from **consecutive thread indices**:
  - Warp 0: threads **0 to 31**
  - Warp 1: threads **32 to 63**
  - Warp *n*: threads **32n** to **32(n + 1) - 1**
- If the block size is **not a multiple of 32**, the last warp is padded with **inactive threads**.
  - Example: 48 threads gives **2 warps**, and the second warp has **16 inactive threads**.

### Linearizing 2D and 3D Blocks

For 2D or 3D thread blocks, CUDA first **linearizes the threads in row-major order**, then partitions them into warps:

- 2D: threads are ordered by `threadIdx.x`, then `threadIdx.y`.
- 3D: threads are ordered by `threadIdx.x`, then `threadIdx.y`, then `threadIdx.z`.
- Threads with consecutive `threadIdx.x` values end up in the same warp whenever possible.

### SIMD Execution and the Von Neumann Model

- In the classic **Von Neumann model**, a CPU executes instructions using a **Control Unit**, **ALU**, **Registers**, and **Memory**.
- In a GPU, **one Control Unit** dispatches the **same instruction** to multiple processing units, and each processing unit executes **one thread of the warp**.
- This is **SIMD (Single Instruction, Multiple Data)**: the same instruction, different data per thread.
- Sharing one Control Unit across many execution units **reduces hardware cost and power consumption**, because instruction fetch and control logic are amortized.
- **SIMT (Single Instruction, Multiple Threads)** is CUDA's programming model on top of SIMD hardware: you write normal scalar thread code, and the hardware automatically groups threads into warps that execute in lockstep.
- CUDA also provides **warp-level primitives** for efficient communication and synchronization between threads in the same warp.

### Flynn's Taxonomy

| Class    | Meaning                              | Example   |
| -------- | ------------------------------------ | --------- |
| **SISD** | Single Instruction, Single Data      | Classic CPU core |
| **SIMD** | Single Instruction, Multiple Data    | GPU warps |
| **MISD** | Multiple Instruction, Single Data    | Rare      |
| **MIMD** | Multiple Instruction, Multiple Data  | Multicore CPUs |

**Summary:** Blocks are carved into warps of 32 consecutive threads (after row-major linearization for 2D/3D blocks), and each warp executes one instruction at a time on SIMD hardware. SIMT lets you write scalar code while the hardware handles the lockstep execution.

---

## 5. Control Divergence

### How Divergence Happens

- SIMD/SIMT is most efficient when **all threads in a warp follow the same execution path**.
- **Control divergence** occurs when threads in the **same warp take different execution paths**.
- For an `if-else`, the GPU executes **one branch at a time**, while threads on the other branch are **masked off (inactive)**.
- After both branches finish, the warp **reconverges** and continues together.
- Divergence increases execution time because it requires **multiple execution passes**, reducing parallel efficiency.
- Divergence also occurs in **loops**: when threads execute different numbers of iterations, threads that finish early sit **inactive** while the rest continue.

### Independent Thread Scheduling (Volta+)

- **Pascal and earlier**: divergent branches execute **sequentially**.
- **Volta and newer**: divergent branches **may be interleaved** thanks to **Independent Thread Scheduling**.
- Do **not assume** threads automatically reconverge after divergence.
- Use **`__syncwarp()`** when warp-level synchronization is required.

### Common Causes of Divergence

- Branches based on **`threadIdx`** values.
- Loops whose iteration count depends on **`threadIdx`**.
- Boundary checks such as `if (i < n)`, used when the data size is not a multiple of the block size.

### Boundary Divergence

- **Boundary divergence** only affects the **last partially active warp**.
- As the input size grows, **fewer warps diverge** (proportionally), so the performance impact becomes **smaller**.
- Boundary checks like `if (i < n)` are common and usually have **low overhead** for large datasets.
- The same idea applies to **2D data**: only blocks touching the image boundaries may diverge.

**Summary:** Divergence forces a warp to execute branch paths in multiple passes, so keep threads in a warp on the same path when you can. Boundary-check divergence is usually acceptable because it only touches the last warp, and on Volta and newer you must use `__syncwarp()` rather than assuming reconvergence.

---

## 6. Warp Scheduling and Latency Tolerance

### Latency Hiding

- An SM usually has **more resident warps than it can execute simultaneously**.
- When one warp **stalls** (for example, waiting on global memory), the scheduler **switches to another ready warp**.
- This rapid switching is called **fine-grained multithreading**.
- Hiding memory latency by executing other ready warps is called **latency hiding** or **latency tolerance**.
- Keeping many ready warps resident on an SM helps **maximize hardware utilization**.

**Analogy:** Think of a post office. A customer filling out a form is like a warp waiting for memory. Instead of waiting, the clerk serves the next ready customer, and the first customer resumes when they are done with the form. The GPU hides latency the same way: it switches to ready warps instead of idling.

### Zero-Overhead Scheduling

- GPUs use **zero-overhead scheduling** to switch between warps waiting on long-latency operations.
- Unlike CPUs, GPUs **do not save and restore thread context** when switching warps.
- Each warp's execution state is already stored in **hardware registers**, so warp switching is nearly instantaneous.
- Having **many resident warps** increases the chance that some warp is ready to execute, which improves latency hiding.

Example (H100): an SM has **128 streaming processors** but can keep **2048 threads (64 warps)** resident. That oversubscription is what makes latency hiding work.

**Summary:** SMs deliberately hold far more warps than they can run at once. Because every warp's state lives permanently in registers, switching costs nothing, and stalls on memory are hidden by simply running whichever warps are ready.

---

## 7. Resource Partitioning and Occupancy

### What Limits Occupancy

**Occupancy = active warps on an SM / maximum supported warps.**

SM resources are **dynamically partitioned** among thread blocks, and occupancy is limited by whichever resource runs out first:

- Registers
- Shared memory
- Thread slots
- Block slots

Smaller blocks allow **more blocks per SM**, while larger blocks allow **fewer blocks per SM**. High occupancy generally improves **latency hiding**, but **100% occupancy is not always achievable** due to resource limits. Occupancy can also drop when the block size **does not divide evenly** into the available thread slots.

### Block Slots vs. Thread Slots

Occupancy may be limited by a resource other than thread slots:

- An H100 SM supports **2048 threads** but only **32 block slots**.
- Blocks of **32 threads** would need **64 blocks** to fill 2048 threads, but only **32 blocks** fit.
- The result is **1024 active threads, or 50% occupancy**.

### Registers and the Performance Cliff

- More **registers per thread** means **fewer threads and blocks** can run simultaneously.
- **Rule of thumb (H100):**
  - **32 or fewer registers/thread**: can achieve **100% occupancy** (2048 threads/SM).
  - **64 registers/thread**: at most **1024 threads/SM (50% occupancy)**.
- **Register spilling** stores excess registers in memory, increasing memory accesses and reducing performance.
- A small increase in resource usage can cause a sudden large drop in occupancy, known as a **performance cliff**.
  - Example: going from **31 to 33 registers/thread** may reduce active blocks from **4 to 3**, dropping occupancy from **100% to 75%**.

> **Exam tip (H100):** 65,536 registers/SM ÷ 2048 max threads/SM = **32 registers/thread** to maintain full occupancy. Exceeding this limit can reduce occupancy.

NVIDIA provides tools to estimate occupancy:

- **Occupancy Calculator** (Nsight Compute)
- CUDA Occupancy API, for example `cudaOccupancyMaxActiveBlocksPerMultiprocessor()`

**Summary:** Occupancy is the fraction of an SM's warp capacity you actually use, and it is bounded by registers, shared memory, thread slots, and block slots. Watch for performance cliffs: a couple of extra registers per thread can knock out an entire block's worth of occupancy.

---

## 8. Querying Device Properties

### The Query API

CUDA applications can **query GPU hardware properties at runtime**, which lets the same code adapt to different GPUs.

- **Compute Capability (CC)** identifies a GPU architecture and its supported features. Higher CC generally means **more resources and capabilities**.
- `cudaGetDeviceCount()` returns how many CUDA-capable GPUs are available.
- `cudaGetDeviceProperties()` queries the properties of each GPU.
- The results are stored in a `cudaDeviceProp` struct.

### Important Device Properties

| Property              | Meaning                                                        |
| --------------------- | -------------------------------------------------------------- |
| `maxThreadsPerBlock`  | Maximum threads allowed per block                              |
| `multiProcessorCount` | Number of SMs in the GPU                                       |
| `clockRate`           | GPU clock frequency; with SM count, estimates compute throughput |
| `maxThreadsDim[3]`    | Maximum block dimensions `(x, y, z)`                           |
| `maxGridSize[3]`      | Maximum grid dimensions `(x, y, z)`                            |
| `regsPerBlock`        | Maximum registers available to a thread block                  |
| `warpSize`            | Warp size (typically 32)                                       |

### Why Query Device Properties?

- Adapt kernels to different GPU architectures.
- Select valid block and grid sizes.
- Estimate occupancy and register limitations.
- Choose the most capable GPU when multiple CUDA devices are available.

**Summary:** Query device properties at runtime with `cudaGetDeviceCount()` and `cudaGetDeviceProperties()` so your launch configurations and occupancy assumptions match the actual hardware instead of being hard-coded for one GPU.
