# Programming Massively Parallel Processors — Chapter 1: Introduction

## Index

1. [The End of Frequency Scaling (2003)](#1-the-end-of-frequency-scaling-2003)
2. [Two Processor Design Philosophies](#2-two-processor-design-philosophies)
   - [Multi-core (CPU)](#multi-core-cpu)
   - [Many-thread (GPU)](#many-thread-gpu)
   - [CPU vs. GPU Comparison](#cpu-vs-gpu-comparison)
3. [Latency vs. Throughput](#3-latency-vs-throughput)
4. [Why GPU Computing Became Mainstream](#4-why-gpu-computing-became-mainstream)
5. [Why Applications Need More Speed](#5-why-applications-need-more-speed)
6. [Amdahl's Law](#6-amdahls-law)
   - [Formula](#formula)
   - [Worked Examples](#worked-examples)
7. [Memory Bandwidth: The Hidden Bottleneck](#7-memory-bandwidth-the-hidden-bottleneck)
8. [Challenges of Parallel Programming](#8-challenges-of-parallel-programming)
9. [Related Parallel Programming APIs](#9-related-parallel-programming-apis)
   - [OpenMP](#openmp)
   - [MPI (Message Passing Interface)](#mpi-message-passing-interface)
   - [OpenCL (Open Computing Language)](#opencl-open-computing-language)

---

## 1. The End of Frequency Scaling (2003)

The key idea is that **CPU clock speeds stopped increasing significantly around 2003** because of **power consumption and heat dissipation**. This forced a major change in both hardware and software.

1. **Clock speed hit a wall**
    - Before 2003, CPUs became faster mainly by increasing their clock frequency (GHz).
    - Higher frequencies produced too much heat and consumed too much power.
    - Manufacturers could no longer keep boosting performance this way.

2. **CPU manufacturers switched to multi-core processors**
    - Instead of making **one core faster**, they started putting **multiple cores** on a single chip.
    - A modern CPU can be thought of as several CPUs working together.

3. **Software had to change**
    - Old programs were mostly **sequential**: one instruction after another, one **thread of execution**.
    - A sequential program can only use **one CPU core** effectively.

4. **Programs needed to become parallel**
    - Developers began dividing applications into **multiple threads**.
    - Different threads can run **simultaneously on different CPU cores**.
    - This is called **parallel programming** (or concurrent programming in a broader sense).

5. **Performance improvements now depend on parallelism**
    - Sequential programs no longer automatically become much faster with each new CPU generation.
    - To benefit from modern hardware, applications must perform work in parallel.

**Summary:** After clock speeds reached their limit due to power and heat, CPU manufacturers began adding multiple cores instead of increasing frequency. This allowed several instructions or threads to run at the same time. As a result, software had to be redesigned to use parallel programming and multiple threads. Since then, performance gains have depended more on parallelism than on higher clock speeds.

---

## 2. Two Processor Design Philosophies

### Multi-core (CPU)

Multi-core processors improve the performance of sequential programs by using multiple powerful CPU cores.
Each core is optimized for complex tasks and can run one or more hardware threads.
The number of CPU cores has steadily increased with each processor generation.
They are designed to maximize single-thread performance while supporting parallel execution.

### Many-thread (GPU)

Many-thread processors, such as GPUs, focus on maximizing throughput using thousands of lightweight threads.
They execute many simple operations in parallel instead of optimizing a single thread.
GPUs provide much higher floating-point performance than CPUs for parallel workloads.
They are ideal for compute-intensive tasks such as deep learning, scientific computing, and graphics.

### CPU vs. GPU Comparison

| Multi-core (CPU)                                      | Many-thread (GPU)                                               |
| ----------------------------------------------------- | --------------------------------------------------------------- |
| Uses a small number of powerful cores.                | Uses thousands of lightweight threads.                          |
| Optimized for low latency and sequential performance. | Optimized for high throughput and parallel workloads.           |
| Best for complex, general-purpose tasks.              | Best for compute-intensive parallel tasks (e.g., AI, graphics). |
| Focuses on fast individual threads.                   | Focuses on executing many threads simultaneously.               |

---

## 3. Latency vs. Throughput

| CPU (Latency-oriented)                                     | GPU (Throughput-oriented)                                           |
| ---------------------------------------------------------- | ------------------------------------------------------------------- |
| Optimized to minimize the execution time of a single task. | Optimized to maximize the number of tasks completed simultaneously. |
| Uses large caches and complex control logic.               | Uses many arithmetic units and higher memory bandwidth.             |
| Best for sequential and branch-heavy workloads.            | Best for highly parallel, compute-intensive workloads.              |
| **Tradeoff:** Faster per thread, fewer parallel units.     | **Tradeoff:** Slower per thread, much higher overall throughput.    |

| Reducing Latency                                                    | Increasing Throughput                                            |
| ------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Requires complex hardware, larger chip area, and much higher power. | Achieved by adding more arithmetic units with proportional cost. |
| Benefits a single thread by making it execute faster.               | Benefits many threads by increasing total work completed.        |
| Expensive and scales poorly.                                        | More cost-effective and scales well.                             |
| **Tradeoff:** Faster individual execution.                          | **Tradeoff:** Higher overall performance for parallel workloads. |

---

## 4. Why GPU Computing Became Mainstream

| Installed Base                                                             | GPGPU (Before CUDA)                                                          | CUDA (After 2007)                                                                    |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| A large installed base makes software development economically worthwhile. | GPUs were difficult to program using graphics APIs like OpenGL and Direct3D. | CUDA enabled general-purpose GPU programming with a simple programming model.        |
| Most PCs already include GPUs, making them widely accessible.              | Computations had to be expressed as graphics operations.                     | Developers could write compute kernels directly, greatly expanding GPU applications. |

- **CUDA (2007):** NVIDIA introduced CUDA, making GPUs easy to program for general-purpose computing using familiar C/C++ instead of graphics APIs like OpenGL. This made GPU computing mainstream.

- **FPGAs (Field-Programmable Gate Arrays):** Reconfigurable hardware chips that can be programmed to implement custom digital circuits, providing high performance and low latency for specialized tasks such as networking, signal processing, and AI inference.

---

## 5. Why Applications Need More Speed

The book argues that **future applications will need much more computing power**, even if today's software already seems fast.

- Future "super-applications" (e.g., molecular biology simulations, AI, climate modeling) require enormous computation.

- Simulations can model systems that physical instruments cannot directly observe.

- Parallel computing, especially on GPUs, provides **10×–100× speedups** for suitable applications.

- Therefore, continued performance growth depends on exploiting **parallelism**, not higher CPU clock speeds.

---

## 6. Amdahl's Law

- **Amdahl's Law:** The maximum speedup is limited by the fraction of a program that **cannot be parallelized**.

- The larger the parallel portion, the greater the overall speedup.

- Even an infinitely fast GPU cannot accelerate the sequential part.

- Therefore, maximizing the parallel fraction is essential for high performance.

### Formula

$$
S = \frac{1}{(1 - P) + \frac{P}{N}}
$$

where:

- $S$ = overall speedup

- $P$ = parallelizable fraction

- $N$ = speedup of the parallel part

### Worked Examples

**Example 1 — 30% parallelizable, 100× faster**

Parallel execution time:

$$
\frac{0.30}{100} = 0.003
$$

Total execution time:

$$
0.70 + 0.003 = 0.703
$$

Overall speedup:

$$
S = \frac{1}{0.703} \approx 1.42\times
$$

**Example 2 — 30% parallelizable, infinite speedup**

$$
S = \frac{1}{0.70 + 0} = \frac{1}{0.70} \approx 1.43\times
$$

**Example 3 — 99% parallelizable, 100× faster**

Parallel execution time:

$$
\frac{0.99}{100} = 0.0099
$$

Total execution time:

$$
0.01 + 0.0099 = 0.0199
$$

Overall speedup:

$$
S = \frac{1}{0.0199} \approx 50\times
$$

**Key takeaway:** Even extremely fast parallel hardware provides limited benefit unless **most of the application is parallelizable**.

---

## 7. Memory Bandwidth: The Hidden Bottleneck

- **Memory bandwidth often limits speedup**, even when computation can be parallelized.

- Simply parallelizing code usually reaches only about **10× speedup** because DRAM becomes the bottleneck.

- Using **fast on-chip GPU memories** (shared memory, caches, registers) reduces expensive DRAM accesses and increases performance.

- Achieving high speedups requires **optimizing memory access patterns**, not just adding more parallel threads.

---

## 8. Challenges of Parallel Programming

**Algorithm and work efficiency:**

- **Design efficient parallel algorithms** without increasing total work.

- **Avoid extra or redundant computation**, which can reduce performance.

- **Manage memory bottlenecks**, since many applications are memory-bound rather than compute-bound.

**Load balance and synchronization:**

- **Load imbalance / input sensitivity:** Uneven or unpredictable input data can distribute work unevenly across threads, leaving some threads idle while others do more work.

- **Synchronization overhead:** Threads often need barriers, locks, or atomic operations, causing some threads to wait for others instead of doing useful work.

- **Communication:** Some applications require frequent thread collaboration, making parallelization harder.

- **Embarrassingly parallel vs. synchronized tasks:** Some problems require almost no communication between threads (easy to parallelize), while others require frequent synchronization (harder and slower).

**The good news:**

- **Reuse of parallel patterns:** Many of these challenges can be addressed using well-known parallel programming patterns and techniques.

---

## 9. Related Parallel Programming APIs

### OpenMP

- **OpenMP** is a compiler-based API for parallel programming on **shared-memory CPUs**.

- It uses simple compiler directives (`#pragma omp`) to parallelize loops and tasks.

- The runtime automatically manages threads, making CPU parallelization easier with minimal code changes.

### MPI (Message Passing Interface)

- Used for **distributed-memory systems** (clusters), where each node has its own memory.

- Processes communicate by **explicitly sending and receiving messages**.

- Standard for **high-performance computing (HPC)** and large supercomputers.

- Often combined with **CUDA** for multi-GPU clusters.

### OpenCL (Open Computing Language)

- An **open standard** for parallel programming across CPUs, GPUs, and other accelerators.

- Similar to CUDA, but **vendor-independent** (works on hardware from different manufacturers).

- Provides portability across devices, though performance tuning is still hardware-specific.

- CUDA programmers can learn OpenCL easily because the programming concepts are very similar.
