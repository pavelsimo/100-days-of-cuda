# Programming Massively Parallel Processors: Chapter 1, Introduction

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

The key idea is that **CPU clock speeds stopped increasing significantly around 2003** because of **power consumption and heat dissipation**. This forced a major change in both hardware and software, and it is basically the reason this book exists.

1. **Clock speed hit a wall**
    - Before 2003, CPUs got faster mostly by cranking up the clock frequency (GHz).
    - Higher frequencies produced too much heat and consumed too much power.
    - At some point manufacturers simply could not keep boosting performance this way.

2. **CPU manufacturers switched to multi-core processors**
    - Instead of making **one core faster**, they started putting **multiple cores** on a single chip.
    - You can think of a modern CPU as several CPUs working together.

3. **Software had to change**
    - Old programs were mostly **sequential**: one instruction after another, one **thread of execution**.
    - Here is the catch: a sequential program can only use **one CPU core** effectively, no matter how many cores you have.

4. **Programs needed to become parallel**
    - Developers began dividing applications into **multiple threads**.
    - Different threads can run **simultaneously on different CPU cores**.
    - This is called **parallel programming** (or concurrent programming in a broader sense).

5. **Performance improvements now depend on parallelism**
    - The free lunch is over: sequential programs no longer automatically get much faster with each new CPU generation.
    - To benefit from modern hardware, applications must do their work in parallel.

**Summary:** Around 2003, clock speeds hit a wall because of power and heat, so CPU manufacturers started adding cores instead of GHz. That shifted the burden to software: programs had to be redesigned around threads and parallelism, and ever since then, performance gains have come from parallelism rather than faster clocks.

---

## 2. Two Processor Design Philosophies

CPUs and GPUs are not just faster and slower versions of each other. They are built around two genuinely different design philosophies.

### Multi-core (CPU)

Multi-core processors improve the performance of sequential programs by using a handful of powerful cores. Each core is optimized for complex tasks and can run one or more hardware threads. The number of cores has steadily grown with each generation, but the philosophy stays the same: make each individual thread as fast as possible, and support parallelism on top of that.

### Many-thread (GPU)

Many-thread processors, such as GPUs, go the opposite way: instead of a few heavyweight cores, they use thousands of lightweight threads and focus on total throughput. No single thread is impressive on its own, but together they get through an enormous amount of work. GPUs deliver much higher floating-point performance than CPUs for parallel workloads, which makes them ideal for deep learning, scientific computing, and graphics.

### CPU vs. GPU Comparison

| Multi-core (CPU)                                      | Many-thread (GPU)                                               |
| ----------------------------------------------------- | --------------------------------------------------------------- |
| Uses a small number of powerful cores.                | Uses thousands of lightweight threads.                          |
| Optimized for low latency and sequential performance. | Optimized for high throughput and parallel workloads.           |
| Best for complex, general-purpose tasks.              | Best for compute-intensive parallel tasks (e.g., AI, graphics). |
| Focuses on fast individual threads.                   | Focuses on executing many threads simultaneously.               |

**Summary:** CPUs bet on a few fast threads, GPUs bet on a huge number of slow ones. Neither is better in general. It depends entirely on the workload.

---

## 3. Latency vs. Throughput

The CPU vs. GPU split really comes down to one question: do you want to finish **one task as fast as possible** (latency), or **as many tasks as possible** (throughput)?

| CPU (Latency-oriented)                                     | GPU (Throughput-oriented)                                           |
| ---------------------------------------------------------- | ------------------------------------------------------------------- |
| Optimized to minimize the execution time of a single task. | Optimized to maximize the number of tasks completed simultaneously. |
| Uses large caches and complex control logic.               | Uses many arithmetic units and higher memory bandwidth.             |
| Best for sequential and branch-heavy workloads.            | Best for highly parallel, compute-intensive workloads.              |
| **Tradeoff:** Faster per thread, fewer parallel units.     | **Tradeoff:** Slower per thread, much higher overall throughput.    |

There is also an economic angle: reducing latency is expensive, while increasing throughput scales nicely.

| Reducing Latency                                                    | Increasing Throughput                                            |
| ------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Requires complex hardware, larger chip area, and much higher power. | Achieved by adding more arithmetic units with proportional cost. |
| Benefits a single thread by making it execute faster.               | Benefits many threads by increasing total work completed.        |
| Expensive and scales poorly.                                        | More cost-effective and scales well.                             |
| **Tradeoff:** Faster individual execution.                          | **Tradeoff:** Higher overall performance for parallel workloads. |

**Summary:** Making one thread faster costs a lot of silicon and power for diminishing returns. Adding more arithmetic units for more threads is cheap and scales well. That is why throughput-oriented designs like GPUs win so big on parallel workloads.

---

## 4. Why GPU Computing Became Mainstream

Great hardware alone is not enough. Two more things had to happen: GPUs had to be everywhere, and they had to be easy to program.

| Installed Base                                                             | GPGPU (Before CUDA)                                                          | CUDA (After 2007)                                                                    |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| A large installed base makes software development economically worthwhile. | GPUs were difficult to program using graphics APIs like OpenGL and Direct3D. | CUDA enabled general-purpose GPU programming with a simple programming model.        |
| Most PCs already include GPUs, making them widely accessible.              | Computations had to be expressed as graphics operations.                     | Developers could write compute kernels directly, greatly expanding GPU applications. |

- **CUDA (2007):** NVIDIA introduced CUDA, letting developers program GPUs with familiar C/C++ instead of pretending their computation was a graphics operation. This is the moment GPU computing went mainstream.

- **FPGAs (Field-Programmable Gate Arrays):** Worth knowing as a contrast: reconfigurable chips that can be programmed to implement custom digital circuits. They offer high performance and low latency for specialized tasks such as networking, signal processing, and AI inference.

**Summary:** GPUs were already in most PCs, and CUDA made them programmable with plain C/C++. Availability plus programmability is what turned GPU computing from a research trick into a mainstream platform.

---

## 5. Why Applications Need More Speed

You might wonder if we really need all this speed when today's software already feels fast. The book argues yes, very much so.

- Future "super-applications" (molecular biology simulations, AI, climate modeling) require enormous amounts of computation.

- Simulations are especially interesting: they can model systems that physical instruments cannot directly observe.

- Parallel computing, especially on GPUs, provides **10x to 100x speedups** for suitable applications.

- The conclusion is the same as in section 1: continued performance growth depends on exploiting **parallelism**, not on higher CPU clock speeds.

---

## 6. Amdahl's Law

Amdahl's Law is the reality check of parallel computing: **the maximum speedup is limited by the fraction of a program that cannot be parallelized**.

- The larger the parallel portion, the greater the overall speedup.

- Even an infinitely fast GPU cannot accelerate the sequential part. It just sits there, waiting.

- So maximizing the parallel fraction is essential for high performance.

### Formula

$$
S = \frac{1}{(1 - P) + \frac{P}{N}}
$$

where:

- $S$ = overall speedup

- $P$ = parallelizable fraction

- $N$ = speedup of the parallel part

### Worked Examples

**Example 1: 30% parallelizable, 100x faster**

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

**Example 2: 30% parallelizable, infinite speedup**

$$
S = \frac{1}{0.70 + 0} = \frac{1}{0.70} \approx 1.43\times
$$

Notice this one: even with an *infinitely* fast parallel part, we barely improve on Example 1. The sequential 70% dominates everything.

**Example 3: 99% parallelizable, 100x faster**

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

**Key takeaway:** Even extremely fast parallel hardware provides limited benefit unless **most of the application is parallelizable**. Going from 30% to 99% parallel turns a 1.4x speedup into a 50x one.

---

## 7. Memory Bandwidth: The Hidden Bottleneck

Here is a trap that catches a lot of beginners: you parallelize your code perfectly and still only get a modest speedup. The culprit is usually memory.

- **Memory bandwidth often limits speedup**, even when the computation itself parallelizes beautifully.

- Simply parallelizing code usually tops out around **10x speedup** because DRAM becomes the bottleneck. The arithmetic units are starving while they wait for data.

- Using **fast on-chip GPU memories** (shared memory, caches, registers) reduces expensive DRAM accesses and unlocks much more performance.

- The lesson: high speedups come from **optimizing memory access patterns**, not just from throwing more threads at the problem. This theme comes back again and again in later chapters.

---

## 8. Challenges of Parallel Programming

Parallel programming is powerful but not free. Here are the main things that make it hard.

**Algorithm and work efficiency:**

- **Design efficient parallel algorithms** without increasing total work. Some parallel algorithms do more total work than their sequential counterparts, which eats into the speedup.

- **Avoid extra or redundant computation** wherever possible.

- **Manage memory bottlenecks**, since many applications are memory-bound rather than compute-bound (see section 7).

**Load balance and synchronization:**

- **Load imbalance / input sensitivity:** Uneven or unpredictable input data can distribute work unevenly across threads, leaving some threads idle while others do all the heavy lifting.

- **Synchronization overhead:** Threads often need barriers, locks, or atomic operations, which means some threads spend time waiting instead of doing useful work.

- **Communication:** Some applications require frequent thread collaboration, which makes parallelization harder.

- **Embarrassingly parallel vs. synchronized tasks:** Some problems need almost no communication between threads (easy mode), while others require constant synchronization (hard mode, and slower).

**The good news:**

- Many of these challenges have been solved before. **Well-known parallel patterns and techniques** address most of them, and learning those patterns is a big part of what this book is about.

---

## 9. Related Parallel Programming APIs

CUDA is not the only game in town. It helps to know the neighbors.

### OpenMP

- **OpenMP** is a compiler-based API for parallel programming on **shared-memory CPUs**.

- You sprinkle simple compiler directives (`#pragma omp`) over your loops and tasks, and the runtime handles the threads for you.

- Great for parallelizing CPU code with minimal changes.

### MPI (Message Passing Interface)

- Used for **distributed-memory systems** (clusters), where each node has its own memory.

- Processes communicate by **explicitly sending and receiving messages**. Nothing is shared, everything is a message.

- The standard for **high-performance computing (HPC)** and large supercomputers.

- Often combined with **CUDA** for multi-GPU clusters.

### OpenCL (Open Computing Language)

- An **open standard** for parallel programming across CPUs, GPUs, and other accelerators.

- Similar to CUDA in spirit, but **vendor-independent**, so it runs on hardware from different manufacturers.

- Provides portability across devices, though performance tuning is still hardware-specific.

- The concepts map so closely to CUDA that once you know one, picking up the other is easy.
