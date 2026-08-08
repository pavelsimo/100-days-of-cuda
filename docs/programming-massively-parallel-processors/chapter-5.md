# Programming Massively Parallel Processors: Chapter 5, Memory Architecture and Data Locality

1. [Memory Bandwidth as a Performance Limiter](#1-memory-bandwidth-as-a-performance-limiter)
   - [Compute-bound vs. Memory-bound](#compute-bound-vs-memory-bound)
   - [Arithmetic Intensity](#arithmetic-intensity)
   - [Hardware Threshold and the Roofline Model](#hardware-threshold-and-the-roofline-model)
2. [Arithmetic Intensity Examples](#2-arithmetic-intensity-examples)
   - [Simple Vector Operations](#simple-vector-operations)
   - [Vector Addition](#vector-addition)
   - [Bandwidth Utilization](#bandwidth-utilization)
3. [Matrix Multiplication](#3-matrix-multiplication)
   - [Ideal Arithmetic Intensity](#ideal-arithmetic-intensity)
   - [Why the Naive Kernel Is Memory-bound](#why-the-naive-kernel-is-memory-bound)
4. [Data Locality and Reuse](#4-data-locality-and-reuse)
5. [CUDA Memory Types](#5-cuda-memory-types)
   - [Registers](#registers)
   - [Shared Memory](#shared-memory)
   - [Local Memory](#local-memory)
   - [CUDA Variable Declarations](#cuda-variable-declarations)
   - [Scope and Lifetime](#scope-and-lifetime)
6. [Tiling for Reduced Memory Traffic](#6-tiling-for-reduced-memory-traffic)
   - [How Tiling Works](#how-tiling-works)
   - [Data Reuse and Locality](#data-reuse-and-locality)
7. [Tiled Matrix Multiplication Kernel](#7-tiled-matrix-multiplication-kernel)
   - [Thread-to-Output Mapping](#thread-to-output-mapping)
   - [Kernel Phases](#kernel-phases)
   - [Synchronization](#synchronization)
   - [Strip-Mining](#strip-mining)
   - [Performance Impact](#performance-impact)
   - [Optimized Libraries](#optimized-libraries)
   - [Shared-Memory and Register Tiling](#shared-memory-and-register-tiling)
8. [Boundary Handling](#8-boundary-handling)
   - [Partial Edge Tiles](#partial-edge-tiles)
   - [Rectangular Matrices](#rectangular-matrices)
9. [Memory Usage and Occupancy](#9-memory-usage-and-occupancy)
   - [Shared-Memory Limits](#shared-memory-limits)
   - [Dynamic Shared Memory](#dynamic-shared-memory)
10. [Summary](#10-summary)

---

## 1. Memory Bandwidth as a Performance Limiter

GPU performance has two main hardware limits:

- **Peak compute throughput**: arithmetic operations per second, measured in FLOP/s.
- **Peak memory bandwidth**: bytes transferred per second, measured in GB/s or TB/s.

### Compute-bound vs. Memory-bound

| Compute-bound | Memory-bound |
| --- | --- |
| Arithmetic throughput is the bottleneck. | Memory bandwidth is the bottleneck. |
| Compute units stay busy. | Compute units may wait for data. |
| Many operations per byte transferred. | Few operations per byte transferred. |

Thousands of threads can hide **memory latency**, but they cannot remove a **bandwidth limit**.

### Arithmetic Intensity

Arithmetic intensity measures useful computation per byte transferred from global memory:

$$
\text{Arithmetic intensity}
= \frac{\text{Floating-point operations}}{\text{Bytes transferred from global memory}}
$$

Unit: **FLOP/B**

- High FLOP/B: more likely compute-bound.
- Low FLOP/B: more likely memory-bound.

### Hardware Threshold and the Roofline Model

The approximate boundary between memory-bound and compute-bound execution is:

$$
\text{Threshold}
= \frac{\text{Peak compute throughput}}{\text{Peak memory bandwidth}}
$$

H100 example from the book:

$$
\frac{66.9\ \text{TFLOP/s}}{3.35\ \text{TB/s}}
\approx 20\ \text{FLOP/B}
$$

- Below 20 FLOP/B: likely memory-bound.
- Above 20 FLOP/B: likely compute-bound.
- The threshold depends on the GPU.

Peak values are limits, not guaranteed performance. The **Roofline Model** expresses the limit as:

$$
\text{Attainable performance}
\leq \min(\text{Peak compute},\ \text{Arithmetic intensity} \times \text{Peak bandwidth})
$$

**Summary:** Arithmetic intensity connects an algorithm's memory traffic to the GPU's compute and bandwidth limits.

---

## 2. Arithmetic Intensity Examples

Assume one `float` is **4 bytes**.

### Simple Vector Operations

| Operation | FLOPs | Global-memory traffic | Intensity |
| --- | ---: | ---: | ---: |
| `x[i] = y[i]` | 0 | 8 B | 0 FLOP/B |
| `x[i] = y[i] + 1.f` | 1 | 8 B | 0.125 FLOP/B |
| `x[i] = y[i] + z[i]` | 1 | 12 B | 0.083 FLOP/B |

More computation does not always mean higher arithmetic intensity. Additional memory accesses can lower it.

### Vector Addition

```cpp
C[i] = A[i] + B[i];
```

Per element:

- 1 FLOP: one addition.
- 12 B: two 4-byte reads and one 4-byte write.

$$
\text{Arithmetic intensity}
= \frac{1\ \text{FLOP}}{12\ \text{B}}
\approx 0.083\ \text{FLOP/B}
$$

On the example H100, $0.083 \ll 20$ FLOP/B. Vector addition is strongly memory-bound.

### Bandwidth Utilization

For $10^9$ elements:

$$
12\ \text{B} \times 10^9 = 12\ \text{GB}
$$

If the kernel takes 4 ms:

$$
\text{Effective bandwidth}
= \frac{12\ \text{GB}}{0.004\ \text{s}}
= 3\ \text{TB/s}
$$

Relative to 3.35 TB/s peak bandwidth:

$$
\frac{3}{3.35} \approx 90\%
$$

The bandwidth-only lower bound is:

$$
\frac{12\ \text{GB}}{3.35\ \text{TB/s}} \approx 3.6\ \text{ms}
$$

At 4 ms, little improvement is possible without reducing memory traffic.

**Summary:** Vector addition performs too little work per byte to use much of the GPU's compute throughput.

---

## 3. Matrix Multiplication

### Ideal Arithmetic Intensity

Multiplying two $N \times N$ matrices requires approximately:

$$
2N^3\ \text{FLOP}
$$

With ideal data reuse, each input matrix is read once and the output matrix is written once:

$$
3N^2 \times 4\ \text{B} = 12N^2\ \text{B}
$$

Therefore:

$$
\text{Arithmetic intensity}
= \frac{2N^3}{12N^2}
= \frac{N}{6}\ \text{FLOP/B}
$$

For $N = 1024$:

$$
\frac{1024}{6} \approx 171\ \text{FLOP/B}
$$

This is above the H100 threshold of 20 FLOP/B, so matrix multiplication can be compute-bound with sufficient reuse.

### Why the Naive Kernel Is Memory-bound

Naive inner loop:

```cpp
Pvalue += M[row * width + k] * N[k * width + col];
```

Each iteration performs:

- 2 FLOPs: one multiplication and one addition.
- 8 B: two global-memory reads.

Ignoring the small amortized cost of the final output write:

$$
\text{Arithmetic intensity}
\approx \frac{2\ \text{FLOP}}{8\ \text{B}}
= 0.25\ \text{FLOP/B}
$$

Bandwidth-limited throughput on the example H100:

$$
3.35\ \text{TB/s} \times 0.25\ \text{FLOP/B}
\approx 0.84\ \text{TFLOP/s}
$$

This is far below the 66.9 TFLOP/s compute peak.

**Summary:** Matrix multiplication has high theoretical arithmetic intensity, but a naive kernel loses it by repeatedly loading the same values from global memory.

---

## 4. Data Locality and Reuse

Main optimization goal:

> Reduce global-memory traffic by reusing data in fast on-chip memory.

- **Data locality**: keep frequently used values close to the compute units.
- **Reuse**: load a value from global memory once, then use it for multiple operations.
- **Shared memory, caches, and registers**: faster on-chip storage for reused data.
- **Tiling**: divide data into smaller blocks that fit on-chip and can be shared by threads.

For matrix multiplication, tiling lets many threads reuse the same matrix values. This raises arithmetic intensity and moves performance closer to the compute limit.

**Summary:** Effective CUDA optimization depends on moving less data, not only on launching more threads.


---

## 5. CUDA Memory Types

CUDA provides different memory types with different **speeds, scopes, and lifetimes**:

| Memory | Scope | Location | Speed |
| --- | --- | --- | --- |
| **Registers** | Thread | On-chip | Fastest |
| **Shared memory** | Block | On-chip | Very fast |
| **Local memory** | Thread | Global DRAM | Slow |
| **Global memory** | Grid | Off-chip DRAM | Slow |
| **Constant memory** | Grid | Cached on-chip | Fast for reads |

### Registers

- Private to each thread.
- Used for frequently accessed variables.
- Extremely fast and do not consume global-memory bandwidth.
- Hold each warp's execution state, which makes warp switching fast.

Register capacity is limited per SM, so high register use per thread can reduce occupancy.

### Shared Memory

- Shared by all threads in a block.
- Located on-chip and much faster than global memory.
- Useful for data reuse and cooperation between threads.

```text
Thread ─┐
Thread ─┼── Shared Memory
Thread ─┤       │
Thread ─┘       ↓
             Global Memory
```

### Local Memory

Despite the name, **local memory is stored in global memory**. It is private to one thread and has similar latency to global memory. It is commonly used for register spills or thread-local arrays that the compiler cannot place in registers.

The basic goal is to move frequently reused data out of global memory:

$$
\text{Global Memory} \rightarrow \text{Shared Memory / Registers}
$$

This reduces global-memory traffic and raises effective arithmetic intensity.

### CUDA Variable Declarations

Where a CUDA variable lives depends on **how it is declared**:

| Declaration | Memory | Scope | Lifetime |
| --- | --- | --- | --- |
| Automatic scalar | Register | Thread | Grid |
| Automatic array | Local | Thread | Grid |
| `__shared__` | Shared | Block | Grid |
| `__device__` | Global | Grid | Application |
| `__constant__` | Constant | Grid | Application |

Automatic scalars normally use registers:

```c
float x;
```

Automatic arrays usually go into **local memory**, which is much slower. The compiler may still place small arrays with constant indexing in registers.

A shared declaration creates one copy per block for the duration of that block:

```c
__shared__ float tile[256];
```

Shared variables are commonly used to cache global-memory data for reuse by the block. Constant variables instead hold read-only data that is visible to every thread for the application's lifetime:

```c
__constant__ float data[256];
```

Kernel code cannot modify constant memory, and its cache provides fast access for read-only kernel inputs.

### Scope and Lifetime

**Scope** tells us *who can access the variable*:

$$
\text{Thread} \rightarrow \text{Block} \rightarrow \text{Grid}
$$

**Lifetime** tells us *how long the variable exists*, such as one thread or block execution, or the entire application.

---

## 6. Tiling for Reduced Memory Traffic

**Tiling** splits data into small pieces called **tiles** that fit in fast shared memory:

$$
\text{Global Memory} \rightarrow \text{Shared Memory} \rightarrow \text{Reuse}
$$

### How Tiling Works

In naive matrix multiplication, threads repeatedly load the same matrix elements from global memory. Tiling avoids much of that repeated work:

1. Threads cooperatively load one tile each from $M$ and $N$ into shared memory.
2. The block synchronizes.
3. Threads reuse the tiles for multiple calculations.
4. The block repeats the process for the next tile pair.

For a `TILE_WIDTH` by `TILE_WIDTH` tile, global-memory traffic can fall by roughly a factor of `TILE_WIDTH`. With a width of 32, that is about $1/32$ of the naive input traffic.

### Data Reuse and Locality

The shared-memory arrays are reused across phases, so only a small portion of each matrix must be on-chip at once. A $2 \times 2$ tile reuses each loaded value twice; a tile of width $T$ can reuse it up to $T$ times.

> Focus computation on a small subset of data, keep it in fast on-chip memory, and reuse it before loading new data.

The key idea is simple: tiling trades limited shared-memory capacity for much less global-memory traffic.

---

## 7. Tiled Matrix Multiplication Kernel

Each thread computes one element of $P$, while the block cooperates to load tiles into shared memory.

### Thread-to-Output Mapping

For tile width $T$:

$$
Row = blockIdx.y \times T + threadIdx.y
$$

$$
Col = blockIdx.x \times T + threadIdx.x
$$

The thread accumulates $P[Row][Col]$ across all tile pairs.

### Kernel Phases

For every tile pair:

```c
Mds[ty][tx] = M[Row * width + ph * T + tx];
Nds[ty][tx] = N[(ph * T + ty) * width + Col];

__syncthreads();

for (int k = 0; k < T; k++)
    Pvalue += Mds[ty][k] * Nds[k][tx];

__syncthreads();
```

Each phase:

1. Threads cooperatively load tiles from $M$ and $N$.
2. `__syncthreads()` ensures the tiles are fully loaded.
3. Threads compute using the shared-memory tiles.
4. Synchronize again before replacing them with the next tiles.

Number of phases:

$$
\frac{width}{TILE_WIDTH}
$$

Finally:

```c
P[Row * width + Col] = Pvalue;
```

### Synchronization

Each of the $T^2$ threads loads:

- One element of $M$.
- One element of $N$.

into shared memory.

Two synchronizations are required per phase:

```c
// Load tiles
__syncthreads();   // Wait until all data is loaded

// Compute using tiles
__syncthreads();   // Wait until everyone is done before overwriting
```

The first handles a **read-after-write** dependency. Threads cannot read the tile until other threads finish writing it.

The second handles a **write-after-read** dependency. Threads cannot overwrite the tile with the next phase until everyone finishes reading it.

### Strip-Mining

Tiling breaks a large dot-product loop into smaller **phases**, a technique called **strip-mining**:

$$
\text{Large loop} \rightarrow \text{multiple tile-sized phases}
$$

Each phase loads a small chunk into shared memory, computes with it, then moves to the next chunk.

### Performance Impact

Tiling can reduce input reads by roughly `TILE_WIDTH`. For a $32 \times 32$ tile:

$$
0.25\text{ FLOP/B}\times32
=8\text{ FLOP/B}
$$

With H100 bandwidth:

$$
3.35\text{ TB/s}\times8\text{ FLOP/B}
=26.8\text{ TFLOP/s}
$$

This is a theoretical $32\times$ improvement over the naive limit of $0.84\ \text{TFLOP/s}$. Since $8\ \text{FLOP/B}$ is still below the example H100 threshold of $20\ \text{FLOP/B}$, this basic tiled kernel remains memory-bound.

### Optimized Libraries

Even with tiling, the kernel may still be below peak performance. Highly optimized libraries already implement more advanced matrix multiplication techniques:

- **cuBLAS**: NVIDIA's optimized BLAS library for operations such as GEMM.
- **CUTLASS**: CUDA templates for building optimized matrix multiplication kernels.

These libraries combine techniques such as tiling, register reuse, shared memory, and Tensor Cores to get much closer to peak GPU performance.

For real applications, it is usually better to use cuBLAS or CUTLASS than to write a basic matrix multiplication kernel from scratch.

### Shared-Memory and Register Tiling

Tiling means keeping frequently reused data in **faster memory**, not only shared memory.

- **Shared-memory tiling:** data reused across threads is stored in shared memory.
- **Register tiling:** data repeatedly used by one thread is kept in registers, such as `Pvalue`.

On GPUs, explicitly using shared memory is important because many threads compete for cache capacity.

---

## 8. Boundary Handling

There is one catch: the basic tiled kernel assumes square matrices whose dimensions are multiples of `TILE_WIDTH`. Other inputs create partially valid edge tiles. For example, a $3 \times 3$ matrix with `TILE_WIDTH = 2` produces threads whose coordinates fall outside the matrix.

Invalid accesses can:

- Read the wrong element due to linearized indexing.
- Access memory outside the allocated array.
- Produce incorrect results or execution errors.

We cannot simply disable threads that do not produce valid output. They may still need to load an element of $M$ or $N$ for another thread.

### Partial Edge Tiles

For partial edge tiles, each global memory access must be checked before loading.

For $M$:

```c
if (Row < width && ph * TILE_WIDTH + tx < width)
    Mds[ty][tx] = M[Row * width + ph * TILE_WIDTH + tx];
else
    Mds[ty][tx] = 0.0f;
```

For $N$:

```c
if (ph * TILE_WIDTH + ty < width && Col < width)
    Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + Col];
else
    Nds[ty][tx] = 0.0f;
```

Invalid elements are replaced with **zero**, so they do not affect the dot product.

The final result is also written only when valid:

```c
if (Row < width && Col < width)
    P[Row * width + Col] = Pvalue;
```

Number of phases must round up:

$$
\text{phases} =
\left\lceil
\frac{\text{width}}{\text{TILE_WIDTH}}
\right\rceil
$$

Rounding up includes the final partial tile while zero padding makes its invalid elements neutral in the dot product.

### Rectangular Matrices

For rectangular matrices:

$$
M_{j\times k}N_{k\times l}=P_{j\times l}
$$

The same tiled approach works by tracking the output rows $j$, shared inner dimension $k$, and output columns $l$ separately.

---

## 9. Memory Usage and Occupancy

Registers and shared memory reduce global-memory traffic, but both are limited per SM. Greater use per thread or block can leave fewer resources for concurrent work:

$$
\text{More registers/shared memory}
\Rightarrow
\text{fewer resident threads/blocks}
\Rightarrow
\text{lower occupancy}
$$

So there is a balance: use enough on-chip memory to improve reuse, but leave enough occupancy to hide latency.

### Shared-Memory Limits

Shared memory is limited per SM, so high usage can reduce the number of resident threads and therefore **occupancy**.

H100 example:

$$
\frac{228\text{ KB}}{2048\text{ threads}}
\approx 114\text{ B/thread}
$$

For tiled matrix multiplication, each block uses two shared tiles:

$$
2\times TILE_WIDTH^2\times4\text{ B}
$$

Since the block has $T^2$ threads:

$$
\frac{2T^2(4)}{T^2}=8\text{ B/thread}
$$

At 8 B per thread, shared memory does not limit occupancy in this example.

For comparison, a kernel using 38 KB per 256-thread block requires about:

$$
\frac{38\text{ KB}}{256}\approx152\text{ B/thread}
$$

Maximum resident threads:

$$
\frac{228\text{ KB}}{152\text{ B/thread}}
\approx1536
$$

Therefore:

$$
\text{Occupancy}=\frac{1536}{2048}=75%
$$

### Dynamic Shared Memory

Static shared memory has a compile-time size:

```c
__shared__ float Mds[32][32];
```

Dynamic shared memory allows the size to be selected at runtime:

```c
extern __shared__ char shared[];
```

The size is specified as the **third kernel launch parameter**:

```c
kernel<<<grid, block, size>>>(...);
```

For two $32 \times 32$ `float` tiles:

$$
size=2\times32\times32\times4
=8192\text{ B}
$$

This allows shared-memory usage to change with the device or kernel configuration without recompiling.

---

## 10. Summary

- Arithmetic intensity relates useful FLOPs to bytes transferred from global memory.
- High intensity tends toward compute-bound execution; low intensity tends toward memory-bound execution.
- Registers, shared memory, and constant memory reduce global-memory traffic when data is reused.
- Tiling improves locality by loading small subsets into shared memory and reusing them.
- Cooperative tile loading requires `__syncthreads()` before use and before overwrite.
- Bounds checks and zero padding make partial tiles safe without skipping required barriers.
- Register and shared-memory use must be balanced against occupancy.
- Dynamic shared memory allows its size to be chosen at launch time.

**Main idea:** maximize **data locality and reuse** to reduce expensive global memory traffic, while balancing on-chip memory usage against occupancy.
