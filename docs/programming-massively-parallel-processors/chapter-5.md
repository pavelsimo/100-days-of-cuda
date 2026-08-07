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
