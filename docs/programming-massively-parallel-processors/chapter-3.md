> Dear reader: These notes were created with the help of AI, with me cherry-picking the parts of the book I found most relevant. I also reviewed the content to make sure no AI hallucinations slipped through. I hope you find them useful. Happy reading! :)

# Programming Massively Parallel Processors: Chapter 3, Multidimensional Grids and Data

1. [Grid and Block Organization](#1-grid-and-block-organization)
   - [The Two-Level Hierarchy](#the-two-level-hierarchy)
   - [Execution Configuration](#execution-configuration)
   - [The 1D Shortcut](#the-1d-shortcut)
2. [Dynamic Grid Sizing](#2-dynamic-grid-sizing)
3. [Built-in Variables Inside a Kernel](#3-built-in-variables-inside-a-kernel)
4. [CUDA Limits](#4-cuda-limits)
   - [Grid Limits](#grid-limits)
   - [Block Limits](#block-limits)
   - [Rules Worth Remembering](#rules-worth-remembering)
5. [Mapping Threads to Data](#5-mapping-threads-to-data)
   - [Choosing a Grid Shape](#choosing-a-grid-shape)
   - [Computing Global Coordinates](#computing-global-coordinates)
   - [Worked Example: A 2D Image](#worked-example-a-2d-image)
6. [Linearizing Multidimensional Arrays](#6-linearizing-multidimensional-arrays)
   - [Why Flattening Is Necessary](#why-flattening-is-necessary)
   - [Row-Major vs. Column-Major](#row-major-vs-column-major)
7. [3D Grids and Data](#7-3d-grids-and-data)
8. [BLAS: Basic Linear Algebra Subprograms](#8-blas-basic-linear-algebra-subprograms)
   - [The Three BLAS Levels](#the-three-blas-levels)

---

## 1. Grid and Block Organization

The key idea of this chapter is that CUDA organizes threads in a **two-level hierarchy** that can be up to **three-dimensional**, and your job is to map that hierarchy onto the shape of your data.

### The Two-Level Hierarchy

1. **A grid is a 3D array of blocks**
    - The grid is the top level: every kernel launch creates exactly one grid.

2. **A block is a 3D array of threads**
    - Blocks are the second level, and every block in a grid has the same dimensions.

3. **Unused dimensions are simply set to 1**
    - A "1D grid" is really a 3D grid with `y = 1` and `z = 1`.

4. **Dimensions are specified with the `dim3` type**

```cpp
dim3 grid(x, y, z);
dim3 block(x, y, z);
kernel<<<grid, block>>>();
```

### Execution Configuration

The `<<<...>>>` syntax between the kernel name and its arguments is the **execution configuration**:

```cpp
kernel<<<gridDim, blockDim>>>();
```

- The first parameter (`gridDim`) is the **number of blocks** in the grid.
- The second parameter (`blockDim`) is the **number of threads per block**.
- Total threads = `gridDim.x * blockDim.x` (for the 1D case).

Example:

```cpp
kernel<<<32, 128>>>();
```

- 32 blocks
- 128 threads per block
- Total threads = 32 × 128 = **4096**

### The 1D Shortcut

For one-dimensional launches you do not need `dim3` at all. Instead of:

```cpp
dim3 grid(16, 1, 1);
dim3 block(256, 1, 1);
kernel<<<grid, block>>>();
```

you can simply write:

```cpp
kernel<<<16, 256>>>();
```

CUDA assumes `y = 1` and `z = 1` for both the grid and the block.

**Summary:** A kernel launch creates a grid of blocks, each block a (up to 3D) array of threads. `dim3` describes both levels, and plain integers are a convenient shortcut for the 1D case.

---

## 2. Dynamic Grid Sizing

A common and important pattern: **fix the block size, and let the number of blocks scale with the input**. This makes the same kernel work for any input size.

```cpp
kernel<<<ceil(n / 256.0), 256>>>();
```

- Block size stays constant at 256 threads.
- The number of blocks grows with `n`, so there are always enough threads to cover every element.
- `ceil()` is essential: without it, a partial last block would be dropped and the final elements would never be processed.

Examples:

- `n = 1000` → `ceil(1000 / 256.0)` = **4 blocks**
- `n = 4000` → `ceil(4000 / 256.0)` = **16 blocks**

**Summary:** Hard-coding the grid size ties your kernel to one input size. Computing the block count with `ceil(n / blockSize)` makes the launch configuration scale automatically, at the small cost of a few extra threads that must be masked off with a bounds check.

---

## 3. Built-in Variables Inside a Kernel

Inside a kernel, CUDA automatically provides a set of built-in variables that tell each thread where it lives in the hierarchy:

| Variable    | Meaning                                    | Set by                       |
| ----------- | ------------------------------------------ | ---------------------------- |
| `gridDim`   | Number of blocks in the grid (`.x .y .z`)  | The execution configuration  |
| `blockDim`  | Number of threads per block (`.x .y .z`)   | The execution configuration  |
| `blockIdx`  | This block's index within the grid         | The runtime, per block       |
| `threadIdx` | This thread's index within its block       | The runtime, per thread      |

Properties:

- They are **read-only**.
- They are **automatically initialized** from the launch configuration; you never assign them yourself.

**Summary:** Every thread runs the same kernel code, and these built-in variables are what let each thread figure out which piece of the data is its own.

---

## 4. CUDA Limits

The hierarchy is not unlimited. Both grids and blocks have hard caps you need to respect.

### Grid Limits

- `gridDim.x`: **1 to 2³¹ − 1**
- `gridDim.y`: **1 to 65,535 (2¹⁶ − 1)**
- `gridDim.z`: **1 to 65,535 (2¹⁶ − 1)**

Note the asymmetry: the `x` dimension is enormous, while `y` and `z` are much smaller.

### Block Limits

- A block can have at most **1024 threads in total**.
- Those threads can be distributed across `x`, `y`, and `z` in any way, as long as the product stays ≤ 1024.

Valid examples:

```cpp
(512, 1, 1)   // 512 threads
(8, 16, 4)    // 512 threads
(32, 16, 2)   // 1024 threads
```

Invalid:

```cpp
(32, 32, 2)   // 2048 threads > 1024
```

### Rules Worth Remembering

- **Every block in a grid has the same block dimensions.**
- **Grid and block dimensionality do not have to match.** A 2D grid of 3D blocks is perfectly valid.
- All threads in the same block share the same `blockIdx`.
- Block indices range from `0` to `gridDim - 1` in each dimension:
    - `blockIdx.x = 0 ... gridDim.x - 1`
    - `blockIdx.y = 0 ... gridDim.y - 1`
    - `blockIdx.z = 0 ... gridDim.z - 1`

**Summary:** 1024 threads per block is the hard ceiling, the grid's `x` dimension is effectively unbounded while `y` and `z` cap at 65,535, and every block in a grid is the same shape.

---

## 5. Mapping Threads to Data

This is the heart of the chapter: the point of multidimensional grids is to make the thread organization **mirror the data organization**.

### Choosing a Grid Shape

Choose the grid layout to match the structure of the data:

| Data                        | Grid    |
| --------------------------- | ------- |
| Vectors (1D)                | 1D grid |
| Images, matrices (2D)       | 2D grid |
| Volumes, simulations (3D)   | 3D grid |

### Computing Global Coordinates

Each thread typically processes **one data element**, and it finds that element by combining `blockIdx`, `blockDim`, and `threadIdx` into global coordinates:

```cpp
row = blockIdx.y * blockDim.y + threadIdx.y;
col = blockIdx.x * blockDim.x + threadIdx.x;
```

Notice the convention: **`y` maps to rows and `x` maps to columns**. Mixing these up is a classic source of bugs.

### Worked Example: A 2D Image

Process a **62 × 76** image with **16 × 16** blocks:

- Grid size: `ceil(76/16) × ceil(62/16)` = **5 × 4**
- Total blocks: **20**
- Total threads: 20 × 256 = 5120, but the image only has 62 × 76 = 4712 pixels.

Here is the catch: the edge blocks hang over the boundary of the image, so some threads have no pixel to process. Every thread must **check bounds** before touching memory:

```cpp
if (row < height && col < width) {
    // safe to process pixel (row, col)
}
```

**Summary:** Match the grid dimensionality to the data, compute each thread's global coordinates from `blockIdx * blockDim + threadIdx`, and always guard with a bounds check because `ceil()` rounds the grid up past the edge of the data.

---

## 6. Linearizing Multidimensional Arrays

Here is a surprise for newcomers: even though grids and blocks are multidimensional, **device memory is flat**. Your 2D and 3D arrays have to be flattened by hand.

### Why Flattening Is Necessary

- Device memory is a **flat (1D)** address space, not truly multidimensional.
- **Dynamically allocated 2D arrays** cannot be indexed as `A[row][col]`, because the compiler does not know the number of columns at compile time.
- So CUDA programmers **linearize (flatten)** multidimensional arrays into 1D and compute the index manually:

```cpp
index = row * width + col;
```

### Row-Major vs. Column-Major

There are two conventions for laying out a 2D array in flat memory:

| Row-major (C/C++, CUDA)          | Column-major (Fortran, MATLAB)     |
| -------------------------------- | ---------------------------------- |
| Rows are stored consecutively.   | Columns are stored consecutively.  |
| `index = row * width + col;`     | `index = col * height + row;`      |

**Note:** CUDA, like C/C++, uses **row-major layout** by default. Keep this in mind when interfacing with libraries or languages that assume column-major.

**Summary:** Multidimensional arrays are an illusion the programmer maintains: memory is 1D, so you compute `row * width + col` yourself, and CUDA follows C's row-major convention.

---

## 7. 3D Grids and Data

Everything from the 2D case extends naturally to 3D. The mental model: a 3D array is a **stack of 2D planes**, and the `z` dimension selects the plane.

Each thread computes three global coordinates:

```cpp
int plane = blockIdx.z * blockDim.z + threadIdx.z;
int row   = blockIdx.y * blockDim.y + threadIdx.y;
int col   = blockIdx.x * blockDim.x + threadIdx.x;
```

The flattened row-major index adds one more term for the plane:

```cpp
int index = plane * rows * cols + row * cols + col;
```

Two things to remember:

- Launch enough blocks in **all three dimensions** (`x`, `y`, and `z`) to cover the data.
- Check **all three boundaries** before accessing memory:

```cpp
if (plane < depth && row < rows && col < cols) {
    // safe to access element (plane, row, col)
}
```

**Summary:** A 3D array is just a stack of 2D planes in flat memory. The indexing pattern generalizes directly: one extra coordinate, one extra term in the index formula, and one extra bounds check.

---

## 8. BLAS: Basic Linear Algebra Subprograms

Matrix and vector operations are so common that they were standardized decades ago into **BLAS**, and it is worth knowing the vocabulary because GPU libraries (like cuBLAS) are organized around it.

- BLAS is the standard library interface for **high-performance linear algebra**.
- It is widely used in **scientific computing, AI, and engineering**.
- It provides optimized implementations of common vector and matrix operations.
- Higher BLAS levels perform more computation per data element and are usually more efficient.

### The Three BLAS Levels

| Level | Operations        | Example                        | Formula          |
| ----- | ----------------- | ------------------------------ | ---------------- |
| 1     | **Vector-vector** | Vector addition, dot product   | `y = αx + y`     |
| 2     | **Matrix-vector** | Multiplying a matrix by a vector | `y = αAx + βy` |
| 3     | **Matrix-matrix** | Matrix multiplication          | `C = αAB + βC`   |

- **Level 1** does the least work per element: for vectors of length `n`, it performs on the order of `n` operations.
- **Level 2** steps up to matrix-vector products.
- **Level 3** is the most computationally intensive and the best suited for GPUs because it performs enough arithmetic per byte of data to keep all those threads busy.

**Note:** Matrix multiplication is a **Level 3 BLAS** operation, making it a key workload for CUDA.

**Summary:** BLAS classifies linear algebra into three levels: vector-vector, matrix-vector, and matrix-matrix. The higher the level, the more computation per data element, which is exactly what GPUs thrive on; that is why matrix multiplication (Level 3) is such a central CUDA workload.
