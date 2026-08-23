> Dear reader: These notes were created with the help of AI, with me cherry-picking the parts of the book I found most relevant. I also reviewed the content to make sure no AI hallucinations slipped through. I hope you find them useful. Happy reading! :)

# Programming Massively Parallel Processors: Chapter 7, Convolution and Constant Memory

1. [Convolution Basics](#1-convolution-basics)
   - [One-Dimensional Convolution](#one-dimensional-convolution)
   - [Boundary Conditions](#boundary-conditions)
   - [Two-Dimensional Convolution](#two-dimensional-convolution)
2. [Basic Parallel Convolution](#2-basic-parallel-convolution)
   - [Mapping Threads to Outputs](#mapping-threads-to-outputs)
   - [The Basic Kernel](#the-basic-kernel)
   - [Boundary Divergence](#boundary-divergence)
3. [Memory Bandwidth Considerations](#3-memory-bandwidth-considerations)
   - [Ideal Arithmetic Intensity](#ideal-arithmetic-intensity)
   - [Traffic in the Basic Kernel](#traffic-in-the-basic-kernel)
4. [Constant Memory and Caching](#4-constant-memory-and-caching)
   - [Why the Filter Fits](#why-the-filter-fits)
   - [Declaring and Initializing Constant Memory](#declaring-and-initializing-constant-memory)
   - [The Constant Cache](#the-constant-cache)
   - [Effect on Arithmetic Intensity](#effect-on-arithmetic-intensity)
5. [Tiled Convolution with Halo Cells](#5-tiled-convolution-with-halo-cells)
   - [Why the Input Tile Is Larger](#why-the-input-tile-is-larger)
   - [Choosing the Thread Block](#choosing-the-thread-block)
   - [The Two Kernel Phases](#the-two-kernel-phases)
   - [Mapping Halo Threads to Outputs](#mapping-halo-threads-to-outputs)
   - [Arithmetic Intensity of Tiled Convolution](#arithmetic-intensity-of-tiled-convolution)
6. [Using Caches for Halo Cells](#6-using-caches-for-halo-cells)
   - [A Smaller Shared-Memory Tile](#a-smaller-shared-memory-tile)
   - [Three Possible Input Paths](#three-possible-input-paths)
   - [Comparing the Two Designs](#comparing-the-two-designs)
7. [Summary](#7-summary)

---

## 1. Convolution Basics

Convolution appears in audio processing, digital recording, image and video processing, computer vision, and many other numerical applications. A filter turns nearby input values into a more useful output. Depending on its weights, it might smooth an image, sharpen details, or highlight edges.

The operation suits a GPU because different output elements can be calculated independently. The performance challenge comes from data reuse: neighboring outputs need many of the same inputs and all of them use the same filter.

A **convolution filter** is the small array of weights applied at every output position. It is also commonly called a convolution kernel. These notes use *filter* so it cannot be confused with a CUDA kernel function.

Convolution can work across several dimensions:

- **1D convolution** is common for audio and other sampled signals.
- **2D convolution** is widely used for images.
- **3D convolution** can process video or volumetric data.

The mechanics are easiest to see in one dimension.

### One-Dimensional Convolution

Let $x$ be the input, $f$ the filter, and $y$ the output. For a filter with radius $r$:

$$
y_i = \sum_{j=-r}^{r} f_{j+r}x_{i+j}
$$

The filter contains $2r+1$ elements. Its width is normally odd, which gives it a clear center that can be aligned with the output position.

Consider this input and filter:

$$
x = [8, 2, 5, 4, 1, 7, 3]
$$

$$
f = [1, 3, 5, 3, 1]
$$

The filter has five elements, so $r=2$. Centering it on $x_2$ gives:

$$
\begin{aligned}
y_2
&= 1(8) + 3(2) + 5(5) + 3(4) + 1(1) \\
&= 8 + 6 + 25 + 12 + 1 \\
&= \boxed{52}
\end{aligned}
$$

Every output repeats the same weighted-sum calculation at a different position. That makes the initial parallel strategy quite natural: assign one output to each thread.

### Boundary Conditions

Near an edge, part of the filter extends beyond the available input. These missing positions are often called **ghost cells**.

Common boundary policies include:

- **Zero padding:** Treat every ghost cell as zero.
- **Edge replication:** Reuse the nearest valid input value.
- **Circular wrapping:** Continue from the opposite end of the input.

The right choice depends on the application. This chapter uses zero padding, so an out-of-bounds position contributes $0 \times f = 0$ to the sum.

Boundary handling also affects performance. Threads near an edge need extra checks, and tiled kernels must account for ghost cells when they fill shared memory.

### Two-Dimensional Convolution

For an image, the filter slides in both directions. Let $N$ be the input, $F$ the filter, and $P$ the output. With horizontal radius $r_x$ and vertical radius $r_y$:

$$
P_{y,x}
= \sum_{j=-r_y}^{r_y}
  \sum_{k=-r_x}^{r_x}
  F_{j+r_y,k+r_x}N_{y+j,x+k}
$$

To calculate one output pixel, we center the filter on the matching input position, multiply corresponding filter and input values, and add every product. A $5 \times 5$ filter has $r_x=r_y=2$ and touches up to 25 input values per output. In the book's worked example, those 25 products sum to $P_{2,2}=321$.

At an edge or corner, some of the requested coordinates fall outside the image. Under zero padding, those positions simply add nothing.

The filter slides only a short distance between adjacent outputs, so their input regions overlap heavily. This spatial reuse is the main opportunity explored throughout the chapter.

---

## 2. Basic Parallel Convolution

The direct CUDA implementation maps one thread to one output pixel. Threads form a 2D grid that mirrors the output image.

### Mapping Threads to Outputs

Each thread finds its output coordinates with the usual CUDA mapping:

```cpp
int outCol = blockIdx.x * blockDim.x + threadIdx.x;
int outRow = blockIdx.y * blockDim.y + threadIdx.y;
```

A $32 \times 32$ block contains 1024 threads and can therefore produce up to 1024 output pixels. A thread centered at `(outRow, outCol)` reads an input window whose top-left coordinate is:

$$
(\text{outRow}-r,\ \text{outCol}-r)
$$

The result goes directly to `(outRow, outCol)` in the output.

### The Basic Kernel

Each thread loops over the filter, checks the corresponding input coordinates, and accumulates one result in a register. The central part of the kernel looks like this:

```cpp
int outCol = blockIdx.x * blockDim.x + threadIdx.x;
int outRow = blockIdx.y * blockDim.y + threadIdx.y;

if (outRow < height && outCol < width) {
    float value = 0.0f;

    for (int fRow = 0; fRow < FILTER_DIM; ++fRow) {
        for (int fCol = 0; fCol < FILTER_DIM; ++fCol) {
            int inRow = outRow - FILTER_RADIUS + fRow;
            int inCol = outCol - FILTER_RADIUS + fCol;

            if (inRow >= 0 && inRow < height &&
                inCol >= 0 && inCol < width) {
                value += F[fRow][fCol]
                       * N[inRow * width + inCol];
            }
        }
    }

    P[outRow * width + outCol] = value;
}
```

Keeping `value` in a register avoids writing partial sums to global memory. Only the completed output is stored.

The mapping is simple, but the memory behavior is not ideal. Nearby threads repeatedly fetch overlapping input regions, and every thread walks through the same filter values.

### Boundary Divergence

The inner bounds check implements zero padding. Near an image edge, some threads in a warp use a valid input while others skip an out-of-bounds position. Their control paths diverge for that loop iteration.

For a large image and a relatively small filter, boundary threads make up a small fraction of the grid, so this divergence is usually modest. Repeated global-memory traffic is the more important performance concern.

---

## 3. Memory Bandwidth Considerations

Arithmetic intensity measures how much useful computation an algorithm performs per byte moved from global memory:

$$
\text{Arithmetic intensity}
= \frac{\text{floating-point operations}}
        {\text{bytes transferred from global memory}}
$$

A kernel with low arithmetic intensity is more likely to be memory-bound. Increasing the filter size adds computation, but whether the kernel benefits depends on how often it reloads the same data.

### Ideal Arithmetic Intensity

Assume an $n \times n$ input, an $m \times m$ filter, 4-byte floating-point values, and an output with roughly the same dimensions as the input.

Using the convention that one multiply-add counts as two FLOPs, the convolution performs approximately:

$$
2n^2m^2 \text{ FLOPs}
$$

In the ideal case, each input is loaded once and each output is stored once. For a large image, the one-time filter transfer is negligible, so global-memory traffic is approximately:

$$
4n^2 + 4n^2 = 8n^2 \text{ bytes}
$$

This gives an ideal arithmetic intensity of:

$$
\boxed{
\frac{2n^2m^2}{8n^2}
= \frac{m^2}{4}
}
\text{ FLOP/B}
$$

| Filter | Ideal arithmetic intensity |
| --- | ---: |
| $3 \times 3$ | $2.25$ FLOP/B |
| $11 \times 11$ | $30.25$ FLOP/B |

Small filters are more likely to be memory-bound. Larger filters offer enough work per input value to move the bottleneck toward compute throughput, provided the implementation captures the available reuse.

### Traffic in the Basic Kernel

The basic kernel does not approach the ideal. For each filter position, a thread may fetch:

- One 4-byte input value.
- One 4-byte filter value.

That is 8 bytes from global memory for one multiplication and one addition:

$$
\boxed{
\frac{2\text{ FLOPs}}{8\text{ B}}
= 0.25\text{ FLOP/B}
}
$$

This simplified estimate ignores the final output store and assumes neither load is served by a cache. Its purpose is to expose the central problem: a naive implementation repeatedly requests both shared inputs and identical filter values from DRAM.

The first easy target is the small, read-only filter.

---

## 4. Constant Memory and Caching

CUDA constant memory is a good home for data that is small, read-only during kernel execution, and shared across the entire grid. A convolution filter has exactly those properties.

### Why the Filter Fits

Filters are usually much smaller than their inputs. A typical filter might be $7 \times 7$ or smaller, every block uses the same values, and the weights do not change during one kernel launch.

CUDA exposes a 64 KB constant-memory space. It lives in device memory, but a dedicated on-chip cache makes repeated access much cheaper when threads use it well.

Constant memory is visible to every block and cannot be modified by device threads during the kernel. The host initializes it before launch.

### Declaring and Initializing Constant Memory

A constant-memory object is declared at global scope:

```cpp
#define FILTER_RADIUS 2
#define FILTER_DIM (2 * FILTER_RADIUS + 1)

__constant__ float F[FILTER_DIM][FILTER_DIM];
```

Because `F` is a device symbol, the kernel can refer to it directly instead of receiving a filter pointer as an argument.

The host copies the weights with `cudaMemcpyToSymbol`:

```cpp
cudaMemcpyToSymbol(
    F,
    F_h,
    FILTER_DIM * FILTER_DIM * sizeof(float)
);
```

```text
Host filter
     |
     | cudaMemcpyToSymbol
     v
Device constant memory
```

### The Constant Cache

Threads in a warp usually advance through the filter loops together. On one iteration they all request `F[0][0]`, on the next they all request `F[0][1]`, and so on.

```text
Thread 0  --+
Thread 1  --+
Thread 2  --+--> F[fRow][fCol]
...         |
Thread 31 --+
               constant cache
```

The constant cache can broadcast one value to the entire warp when its threads request the same address. This access pattern is a particularly good match for convolution. If threads in the warp request many different constant addresses, the accesses are less efficient.

Normal global-memory loads also benefit from transparent caches. L1 is close to an SM, while L2 is larger and shared across the GPU. Unlike shared memory, these caches do not require explicit copies:

```text
Global load
    |
    +--> cache hit: return cached value
    |
    +--> cache miss: fetch from DRAM
```

Caching may capture some reuse between neighboring threads, but its behavior is less directly controlled than an explicitly managed shared-memory tile.

### Effect on Arithmetic Intensity

If the filter stays in the constant cache, its repeated requests generate little DRAM traffic. The simplified per-filter-position estimate then counts only the 4-byte input load:

$$
\boxed{
\frac{2\text{ FLOPs}}{4\text{ B}}
= 0.5\text{ FLOP/B}
}
$$

Moving the filter from ordinary global-memory traffic to constant memory roughly doubles this estimate from 0.25 to 0.5 FLOP/B. The exact result depends on cache hits, boundary behavior, and output stores, but the direction is clear.

The remaining opportunity is larger: neighboring outputs still reuse many input values. Shared-memory tiling lets us manage that reuse explicitly.

---

## 5. Tiled Convolution with Halo Cells

Neighboring output elements overlap heavily in the input values they use. In a basic convolution kernel, each thread reads those values from global memory for itself. That means the same input value may travel from DRAM many times.

Tiling gives the block a better arrangement. Its threads load a region of the input into shared memory together, synchronize, and then reuse that local copy while calculating several outputs.

```text
Global memory
      |
      | cooperative load
      v
Shared-memory input tile
      |
      | repeated reuse
      v
Output tile
```

The filter can stay in constant memory, so this section focuses on reducing traffic for the input array.

### Why the Input Tile Is Larger

An **output tile** is the group of output elements produced by one block. The corresponding **input tile** contains every input element needed to produce those outputs.

These two regions are not the same size. An output near a tile boundary still needs neighboring input values outside the output region. The extra border is called the **halo**.

```text
              input tile
     +-------------------------+
     |          halo           |
     |    +---------------+    |
     |    |               |    |
     |    |  output tile  |    |
     |    |               |    |
     |    +---------------+    |
     |          halo           |
     +-------------------------+
```

For a filter with radius $r$, the input tile extends by $r$ elements on every side:

$$
\text{input tile width}
= \text{output tile width} + 2r
$$

Since a filter of width $m$ has $m = 2r + 1$, we can also write:

$$
\text{input tile width}
= \text{output tile width} + m - 1
$$

Suppose the filter is $5 \times 5$, so $r=2$.

- A $4 \times 4$ output tile needs an $8 \times 8$ input tile. That is 64 input values for only 16 outputs.
- A $16 \times 16$ output tile needs a $20 \times 20$ input tile. That is 400 input values for 256 outputs.

The second tile loads more data overall, but its halo is a smaller fraction of the useful region. This is one reason larger tiles can provide better reuse.

Halo cells and ghost cells sound similar, but they describe different situations:

- A **halo cell** lies outside a block's output tile but may still be inside the image.
- A **ghost cell** lies outside the image itself and contributes zero when zero padding is used.

### Choosing the Thread Block

The mismatched tile sizes create a choice about how many threads to launch.

| Block matches | Loading behavior | Computation behavior |
| --- | --- | --- |
| Input tile | Each thread loads one input value. | Threads in the halo do not produce outputs. |
| Output tile | Every thread produces one output. | Some threads must load more than one input value. |

The first design is easier to follow, so the initial tiled kernel uses a block with the same dimensions as the input tile:

```cpp
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)

__shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
```

For a $5 \times 5$ filter, `FILTER_RADIUS` is 2 and a $32 \times 32$ input tile produces a $28 \times 28$ output tile.

### The Two Kernel Phases

The kernel first fills shared memory and then performs the convolution.

#### Phase 1: Load the Input Tile

Each thread maps to one input position. If that position is part of the image, the thread copies it into shared memory. If it is a ghost cell, the thread stores zero instead.

```cpp
if (inRow >= 0 && inRow < height &&
    inCol >= 0 && inCol < width) {
    N_s[threadIdx.y][threadIdx.x] = N[inRow * width + inCol];
} else {
    N_s[threadIdx.y][threadIdx.x] = 0.0f;
}

__syncthreads();
```

The barrier is essential. Without it, a thread could begin the convolution while another thread is still filling a shared-memory value it needs.

#### Phase 2: Produce the Output Tile

Only the inner threads have corresponding output elements. Those threads walk over the filter and read the matching input window from shared memory:

```cpp
float value = 0.0f;

for (int fRow = 0; fRow < FILTER_DIM; ++fRow) {
    for (int fCol = 0; fCol < FILTER_DIM; ++fCol) {
        value += F[fRow][fCol]
               * N_s[tileRow + fRow][tileCol + fCol];
    }
}
```

With an $8 \times 8$ input tile and a $3 \times 3$ filter, the block launches 64 threads but produces a $6 \times 6$ output tile:

```text
H H H H H H H H
H O O O O O O H
H O O O O O O H
H O O O O O O H
H O O O O O O H
H O O O O O O H
H O O O O O O H
H H H H H H H H

H = helps load the halo, then becomes inactive
O = loads an input and computes an output
```

The halo threads are useful during loading, but they still consume block resources after that phase.

### Mapping Halo Threads to Outputs

Because the active output region begins inside the halo, the local output coordinates are shifted by the filter radius:

```cpp
int tileCol = threadIdx.x - FILTER_RADIUS;
int tileRow = threadIdx.y - FILTER_RADIUS;
```

For $r=1$, thread `(1, 1)` produces local output `(0, 0)`, and thread `(6, 6)` produces local output `(5, 5)` in the $8 \times 8$ example.

The block index advances in units of `OUT_TILE_DIM`, not `IN_TILE_DIM`, because neighboring blocks must produce adjacent output tiles:

```cpp
int outCol = blockIdx.x * OUT_TILE_DIM + tileCol;
int outRow = blockIdx.y * OUT_TILE_DIM + tileRow;
```

The full workflow is now:

1. Every thread loads one input or a zero into shared memory.
2. The block synchronizes.
3. Inner threads calculate outputs from the shared tile.
4. Valid outputs are written to global memory.

The important win is that input values are fetched from global memory once per tile and then reused from shared memory by several threads.

### Arithmetic Intensity of Tiled Convolution

Tiling is useful because it changes how much work the kernel gets from each byte fetched from DRAM. Arithmetic intensity lets us estimate that improvement.

Let:

- $t$ be the output tile width.
- $m$ be the filter width.
- $t+m-1$ be the input tile width.

We will count a multiply and an addition as two floating-point operations, and assume each value is a 4-byte `float`.

#### Deriving the Intensity

One output performs $m^2$ multiply-add operations, or $2m^2$ FLOPs. A tile contains $t^2$ outputs, so the block performs:

$$
2t^2m^2 \text{ FLOPs}
$$

The block loads $(t+m-1)^2$ input values and stores $t^2$ output values. Ignoring the filter traffic because the filter is cached in constant memory, the total global-memory traffic is:

$$
4\left[(t+m-1)^2+t^2\right] \text{ bytes}
$$

The tiled kernel therefore has an arithmetic intensity of:

$$
\boxed{
\frac{2t^2m^2}
{4\left[(t+m-1)^2+t^2\right]}
}
\text{ FLOP/B}
$$

#### Why Larger Tiles Help

As $t$ grows, the halo becomes small relative to the output tile. The intensity approaches:

$$
\lim_{t \to \infty}
\frac{2t^2m^2}
{4\left[(t+m-1)^2+t^2\right]}
= \frac{m^2}{4}
\text{ FLOP/B}
$$

For a $5 \times 5$ filter, the ideal value is:

$$
\frac{5^2}{4}=6.25\text{ FLOP/B}
$$

A $28 \times 28$ output tile reaches about $5.42$ FLOP/B, already fairly close to that limit.

This affects the likely bottleneck:

- Small filters and small tiles often remain memory-bound.
- Larger filters perform more computation per input value and may become compute-bound.
- Larger tiles reduce halo overhead and move the kernel closer to its ideal intensity.

#### Limits of Explicit Halo Tiling

A larger tile is not automatically better. Hardware limits and block efficiency eventually get in the way:

- Shared-memory use grows with the area of the input tile.
- The block cannot exceed the device's maximum thread count.
- Halo threads take up execution resources even though they do not compute outputs.
- A wider filter creates a wider halo and leaves fewer productive threads when the input tile size stays fixed.
- A $28 \times 28$ output region inside a $32 \times 32$ block leaves partially active warps during computation.

The explicit-halo design makes data movement predictable, but it pays for that control with inactive threads and a larger shared-memory tile.

---

## 6. Using Caches for Halo Cells

There is another way to handle the halo. A cell that belongs to one block's halo is often an ordinary internal cell for a neighboring block. If a nearby block accessed it recently, the value may already be in the L2 cache.

That suggests a simpler division of labor:

```text
Internal tile  -> shared memory
Halo cells     -> global memory, with possible cache hits
Filter         -> constant memory
```

This design does not depend on a cache hit for correctness. A miss simply means the hardware must fetch the halo value from DRAM.

### A Smaller Shared-Memory Tile

The block, output tile, and shared-memory tile can now use the same dimensions:

```cpp
#define TILE_DIM 32

__shared__ float N_s[TILE_DIM][TILE_DIM];
```

Each thread loads one internal input value and, except for threads beyond the image boundary, produces one output:

```cpp
if (row < height && col < width) {
    N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
} else {
    N_s[threadIdx.y][threadIdx.x] = 0.0f;
}

__syncthreads();
```

### Three Possible Input Paths

While applying the filter, a thread determines where each requested input value belongs:

1. If it falls inside the block's internal tile, read it from shared memory.
2. If it falls outside the tile but remains inside the image, read it from global memory. Ideally, L2 serves the request.
3. If it falls outside the image, use zero for the ghost cell.

```text
Requested input
      |
      +-- inside tile --------> shared memory
      |
      +-- halo inside image --> global memory or cache
      |
      +-- outside image ------> zero
```

The access decision adds control flow inside the filter loops, but the block no longer carries threads that become inactive during computation.

### Comparing the Two Designs

| Explicit halo in shared memory | Cached halo |
| --- | --- |
| Loads the complete input tile into shared memory. | Loads only the internal tile into shared memory. |
| Some threads load halo values but produce no output. | Every thread loads an internal value and produces an output. |
| Halo reuse is explicit and predictable. | Halo performance depends partly on cache behavior. |
| Input and output tiles have different dimensions. | Block, shared tile, and output tile have matching dimensions. |
| Uses more shared memory per output. | Uses less shared memory but may issue extra global loads. |

Neither version wins in every situation. The better choice depends on filter size, tile dimensions, shared-memory pressure, cache effectiveness, and the target GPU. Profiling is the reliable way to decide.

---

## 7. Summary

Convolution calculates each output independently, so assigning one output to each thread is easy. Making that kernel efficient is mostly a question of where its repeatedly used data lives.

The chapter's optimization path is:

```text
Basic kernel
    |  input and filter repeatedly requested from global memory
    v
Constant-memory filter
    |  warp-wide reuse of small, read-only weights
    v
Shared-memory input tile
    |  explicit reuse of overlapping input regions
    v
Cached halo cells
       fewer inactive threads and a smaller shared tile
```

The main lessons are:

- Zero padding turns out-of-bounds ghost cells into zero contributions, at the cost of some boundary checks and divergence.
- Arithmetic intensity reveals why repeated global loads make the basic kernel memory-bound.
- Constant memory suits a small, read-only filter that every warp reads in the same order.
- An output tile needs a wider input tile because outputs near its boundary depend on halo cells.
- Explicit halo tiling makes reuse predictable, but some threads load data without producing outputs.
- Larger tiles reduce proportional halo overhead, though threads, shared-memory capacity, and occupancy limit their size.
- A cache-based halo design lets every thread produce an output, but its performance depends more on cache hits.

The same principles carry over to 3D convolution. There are more indices and nested loops, but the core question is unchanged: how can overlapping inputs be reused without repeatedly fetching them from DRAM?
