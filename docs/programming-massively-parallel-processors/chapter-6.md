> Dear reader: These notes were created with the help of AI, with me cherry-picking the parts of the book I found most relevant. I also reviewed the content to make sure no AI hallucinations slipped through. I hope you find them useful. Happy reading! :)

# Programming Massively Parallel Processors: Chapter 6, Performance Considerations

1. [Coalesced Global Memory Access](#1-coalesced-global-memory-access)
   - [Why Access Patterns Matter](#why-access-patterns-matter)
   - [DRAM Bursts and Cache Lines](#dram-bursts-and-cache-lines)
   - [Coalescing Within a Warp](#coalescing-within-a-warp)
   - [Alignment](#alignment)
   - [Row-Major Matrix Access](#row-major-matrix-access)
   - [Column-Major Access and Corner Turning](#column-major-access-and-corner-turning)
   - [Coalesced Stores](#coalesced-stores)
2. [Hiding Memory Latency](#2-hiding-memory-latency)
   - [Channels, Banks, and Bandwidth](#channels-banks-and-bandwidth)
   - [High Bandwidth Memory](#high-bandwidth-memory)
   - [Why DRAM Needs Many Banks](#why-dram-needs-many-banks)
   - [Bank Conflicts and Interleaving](#bank-conflicts-and-interleaving)
   - [Occupancy and Memory-Level Parallelism](#occupancy-and-memory-level-parallelism)
3. [Vector Loads and Stores](#3-vector-loads-and-stores)
   - [Using `float4`](#using-float4)
   - [When Vector Access Helps](#when-vector-access-helps)
4. [Shared-Memory Bank Conflicts](#4-shared-memory-bank-conflicts)
   - [Conflict-Free Access](#conflict-free-access)
   - [How Conflicts Serialize Access](#how-conflicts-serialize-access)
   - [Avoiding Conflicts with Padding](#avoiding-conflicts-with-padding)
5. [Thread Coarsening](#5-thread-coarsening)
   - [How Coarsening Works](#how-coarsening-works)
   - [Where It Helps](#where-it-helps)
   - [Common Pitfalls](#common-pitfalls)
6. [Loop Unrolling](#6-loop-unrolling)
   - [Why Unrolling Helps](#why-unrolling-helps)
   - [Compiler Support](#compiler-support)
   - [Register Promotion](#register-promotion)
   - [Tradeoffs](#tradeoffs)
7. [Double Buffering](#7-double-buffering)
   - [True and False Dependencies](#true-and-false-dependencies)
   - [Alternating Between Two Buffers](#alternating-between-two-buffers)
8. [A Practical Optimization Checklist](#8-a-practical-optimization-checklist)
   - [Compute Utilization](#compute-utilization)
   - [Memory Utilization](#memory-utilization)
   - [Synchronization Latency](#synchronization-latency)
   - [Measure the Bottleneck](#measure-the-bottleneck)

---

## 1. Coalesced Global Memory Access

CUDA performance is usually limited by one architectural resource at a time. We call that resource the **bottleneck**. An optimization only helps when it reduces pressure on the current bottleneck, and it can hurt when it simply moves the problem somewhere else.

Global-memory bandwidth is a common bottleneck because CUDA kernels often move a great deal of data in a short time. Chapter 5 introduced **tiling**, which reduces global-memory traffic by reusing data in shared memory. This chapter adds **memory coalescing**, which makes the remaining traffic more efficient.

### Why Access Patterns Matter

GPU global memory is built from **DRAM**. Each DRAM bit is represented by a tiny electrical charge stored in a capacitor. Reading that charge through long, highly capacitive wires is much slower than performing an arithmetic instruction inside an SM.

Keeping DRAM cells small gives us a lot of storage, but it also means that individual accesses remain relatively slow. The memory system compensates by transferring groups of nearby values and by serving many requests in parallel. Our kernels perform best when their access patterns cooperate with both techniques.

### DRAM Bursts and Cache Lines

When DRAM reads one address, it also reads a group of nearby consecutive addresses. This group is called a **DRAM burst**.

```text
Efficient:   A[0] A[1] A[2] A[3] ...
Inefficient: A[0] A[50] A[117] A[300] ...
```

The first pattern uses most of the data fetched by each burst. The second may need several bursts and then discard most of what they contain.

Modern GPUs also have SRAM-based **L1 and L2 caches**, so not every global-memory load reaches DRAM. Still, the same principle applies because caches transfer data in fixed-size **cache lines**. A kernel gets more value from the memory system when it uses several values from each line it fetches.

### Coalescing Within a Warp

When a warp executes a memory instruction, the GPU examines all addresses requested by its threads. If those addresses fall into a small number of adjacent memory segments, the hardware combines the requests into fewer **memory transactions**.

```text
T0 -> X
T1 -> X + 1
T2 -> X + 2
T3 -> X + 3
```

This is an ideal pattern. Consecutive threads request consecutive values at the same time, so the hardware can move the data efficiently.

A useful way to think about coalescing is carpooling. Each requested value is a passenger, each memory transaction is a car, and memory bandwidth is the road capacity. Filling a few cars is more efficient than sending one almost-empty car per passenger. The requests also need compatible schedules, which is why addresses issued together by one warp matter more than the sequence of addresses visited by one thread over time.

> **Key idea:** Coalescing is determined across the threads of a warp for one memory instruction, not across successive loop iterations of one thread.

### Alignment

Consecutive accesses can still require extra transactions when the first address is not properly aligned.

```text
Aligned:    | requested data        |
Unaligned:       | requested data        |
```

An aligned request touches the minimum number of memory segments or cache lines. An unaligned request may cross a boundary and force the hardware to fetch an additional segment.

CUDA allocations are suitably aligned, but offsets into an allocation can change the alignment of a particular access. This is especially important for vector types such as `float4`, which have stricter alignment requirements than scalar values.

### Row-Major Matrix Access

C and CUDA store multidimensional arrays in **row-major order**, so elements in the same row are consecutive in memory:

```text
N[0][0], N[0][1], N[0][2], N[0][3],
N[1][0], N[1][1], N[1][2], N[1][3]
```

For a matrix with `Width` columns, the linear index is:

$$
\text{index} = \text{row} \times \text{Width} + \text{column}
$$

Vertically adjacent elements such as $N_{0,0}$ and $N_{1,0}$ are separated by one full row in memory.

Consider the access to the second input matrix in a matrix-multiplication kernel:

```cpp
N[k * Width + col]
```

During one iteration, `k` is the same for every thread in a warp while `col` changes across consecutive threads:

```text
Iteration k = 0:
T0 -> N[0][0]
T1 -> N[0][1]
T2 -> N[0][2]
T3 -> N[0][3]

Iteration k = 1:
T0 -> N[1][0]
T1 -> N[1][1]
T2 -> N[1][2]
T3 -> N[1][3]
```

One thread moves down a column as the loop progresses, but the warp accesses one consecutive row during each iteration. The accesses are therefore coalesced.

### Column-Major Access and Corner Turning

Now suppose $N$ is stored in **column-major order**:

```text
N[0][0], N[1][0], N[2][0], N[3][0],
N[0][1], N[1][1], N[2][1], N[3][1]
```

The logical element `N[k][col]` now has the linear index:

$$
\text{index} = \text{col} \times \text{Height} + k
$$

Consecutive threads have consecutive `col` values, but those values are multiplied by the height of the matrix:

```text
T0 -> N[0 * Height + k]
T1 -> N[1 * Height + k]
T2 -> N[2 * Height + k]
T3 -> N[3 * Height + k]
```

The requested addresses are far apart, so the warp needs many memory transactions. The same pattern appears when a row-major matrix is accessed as if it were transposed.

There are three common ways to fix an access pattern like this:

1. Change how threads are mapped to data.
2. Change the data layout.
3. Load a tile into shared memory with coalesced accesses, then read it in the order needed by the computation.

The third approach is called **corner turning**. Consecutive threads load consecutive values from global memory, but place them into shared memory with the opposite orientation. Once the tile is on chip, threads can traverse it in the order required by the algorithm.

Conceptually, corner turning exchanges the roles of `threadIdx.x` and `threadIdx.y` in either the global-memory index or the shared-memory index:

```text
Global memory:  load consecutive addresses
                         |
                         v
Shared memory: store or read the tile in a transposed orientation
```

This works because global memory benefits from coalescing, while shared memory is organized around banks instead. The next challenge is making sure that the new shared-memory pattern does not create bank conflicts.

### Coalesced Stores

Coalescing matters for stores as well as loads. Scattered stores require more memory transactions, and partial updates can create additional traffic. The cost can become more noticeable when ECC protection is enabled.

The goal is the same in both directions:

$$
\text{many per-thread accesses} \longrightarrow \text{few memory transactions}
$$

For accesses to combine well, they should be issued together by a warp and should target nearby addresses.

**Summary:** Make neighboring threads access neighboring addresses. If the algorithm needs a different logical order, use thread remapping, a better data layout, or shared memory to separate the global-memory access pattern from the computation pattern.

---

## 2. Hiding Memory Latency

A DRAM burst transfers several values at once, but burst transfers alone cannot provide enough bandwidth for a modern GPU. The memory system also uses **channels** and **banks** to process many requests concurrently.

### Channels, Banks, and Bandwidth

A **channel** connects the processor to memory through a bus. Each channel contains several independent DRAM banks.

- Multiple channels can transfer data independently.
- Multiple banks let one channel prepare several accesses concurrently.
- Enough outstanding requests are needed to keep all of them busy.

The theoretical bandwidth of one memory bus is:

$$
\text{Bandwidth}
= \text{bytes per transfer}
\times \text{transfers per clock}
\times \text{clock frequency}
$$

For a 64-bit DDR5-6400 bus:

$$
8\ \text{B} \times 2 \times 3.2\ \text{GHz}
= 51.2\ \text{GB/s}
$$

DDR performs two transfers per clock cycle. Since modern GPUs need far more bandwidth than one channel can provide, they use several channels in parallel.

### High Bandwidth Memory

**High Bandwidth Memory (HBM)** places stacks of DRAM close to the processor. The short connections use less power and make very wide memory interfaces practical. Instead of relying only on a very high clock rate, HBM moves a large amount of data on each transfer.

This is how high-end GPUs reach aggregate bandwidth in the terabytes-per-second range.

### Why DRAM Needs Many Banks

A DRAM request has two broad phases:

1. The bank decodes the address and senses the stored charge.
2. The requested data travels across the bus in a much shorter burst.

With only one bank, the bus sits idle during most of the first phase. Multiple banks let the memory controller overlap that waiting time:

```text
Bank 0: prepare -------- transfer
Bank 1:       prepare -------- transfer
Bank 2:             prepare -------- transfer
```

If the ratio of preparation time to burst time is $R$, the maximum bus utilization with one bank is approximately:

$$
\text{Utilization} = \frac{1}{R + 1}
$$

For a ratio of $20:1$:

$$
\frac{1}{21} \approx 4.8\%
$$

A $16\ \text{GB/s}$ channel would deliver only about:

$$
16 \times \frac{1}{21} \approx 0.76\ \text{GB/s}
$$

Keeping the bus fully occupied would require roughly:

$$
\text{banks required} \geq R + 1
$$

This simplified model explains why DRAM systems contain many banks and why a GPU needs many independent requests in flight.

### Bank Conflicts and Interleaving

A **DRAM bank conflict** occurs when concurrent requests need the same bank. That bank can only handle a limited number of operations at once, so the requests wait instead of overlapping.

To spread traffic around, consecutive chunks of the address space are interleaved across channels and banks. For a simplified system with two-element bursts:

```text
M[0],  M[1]  -> Channel 0, Bank 0
M[2],  M[3]  -> Channel 1, Bank 0
M[4],  M[5]  -> Channel 2, Bank 0
M[6],  M[7]  -> Channel 3, Bank 0

M[8],  M[9]  -> Channel 0, Bank 1
M[10], M[11] -> Channel 1, Bank 1
...
```

The actual address mapping depends on the GPU, but the purpose is the same: distribute nearby chunks so several memory components can work at once.

### Occupancy and Memory-Level Parallelism

High occupancy helps hide both instruction latency and memory latency. While one warp waits for data, an SM can schedule another ready warp. A large pool of active threads also creates enough outstanding memory requests to keep many channels and banks busy.

Good global-memory performance therefore needs requests that are:

- **Coalesced**, so each warp uses as few transactions as possible.
- **Distributed**, so the transactions reach several channels and banks.
- **Numerous**, so the hardware has enough independent work to overlap.

These requirements depend on one another. Many active threads are useful only if their requests can use the memory system in parallel. Conversely, a well-banked memory system reaches its potential only when the kernel supplies enough outstanding requests.

Caches can reduce pressure further. For example, blocks in the same block row of tiled matrix multiplication may request the same tile from one input matrix. If those requests occur close together, cached data can satisfy later requests without another trip to DRAM.

**Summary:** The GPU hides long DRAM latency by overlapping requests across channels, banks, and warps. Coalescing reduces the number of transactions, while occupancy and memory-level parallelism keep the remaining transactions moving.

---

## 3. Vector Loads and Stores

A conventional vector-add kernel gives each thread one `float` to process:

- Two 4-byte loads, one from each input.
- One 4-byte store to the output.

The warp may access memory perfectly coalesced, but each thread still executes one memory instruction for every scalar value.

### Using `float4`

A vector type lets one instruction access several consecutive values. A `float4` contains four floats:

$$
4 \times 4\ \text{B} = 16\ \text{B}
$$

```cpp
float4 x4 = reinterpret_cast<const float4*>(x)[i];
float4 y4 = reinterpret_cast<const float4*>(y)[i];

float4 z4;
z4.x = x4.x + y4.x;
z4.y = x4.y + y4.y;
z4.z = x4.z + y4.z;
z4.w = x4.w + y4.w;

reinterpret_cast<float4*>(z)[i] = z4;
```

Each thread now processes four elements, so the kernel needs roughly $N/4$ threads instead of $N$.

### When Vector Access Helps

For eight input floats, two vector loads can replace eight scalar loads:

```text
Scalar version: 8 load instructions for 8 floats
Vector version: 2 load instructions for 8 floats
```

Vector loads and stores can:

- Reduce memory-instruction overhead.
- Move more useful data per instruction.
- Generate substantial memory traffic even when occupancy is limited.
- Make thread coarsening easier to express.

The addresses must satisfy the alignment requirement of the vector type. You also need scalar boundary handling when the element count is not divisible by the vector width. A vector access must never read or write past the valid allocation.

Vector instructions do not repair a poor layout. Threads still need a coalesced access pattern across the warp.

**Summary:** Vector types reduce the instruction cost of moving contiguous data, but they work best on aligned data that is already accessed coalescently.

---

## 4. Shared-Memory Bank Conflicts

Shared memory uses **SRAM**, so its latency is much lower than global DRAM. A typical organization has 32 banks, with consecutive 32-bit words assigned to consecutive banks:

```text
word 0  -> bank 0
word 1  -> bank 1
word 2  -> bank 2
...
word 31 -> bank 31
word 32 -> bank 0
```

For 4-byte values, a useful simplified mapping is:

$$
\text{bank} = \text{word index} \bmod 32
$$

### Conflict-Free Access

When the 32 threads in a warp access 32 consecutive values, every thread reaches a different bank:

```text
T0  -> bank 0
T1  -> bank 1
T2  -> bank 2
...
T31 -> bank 31
```

The banks can serve these accesses in parallel. Shared memory is fast here because the address pattern uses the full banked architecture.

### How Conflicts Serialize Access

A **shared-memory bank conflict** occurs when several threads in a warp access different addresses in the same bank. The bank cannot serve all those addresses simultaneously, so the hardware splits the request into multiple operations.

For a $k$-way conflict:

$$
1\ \text{warp request} \longrightarrow k\ \text{serialized bank operations}
$$

This is separate from a DRAM bank conflict. Both involve serialization, but one happens in fast on-chip shared memory and the other in off-chip global memory.

Reading the same shared-memory address from several threads is a special case because the value can be broadcast. A conflict specifically involves different addresses that map to the same bank.

### Avoiding Conflicts with Padding

Corner turning can create a classic conflict. Consider a $32 \times 32$ shared-memory tile:

```cpp
__shared__ float tile[TILE_DIM][TILE_DIM];  // TILE_DIM = 32

tile[threadIdx.x][threadIdx.y] = value;
```

If threads in a warp have consecutive `threadIdx.x` values and the same `threadIdx.y`, they write down one column:

```text
T0 -> tile[0][0]
T1 -> tile[1][0]
T2 -> tile[2][0]
...
```

Since C uses row-major order, their linear indices are:

$$
0,\ 32,\ 64,\ 96,\ldots
$$

Every index is a multiple of 32, so every thread reaches bank 0. The result is a 32-way bank conflict.

Adding one unused column changes the stride:

```cpp
__shared__ float tile[TILE_DIM][TILE_DIM + 1];
```

Now the row length is 33 and the linear indices are:

$$
0,\ 33,\ 66,\ 99,\ldots
$$

Those values map to banks $0, 1, 2, 3, \ldots$, so the accesses can proceed in parallel.

Padding is not free. It uses extra shared memory and may make address calculations slightly more expensive. More shared memory per block can also reduce occupancy. Measure the whole kernel rather than assuming padding always wins.

**Summary:** Shared memory only delivers its best throughput when a warp spreads its accesses across banks. Padding a tile by one column is a simple way to break an unfortunate power-of-two stride.

---

## 5. Thread Coarsening

Fine-grained kernels normally assign one small unit of work to each thread:

```cpp
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
foo(i);
```

This gives us **transparent scalability**. A large GPU runs many threads concurrently, while a smaller GPU handles the same grid in more waves. Fine-grained work can still have overhead, including repeated data loads, redundant calculations, synchronization, indexing, and instruction management.

### How Coarsening Works

**Thread coarsening** assigns several work units to each thread. The **coarsening factor** tells us how many units one thread processes.

With a factor of 4:

```cpp
unsigned int iStart =
    4 * (blockIdx.x * blockDim.x + threadIdx.x);

for (unsigned int c = 0; c < 4; ++c) {
    unsigned int i = iStart + c;
    foo(i);
}
```

```text
Thread 0 -> i = 0, 1, 2, 3
Thread 1 -> i = 4, 5, 6, 7
Thread 2 -> i = 8, 9, 10, 11
```

The small loop inside each thread is the **coarsening loop**.

### Where It Helps

Coarsening is useful when it removes real parallelization overhead:

- In vector addition, a thread can process four elements with `float4`, reducing memory instructions and indexing work.
- In tiled matrix multiplication, one block can compute a larger output tile, allowing each loaded input tile to contribute to more results.
- In other kernels, a thread may reuse values in registers instead of asking several independent threads to load them again.

The central question is simple: does combining work let one thread reuse data or instructions that separate threads would duplicate?

### Common Pitfalls

#### Coarsening Work That Is Already Cheap

If threads are independent and have little overhead, coarsening may save almost nothing. It can make the kernel more complicated without improving its limiting resource.

#### Removing Too Much Parallelism

A large coarsening factor can leave too few blocks to fill the GPU. Suppose a GPU can keep 264 blocks resident across all its SMs:

$$
\frac{1056\ \text{blocks}}{264\ \text{resident blocks}} = 4\ \text{full waves}
$$

Coarsening by 8 reduces the grid to:

$$
\frac{1056}{8} = 132\ \text{blocks}
$$

That is only half a wave for this example configuration, so a large part of the GPU remains idle.

Partial final waves can also cause a **tail effect**. If coarsening produces 396 blocks:

$$
\frac{396}{264} = 1.5\ \text{waves}
$$

The last half wave underutilizes the machine even though the first wave was full.

#### Increasing Resource Usage

More work per thread usually needs more registers, and larger tiles may need more shared memory per block. Either can reduce occupancy enough to cancel the savings from coarsening.

> **Key idea:** The best coarsening factor depends on the kernel, input size, and GPU. Treat it as a parameter to measure, not a universal constant.

---

## 6. Loop Unrolling

**Loop unrolling** copies the loop body several times so the kernel executes fewer loop iterations.

Original loop:

```cpp
for (unsigned int i = 0; i < 16; ++i) {
    A(i);
    B(i);
}
```

Unrolled by a factor of 4:

```cpp
for (unsigned int i = 0; i < 16; i += 4) {
    A(i);
    B(i);

    A(i + 1);
    B(i + 1);

    A(i + 2);
    B(i + 2);

    A(i + 3);
    B(i + 3);
}
```

The loop now executes four times instead of sixteen.

### Why Unrolling Helps

Every iteration needs instructions to update the counter, test the condition, and branch back to the start. Unrolling reduces how often those instructions execute.

It also exposes more independent work to the compiler. If `B(i)` depends on `A(i)`, the thread might otherwise wait between them:

```text
A(i) -> wait -> B(i)
```

After unrolling, the compiler may schedule independent operations from other iterations during that wait:

```cpp
A(i);
A(i + 1);
A(i + 2);
A(i + 3);

B(i);
B(i + 1);
B(i + 2);
B(i + 3);
```

This extra **instruction-level parallelism** helps hide pipeline latency, especially when the SM has relatively few active warps.

### Compiler Support

The compiler often unrolls small loops whose bounds are known at compile time. CUDA also supports an unroll directive:

```cpp
#pragma unroll 4
for (unsigned int i = 0; i < 16; ++i) {
    A(i);
    B(i);
}
```

To ask the compiler not to unroll a loop:

```cpp
#pragma unroll 1
```

The directive is a request with predictable code-generation intent, but you should still inspect performance rather than assuming the requested factor is best.

### Register Promotion

Unrolling can help the compiler keep a small local array in registers. Consider:

```cpp
int x[4];

for (unsigned int c = 0; c < 4; ++c) {
    foo(x[c]);
}
```

A variable index can make register allocation difficult because registers are not indexed like an array. With full unrolling, each index becomes a compile-time constant:

```cpp
foo(x[0]);
foo(x[1]);
foo(x[2]);
foo(x[3]);
```

The compiler may then represent the elements as separate registers and avoid local-memory traffic.

### Tradeoffs

Unrolling can increase code size and register use. Too much unrolling may reduce occupancy or put pressure on the instruction cache. It is especially useful for small, fixed-size loops and for coarsening loops where the extra work is independent.

**Summary:** Loop unrolling trades a larger loop body for fewer branches, more scheduling freedom, and sometimes better register allocation.

---

## 7. Double Buffering

Suppose threads repeatedly communicate through one shared-memory buffer:

```cpp
for (...) {
    value = buffer[anotherThreadID];

    __syncthreads();

    buffer[myThreadID] = newValue;

    __syncthreads();
}
```

Two barriers are needed because each iteration reads and overwrites the same storage.

### True and False Dependencies

The second barrier ensures that all writes finish before the next iteration reads the new values. This is a **read-after-write dependency**, also called a **true dependency**. The algorithm genuinely needs the new data before it can continue.

The first barrier prevents a fast thread from overwriting an old value before another thread has read it. This is a **write-after-read dependency**. It is often called a **false dependency** because the ordering exists only because the old and new values share the same locations.

### Alternating Between Two Buffers

**Double buffering** removes that false dependency by keeping old and new values in different buffers:

```cpp
for (...) {
    value = inBuffer[anotherThreadID];
    outBuffer[myThreadID] = newValue;

    __syncthreads();
    swap(inBuffer, outBuffer);
}
```

```text
Iteration 0: read Buffer A, write Buffer B
             swap
Iteration 1: read Buffer B, write Buffer A
             swap
Iteration 2: read Buffer A, write Buffer B
```

No thread can overwrite the data another thread is still reading because reads and writes target different storage. Only one barrier per iteration is needed to ensure that the newly written buffer is ready before it becomes the input.

The cost is additional memory. Two shared-memory buffers may also reduce occupancy, so the saved synchronization must be worth the larger allocation.

**Summary:** Double buffering trades memory capacity for fewer synchronization barriers by alternating between an input buffer and an output buffer.

---

## 8. A Practical Optimization Checklist

The optimizations in this chapter fall into three broad goals:

| Goal | Optimization | What it improves |
| --- | --- | --- |
| **Compute utilization** | Occupancy tuning | Gives the SM more warps for hiding pipeline and memory latency |
| | Loop unrolling | Reduces branch overhead and exposes independent instructions |
| | Reduce control divergence | Keeps more lanes in each warp doing useful work |
| **Memory utilization** | Coalesced global access | Reduces memory transactions |
| | Shared-memory tiling | Reduces repeated global-memory traffic |
| | Register tiling | Reduces shared-memory traffic and keeps reused values close to the thread |
| | Vector loads and stores | Moves more data per instruction |
| | Avoid bank conflicts | Preserves parallel shared-memory access |
| | Privatization | Reduces contention on shared or global updates |
| **Synchronization latency** | Warp-level primitives | Replaces some block-wide coordination with cheaper warp-level operations |
| | Double buffering | Removes barriers caused by false dependencies |
| **General** | Thread coarsening | Reduces duplicated work and parallelization overhead |

### Compute Utilization

Occupancy tuning balances three major resources:

```text
threads per block
registers per thread
shared memory per block
          |
          v
resident blocks and warps per SM
```

More active warps give the scheduler more work to choose from, but maximum occupancy is not automatically the fastest configuration. A kernel may run better with lower occupancy if each thread gets enough registers or each block gets enough shared memory to do substantially less work.

Control divergence wastes compute capacity in another way. When threads in one warp follow different branches, some lanes sit idle. You can sometimes reduce divergence by remapping threads, grouping similar work together, or changing the data layout so neighboring threads encounter similar conditions.

### Memory Utilization

Memory optimization has two recurring themes: move data less often, and make each transfer carry as much useful data as possible.

- Map consecutive threads to consecutive global-memory addresses.
- Load reusable data once into shared memory.
- Keep per-thread reuse in registers when practical.
- Use aligned vector accesses for contiguous chunks.
- Change shared-memory layout or add padding to prevent bank conflicts.
- Pack scattered results in shared memory before issuing coalesced stores.

Changing the data layout can sometimes solve the problem at its source. Common examples include matrix transposition, converting an Array of Structures into a Structure of Arrays, and choosing a sparse-matrix format that matches the access pattern.

### Synchronization Latency

Every barrier can leave early-arriving threads idle. Before trying to make a barrier faster, ask whether all those threads need to synchronize at all.

- Use warp-level primitives when communication stays within a warp.
- Privatize partial results, then combine them less frequently.
- Use double buffering when a barrier exists only to protect old data from being overwritten.
- Balance larger cooperative tiles against the extra synchronization and shared memory they require.

### Measure the Bottleneck

Optimization starts with evidence. A useful workflow is:

1. Decide whether the kernel is limited by computation, memory, synchronization, or launch and instruction overhead.
2. Choose an optimization that directly targets that limit.
3. Check its effect on registers, shared memory, occupancy, and the number of active blocks.
4. Measure the complete kernel again on the target GPU and representative input sizes.

Many techniques trade one resource for another. Tiling uses shared memory to reduce DRAM traffic. Unrolling uses code size and registers to reduce branches and expose parallelism. Coarsening reduces thread-level parallelism to gain reuse. Double buffering uses more memory to remove synchronization.

> **Big picture:** CUDA optimization is the practice of finding what makes the GPU wait, then spending the right resource to reduce that wait.
