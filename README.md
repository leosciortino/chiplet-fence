# Chiplet-Level Release/Acquire Fences for AMD MI300X (CDNA3)

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [MI300X Memory Hierarchy](#mi300x-memory-hierarchy)
3. [Chiplet Fence Mechanism](#chiplet-fence-mechanism)
4. [Runtime XCC_ID Detection](#runtime-xcc_id-detection)
5. [Benchmark Suite](#benchmark-suite)
6. [Correctness Argument](#correctness-argument)
7. [Building and Running](#building-and-running)

---

## Problem Statement

The AMD Instinct MI300X is a multi-chiplet GPU composed of **8 XCDs** (Accelerator Complex Dies), each containing its own compute units (CUs) and a private L2 cache. When workgroups on different XCDs need to communicate, the hardware must coordinate across chiplets via the **Infinity Fabric** interconnect. This cross-chiplet coordination is the bottleneck that this project targets.

The standard HIP synchronization primitive `__threadfence()` provides **device-scope** ordering. On MI300X, a device-scope fence must:

1. Drain all pending stores from the issuing CU's L1 caches down to L2.
2. Flush dirty L2 lines to **HBM** (device memory).
3. Send **Infinity Fabric probe** messages to invalidate or update cache lines on all other 7 XCDs.
4. Wait for acknowledgments from all remote chiplets.

This is expensive. For workloads where cooperating workgroups happen to reside on the **same XCD**, the full device-scope fence is overkill. All CUs on the same XCD share a single L2 cache, so data only needs to reach L2 (not HBM) for inter-CU visibility within that chiplet.

This project implements **chiplet-scope release/acquire fences** that exploit this observation: when communicating within a single XCD, we can skip the L2-to-HBM flush and the Infinity Fabric probes entirely, reducing fence latency significantly.

---

## MI300X Memory Hierarchy

The MI300X memory hierarchy has four levels relevant to coherence:

```
                            +-----------+
                            |    HBM    |  <-- Point of coherency ACROSS chiplets
                            +-----------+
                            /    |    \
                     +------+ +------+ +------+
                     | L2_0 | | L2_1 | | L2_7 |  <-- One L2 per XCD
                     +------+ +------+ +------+      Point of coherency WITHIN chiplet
                      / | \    / | \    / | \
                    CU  CU CU CU CU CU CU CU CU     <-- Compute Units
                    |   |  |  |   |  |  |   |  |
                  [V_L1][V_L1] ...                    <-- Vector L1 (per-CU)
                  [S_L1][S_L1] ...                    <-- Scalar L1 (per-CU)
```

### Cache Properties

| Cache | Scope | Write Policy | Coherency Notes |
|-------|-------|-------------|-----------------|
| **Vector L1 (V_L1)** | Per-CU | **Write-through** | Stores pass through to L2 automatically; tracked by `vmcnt` counter. Reads may return stale data if another CU has written to L2 since the line was cached. |
| **Scalar L1 (S_L1)** | Per-CU | **Write-back** | Dirty lines stay in S_L1 until explicitly flushed with `s_dcache_wb`. Without an explicit write-back, stores to scalar addresses may never reach L2. Tracked by `lgkmcnt` counter. |
| **L2** | Per-XCD (shared by all CUs on that XCD) | Write-back to HBM | Point of coherency within a chiplet. All CUs on the same XCD see a consistent view through L2. Data in one XCD's L2 is invisible to CUs on other XCDs. |
| **HBM** | Device-wide | N/A | High Bandwidth Memory. Point of coherency across chiplets. Data must reach HBM (or be probed via Infinity Fabric) for cross-XCD visibility. |

### Key Observations

- **Vector L1 is write-through**: a `global_store` instruction issues a vector store that drains to L2 automatically. The `s_waitcnt vmcnt(0)` instruction waits for all such stores to complete (i.e., reach L2).
- **Scalar L1 is write-back**: the compiler may use scalar stores for uniform data. These dirty lines sit in S_L1 and must be explicitly pushed to L2 with `s_dcache_wb`.
- **L2 is the point of coherency within a chiplet**: once data is in L2, any CU on the same XCD can read it (after invalidating its own stale L1).
- **HBM is the point of coherency across chiplets**: for cross-XCD communication, data must reach HBM or be transferred via Infinity Fabric.

---

## Chiplet Fence Mechanism

### Release Fence (Producer Side)

The chiplet release fence ensures all prior stores from the issuing CU are visible in the shared L2 cache:

```asm
s_waitcnt vmcnt(0)        ; (1) Drain vector stores to L2
s_dcache_wb               ; (2) Write-back dirty scalar L1 lines to L2
s_waitcnt lgkmcnt(0)      ; (3) Wait for the write-back to complete
```

**Why each instruction is needed:**

1. **`s_waitcnt vmcnt(0)`** — The vector memory counter (`vmcnt`) tracks outstanding vector store instructions. Vector L1 is write-through, meaning stores are forwarded to L2, but the store may still be in flight. This wait ensures every vector store has been acknowledged by L2.

2. **`s_dcache_wb`** — The scalar L1 data cache is write-back. If the compiler generated scalar stores, dirty cache lines may sit in S_L1 indefinitely. This instruction initiates a write-back of all dirty lines from S_L1 to L2.

3. **`s_waitcnt lgkmcnt(0)`** — The LDS/GDS/scalar-memory counter (`lgkmcnt`) tracks scalar memory operations, including the `s_dcache_wb` initiated above. This wait ensures the write-back has completed before proceeding.

**What this does NOT do** (and why it is cheaper than `__threadfence()`):
- Does NOT flush L2 to HBM.
- Does NOT send Infinity Fabric probes to other XCDs.
- Does NOT invalidate remote L2 caches.

### Acquire Fence (Consumer Side)

The chiplet acquire fence invalidates the issuing CU's L1 caches so that subsequent reads fetch fresh data from the shared L2:

```asm
buffer_inv sc1            ; (1) Invalidate vector L1 (device scope, sc1=1)
s_dcache_inv              ; (2) Invalidate scalar L1 data / constant cache
s_waitcnt vmcnt(0) lgkmcnt(0)  ; (3) Wait for both invalidations to complete
```

**Why each instruction is needed:**

1. **`buffer_inv sc1`** — Invalidates the vector L1 cache at device scope. After this, any subsequent `global_load` will miss in V_L1 and go to L2, where it will see the producer's data. The `sc1` qualifier specifies the scope; this invalidation is tracked by the `vmcnt` counter.

2. **`s_dcache_inv`** — Invalidates the scalar L1 data cache. Read-side complement of `s_dcache_wb`: the producer wrote back dirty scalar lines to L2, and the consumer must discard its own potentially stale scalar L1 lines so it reads from L2 instead.

3. **`s_waitcnt vmcnt(0) lgkmcnt(0)`** — Waits for both invalidation operations to complete.

**What this does NOT do:**
- Does NOT invalidate L2 (L2 is the coherency point within the XCD — it always has fresh data from all local CUs).
- Does NOT trigger Infinity Fabric traffic.

### Implementation

```cpp
namespace chiplet {

__device__ __forceinline__ void release_fence() {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    asm volatile("s_dcache_wb" ::: "memory");
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}

__device__ __forceinline__ void acquire_fence() {
    asm volatile("buffer_inv sc1" ::: "memory");
    asm volatile("s_dcache_inv" ::: "memory");
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
}

} // namespace chiplet
```

The `"memory"` clobber on each `asm volatile` statement prevents the compiler from reordering memory operations across these instructions and prevents the optimizer from eliding them.

---

## Runtime XCC_ID Detection

To determine which XCD a workgroup is executing on, the implementation reads a hardware status register using `s_getreg_b32`:

```cpp
__device__ __forceinline__ uint32_t get_xcc_id() {
    uint32_t xcc_id;
    asm volatile("s_getreg_b32 %0, hwreg(20, 0, 4)" : "=s"(xcc_id));
    return xcc_id;
}
```

**Decoding `hwreg(20, 0, 4)`:**

| Field | Value | Meaning |
|-------|-------|---------|
| Register ID | 20 | Hardware ID register on gfx942 |
| Bit offset | 0 | XCC_ID field starts at bit 0 |
| Width | 4 | 4 bits, supporting XCC_ID values 0-15 (MI300X uses 0-7) |

This instruction reads directly from a hardware status register — it does not go to memory and has no cache coherence implications. It executes in scalar ALU and returns the physical chiplet ID where the current wavefront is running. The `"=s"` output constraint places the result in an SGPR.

---

## Benchmark Suite

The benchmark suite (`src/bench.hip`) is a single program that exercises three fence implementations against four workloads. Each fence is exposed as a struct-with-static-methods so the benchmarks can be templated on it:

```cpp
struct ChipletFence {
    __device__ static void release() { chiplet::release_fence(); }
    __device__ static void acquire() { chiplet::acquire_fence(); }
};

struct DeviceRelAcqFence {
    __device__ static void release() { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent"); }
    __device__ static void acquire() { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent"); }
};

struct DeviceThreadFence {
    __device__ static void release() { __threadfence(); }
    __device__ static void acquire() { __threadfence(); }
};
```

`DeviceRelAcqFence` is interesting because LLVM's `__builtin_amdgcn_fence` lets us emit *true* release/acquire semantics at agent scope — distinct from `__threadfence()` which is sequentially consistent. Comparing these two device-scope variants isolates the cost of seq-cst vs. release/acquire.

### Synchronization primitives

The benchmarks use custom relaxed primitives (`RelaxedSpinlock`, `RelaxedSemaphore`) built on `__hip_atomic_*` with `__ATOMIC_RELAXED` + `__HIP_MEMORY_SCOPE_AGENT`. By using *relaxed* atomics for the flag/lock variables, all ordering responsibility falls on the explicit fence under test — the atomics provide only atomicity, not ordering.

This is critical for honest measurement: HIP's standard `atomicAdd`/`atomicExch` are also `__ATOMIC_RELAXED` under the hood, but using HIP's typed `__hip_atomic_*` builtins makes the relaxed scope explicit in the source.

### Benchmark 1: Fence-only latency

A single workgroup, single thread, runs `FENCE_NROUNDS = 1024` iterations of:

```
volatile_store(sink, i)
fence (release / acquire / release+acquire)
```

The volatile store is essential — without surrounding memory traffic, `__threadfence()` is elided by `SIMemoryLegalizer` (the AMDGPU memory legalizer only emits cache flushes where they are needed by surrounding memory operations). With a `volatile uint32_t* sink` store on each iteration, the fence has real work to synchronize and the optimizer cannot remove it.

Reports cycles/round and ns/round for each fence type.

### Benchmark 2: Producer-Consumer

Launches 16 blocks of 64 threads. Each block reads its `xcc_id` and races on `RelaxedSpinlock role[xcc_id]`:
- The block that wins `try_lock` becomes the **producer** for that XCD.
- Subsequent blocks on the same XCD become **consumers** (they all wait on the per-XCD semaphore).

Producer writes `buffer[xcc_id] = xcc_id`, executes the release fence, then releases the semaphore. Consumer waits, executes the acquire fence, reads `buffer[xcc_id]`, and times the acquire+read.

Because role election is keyed on `xcc_id`, **the producer and consumer for any given XCD are guaranteed to be on the same XCD by construction** — no `HSA_CU_MASK`, no CPX mode, no retry loop. With 16 blocks distributed across 8 XCDs by the round-robin scheduler, each XCD gets ≥1 producer and ≥1 consumer.

The consumer also verifies `observed == producer_xcc` to catch ordering violations. Reports per-XCD median latency.

### Benchmark 3: Ping-Pong

Same role-election structure as producer-consumer, but runs `PINGPONG_NROUNDS = 1024` rounds with the producer/consumer roles alternating each round. Tracks mismatches (consumer reads a value other than the current round number) — a non-zero mismatch count means the fence failed to enforce ordering.

This is the round-trip latency benchmark: each round measures the cost of one release→acquire handshake.

### Benchmark 4: Fence Storm

Single block, `FENCE_STORM_BLOCK_SIZE = 64` threads, `FENCE_STORM_NROUNDS = 1024` iterations. All 64 threads simultaneously:

```
buffer[threadIdx.x] = threadIdx.x
release()
acquire()
__syncthreads()
```

This stresses the cache/fabric path under contention from many simultaneous fence issuers within a single CU.

---

## Correctness Argument

### Why Chiplet Fences Work Within an XCD

The argument rests on one architectural fact: **L2 is the point of coherency for all CUs on the same XCD.**

1. **Producer executes chiplet release:**
   - `s_waitcnt vmcnt(0)` ensures all vector stores (write-through) have reached L2.
   - `s_dcache_wb` + `s_waitcnt lgkmcnt(0)` ensures all dirty scalar L1 lines have been written back to L2.
   - After this sequence, the producer's data is in L2.

2. **Producer publishes flag:** The flag is written via a relaxed atomic, which goes to L2 (atomics on agent scope target the L2 cache directly). Even though the atomic itself is relaxed, the fence preceding it has already flushed the data to L2, so the flag write cannot become visible before the data.

3. **Consumer observes flag:** The consumer spins on the flag using a relaxed atomic load that reads from L2.

4. **Consumer executes chiplet acquire:**
   - `buffer_inv sc1` invalidates vector L1.
   - `s_dcache_inv` invalidates scalar L1.
   - `s_waitcnt vmcnt(0) lgkmcnt(0)` waits for both invalidations.

5. **Consumer reads data:** The subsequent load misses in L1 (just invalidated) and goes to L2, where it sees the producer's data.

The chain is: **producer stores → L2 (release) → consumer invalidates L1 (acquire) → consumer reads from L2 → sees producer's data.**

### Why Chiplet Fences Fail Across XCDs

When producer and consumer are on different XCDs, they have **separate physical L2 caches**:

1. Producer's chiplet release pushes data to **XCD_0's L2**.
2. Consumer's chiplet acquire invalidates **XCD_3's L1**, causing reads to go to **XCD_3's L2**.
3. XCD_3's L2 does not have the data — it is in XCD_0's L2.
4. The consumer reads stale data from XCD_3's L2.

For cross-XCD communication, data must reach HBM (the shared point of coherency across chiplets), and the consumer must invalidate both L1 and L2. This is what `__threadfence()` does, and what `__builtin_amdgcn_fence(..., "agent")` does at the LLVM level.

The producer-consumer and ping-pong benchmarks **avoid the cross-XCD case by construction**: the role-election logic guarantees that the producer and consumer for each XCD are on the same XCD. This makes the chiplet-fence variant a valid measurement, not an unsafe one.

---

## Building and Running

### Prerequisites

- AMD ROCm (tested with ROCm 6.x)
- AMD Instinct MI300X GPU (gfx942)
- `hipcc` compiler

### Layout

```
src/
├── bench.hip       # Single-file benchmark suite
└── bench.sh        # SLURM script for the AMD HPC Fund cluster
```

### Compiling and Running Locally

```bash
cd src
hipcc bench.hip          # produces ./a.out
./a.out
```

Note: `hipcc` accepts only the `.hip` source file as input; do not pass `.hpp` or other headers as arguments.

### Running on the AMD HPC Fund Cluster

Login:
```bash
ssh user@hpcfund.amd.com
```

Request an interactive node with an MI300X:
```bash
salloc -N 1 -n 4 -p mi3001x -t 00:30:00
```

Verify you got the right GPU:
```bash
rocminfo | grep gfx     # should show gfx942
```

Submit via SLURM batch:
```bash
sbatch bench.sh         # see src/bench.sh
```

The batch script compiles `bench.hip` with `hipcc` and runs the resulting `a.out`. Output is written to `chiplet-fence-<jobid>.out`.

### Expected Output

The program prints three sections per benchmark — one for each fence type (`ChipletFence`, `DeviceRelAcqFence`, `DeviceThreadFence`) — reporting median cycles/round and ns/round over `SAMPLES = 20` runs (with a warmup discarded). Producer-consumer and ping-pong report per-XCD numbers plus a chiplet-average. Ping-pong also reports mismatch counts; a non-zero mismatch indicates an ordering failure and should never occur for chiplet or device fences when role election keeps producer and consumer on the same XCD.
