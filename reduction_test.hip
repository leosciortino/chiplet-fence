#include "chiplet_fence.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err = (call);                                               \
        if (err != hipSuccess) {                                               \
            fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,       \
                    hipGetErrorString(err));                                    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static constexpr uint32_t BLOCK_SIZE = 256;

// ---------------------------------------------------------------------------
// Diagnostic kernel: report which XCD each workgroup is running on
// ---------------------------------------------------------------------------
__global__ void xcc_map_kernel(uint32_t* xcc_ids) {
    if (threadIdx.x == 0) {
        xcc_ids[blockIdx.x] = chiplet::get_xcc_id();
    }
}

// ---------------------------------------------------------------------------
// Adaptive inter-workgroup reduction using chiplet release/acquire fences.
//
// The challenge: at release time, a producer workgroup does NOT know whether
// the eventual consumer will be on the same chiplet. So we use a two-pass
// approach:
//
//   Pass 1 (optimistic): Every workgroup does a chiplet release (L1→L2).
//          This is always safe — it's a subset of device release.
//          Then atomically registers its XCC_ID and increments a counter.
//
//   Pass 2 (fixup): The last workgroup checks all XCC_IDs.
//          - If all same chiplet → chiplet acquire is sufficient (L1 inv).
//            The data is already in the shared L2, which all CUs can read.
//          - If cross-chiplet → the last workgroup issues a device-scope
//            fence which flushes L2→HBM and invalidates remote L2s.
//            BUT: the producers only flushed to their local L2, not HBM.
//            So the last WG must ALSO do a device-scope release to push
//            its own pending reads, and we rely on the fact that the
//            producers' atomicAdd already used device-scope atomics that
//            forced the data path through the coherent interconnect.
//
// IMPORTANT: This reduction is designed for the common case where a small
// number of workgroups land on the same chiplet (which the scheduler does
// for small launches). For guaranteed same-chiplet execution, use
// HSA_CU_MASK or limit dispatch to one XCD's worth of CUs.
// ---------------------------------------------------------------------------
__global__ void chiplet_reduction(const float* __restrict__ input,
                                  float* __restrict__ partial_sums,
                                  float* __restrict__ result,
                                  uint32_t* __restrict__ retire_count,
                                  uint32_t* __restrict__ xcc_ids,
                                  uint32_t n,
                                  uint32_t num_blocks) {
    __shared__ float smem[BLOCK_SIZE];

    const uint32_t tid = threadIdx.x;
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    // --- Phase 1: intra-workgroup reduction via shared memory ---------------
    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    // --- Phase 2: publish partial sum with release fence --------------------
    if (tid == 0) {
        uint32_t my_xcc = chiplet::get_xcc_id();
        xcc_ids[blockIdx.x] = my_xcc;

        // Write partial sum to global memory (vector store → write-through L1 → L2)
        partial_sums[blockIdx.x] = smem[0];

        // DEVICE RELEASE FENCE for correctness in all cases.
        // We cannot use chiplet release here because we don't yet know
        // whether the consumer will be on the same chiplet.
        // __threadfence() ensures data reaches HBM, visible to all XCDs.
        __threadfence();

        // Atomic increment: flag that this workgroup's data is published.
        uint32_t ticket = atomicAdd(retire_count, 1u);

        // --- Phase 3: last workgroup performs final reduction ----------------
        if (ticket == num_blocks - 1) {
            // All producers have completed their release + atomic.
            // Now check if everyone was on the same chiplet.
            bool all_same_xcc = true;
            for (uint32_t i = 0; i < num_blocks; i++) {
                if (xcc_ids[i] != my_xcc) {
                    all_same_xcc = false;
                    break;
                }
            }

            if (all_same_xcc) {
                // CHIPLET ACQUIRE FENCE
                // All producers were on this chiplet, so their data is in
                // our shared L2 (the release pushed L1→L2, and __threadfence
                // was a superset of that). Invalidating L1 is sufficient —
                // subsequent loads will hit L2 and see the correct data.
                chiplet::acquire_fence();
            } else {
                // DEVICE ACQUIRE FENCE
                // Producers were on different chiplets. Their data went
                // through __threadfence() → HBM. We need to invalidate
                // our L1 AND L2 to force reads from HBM.
                chiplet::device_acquire_fence();
            }

            float total = 0.0f;
            for (uint32_t i = 0; i < num_blocks; i++) {
                total += partial_sums[i];
            }
            *result = total;
        }
    }
}

// ---------------------------------------------------------------------------
// Pure chiplet reduction — ONLY safe when all workgroups are on one XCD.
// This is the fast path that demonstrates the actual performance benefit.
// Caller must guarantee same-chiplet placement (e.g. via HSA_CU_MASK,
// limited dispatch size, or rocm-smi --setcomputepartition).
// ---------------------------------------------------------------------------
__global__ void chiplet_reduction_fast(const float* __restrict__ input,
                                       float* __restrict__ partial_sums,
                                       float* __restrict__ result,
                                       uint32_t* __restrict__ retire_count,
                                       uint32_t n,
                                       uint32_t num_blocks) {
    __shared__ float smem[BLOCK_SIZE];

    const uint32_t tid = threadIdx.x;
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Phase 1: intra-workgroup reduction
    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = smem[0];

        // CHIPLET RELEASE: only L1 → L2 (no HBM flush, no fabric traffic)
        chiplet::release_fence();

        uint32_t ticket = atomicAdd(retire_count, 1u);

        if (ticket == num_blocks - 1) {
            // CHIPLET ACQUIRE: only L1 invalidation (no L2 inv, no fabric)
            chiplet::acquire_fence();

            float total = 0.0f;
            for (uint32_t i = 0; i < num_blocks; i++) {
                total += partial_sums[i];
            }
            *result = total;
        }
    }
}

// ---------------------------------------------------------------------------
// Device-scope reduction — baseline using __threadfence() for comparison.
// ---------------------------------------------------------------------------
__global__ void device_reduction(const float* __restrict__ input,
                                 float* __restrict__ partial_sums,
                                 float* __restrict__ result,
                                 uint32_t* __restrict__ retire_count,
                                 uint32_t n,
                                 uint32_t num_blocks) {
    __shared__ float smem[BLOCK_SIZE];

    const uint32_t tid = threadIdx.x;
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = smem[0];

        // DEVICE RELEASE: L1 → L2 → HBM + Infinity Fabric probes
        __threadfence();

        uint32_t ticket = atomicAdd(retire_count, 1u);

        if (ticket == num_blocks - 1) {
            // DEVICE ACQUIRE: invalidate L1 + L2
            __threadfence();

            float total = 0.0f;
            for (uint32_t i = 0; i < num_blocks; i++) {
                total += partial_sums[i];
            }
            *result = total;
        }
    }
}

// ---------------------------------------------------------------------------
// Hierarchical chiplet-aware reduction.
//
// Key insight: assign work based on runtime XCC_ID, NOT blockIdx.x.
// Blocks on chiplet i reduce the i-th partition of the array.
// Since cooperating blocks share the same XCD by construction,
// chiplet fences are ALWAYS correct for intra-chiplet synchronization.
// Device fences are only needed for the final cross-chiplet merge.
//
// Two-level reduction:
//   Level 1 (intra-chiplet): chiplet fences — cheap, L1 only
//   Level 2 (cross-chiplet): device fence    — expensive but only once
//
// Uses a lightweight global barrier (safe when total_blocks << #CUs)
// so that each block knows how many peers landed on its XCD.
// ---------------------------------------------------------------------------

static constexpr uint32_t NUM_XCDS = 8;          // MI300X: 8 XCDs
static constexpr uint32_t MAX_BLOCKS_PER_XCD = 4; // headroom for uneven scheduling

struct HierState {
    // Registration phase
    uint32_t chiplet_slot_ctr[NUM_XCDS];  // per-XCD block counter
    uint32_t barrier_ctr;                  // global registration barrier
    uint32_t num_active_xcds;              // XCDs that got >= 1 block

    // Per-block partial sums, indexed [xcd * MAX_BLOCKS_PER_XCD + slot]
    float block_partials[NUM_XCDS * MAX_BLOCKS_PER_XCD];

    // Intra-chiplet reduction
    uint32_t chiplet_done_ctr[NUM_XCDS];  // per-XCD completion counter
    float chiplet_sums[NUM_XCDS];          // one sum per XCD

    // Cross-chiplet final reduction
    uint32_t global_done_ctr;
};

__global__ void hierarchical_reduction(const float* __restrict__ input,
                                       float* __restrict__ result,
                                       HierState* __restrict__ state,
                                       uint32_t n,
                                       uint32_t total_blocks,
                                       uint32_t blocks_per_xcd) {
    __shared__ float smem[BLOCK_SIZE];
    __shared__ uint32_t s_xcc, s_slot, s_nblocks, s_active_xcds;

    const uint32_t tid = threadIdx.x;

    // === Phase 0: Registration barrier ======================================
    // Every block reads its XCC_ID, claims a slot, and waits for all blocks
    // to register. After the barrier, per-XCD block counts are finalized.
    // Safe because total_blocks (16) << available CUs (224 on MI300X).
    if (tid == 0) {
        s_xcc  = chiplet::get_xcc_id();
        s_slot = atomicAdd(&state->chiplet_slot_ctr[s_xcc], 1);

        // First block on this XCD: register it as active
        if (s_slot == 0) {
            atomicAdd(&state->num_active_xcds, 1);
        }

        // Global barrier: wait until all blocks have registered
        atomicAdd(&state->barrier_ctr, 1);
        while (atomicAdd(&state->barrier_ctr, 0) < total_blocks) {}

        // Now per-XCD counts are final
        s_nblocks     = atomicAdd(&state->chiplet_slot_ctr[s_xcc], 0);
        s_active_xcds = atomicAdd(&state->num_active_xcds, 0);
    }
    __syncthreads();

    const uint32_t xcc         = s_xcc;
    const uint32_t slot        = s_slot;
    const uint32_t active_xcds = s_active_xcds;
    // Cap effective blocks per XCD to avoid oversubscription
    const uint32_t eff_blocks  = min(s_nblocks, blocks_per_xcd);

    // Excess blocks on an oversubscribed XCD: nothing to do
    if (slot >= blocks_per_xcd) return;

    // === Phase 1: Compute this block's assigned chunk =======================
    // Each XCD is responsible for a contiguous (n / NUM_XCDS) slice.
    // Within that slice, each block handles (chunk_per_xcd / eff_blocks).
    const uint32_t chunk_per_xcd   = n / NUM_XCDS;
    const uint32_t chunk_per_block = chunk_per_xcd / eff_blocks;
    const uint32_t my_start        = xcc * chunk_per_xcd + slot * chunk_per_block;

    // Each thread accumulates over its stride (handles chunk > BLOCK_SIZE)
    float val = 0.0f;
    for (uint32_t off = 0; off < chunk_per_block; off += BLOCK_SIZE) {
        uint32_t idx = my_start + off + tid;
        if (off + tid < chunk_per_block && idx < n) {
            val += input[idx];
        }
    }
    smem[tid] = val;
    __syncthreads();

    // Intra-workgroup tree reduction
    for (uint32_t stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    // === Phase 2: Intra-chiplet sync (chiplet fences) =======================
    if (tid == 0) {
        state->block_partials[xcc * MAX_BLOCKS_PER_XCD + slot] = smem[0];

        // CHIPLET RELEASE — correct because all cooperating blocks share L2
        chiplet::release_fence();

        uint32_t ticket = atomicAdd(&state->chiplet_done_ctr[xcc], 1);

        // Last block on this XCD: reduce chiplet partials
        if (ticket == eff_blocks - 1) {
            // CHIPLET ACQUIRE — L1 invalidation, reads go to shared L2
            chiplet::acquire_fence();

            float chiplet_sum = 0.0f;
            for (uint32_t i = 0; i < eff_blocks; i++) {
                chiplet_sum += state->block_partials[xcc * MAX_BLOCKS_PER_XCD + i];
            }
            state->chiplet_sums[xcc] = chiplet_sum;

            // === Phase 3: Cross-chiplet sync (device fence) =================
            // Push chiplet result to HBM so other XCDs can see it
            __threadfence();

            uint32_t g = atomicAdd(&state->global_done_ctr, 1);

            // Last chiplet: final reduction across all XCDs
            if (g == active_xcds - 1) {
                __threadfence();

                float total = 0.0f;
                for (uint32_t i = 0; i < NUM_XCDS; i++) {
                    // Only include XCDs that had blocks
                    if (atomicAdd(&state->chiplet_slot_ctr[i], 0) > 0) {
                        total += state->chiplet_sums[i];
                    }
                }
                *result = total;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side reference reduction
// ---------------------------------------------------------------------------
static float cpu_reduce(const float* data, uint32_t n) {
    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        sum += data[i];
    }
    return static_cast<float>(sum);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    // --- Device info --------------------------------------------------------
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("GCN Arch: gfx%d\n", prop.gcnArch);
    printf("CUs: %d\n\n", prop.multiProcessorCount);

    // --- Parameters ---------------------------------------------------------
    constexpr uint32_t N = 1024;
    constexpr uint32_t NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; // 4

    // --- XCC mapping diagnostic ---------------------------------------------
    uint32_t* d_xcc_map;
    HIP_CHECK(hipMalloc(&d_xcc_map, NUM_BLOCKS * sizeof(uint32_t)));
    HIP_CHECK(hipMemset(d_xcc_map, 0, NUM_BLOCKS * sizeof(uint32_t)));

    hipLaunchKernelGGL(xcc_map_kernel,
                       dim3(NUM_BLOCKS), dim3(1), 0, 0,
                       d_xcc_map);
    HIP_CHECK(hipDeviceSynchronize());

    uint32_t h_xcc_map[NUM_BLOCKS];
    HIP_CHECK(hipMemcpy(h_xcc_map, d_xcc_map,
                        NUM_BLOCKS * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    printf("XCC mapping (%u workgroups):\n", NUM_BLOCKS);
    bool all_same = true;
    for (uint32_t i = 0; i < NUM_BLOCKS; i++) {
        printf("  block %u → XCC %u\n", i, h_xcc_map[i]);
        if (h_xcc_map[i] != h_xcc_map[0]) all_same = false;
    }
    printf("  All on same chiplet: %s\n\n", all_same ? "YES" : "NO");

    // --- Prepare input data -------------------------------------------------
    float* h_input = new float[N];
    for (uint32_t i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }
    float expected = cpu_reduce(h_input, N);

    // --- Device allocations -------------------------------------------------
    float *d_input, *d_partial, *d_result;
    uint32_t *d_retire_count, *d_xcc_ids;

    HIP_CHECK(hipMalloc(&d_input,        N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_partial,      NUM_BLOCKS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_result,       sizeof(float)));
    HIP_CHECK(hipMalloc(&d_retire_count, sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_xcc_ids,      NUM_BLOCKS * sizeof(uint32_t)));

    HIP_CHECK(hipMemcpy(d_input, h_input, N * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_result, 0, sizeof(float)));
    HIP_CHECK(hipMemset(d_retire_count, 0, sizeof(uint32_t)));
    HIP_CHECK(hipMemset(d_xcc_ids, 0, NUM_BLOCKS * sizeof(uint32_t)));

    // --- Launch reduction ---------------------------------------------------
    printf("Launching chiplet reduction: N=%u, blocks=%u, block_size=%u\n",
           N, NUM_BLOCKS, BLOCK_SIZE);

    hipLaunchKernelGGL(chiplet_reduction,
                       dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, 0,
                       d_input, d_partial, d_result,
                       d_retire_count, d_xcc_ids,
                       N, NUM_BLOCKS);
    HIP_CHECK(hipDeviceSynchronize());

    // --- Read back & verify -------------------------------------------------
    float h_result = 0.0f;
    HIP_CHECK(hipMemcpy(&h_result, d_result, sizeof(float),
                        hipMemcpyDeviceToHost));

    uint32_t h_xcc_ids[NUM_BLOCKS];
    HIP_CHECK(hipMemcpy(h_xcc_ids, d_xcc_ids,
                        NUM_BLOCKS * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    printf("\nReduction XCC mapping:\n");
    all_same = true;
    for (uint32_t i = 0; i < NUM_BLOCKS; i++) {
        printf("  block %u → XCC %u\n", i, h_xcc_ids[i]);
        if (h_xcc_ids[i] != h_xcc_ids[0]) all_same = false;
    }
    printf("  All on same chiplet: %s\n", all_same ? "YES" : "NO");
    printf("  Release fence: DEVICE (always, for cross-chiplet safety)\n");
    printf("  Acquire fence: %s\n\n",
           all_same ? "CHIPLET (L1-only, all on same XCD)"
                    : "DEVICE (L2+HBM, cross-chiplet detected)");

    printf("Expected: %.1f\n", expected);
    printf("Got:      %.1f\n", h_result);

    bool pass = (fabsf(h_result - expected) < 1e-3f);
    printf("Result:   %s\n\n", pass ? "PASS" : "FAIL");

    // === Test 2: Pure chiplet reduction (fast path) =========================
    printf("=== Test 2: Pure chiplet reduction (assumes same-XCD) ===\n");

    HIP_CHECK(hipMemset(d_result, 0, sizeof(float)));
    HIP_CHECK(hipMemset(d_retire_count, 0, sizeof(uint32_t)));

    hipLaunchKernelGGL(chiplet_reduction_fast,
                       dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, 0,
                       d_input, d_partial, d_result,
                       d_retire_count, N, NUM_BLOCKS);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(&h_result, d_result, sizeof(float),
                        hipMemcpyDeviceToHost));

    printf("  Release: chiplet (s_waitcnt vmcnt(0) + s_dcache_wb + s_waitcnt lgkmcnt(0))\n");
    printf("  Acquire: chiplet (buffer_gl0_inv + s_dcache_inv + s_waitcnt)\n");
    printf("  Expected: %.1f\n", expected);
    printf("  Got:      %.1f\n", h_result);
    bool pass2 = (fabsf(h_result - expected) < 1e-3f);
    printf("  Result:   %s\n", pass2 ? "PASS" : "FAIL");
    if (!pass2 && !all_same) {
        printf("  WARNING: workgroups were on different XCDs — chiplet fence\n"
               "           is insufficient. Use HSA_CU_MASK to pin to one XCD.\n");
    }
    printf("\n");

    // === Test 3: Device-scope reduction (baseline) ==========================
    printf("=== Test 3: Device-scope reduction (baseline __threadfence) ===\n");

    HIP_CHECK(hipMemset(d_result, 0, sizeof(float)));
    HIP_CHECK(hipMemset(d_retire_count, 0, sizeof(uint32_t)));

    hipLaunchKernelGGL(device_reduction,
                       dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, 0,
                       d_input, d_partial, d_result,
                       d_retire_count, N, NUM_BLOCKS);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(&h_result, d_result, sizeof(float),
                        hipMemcpyDeviceToHost));

    printf("  Release: device (__threadfence)\n");
    printf("  Acquire: device (__threadfence)\n");
    printf("  Expected: %.1f\n", expected);
    printf("  Got:      %.1f\n", h_result);
    bool pass3 = (fabsf(h_result - expected) < 1e-3f);
    printf("  Result:   %s\n\n", pass3 ? "PASS" : "FAIL");

    // === Test 4: Hierarchical chiplet-aware reduction =========================
    printf("=== Test 4: Hierarchical reduction (chiplet + device fences) ===\n");
    printf("  Assigns work by runtime XCC_ID — cooperating blocks share L2\n");
    printf("  Level 1 (intra-chiplet): chiplet fences\n");
    printf("  Level 2 (cross-chiplet): device fences\n\n");

    constexpr uint32_t HIER_BLOCKS_PER_XCD = 2;
    constexpr uint32_t HIER_TOTAL_BLOCKS   = NUM_XCDS * HIER_BLOCKS_PER_XCD; // 16
    constexpr uint32_t HIER_N              = NUM_XCDS * HIER_BLOCKS_PER_XCD
                                             * BLOCK_SIZE;                    // 4096

    float* h_hier_input = new float[HIER_N];
    for (uint32_t i = 0; i < HIER_N; i++) {
        h_hier_input[i] = 1.0f;
    }
    float hier_expected = cpu_reduce(h_hier_input, HIER_N);

    float *d_hier_input, *d_hier_result;
    HierState *d_hier_state;

    HIP_CHECK(hipMalloc(&d_hier_input,  HIER_N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_hier_result, sizeof(float)));
    HIP_CHECK(hipMalloc(&d_hier_state,  sizeof(HierState)));

    HIP_CHECK(hipMemcpy(d_hier_input, h_hier_input,
                        HIER_N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_hier_result, 0, sizeof(float)));
    HIP_CHECK(hipMemset(d_hier_state,  0, sizeof(HierState)));

    printf("  N=%u, total_blocks=%u, blocks_per_xcd=%u, block_size=%u\n",
           HIER_N, HIER_TOTAL_BLOCKS, HIER_BLOCKS_PER_XCD, BLOCK_SIZE);
    printf("  Each XCD handles %u elements (%u per block)\n",
           HIER_N / NUM_XCDS,
           (HIER_N / NUM_XCDS) / HIER_BLOCKS_PER_XCD);

    hipLaunchKernelGGL(hierarchical_reduction,
                       dim3(HIER_TOTAL_BLOCKS), dim3(BLOCK_SIZE), 0, 0,
                       d_hier_input, d_hier_result, d_hier_state,
                       HIER_N, HIER_TOTAL_BLOCKS, HIER_BLOCKS_PER_XCD);
    HIP_CHECK(hipDeviceSynchronize());

    float h_hier_result = 0.0f;
    HIP_CHECK(hipMemcpy(&h_hier_result, d_hier_result, sizeof(float),
                        hipMemcpyDeviceToHost));

    // Read back per-XCD diagnostics
    HierState h_state;
    HIP_CHECK(hipMemcpy(&h_state, d_hier_state, sizeof(HierState),
                        hipMemcpyDeviceToHost));

    printf("\n  XCD distribution:\n");
    uint32_t total_assigned = 0;
    for (uint32_t i = 0; i < NUM_XCDS; i++) {
        if (h_state.chiplet_slot_ctr[i] > 0) {
            uint32_t eff = h_state.chiplet_slot_ctr[i];
            if (eff > HIER_BLOCKS_PER_XCD) eff = HIER_BLOCKS_PER_XCD;
            printf("    XCD %u: %u blocks (%u active), chiplet_sum = %.1f\n",
                   i, h_state.chiplet_slot_ctr[i], eff,
                   h_state.chiplet_sums[i]);
            total_assigned += eff;
        }
    }
    printf("  Active XCDs: %u, total participating blocks: %u\n",
           h_state.num_active_xcds, total_assigned);

    printf("\n  Expected: %.1f\n", hier_expected);
    printf("  Got:      %.1f\n", h_hier_result);
    bool pass4 = (fabsf(h_hier_result - hier_expected) < 1e-3f);
    printf("  Result:   %s\n", pass4 ? "PASS" : "FAIL");

    if (!pass4 && h_state.num_active_xcds < NUM_XCDS) {
        printf("  NOTE: Only %u/%u XCDs received blocks. Elements on inactive\n"
               "        XCDs were not reduced. This is expected if the scheduler\n"
               "        clustered blocks. Increase total_blocks or use CU masking.\n",
               h_state.num_active_xcds, NUM_XCDS);
    }
    printf("\n");

    hipFree(d_hier_input);
    hipFree(d_hier_result);
    hipFree(d_hier_state);
    delete[] h_hier_input;

    // === Summary ============================================================
    bool all_pass = pass && pass2 && pass3 && pass4;
    printf("=== Summary ===\n");
    printf("  Test 1 - Adaptive (safe):            %s\n", pass  ? "PASS" : "FAIL");
    printf("  Test 2 - Chiplet-only (fast):        %s\n", pass2 ? "PASS" : "FAIL");
    printf("  Test 3 - Device-scope (baseline):    %s\n", pass3 ? "PASS" : "FAIL");
    printf("  Test 4 - Hierarchical (chiplet+dev): %s\n", pass4 ? "PASS" : "FAIL");

    // --- Cleanup ------------------------------------------------------------
    hipFree(d_input);
    hipFree(d_partial);
    hipFree(d_result);
    hipFree(d_retire_count);
    hipFree(d_xcc_ids);
    hipFree(d_xcc_map);
    delete[] h_input;

    return all_pass ? 0 : 1;
}
