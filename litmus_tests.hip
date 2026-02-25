// ============================================================================
// AMD CDNA3 (MI300X) Coherence Litmus Tests
// ============================================================================
//
// Each test is a minimal concurrent program that probes a specific ordering
// property of the cache hierarchy. Tests are grouped by what they verify:
//
//   Group A: Correct fence sequences (should NEVER fail)
//   Group B: Intentionally broken sequences (MUST fail to confirm the model)
//   Group C: Cross-scope boundary tests (chiplet vs device)
//   Group D: Compound / multi-level tests
//
// Notation:
//   CU_i(XCD_j) = compute unit i on chiplet j
//   V_L1 = vector L1 cache (write-through, per-CU)
//   S_L1 = scalar L1 data cache (write-back, per-CU)
//   L2   = shared L2 cache (per-XCD, point of coherency within chiplet)
//   HBM  = device memory (point of coherency across chiplets)
//
// Each test records observed register values and checks against
// expected allowed/forbidden outcomes.
// ============================================================================

#include "chiplet_fence.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err = (call);                                               \
        if (err != hipSuccess) {                                               \
            fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,       \
                    hipGetErrorString(err));                                    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Outcome counters for statistical testing (run many iterations)
struct LitmusResult {
    uint32_t r0;
    uint32_t r1;
    uint32_t r2;  // used by IRIW tests
    uint32_t r3;
};

// ============================================================================
// GROUP A: Message Passing (MP) — the fundamental release/acquire test
// ============================================================================
//
// Producer writes data, then sets a flag.
// Consumer spins on the flag, then reads data.
// Correct fences guarantee: if flag==1 is observed, data==42 must hold.
//
//   CU 0 (producer):              CU 1 (consumer):
//   global_store [x], 42          loop:
//   <release fence>                 global_load r0, [flag] (glc=1)
//   global_store [flag], 1          if r0 == 0: goto loop
//                                 <acquire fence>
//                                 global_load r1, [x]
//
// Forbidden outcome: r0=1, r1=0  (saw flag but missed data)
// ============================================================================

// --- A1: MP with chiplet fences (same XCD) — should PASS -------------------
__global__ void litmus_mp_chiplet_release(int* x, int* flag, int* ready) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Wait for consumer to be ready (avoids race on startup)
        while (atomicAdd(ready, 0) == 0) {}

        // Store data
        *x = 42;

        // CHIPLET RELEASE: drain vector stores to L2, flush scalar L1
        chiplet::release_fence();

        // Publish flag (atomics bypass L1, go to L2)
        atomicExch(flag, 1);
    }
}

__global__ void litmus_mp_chiplet_acquire(int* x, int* flag, int* ready,
                                          LitmusResult* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Signal ready
        atomicExch(ready, 1);

        // Spin on flag (atomic load — always reads from L2, bypasses L1)
        while (atomicAdd(flag, 0) == 0) {}
        int r0 = 1;  // we know flag == 1

        // CHIPLET ACQUIRE: invalidate vector L1 + scalar L1
        chiplet::acquire_fence();

        // Read data — should miss in L1, go to L2, see 42
        int r1 = *x;

        out->r0 = r0;
        out->r1 = r1;
    }
}

// --- A2: MP with device fences (always correct) — baseline -----------------
__global__ void litmus_mp_device_release(int* x, int* flag, int* ready) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        while (atomicAdd(ready, 0) == 0) {}
        *x = 42;
        __threadfence();
        atomicExch(flag, 1);
    }
}

__global__ void litmus_mp_device_acquire(int* x, int* flag, int* ready,
                                         LitmusResult* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicExch(ready, 1);
        while (atomicAdd(flag, 0) == 0) {}
        __threadfence();
        int r1 = *x;
        out->r0 = 1;
        out->r1 = r1;
    }
}

// ============================================================================
// GROUP B: Broken fence sequences — MUST observe forbidden outcomes
// These confirm the model is realistic (not vacuously passing everything)
// ============================================================================

// --- B1: MP with NO fence at all -------------------------------------------
// Producer: store x; store flag  (no fence between them)
// Consumer: load flag; load x    (no fence between them)
//
// r0=1, r1=0 should be POSSIBLE:
//   - CU 1's L1 may have stale x=0 even after seeing flag=1
//   - Without invalidation, L1 serves the stale copy
__global__ void litmus_mp_no_fence_producer(int* x, int* flag, int* ready) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        while (atomicAdd(ready, 0) == 0) {}
        *x = 42;
        // NO FENCE — stores may not be ordered w.r.t. each other in L2
        atomicExch(flag, 1);
    }
}

__global__ void litmus_mp_no_fence_consumer(int* x, int* flag, int* ready,
                                            LitmusResult* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicExch(ready, 1);
        while (atomicAdd(flag, 0) == 0) {}
        // NO FENCE — L1 not invalidated, may see stale x
        int r1 = *x;
        out->r0 = 1;
        out->r1 = r1;
    }
}

// --- B2: MP with acquire but NO release ------------------------------------
// Producer does NOT flush L1 → L2 before publishing flag
// Consumer correctly invalidates L1
//
// r0=1, r1=0 should be POSSIBLE if producer used scalar stores:
//   - Scalar L1 is write-back → data may sit dirty in producer's S_L1
//   - Consumer invalidates own L1, reads from L2 → but L2 never got the data
__global__ void litmus_mp_missing_release_producer(int* x, int* flag,
                                                   int* ready) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        while (atomicAdd(ready, 0) == 0) {}
        *x = 42;
        // NO RELEASE FENCE — skip s_waitcnt + s_dcache_wb
        // (vector stores are write-through so they reach L2 anyway,
        //  but if the compiler uses scalar stores, data stays in S_L1)
        atomicExch(flag, 1);
    }
}

// --- B3: MP with release but NO acquire ------------------------------------
// Producer correctly flushes to L2
// Consumer does NOT invalidate L1 → reads stale cached data
//
// r0=1, r1=0 should be POSSIBLE:
//   - Data is in L2 (producer flushed) but consumer's V_L1 has stale copy
//   - Without buffer_gl0_inv, the load hits in L1 and returns old value
__global__ void litmus_mp_missing_acquire_consumer(int* x, int* flag,
                                                   int* ready,
                                                   LitmusResult* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicExch(ready, 1);
        while (atomicAdd(flag, 0) == 0) {}
        // NO ACQUIRE FENCE — L1 not invalidated
        int r1 = *x;
        out->r0 = 1;
        out->r1 = r1;
    }
}

// ============================================================================
// GROUP C: Cross-scope boundary tests
// ============================================================================

// --- C1: MP chiplet fence across different XCDs — should FAIL --------------
// Producer on XCD 0, consumer on XCD 3
// Chiplet release only pushes data to XCD 0's L2
// Chiplet acquire only invalidates XCD 3's L1 (reads from XCD 3's L2)
// XCD 3's L2 does NOT have the data → consumer sees stale value
//
// r0=1, r1=0 is EXPECTED: chiplet fence is insufficient cross-XCD
//
// NOTE: This test requires pinning blocks to specific XCDs (HSA_CU_MASK)
// to guarantee cross-XCD placement. Without pinning, the scheduler may
// place both blocks on the same XCD, making the test vacuously pass.
__global__ void litmus_mp_cross_xcd_chiplet(int* x, int* flag, int* ready,
                                            LitmusResult* out,
                                            uint32_t* xcc_ids) {
    uint32_t my_xcc = chiplet::get_xcc_id();

    if (threadIdx.x == 0) {
        xcc_ids[blockIdx.x] = my_xcc;

        if (blockIdx.x == 0) {
            // Producer
            while (atomicAdd(ready, 0) == 0) {}
            *x = 42;
            chiplet::release_fence();  // only L1 → local L2
            atomicExch(flag, 1);
        } else {
            // Consumer
            atomicExch(ready, 1);
            while (atomicAdd(flag, 0) == 0) {}
            chiplet::acquire_fence();  // only invalidates local L1
            int r1 = *x;              // reads from LOCAL L2 → stale!
            out->r0 = 1;
            out->r1 = r1;
        }
    }
}

// --- C2: MP device fence across different XCDs — should PASS ---------------
// Same as C1 but with __threadfence() (flushes L2 → HBM)
// Data reaches HBM → consumer's L2 miss goes to HBM → sees fresh data
__global__ void litmus_mp_cross_xcd_device(int* x, int* flag, int* ready,
                                           LitmusResult* out) {
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) {
            while (atomicAdd(ready, 0) == 0) {}
            *x = 42;
            __threadfence();  // L1 → L2 → HBM
            atomicExch(flag, 1);
        } else {
            atomicExch(ready, 1);
            while (atomicAdd(flag, 0) == 0) {}
            __threadfence();  // invalidate L1 + L2
            int r1 = *x;     // miss in L2 → goes to HBM → sees 42
            out->r0 = 1;
            out->r1 = r1;
        }
    }
}

// ============================================================================
// GROUP D: Store Buffering (SB) and IRIW
// ============================================================================

// --- D1: Store Buffering within same XCD -----------------------------------
//
//   CU 0:                          CU 1:
//   global_store [x], 1            global_store [y], 1
//   <chiplet release+acquire>      <chiplet release+acquire>
//   global_load r0, [y]            global_load r1, [x]
//
// Can r0=0, r1=0? Both CUs wrote but neither sees the other's write.
//
// Within same XCD with FULL chiplet release+acquire between store and load:
// The release pushes stores to L2, the acquire invalidates L1.
// L2 provides total store order within XCD → r0=0,r1=0 should be FORBIDDEN.
__global__ void litmus_sb_chiplet(int* x, int* y, LitmusResult* out,
                                  int* barrier) {
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) {
            // Wait for both blocks to be ready
            atomicAdd(barrier, 1);
            while (atomicAdd(barrier, 0) < 2) {}

            *x = 1;
            chiplet::release_fence();
            chiplet::acquire_fence();
            out->r0 = *y;

        } else if (blockIdx.x == 1) {
            atomicAdd(barrier, 1);
            while (atomicAdd(barrier, 0) < 2) {}

            *y = 1;
            chiplet::release_fence();
            chiplet::acquire_fence();
            out->r1 = *x;
        }
    }
}

// --- D2: IRIW (Independent Reads of Independent Writes) --------------------
//
//   CU 0:          CU 1:          CU 2:            CU 3:
//   store [x],1    store [y],1    load r0,[x]      load r2,[y]
//                                 <fence>          <fence>
//                                 load r1,[y]      load r3,[x]
//
// Forbidden (under multi-copy atomicity): r0=1,r1=0,r2=1,r3=0
// This means CU 2 saw x before y, but CU 3 saw y before x.
// If L2 provides a single total order (multi-copy atomic within XCD),
// this outcome is impossible when all 4 CUs are on the same XCD.
//
// NOTE: This test needs 4 workgroups. With chiplet fences on same XCD,
// L2's total order should prevent this outcome.
__global__ void litmus_iriw_chiplet(int* x, int* y, LitmusResult* out,
                                    int* barrier) {
    if (threadIdx.x == 0) {
        // Barrier: all 4 blocks start together
        atomicAdd(barrier, 1);
        while (atomicAdd(barrier, 0) < 4) {}

        if (blockIdx.x == 0) {
            *x = 1;
            chiplet::release_fence();
        } else if (blockIdx.x == 1) {
            *y = 1;
            chiplet::release_fence();
        } else if (blockIdx.x == 2) {
            chiplet::acquire_fence();
            out->r0 = *x;
            chiplet::release_fence();
            chiplet::acquire_fence();
            out->r1 = *y;
        } else if (blockIdx.x == 3) {
            chiplet::acquire_fence();
            out->r2 = *y;
            chiplet::release_fence();
            chiplet::acquire_fence();
            out->r3 = *x;
        }
    }
}

// ============================================================================
// Test Runner
// ============================================================================

struct TestConfig {
    const char* name;
    bool expect_always_42;  // true = r1 must always be 42 when r0=1
};

static void run_mp_test(const char* name,
                        void (*producer)(int*, int*, int*),
                        void (*consumer)(int*, int*, int*, LitmusResult*),
                        bool expect_correct, int iterations) {
    int *d_x, *d_flag, *d_ready;
    LitmusResult *d_out;
    LitmusResult h_out;

    HIP_CHECK(hipMalloc(&d_x,     sizeof(int)));
    HIP_CHECK(hipMalloc(&d_flag,  sizeof(int)));
    HIP_CHECK(hipMalloc(&d_ready, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_out,   sizeof(LitmusResult)));

    int violations = 0;

    for (int i = 0; i < iterations; i++) {
        HIP_CHECK(hipMemset(d_x,     0, sizeof(int)));
        HIP_CHECK(hipMemset(d_flag,  0, sizeof(int)));
        HIP_CHECK(hipMemset(d_ready, 0, sizeof(int)));
        HIP_CHECK(hipMemset(d_out,   0, sizeof(LitmusResult)));

        // Launch producer and consumer on separate streams for concurrency
        hipStream_t s0, s1;
        HIP_CHECK(hipStreamCreate(&s0));
        HIP_CHECK(hipStreamCreate(&s1));

        hipLaunchKernelGGL(consumer, dim3(1), dim3(1), 0, s1,
                           d_x, d_flag, d_ready, d_out);
        hipLaunchKernelGGL(producer, dim3(1), dim3(1), 0, s0,
                           d_x, d_flag, d_ready);

        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipStreamDestroy(s0));
        HIP_CHECK(hipStreamDestroy(s1));

        HIP_CHECK(hipMemcpy(&h_out, d_out, sizeof(LitmusResult),
                            hipMemcpyDeviceToHost));

        if (h_out.r0 == 1 && h_out.r1 != 42) {
            violations++;
        }
    }

    printf("  %-45s ", name);
    if (expect_correct) {
        printf("violations: %d/%d  %s\n", violations, iterations,
               violations == 0 ? "[PASS - no violations]"
                               : "[FAIL - unexpected violation!]");
    } else {
        printf("violations: %d/%d  %s\n", violations, iterations,
               violations > 0 ? "[EXPECTED - fence is insufficient]"
                              : "[WEAK - no violations seen (may need more iters)]");
    }

    hipFree(d_x);
    hipFree(d_flag);
    hipFree(d_ready);
    hipFree(d_out);
}

int main() {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s (gfx%d, %d CUs)\n\n", prop.name, prop.gcnArch,
           prop.multiProcessorCount);

    constexpr int ITERS = 1000;

    printf("=== GROUP A: Correct fence sequences (expect 0 violations) ===\n");
    run_mp_test("A1: MP + chiplet release/acquire (same XCD)",
                litmus_mp_chiplet_release, litmus_mp_chiplet_acquire,
                true, ITERS);
    run_mp_test("A2: MP + device release/acquire (baseline)",
                litmus_mp_device_release, litmus_mp_device_acquire,
                true, ITERS);

    printf("\n=== GROUP B: Broken sequences (expect violations) ===\n");
    run_mp_test("B1: MP + NO fence at all",
                litmus_mp_no_fence_producer, litmus_mp_no_fence_consumer,
                false, ITERS);
    run_mp_test("B2: MP + acquire only (missing release)",
                litmus_mp_missing_release_producer, litmus_mp_chiplet_acquire,
                false, ITERS);
    run_mp_test("B3: MP + release only (missing acquire)",
                litmus_mp_chiplet_release, litmus_mp_missing_acquire_consumer,
                false, ITERS);

    printf("\n=== GROUP C: Cross-scope tests ===\n");
    printf("  (C1/C2 require cross-XCD placement — results depend on scheduler)\n");

    // C1/C2 use 2 blocks in one kernel, not separate producer/consumer
    {
        int *d_x, *d_flag, *d_ready;
        uint32_t *d_xcc_ids;
        LitmusResult *d_out;
        LitmusResult h_out;

        HIP_CHECK(hipMalloc(&d_x,       sizeof(int)));
        HIP_CHECK(hipMalloc(&d_flag,    sizeof(int)));
        HIP_CHECK(hipMalloc(&d_ready,   sizeof(int)));
        HIP_CHECK(hipMalloc(&d_xcc_ids, 2 * sizeof(uint32_t)));
        HIP_CHECK(hipMalloc(&d_out,     sizeof(LitmusResult)));

        // C1: chiplet fence cross-XCD
        int c1_violations = 0;
        int c1_cross_xcd = 0;
        for (int i = 0; i < ITERS; i++) {
            HIP_CHECK(hipMemset(d_x, 0, sizeof(int)));
            HIP_CHECK(hipMemset(d_flag, 0, sizeof(int)));
            HIP_CHECK(hipMemset(d_ready, 0, sizeof(int)));
            HIP_CHECK(hipMemset(d_out, 0, sizeof(LitmusResult)));

            hipLaunchKernelGGL(litmus_mp_cross_xcd_chiplet,
                               dim3(2), dim3(1), 0, 0,
                               d_x, d_flag, d_ready, d_out, d_xcc_ids);
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(&h_out, d_out, sizeof(LitmusResult),
                                hipMemcpyDeviceToHost));
            uint32_t xcc[2];
            HIP_CHECK(hipMemcpy(xcc, d_xcc_ids, 2 * sizeof(uint32_t),
                                hipMemcpyDeviceToHost));

            if (xcc[0] != xcc[1]) {
                c1_cross_xcd++;
                if (h_out.r0 == 1 && h_out.r1 != 42) c1_violations++;
            }
        }
        printf("  C1: MP + chiplet fence cross-XCD             "
               "cross-xcd runs: %d/%d, violations: %d  %s\n",
               c1_cross_xcd, ITERS, c1_violations,
               c1_cross_xcd == 0 ? "[all same XCD — inconclusive]"
               : (c1_violations > 0 ? "[EXPECTED - chiplet fence insufficient]"
                                    : "[no violations yet]"));

        // C2: device fence cross-XCD
        HIP_CHECK(hipMemset(d_x, 0, sizeof(int)));
        HIP_CHECK(hipMemset(d_flag, 0, sizeof(int)));
        HIP_CHECK(hipMemset(d_ready, 0, sizeof(int)));
        HIP_CHECK(hipMemset(d_out, 0, sizeof(LitmusResult)));

        int c2_violations = 0;
        for (int i = 0; i < ITERS; i++) {
            HIP_CHECK(hipMemset(d_x, 0, sizeof(int)));
            HIP_CHECK(hipMemset(d_flag, 0, sizeof(int)));
            HIP_CHECK(hipMemset(d_ready, 0, sizeof(int)));
            HIP_CHECK(hipMemset(d_out, 0, sizeof(LitmusResult)));

            hipLaunchKernelGGL(litmus_mp_cross_xcd_device,
                               dim3(2), dim3(1), 0, 0,
                               d_x, d_flag, d_ready, d_out);
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(&h_out, d_out, sizeof(LitmusResult),
                                hipMemcpyDeviceToHost));
            if (h_out.r0 == 1 && h_out.r1 != 42) c2_violations++;
        }
        printf("  C2: MP + device fence cross-XCD              "
               "violations: %d/%d  %s\n",
               c2_violations, ITERS,
               c2_violations == 0 ? "[PASS]" : "[FAIL!]");

        hipFree(d_x); hipFree(d_flag); hipFree(d_ready);
        hipFree(d_xcc_ids); hipFree(d_out);
    }

    printf("\n=== GROUP D: Store Buffering and IRIW ===\n");

    // D1: Store Buffering
    {
        int *d_x, *d_y, *d_barrier;
        LitmusResult *d_out;
        LitmusResult h_out;

        HIP_CHECK(hipMalloc(&d_x, sizeof(int)));
        HIP_CHECK(hipMalloc(&d_y, sizeof(int)));
        HIP_CHECK(hipMalloc(&d_barrier, sizeof(int)));
        HIP_CHECK(hipMalloc(&d_out, sizeof(LitmusResult)));

        int sb_both_zero = 0;
        for (int i = 0; i < ITERS; i++) {
            HIP_CHECK(hipMemset(d_x, 0, sizeof(int)));
            HIP_CHECK(hipMemset(d_y, 0, sizeof(int)));
            HIP_CHECK(hipMemset(d_barrier, 0, sizeof(int)));
            HIP_CHECK(hipMemset(d_out, 0, sizeof(LitmusResult)));

            hipLaunchKernelGGL(litmus_sb_chiplet,
                               dim3(2), dim3(1), 0, 0,
                               d_x, d_y, d_out, d_barrier);
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(&h_out, d_out, sizeof(LitmusResult),
                                hipMemcpyDeviceToHost));
            if (h_out.r0 == 0 && h_out.r1 == 0) sb_both_zero++;
        }
        printf("  D1: SB + chiplet fences (same XCD)           "
               "r0=0,r1=0: %d/%d  %s\n",
               sb_both_zero, ITERS,
               sb_both_zero == 0 ? "[PASS - L2 provides TSO-like order]"
                                 : "[INTERESTING - L2 allows SB reorder]");

        hipFree(d_x); hipFree(d_y); hipFree(d_barrier); hipFree(d_out);
    }

    // D2: IRIW
    {
        int *d_x, *d_y, *d_barrier;
        LitmusResult *d_out;
        LitmusResult h_out;

        HIP_CHECK(hipMalloc(&d_x, sizeof(int)));
        HIP_CHECK(hipMalloc(&d_y, sizeof(int)));
        HIP_CHECK(hipMalloc(&d_barrier, sizeof(int)));
        HIP_CHECK(hipMalloc(&d_out, sizeof(LitmusResult)));

        int iriw_violation = 0;
        for (int i = 0; i < ITERS; i++) {
            HIP_CHECK(hipMemset(d_x, 0, sizeof(int)));
            HIP_CHECK(hipMemset(d_y, 0, sizeof(int)));
            HIP_CHECK(hipMemset(d_barrier, 0, sizeof(int)));
            HIP_CHECK(hipMemset(d_out, 0, sizeof(LitmusResult)));

            hipLaunchKernelGGL(litmus_iriw_chiplet,
                               dim3(4), dim3(1), 0, 0,
                               d_x, d_y, d_out, d_barrier);
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(&h_out, d_out, sizeof(LitmusResult),
                                hipMemcpyDeviceToHost));
            // Forbidden: r0=1,r1=0,r2=1,r3=0
            if (h_out.r0 == 1 && h_out.r1 == 0 &&
                h_out.r2 == 1 && h_out.r3 == 0) {
                iriw_violation++;
            }
        }
        printf("  D2: IRIW + chiplet fences (same XCD)         "
               "violations: %d/%d  %s\n",
               iriw_violation, ITERS,
               iriw_violation == 0
                   ? "[PASS - L2 is multi-copy atomic within XCD]"
                   : "[VIOLATION - L2 not multi-copy atomic!]");

        hipFree(d_x); hipFree(d_y); hipFree(d_barrier); hipFree(d_out);
    }

    printf("\nDone.\n");
    return 0;
}
