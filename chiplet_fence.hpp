#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace chiplet {

// ---------------------------------------------------------------------------
// XCC Topology Detection
// ---------------------------------------------------------------------------

// Read the XCC_ID (chiplet ID) from HW_REG_HW_ID1.
// On MI300X: 8 XCDs (XCC_ID 0-7), field at bits [23:20] of HW_REG_HW_ID1.
// Uses s_getreg_b32 with hwreg(HW_REG_HW_ID1, offset=20, width=4).
__device__ __forceinline__ uint32_t get_xcc_id() {
    uint32_t xcc_id;
    // HW_REG_HW_ID1 = register 23 on gfx940/gfx942 (CDNA3)
    // Extracts 4 bits starting at bit 20 → XCC_ID
    asm volatile("s_getreg_b32 %0, hwreg(23, 20, 4)" : "=s"(xcc_id));
    return xcc_id;
}

// Check if two XCC IDs refer to the same chiplet.
__device__ __forceinline__ bool same_chiplet(uint32_t xcc_a, uint32_t xcc_b) {
    return xcc_a == xcc_b;
}

// ---------------------------------------------------------------------------
// Chiplet-Level Release Fence
// ---------------------------------------------------------------------------
// Ensures all prior stores from this CU are visible in the shared L2 cache
// (the point of coherency for all CUs on the same XCD).
//
// Why this works within a chiplet:
//   - Vector L1 is write-through → s_waitcnt vmcnt(0) drains stores to L2
//   - Scalar L1 is write-back   → s_dcache_wb pushes dirty lines to L2
//   - After this fence, any CU on the same XCD can see the data via L2
//
// This is cheaper than __threadfence() (device scope) which flushes L2 → HBM
// and triggers cross-chiplet Infinity Fabric probes.

__device__ __forceinline__ void release_fence() {
    // 1. Drain all pending vector stores so they reach the shared L2
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    // 2. Write back dirty scalar L1 data cache lines to L2
    asm volatile("s_dcache_wb" ::: "memory");
    // 3. Wait for the write-back to complete
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}

// ---------------------------------------------------------------------------
// Chiplet-Level Acquire Fence
// ---------------------------------------------------------------------------
// Invalidates local L1 caches so that subsequent reads fetch fresh data from
// the shared L2 (which contains the releasing CU's stores).
//
// Why this works within a chiplet:
//   - buffer_gl0_inv invalidates the vector L1 cache
//   - s_dcache_inv   invalidates the scalar L1 / constant cache
//   - Subsequent loads miss in L1 and go to L2, which is coherent
//
// On the acquire side, this avoids the expensive L2 invalidation and
// cross-chiplet traffic that device-scope fences require.

__device__ __forceinline__ void acquire_fence() {
    // 1. Invalidate the vector L1 (GL0) cache
    asm volatile("buffer_gl0_inv" ::: "memory");
    // 2. Invalidate the scalar L1 data / constant cache
    asm volatile("s_dcache_inv" ::: "memory");
    // 3. Wait for both invalidations to complete
    //    buffer_gl0_inv → tracked by vmcnt, s_dcache_inv → tracked by lgkmcnt
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
}

// ---------------------------------------------------------------------------
// Device-Level Fence (fallback for cross-chiplet communication)
// ---------------------------------------------------------------------------
// Falls back to HIP's built-in __threadfence() which provides device-scope
// ordering. This is the expensive path that flushes through L2 to HBM and
// invalidates remote chiplets' caches via Infinity Fabric.

__device__ __forceinline__ void device_release_fence() {
    __threadfence();
}

__device__ __forceinline__ void device_acquire_fence() {
    __threadfence();
}

// ---------------------------------------------------------------------------
// Adaptive Fence (runtime scope selection)
// ---------------------------------------------------------------------------
// Compares XCC_IDs at runtime and picks the cheapest correct fence:
//   - Same chiplet → chiplet fence (L1 only, no fabric traffic)
//   - Different chiplet → device fence (L2 + HBM + fabric probes)

__device__ __forceinline__ void adaptive_release_fence(uint32_t my_xcc,
                                                       uint32_t target_xcc) {
    if (same_chiplet(my_xcc, target_xcc)) {
        release_fence();
    } else {
        device_release_fence();
    }
}

__device__ __forceinline__ void adaptive_acquire_fence(uint32_t my_xcc,
                                                       uint32_t source_xcc) {
    if (same_chiplet(my_xcc, source_xcc)) {
        acquire_fence();
    } else {
        device_acquire_fence();
    }
}

} // namespace chiplet
