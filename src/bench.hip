/**
        AMD Chiplet Fence Benchmarks and Litmus Tests

        Author: David Yue

        No portion of this code was AI generated. However, AI was used to
        assist in understanding HIP APIs and debugging.
*/

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <vector>

#define NUM_XCCS               8
#define FENCE_NROUNDS          1024
#define PINGPONG_NROUNDS       1024
#define FENCE_STORM_BLOCK_SIZE 64
#define FENCE_STORM_NROUNDS    1024
#define SAMPLES                20

#define HIP_CHECK(call)                                                                                                \
        do {                                                                                                           \
                hipError_t err = (call);                                                                               \
                if (err != hipSuccess) {                                                                               \
                        fprintf (stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString (err));        \
                        exit (1);                                                                                      \
                }                                                                                                      \
        } while (0)

namespace chiplet
{

__device__ __forceinline__ uint32_t get_xcc_id ()
{
        uint32_t xcc_id;
        asm volatile ("s_getreg_b32 %0, hwreg(20, 0, 4)" : "=s"(xcc_id));
        return xcc_id;
}

__device__ __forceinline__ void release_fence ()
{
        asm volatile ("s_waitcnt vmcnt(0)" ::: "memory");
        asm volatile ("s_dcache_wb" ::: "memory");
        asm volatile ("s_waitcnt lgkmcnt(0)" ::: "memory");
}

__device__ __forceinline__ void acquire_fence ()
{
        asm volatile ("buffer_inv sc1" ::: "memory");
        asm volatile ("s_dcache_inv" ::: "memory");
        asm volatile ("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
}
} // namespace chiplet

struct RelaxedSpinlock {
        int32_t lock;
};

struct RelaxedSemaphore {
        uint32_t count;
        RelaxedSpinlock lock;
};

template <typename T>
__device__ __forceinline__ T relaxed_atomic_load (T *address)
{
        return __hip_atomic_load (address, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T>
__device__ __forceinline__ void relaxed_atomic_store (T *address, T value)
{
        return __hip_atomic_store (address, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T>
__device__ __forceinline__ bool relaxed_compare_xchg (T *address, T compare, T value)
{
        return __hip_atomic_compare_exchange_weak (
                address, &compare, value, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void relaxed_spinlock_lock (RelaxedSpinlock *lock)
{
        while (!relaxed_compare_xchg (&lock->lock, 0, 1))
                ;
}

__device__ __forceinline__ bool relaxed_spinlock_try_lock (RelaxedSpinlock *lock)
{
        return relaxed_compare_xchg (&lock->lock, 0, 1);
}

__device__ __forceinline__ void relaxed_spinlock_unlock (RelaxedSpinlock *lock)
{
        relaxed_atomic_store (&lock->lock, 0);
}

void host_relaxed_spinlock_init (RelaxedSpinlock *lock)
{
        lock->lock = 0;
}

__device__ __forceinline__ void relaxed_sem_acquire (RelaxedSemaphore *sem)
{
        while (1) {
                while (relaxed_atomic_load<uint32_t> (&sem->count) == 0)
                        ;

                relaxed_spinlock_lock (&sem->lock);

                uint32_t count = relaxed_atomic_load (&sem->count);

                if (count == 0) {
                        relaxed_spinlock_unlock (&sem->lock);
                        continue;
                }

                relaxed_atomic_store (&sem->count, count - 1);
                relaxed_spinlock_unlock (&sem->lock);

                break;
        }
}

__device__ __forceinline__ void relaxed_sem_release (RelaxedSemaphore *sem)
{
        __hip_atomic_fetch_add (&sem->count, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

void host_relaxed_sem_init (RelaxedSemaphore *sem, uint32_t count)
{
        sem->count = count;
        host_relaxed_spinlock_init (&sem->lock);
}

struct ChipletFence {
        __device__ static void release () { chiplet::release_fence (); }
        __device__ static void acquire () { chiplet::acquire_fence (); }
};

struct DeviceRelAcqFence {
        __device__ static void release () { __builtin_amdgcn_fence (__ATOMIC_RELEASE, "agent"); }
        __device__ static void acquire () { __builtin_amdgcn_fence (__ATOMIC_ACQUIRE, "agent"); }
};

struct DeviceThreadFence {
        __device__ static void release () { __threadfence (); }
        __device__ static void acquire () { __threadfence (); }
};

// ============================================================================
// Fence-only kernels
// ============================================================================

template <typename Fence>
__global__ void fence_release_only_benchmark (long long *total_cycles, volatile uint32_t *sink)
{
        if (blockIdx.x == 0 && threadIdx.x == 0) {
                long long start = clock64 ();
                for (int i = 0; i < FENCE_NROUNDS; i++) {
                        *sink = i;
                        Fence::release ();
                }
                long long end = clock64 ();
                *total_cycles = end - start;
        }
}

template <typename Fence>
__global__ void fence_acquire_only_benchmark (long long *total_cycles, volatile uint32_t *sink)
{
        if (blockIdx.x == 0 && threadIdx.x == 0) {
                long long start = clock64 ();
                for (int i = 0; i < FENCE_NROUNDS; i++) {
                        *sink = i;
                        Fence::acquire ();
                }
                long long end = clock64 ();
                *total_cycles = end - start;
        }
}

template <typename Fence>
__global__ void fence_release_acquire_benchmark (long long *total_cycles, volatile uint32_t *sink)
{
        if (blockIdx.x == 0 && threadIdx.x == 0) {
                long long start = clock64 ();
                for (int i = 0; i < FENCE_NROUNDS; i++) {
                        *sink = i;
                        Fence::release ();
                        Fence::acquire ();
                }
                long long end = clock64 ();
                *total_cycles = end - start;
        }
}

// ============================================================================
// Producer-Consumer kernel
// ============================================================================

struct ProducerConsumerResult {
        uint32_t producer_xcc;
        uint32_t consumer_xcc;
        long long clock64_delta;
};

template <typename Fence>
__global__ void producer_consumer_benchmark (uint32_t buffer[NUM_XCCS],
                                             RelaxedSpinlock role[NUM_XCCS],
                                             RelaxedSemaphore sems[NUM_XCCS],
                                             ProducerConsumerResult result[NUM_XCCS])
{
        if (threadIdx.x != 0) return;

        uint32_t xcc_id = chiplet::get_xcc_id ();

        if (relaxed_spinlock_try_lock (&role[xcc_id])) {
                buffer[xcc_id] = xcc_id;
                Fence::release ();
                relaxed_sem_release (&sems[xcc_id]);
        } else {
                relaxed_sem_acquire (&sems[xcc_id]);
                long long start = clock64 ();
                Fence::acquire ();
                uint32_t observed = buffer[xcc_id];
                long long end = clock64 ();

                result[xcc_id].producer_xcc  = observed;
                result[xcc_id].consumer_xcc  = xcc_id;
                result[xcc_id].clock64_delta = end - start;
        }
}

// ============================================================================
// Ping-Pong kernel
// ============================================================================

struct PingPongResult {
        uint32_t mismatches;
        long long clock64_delta;
};

template <typename Fence>
__global__ void ping_pong_benchmark (uint32_t buffer[NUM_XCCS],
                                     RelaxedSpinlock roles[NUM_XCCS],
                                     RelaxedSemaphore sems[2 * NUM_XCCS],
                                     PingPongResult results[NUM_XCCS])
{
        if (threadIdx.x != 0) return;

        uint32_t xcc_id = chiplet::get_xcc_id ();
        uint32_t me = relaxed_spinlock_try_lock (&roles[xcc_id]) ? 0u : 1u;

        for (int i = 0; i < PINGPONG_NROUNDS; i++) {
                uint32_t sem_idx = xcc_id * 2 + (i % 2);

                if ((i % 2) == me) {
                        buffer[xcc_id] = i;
                        Fence::release ();
                        relaxed_sem_release (&sems[sem_idx]);
                } else {
                        relaxed_sem_acquire (&sems[sem_idx]);
                        long long start = clock64 ();
                        Fence::acquire ();
                        uint32_t observed = buffer[xcc_id];
                        long long end = clock64 ();

                        if (observed != (uint32_t)i)
                                results[xcc_id].mismatches++;
                        results[xcc_id].clock64_delta += end - start;
                }
        }
}

// ============================================================================
// Fence-Storm kernel
// ============================================================================

struct FenceStormResult {
        long long clock64_delta;
};

template <typename Fence>
__global__ void fence_storm_benchmark (uint32_t buffer[FENCE_STORM_BLOCK_SIZE],
                                       FenceStormResult *result)
{
        long long start = clock64 ();
        for (int i = 0; i < FENCE_STORM_NROUNDS; i++) {
                buffer[threadIdx.x] = threadIdx.x;
                Fence::release ();
                Fence::acquire ();
                __syncthreads ();
        }
        long long end = clock64 ();

        if (threadIdx.x == 0)
                result->clock64_delta = end - start;
}

// ============================================================================
// Host helpers
// ============================================================================

template <typename T>
T *to_device (const T *host, size_t count)
{
        void *deviceMem = nullptr;
        HIP_CHECK (hipMalloc (&deviceMem, count * sizeof (T)));
        HIP_CHECK (hipMemcpy (deviceMem, host, count * sizeof (T), hipMemcpyHostToDevice));
        return (T *)deviceMem;
}

template <typename T>
void to_host (T *host, const T *device, size_t count)
{
        HIP_CHECK (hipMemcpy (host, device, count * sizeof (T), hipMemcpyDeviceToHost));
}

template <typename T>
T *device_zero_init (size_t count)
{
        void *deviceMem = nullptr;
        HIP_CHECK (hipMalloc (&deviceMem, count * sizeof (T)));
        HIP_CHECK (hipMemset (deviceMem, 0, count * sizeof (T)));
        return (T *)deviceMem;
}

static int g_clock_khz = 0;

static void cache_clock_rate ()
{
        if (g_clock_khz == 0)
                HIP_CHECK (hipDeviceGetAttribute (&g_clock_khz, hipDeviceAttributeClockRate, 0));
}

void compute_stats (long long total_cycle_delta, size_t num_rounds, double &cycles_per_round, double &ns_per_round)
{
        cache_clock_rate ();
        cycles_per_round = (double)total_cycle_delta / num_rounds;
        ns_per_round     = cycles_per_round * 1e6 / g_clock_khz;
}

template <typename T>
T median (std::vector<T> v)
{
        std::sort (v.begin (), v.end ());
        return v[v.size () / 2];
}

// ============================================================================
// Fence-only launcher with median-of-SAMPLES
// ============================================================================

template <typename Fence>
void launch_fenceonly_benchmark ()
{
        long long *d_total_cycles = device_zero_init<long long> (1);
        uint32_t *d_sink          = device_zero_init<uint32_t> (1);

        auto run_one = [&](auto kernel, const char *label) {
                kernel<<<1, 64>>> (d_total_cycles, (volatile uint32_t *)d_sink); // warmup
                HIP_CHECK (hipDeviceSynchronize ());

                std::vector<long long> samples;
                samples.reserve (SAMPLES);
                for (int s = 0; s < SAMPLES; s++) {
                        kernel<<<1, 64>>> (d_total_cycles, (volatile uint32_t *)d_sink);
                        HIP_CHECK (hipDeviceSynchronize ());
                        long long h;
                        to_host (&h, d_total_cycles, 1);
                        samples.push_back (h);
                }

                long long med = median (samples);
                double cycles, ns;
                compute_stats (med, FENCE_NROUNDS, cycles, ns);
                printf ("  %-20s ns/round: %10.3lf  cycles/round: %10.3lf\n", label, ns, cycles);
        };

        run_one (fence_release_only_benchmark<Fence>,    "Release only");
        run_one (fence_acquire_only_benchmark<Fence>,    "Acquire only");
        run_one (fence_release_acquire_benchmark<Fence>, "Release+Acquire");

        HIP_CHECK (hipFree (d_total_cycles));
        HIP_CHECK (hipFree (d_sink));
}

// ============================================================================
// Producer-Consumer launcher with median-of-SAMPLES
// ============================================================================

template <typename Fence>
void launch_producer_consumer_benchmark ()
{
        RelaxedSpinlock h_roles[NUM_XCCS];
        RelaxedSemaphore h_sems[NUM_XCCS];

        std::vector<long long> samples[NUM_XCCS];
        for (int i = 0; i < NUM_XCCS; i++) samples[i].reserve (SAMPLES + 1);

        for (int s = -1; s < SAMPLES; s++) {
                for (int i = 0; i < NUM_XCCS; i++) {
                        host_relaxed_sem_init (&h_sems[i], 0);
                        host_relaxed_spinlock_init (&h_roles[i]);
                }

                uint32_t *d_buffer       = device_zero_init<uint32_t> (NUM_XCCS);
                RelaxedSpinlock *d_roles = to_device (h_roles, NUM_XCCS);
                RelaxedSemaphore *d_sems = to_device (h_sems, NUM_XCCS);

                ProducerConsumerResult h_init[NUM_XCCS] = {};
                ProducerConsumerResult *d_results = to_device (h_init, NUM_XCCS);

                producer_consumer_benchmark<Fence><<<16, 64>>> (d_buffer, d_roles, d_sems, d_results);
                HIP_CHECK (hipDeviceSynchronize ());

                ProducerConsumerResult h_results[NUM_XCCS];
                to_host (h_results, d_results, NUM_XCCS);

                HIP_CHECK (hipFree (d_buffer));
                HIP_CHECK (hipFree (d_results));
                HIP_CHECK (hipFree (d_roles));
                HIP_CHECK (hipFree (d_sems));

                if (s < 0) continue; // discard warmup

                for (int i = 0; i < NUM_XCCS; i++) {
                        bool pass = h_results[i].consumer_xcc == (uint32_t)i
                                 && h_results[i].consumer_xcc == h_results[i].producer_xcc;
                        if (!pass)
                                fprintf (stderr, "  XCC %d: FAIL on sample %d\n", i, s);
                        samples[i].push_back (h_results[i].clock64_delta);
                }
        }

        double total_ns = 0, total_cycles = 0;
        for (int i = 0; i < NUM_XCCS; i++) {
                long long med = median (samples[i]);
                double cycles, ns;
                compute_stats (med, 1, cycles, ns);
                printf ("  XCC %d: ns/round: %10.3lf  cycles/round: %10.3lf  (median of %d)\n",
                        i, ns, cycles, SAMPLES);
                total_ns += ns;
                total_cycles += cycles;
        }
        printf ("  Average over %d chiplets: ns/round: %10.3lf  cycles/round: %10.3lf\n",
                NUM_XCCS, total_ns / NUM_XCCS, total_cycles / NUM_XCCS);
}

// ============================================================================
// Ping-Pong launcher with median-of-SAMPLES + mismatch tracking
// ============================================================================

template <typename Fence>
void launch_ping_pong_benchmark ()
{
        RelaxedSpinlock h_roles[NUM_XCCS];
        RelaxedSemaphore h_sems[2 * NUM_XCCS];

        std::vector<long long> samples[NUM_XCCS];
        uint64_t total_mismatches[NUM_XCCS] = {};
        for (int i = 0; i < NUM_XCCS; i++) samples[i].reserve (SAMPLES + 1);

        for (int s = -1; s < SAMPLES; s++) {
                for (int i = 0; i < 2 * NUM_XCCS; i++) host_relaxed_sem_init (&h_sems[i], 0);
                for (int i = 0; i < NUM_XCCS; i++)     host_relaxed_spinlock_init (&h_roles[i]);

                uint32_t *d_buffer       = device_zero_init<uint32_t> (NUM_XCCS);
                RelaxedSpinlock *d_roles = to_device (h_roles, NUM_XCCS);
                RelaxedSemaphore *d_sems = to_device (h_sems, 2 * NUM_XCCS);

                PingPongResult h_init[NUM_XCCS] = {};
                PingPongResult *d_results = to_device (h_init, NUM_XCCS);

                ping_pong_benchmark<Fence><<<16, 64>>> (d_buffer, d_roles, d_sems, d_results);
                HIP_CHECK (hipDeviceSynchronize ());

                PingPongResult h_results[NUM_XCCS];
                to_host (h_results, d_results, NUM_XCCS);

                HIP_CHECK (hipFree (d_buffer));
                HIP_CHECK (hipFree (d_results));
                HIP_CHECK (hipFree (d_roles));
                HIP_CHECK (hipFree (d_sems));

                if (s < 0) continue;

                for (int i = 0; i < NUM_XCCS; i++) {
                        samples[i].push_back (h_results[i].clock64_delta);
                        total_mismatches[i] += h_results[i].mismatches;
                }
        }

        double total_ns = 0, total_cycles = 0;
        for (int i = 0; i < NUM_XCCS; i++) {
                long long med = median (samples[i]);
                double cycles, ns;
                // PINGPONG_NROUNDS / 2 because each thread is consumer for half the iterations
                compute_stats (med, PINGPONG_NROUNDS / 2, cycles, ns);
                printf ("  XCD %d: ns/round: %10.3lf  cycles/round: %10.3lf  mismatches: %llu  (median of %d)\n",
                        i, ns, cycles, (unsigned long long)total_mismatches[i], SAMPLES);
                total_ns += ns;
                total_cycles += cycles;
        }
        printf ("  Average over %d chiplets: ns/round: %10.3lf  cycles/round: %10.3lf\n",
                NUM_XCCS, total_ns / NUM_XCCS, total_cycles / NUM_XCCS);
}

// ============================================================================
// Fence-Storm launcher with median-of-SAMPLES
// ============================================================================

template <typename Fence>
void launch_fence_storm_benchmark ()
{
        std::vector<long long> samples;
        samples.reserve (SAMPLES + 1);

        for (int s = -1; s < SAMPLES; s++) {
                uint32_t *d_buffer        = device_zero_init<uint32_t> (FENCE_STORM_BLOCK_SIZE);
                FenceStormResult *d_result = device_zero_init<FenceStormResult> (1);

                fence_storm_benchmark<Fence><<<1, FENCE_STORM_BLOCK_SIZE>>> (d_buffer, d_result);
                HIP_CHECK (hipDeviceSynchronize ());

                FenceStormResult h_result;
                to_host (&h_result, d_result, 1);

                HIP_CHECK (hipFree (d_buffer));
                HIP_CHECK (hipFree (d_result));

                if (s < 0) continue; // discard warmup

                samples.push_back (h_result.clock64_delta);
        }

        long long med = median (samples);
        double cycles, ns;
        compute_stats (med, FENCE_STORM_NROUNDS, cycles, ns);
        printf ("  Fence Storm: ns/round: %10.3lf  cycles/round: %10.3lf  (median of %d)\n",
                ns, cycles, SAMPLES);
}

// ============================================================================
// main
// ============================================================================

int main ()
{
        hipDeviceProp_t prop;
        HIP_CHECK (hipGetDeviceProperties (&prop, 0));
        printf ("Device: %s, %d CUs, %d MHz\n\n", prop.name, prop.multiProcessorCount, prop.clockRate / 1000);

        printf ("Fence Only: Chiplet Fence\n");
        launch_fenceonly_benchmark<ChipletFence> ();
        printf ("Fence Only: Device Rel+Acq\n");
        launch_fenceonly_benchmark<DeviceRelAcqFence> ();
        printf ("Fence Only: Device Thread Fence\n");
        launch_fenceonly_benchmark<DeviceThreadFence> ();

        printf ("\nProducer-Consumer: Chiplet Fence\n");
        launch_producer_consumer_benchmark<ChipletFence> ();
        printf ("Producer-Consumer: Device Rel+Acq\n");
        launch_producer_consumer_benchmark<DeviceRelAcqFence> ();
        printf ("Producer-Consumer: Device Thread Fence\n");
        launch_producer_consumer_benchmark<DeviceThreadFence> ();

        printf ("\nPing-Pong: Chiplet Fence\n");
        launch_ping_pong_benchmark<ChipletFence> ();
        printf ("Ping-Pong: Device Rel+Acq\n");
        launch_ping_pong_benchmark<DeviceRelAcqFence> ();
        printf ("Ping-Pong: Device Thread Fence\n");
        launch_ping_pong_benchmark<DeviceThreadFence> ();

        printf ("\nFence Storm: Chiplet Fence\n");
        launch_fence_storm_benchmark<ChipletFence> ();
        printf ("Fence Storm: Device Rel+Acq\n");
        launch_fence_storm_benchmark<DeviceRelAcqFence> ();
        printf ("Fence Storm: Device Thread Fence\n");
        launch_fence_storm_benchmark<DeviceThreadFence> ();

        return 0;
}