#import "@preview/cetz:0.4.2": canvas, draw, tree
#import "@preview/cetz-plot:0.1.3": plot, chart
#import "slides_template.typ": *


#set document(
  author: "David Yue, Leo Sciortino",
  date: datetime(year: 2026, month: 2, day: 23),
  title: "Chiplet-Level Release/Acquire on AMD MI300",
)
#show: conf

#let Accent = rgb("#2F5597")
#let Dark = rgb("#222222")
#let Light = rgb("#F7F7F7")
#let Red = rgb("#C00000")
#let Green = rgb("#00B050")
#set text(fill: Dark)

// ----------------------------
// Title slide
// ----------------------------
#title-slide(
  "Chiplet-Level Release/Acquire on AMD MI300X",
  "CS7810: Advanced Architecture: Final Project",
  "Presented By: David Yue, Leo Sciortino",
)

// ----------------------------
// Content
// ----------------------------
#body-frame("GPU Memory Models", subtitle: "Background", [
  - GPUs implement a relaxed consistency model allowing for low coherence overhead
  - Therefore to guarantee memory orderings we must use fence or release and acquire semantics
  #align(center)[*Release Acquire*]
  #grid(
    columns: (1fr, 0.2fr, 1fr),
    align: horizon,
    [
      #align(center)[Producer]
      #block(fill: luma(245), inset: 10pt, width: 100%, radius: 4pt)[
        `data = 42;` \
        `Release: flag = 1;` \
      ]
    ],
    [
      #v(1.2em) // Align arrow with the fence line
      #canvas(length: 1cm, {
        import draw: *
        line((-1, -2), (2.5, -1), stroke: (paint: red, thickness: 2pt), mark: (end: ">"))
        content((0.5, 0), anchor: "south", padding: .1, text(fill: red, size: 18pt)[`flag`])
      })
    ],
    [
      #align(center)[Consumer]
      #block(fill: luma(245), inset: 10pt, width: 100%, radius: 4pt)[
        `Acquire: while (flag != 1)` \
        `result = data;`
      ]
    ],
  )

  #v(-0.5em)
  #align(center)[
    #text(size: 18pt, style: "italic", fill: gray.darken(20%))[
      The Release fence ensures "data" is visible before "flag" is seen by the Acquire fence.
    ]
  ]
  - We target producer consumer since we commonly see this pattern in machine learning workloads that use reductions

])

#body-frame("AMD MI300X", subtitle: "Background", [
  #columns(2)[
    #figure(image("mi300.png", width: 80%), caption: [CDNA 3 Architecture \
      #text(
        size: 9pt,
        style: "italic",
      )[Source: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf]])


    #colbreak()
    #figure(
      image("mi300_block.png", width: 51%),
      caption: [MI300x \ #text(size: 9pt, style: "italic")[Source: https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html]
      ],
    )

  ]])
#body-frame(
  "The Multi-Chiplet Bottleneck",
  subtitle: "Problem Statement",
  [
    *The MI300X Specifications:*
    - 8 XCDs/XCCs (accelerated compute dies)
    - Each XCD has 28 CU (Compute units, known as SMs on Nvidia)
    - 32KB L1 per CU
    - 4MB shared L2 across CUs on the same XCD
    - 256MB LLC aka MALL (memory attached LLC)
    - 192 GB of HBM3 shared coherently between CPU and GPU
    - Each XCC has its own L2 cache //; Infinity Fabric connects these partitions.

  ],
)

#body-frame("Current Fences and the Problem", subtitle: "", [
  *Current HIP Fences*
  #slide-footnote[https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html]
  - `__threadfence_block()`: orders memory accesses for all threads within a thread block
  - `__threadfence()`: orders memory accesses for all threads on a device.
  - `__threadfence_system()`: orders memory accesses for all threads in the system, making writes to memory visible to other devices and the host

  *The Performance Gap:*
  - *The Cost:* Device scope of `__threadfence()` triggers fabric-wide probes and L2 flushes to HBM.
  - *Observation:* We can force kernels to exhibit high spatial locality within an XCD
])

#body-frame(
  "The Chiplet Fence",
  subtitle: "Solution",
  [
    Develop a *Software-Defined Chiplet Fence* that provides Release/Acquire semantics restricted to the XCC boundary.

    *Motivation*: Imagine a workload that *spans multiple chiplets* that also performs heavy reductions. If the reductions can be *done locally at the chiplet level*, we can reduce data movement across chiplets by reducing locally instead of globally (device level)$arrow$ this requires a chiplet level fence.

    *Key Innovation:*
    - Use `XCC_ID` state registers to detect locality at runtime.
    - Exploit L2 as the unified Point of Coherency for all CUs on a die.
    - Avoid L2 $arrow$ HBM write-backs by keeping synchronization data in-cache.
  ],
)
#slide[
  #align(center + horizon)[
    We looked at the *_AMD Instinct MI300_ Instruction Set Architecture Reference Guide*. Here is what we found...
  ]
]
#body-frame("MI300 Chiplet Hierarchy & Fence Mechanism", subtitle: "XCC Logical Layout & Sync Path")[
  #set text(size: 20pt) // Smaller base size for the diagram labels
  #align(center + horizon)[
    #canvas({
      import draw: *
      // --- Data Flow Arrows ---
      let arrow_p = (stroke: 2pt + blue, mark: (end: ">"))
      let arrow_c = (stroke: 2pt + orange, mark: (end: ">"))

      // --- COORDINATE SCALE: (-20, -10) to (20, 10) ---

      // Outer Chiplet Boundary
      rect((-18, -7), (18, 7), stroke: (thickness: 2pt, paint: Dark), fill: Light, name: "xcc")
      content((0, 6.2), [*XCC (Accelerator Complex Die)*])

      // Shared L2 Cache - The Point of Coherency
      rect((-14, -5.5), (14, -3), fill: Accent.lighten(80%), stroke: (thickness: 2pt, paint: Accent), name: "l2")
      content("l2", [*Shared L2 Cache (Point of Coherency)*])

      // Compute Unit 0 (Producer)
      rect((-16, -2), (-1, 5), stroke: (dash: "dashed", paint: gray), name: "cu0")
      content((-8.5, 4.3), [CU 0 (Producer)])

      // Caches for CU 0
      rect((-14.5, -1), (-9, 2.5), fill: white, stroke: Red, name: "s0")
      content("s0", [Scalar L1\ #text(size: 16pt)[(Write-Back)]])
      line((-11.7, -1), (-11.7, -3), ..arrow_p)

      content((-11.7, -2), frame: "rect", fill: white, stroke: none, padding: .1, text(
        size: 14pt,
        fill: blue.darken(20%),
      )[`S_ATOMIC_*`])

      rect((-8, -1), (-2.5, 2.5), fill: white, stroke: Green, name: "v0")
      content("v0", [Vector L1\ #text(size: 16pt)[(Write-Thru)]])

      // Compute Unit 1 (Consumer)
      rect((1, -2), (16, 5), stroke: (dash: "dashed", paint: gray), name: "cu1")
      content((8.5, 4.3), [CU 1 (Consumer)])

      // Caches for CU 1
      rect((2.5, -1), (8, 2.5), fill: white, stroke: Red, name: "s1")
      content("s1", [Scalar L1\ #text(size: 16pt)[(Write-Back)]])


      rect((9, -1), (14.5, 2.5), fill: white, stroke: Green, name: "v1")
      content("v1", [Vector L1\ #text(size: 16pt)[(Write-Thru)]])


      // Producer: Release Path
      line((-5.25, -1), (-5.25, -3), ..arrow_p)
      content((-5.25, -2), frame: "rect", fill: white, stroke: none, padding: .1, text(
        size: 14pt,
        fill: blue.darken(20%),
      )[`S_WAITCNT VSCNT(0)`])

      // Consumer: Acquire Path
      line((11.75, -3), (11.75, -1), ..arrow_c)
      content((11.75, -2), frame: "rect", fill: white, stroke: none, padding: .1, text(
        size: 14pt,
        fill: orange.darken(20%),
      )[`GLOBAL_LOAD`])

      // --- The "Barrier" - Infinity Fabric ---
      line((-18, -8.5), (18, -8.5), stroke: (thickness: 4pt, paint: gray.lighten(50%)), name: "fabric")
      content((0, -8.5), frame: "rect", fill: white, stroke: gray, [*MALL/HBM (Global Scope - AVOID)*])

      // Connection to Fabric (Show it's NOT used)
      line((0, -5.5), (0, -8.2), stroke: (dash: "dashed", paint: Red), mark: (end: "x"))
      content((0, -6.25), text(size: 14pt, fill: Red)[*Global Scope (L2 Flush to HBM)*])
    })
  ]
]

#body-frame(
  "Instruction Types",
  subtitle: "ISA Reference",
  [
    There are 2 types of instructions: scalar instructions, vector instructions
    - Vector Instruction: Generally used for high throughput and when each thread needs a different piece or data
      - Global Instructions: on-device HBM, host memory
      - Buffer Instructions: local data share (LDS) memory
      - Scratch Instructions: per thread private memory
      - Flat Instructions: views all memory units above as a single address space
    - Scalar instructions are unified.
      - Generally used when all threads need the same data broadcasted (loop bounds, locks, kernel parameters, locks)
  ],
)


// #body-frame(
//   "Instruction Breakdown",
//   subtitle: "ISA Reference",
//   [
//     #set text(size: 18pt)
//     #table(
//       columns: (1.5fr, 1.5fr, 4fr),
//       inset: 10pt,
//       align: horizon,
//       stroke: 0.5pt + Dark,
//       [*Instruction*], [*Hierarchy Level*], [*Functional Role in the Fence*],
//       [`S_WAITCNT VMCNT(0)`],
//       [CU $arrow$ L2],
//       [*The Producer Fence:* Pauses execution until all vector stores have cleared the CU's write-combining buffers and reached the XCC's shared L2.],

//       [`S_DCACHE_INV`],
//       [Scalar L1],
//       [*Scalar Invalidation:* Manually clears the Scalar Data Cache. Necessary because the S-Cache is Write-Back and not hardware-coherent with vector stores.],

//       [`S_WAITCNT LGKMCNT(0)`],
//       [Control Flow],
//       [*Invalidation Gate:* Ensures the `S_DCACHE_INV` or `S_ICACHE_INV` operation has completed before the next scalar/instruction load is issued.],

//       [`GLOBAL_LOAD (GLC=1)`],
//       [Vector L1],
//       [*Invalidate-on-Read:* Forces the CU to ignore the current Vector L1 line and fetch fresh data from L2, then populates L1 for subsequent hits.],

//       [`S_GETREG (HW_ID1)`],
//       [CU State],
//       [*Topology Awareness:* Reads the physical `XCC_ID`. Used at runtime to decide between the "Fast Chiplet Fence" or a "Global Fabric Fence."],

//       [`SC0` Bit (Scope)],
//       [L2 / Die],
//       [*Traffic Control:* Sets scope to *Workgroup*. On MI300, this restricts coherency logic to the local L2, preventing Infinity Fabric probes.],
//     )
//   ],
// )

#body-frame(
  "Release/Acquire Implementation",
  [*Release*
    - `S_WAITCNT VMCNT(0)`: flushes L1 Vector stores to L2 (L1 vector cache is write through)
    - `S_DCACHE_WB`: write back from scalar L1 data cache to L2 (L1 scalar cache is write back)
    - `S_WAITCNT LGKM_CNT(0)`: wait for `S_DCACHE_WB` to finish
    // - `S_Store`: Stores to L2 (write through)
    *Acquire*
    - `BUFFER_INV`: Invalidates the L1 Vector Cache
    - `S_DCACHE_INV`: Invalidate the L1 Scalar/Constant Cache
    - `S_WAITCNT LGKM_CNT(0) VMCNT(0)`: Waits on the `S_DCACHE_INV` and `BUFFER_INV`
  ],
)
#slide[
  #align(horizon)[
    #text(fill: Accent)[#heading("How do we spin on a flag at the chiplet level?")]

    We need a way to write and read directly from L2. Need to find a way to implement a synchronization flag.
  ]
]
#body-frame("Can we use scalar memory instructions for flags?", subtitle: "Synchronization Flag", [

  Yes, we use scalar atomics with Global Level Coherence microcode bit set. Atomics will force a miss on L1 and read from L2 or data fabric. The ISA manual says: "No L1 persistence across wavefronts"

  // However, these atomic instructions are asynchronous, must still wait for completion with `S_WAITCNT LGKM_CNT(0)`.

  Do we have to spin? Can we sleep instead? No, there is no sleep/wakeup support across work groups, only within a work group
])

// #body-frame("XCD-Aware Fencing: Implementation & Plan", [
//   #set text(size: 25pt)
//   #grid(
//     columns: (1.2fr, 1fr),
//     column-gutter: 2em,
//     [
//       === XCD Explicit Work Scheduling
//       - *Manual Dispatch:* Can no longer naively let the scheduler handle WG/TB placement.
//       - *Logic:* Code must detect if a TB is on a specific XCD to assign work.
//       - *Risks:* - Potential deadlocks if occupancy is too high.
//         - Inefficiency with uneven workloads.

//       === Chiplet Swizzling
//       - *Strategy:* Remap TB IDs so the round-robin scheduler (ideally) stacks work on the same XCD.
//       - *Issue:* No hardware guarantee; correctness issues persist.
//     ],
//     [
//       === Plan / Deliverables
//       - *ISA Inspection:* Analyze assembly generated by current `threadfence`.
//       - *Implementation:* Write the chiplet-level release/acquire assembly sequence.
//       - *Micro-benchmarking:* Validate latency/throughput of the new sequence.
//       - *Kernel Integration:* Test in real reduction kernels to measure speedup.
//     ],
//   )
// ])
// #body-frame("Don't we need WG/TB on the same XCD to use chiplet relase/acquire?",
//   [
//     *XCD explicit work Scheduling*

//     - We can no longer naively write thread blocks and let the WG/TB scheduler dispatch our work
//     - In the code we now need to say if WG/TB is on a specific XCD do some specific work
//     - This may not work well for uneven workloads
//     - Could also see deadlock if occupancy is too high
//     *Chiplet Swizzling*
//     - remap TB/WG ids such that round robin scheduler most likely puts two TB/WG on the same XCD
//     - This has correctness issues since this isn't guaranteed
//    ]
// )

// #body-frame("Plan/Deliverables", [
//   - Inspect the assembly that current thread fences generate
//   - Implement our chiplet level release/acquire assembly sequence
//   - Micro benchmark our new release/acquire
//   - Test it in real reduce kernels
// ])
// ----------------------------
// Results
// ----------------------------
#slide[
  #align(center + horizon)[
    #text(fill: Accent)[#heading("Results", depth: 1)]
  ]
]


#body-frame("Same XCD Producer-Consumer Test Across Blocks", subtitle: "Litmus Tests")[
  #set text(size: 23pt)
  #columns(2)[
  The following code is run 10,000 times. It passes this litmus test.

  Things to note:
  - HIP Atomics are relaxed by default. #slide-footnote[https://github.com/ROCm/HIP/blob/master/include/hip/hcc_detail/hip_atomic.h] So the chiplet fence is providing the memory ordering, not the atomic write.
  - The consumer writes it's own XCC ID as the data. We later check that what the consumer wrote matches what the producer wrote.
    
  #colbreak()
  ```
  if threadIdx.x != 0:
    return
    
  xcc_id = get_xcc_id()

  Wait for 2 thread blocks to reach this point
  
  if blockIdx.x == 0:
    write (xcc_id)
    chiplet_release()
    atomic write: flag = 1
  else:
    while flag != 1:
      continue

    chiplet_acquire()
    producer_xcc_id = read()
  ```
]
  
]

#body-frame("Chiplet Fence Latency Benchmark", subtitle: "Microbenchmarks")[
  We run the following code where `op` is one of `chiplet_release`, `chiplet_acquire` and both.
  
  ```
  start = timestamp()
  for (i = 0; i < 1024; i++)
    op()
  end = timestamp()
  ```
]

#body-frame("Chiplet Fence Latency Results", subtitle: "Microbenchmark Results")[
  #set text(size: 22pt)
  1024 iterations per measurement. Averaged over 10 samples.

  #align(center)[
    #canvas(length: 1cm, {
      chart.columnchart(
        size: (24, 8),
        label-key: 0,
        value-key: 1,
        bar-style: i => (stroke: none, fill: if i == 0 { Accent } else if i == 1 { Red } else if i == 2 { Red.lighten(30%) } else { gray }),
        x-label: [Fence Type],
        y-label: [Cycles / op],
        (
          ([Release], 50.6),
          ([Acquire], 346.6),
          ([Rel+Acq], 402.5),
        ),
      )
    })
  ]

  // #text(size: 18pt, style: "italic", fill: gray.darken(20%))[
  //   Device fence shows 0 cycles in isolation --- its cost manifests in cross-CU synchronization (see Ping-Pong).
  // ]
]

#body-frame("Ping Pong Latency Benchmark", subtitle: "Microbenchmarks")[
  We run the following code where `op` is one of `chiplet_release`, `chiplet_acquire` and both. This measures the round trip latency of a release-acquire handshake. A similar benchmark is run with a device fence, `__threadfence()`.
  
  ```
  me = blockIdx.x
  for (round = 0; round < 512; round++)
    producer = round & 1

    if me == producer:
      write(round)
      chiplet_release()
    else:
      chiplet_acquire()
      read()
      
    
  ```
]

#body-frame("Ping-Pong Latency", subtitle: "Microbenchmark Results")[
  #set text(size: 22pt)
  512 rounds, two workgroups exchanging a flag

  #grid(
    columns: (1fr, 1fr),
    column-gutter: 2em,
    [
      #align(center)[
        *Cycles / round*
        #canvas(length: 1cm, {
          chart.columnchart(
            size: (10, 8),
            label-key: 0,
            value-key: 1,
            bar-style: i => (stroke: none, fill: if i == 0 { Accent } else { Red }),
            y-label: [Cycles],
            (
              ([Chiplet], 2713.2),
              ([Device], 4016.2),
            ),
          )
        })
      ]
    ],
    [
      #align(center)[
        *Nanoseconds / round*
        #canvas(length: 1cm, {
          chart.columnchart(
            size: (10, 8),
            label-key: 0,
            value-key: 1,
            bar-style: i => (stroke: none, fill: if i == 0 { Accent } else { Red }),
            y-label: [ns],
            (
              ([Chiplet], 1292.0),
              ([Device], 1912.5),
            ),
          )
        })
      ]
    ],
  )

  #align(center)[
    #text(size: 28pt, fill: Green)[*1.48x faster* with chiplet fence]
  ]
]
#body-frame("Fence Storm Latency Benchmark", subtitle: "Microbenchmarks")[
  One thread in each block writes some data to a shared memory region (non overlapping writes). Then issues a chiplet release and acquire. Stress the L2 cache and measure performance. 
  ```
  for (i = 0; i < 256; i++)
    if threadIdx.x == 0:
      write to shared region
      chiplet_release();
      chiplet_acquire();

    syncthreads() // sync entire block.
    
  ```
  This code is run with 1,4,16,64 and 128 blocks. A similar benchmark is made for `__threadfence()`
]


#body-frame("Fence Storm", subtitle: "Microbenchmark Results")[
  #set text(size: 20pt)
  256 iterations, varying number of concurrent workgroups (cycles/op)

  #align(center)[
    #canvas(length: 1cm, {
      plot.plot(
        size: (24, 9),
        x-label: [Workgroups],
        y-label: [Cycles / op],
        x-tick-step: none,
        x-ticks: ((0, [1]), (1, [4]), (2, [16]), (3, [64]), (4, [128])),
        y-tick-step: 1000,
        y-min: 0,
        y-max: 7000,
        legend: "north-west",
        {
          plot.add(
            ((0, 691.2), (1, 695.0), (2, 737.7), (3, 1138.9), (4, 2134.9)),
            mark: "o",
            mark-size: 0.2,
            style: (stroke: (paint: Accent, thickness: 3pt)),
            label: [Chiplet Fence],
          )
          plot.add(
            ((0, 849.6), (1, 788.7), (2, 1237.7), (3, 4036.6), (4, 6188.1)),
            mark: "triangle",
            mark-size: 0.2,
            style: (stroke: (paint: Red, thickness: 3pt)),
            label: [Device Fence],
          )
        },
      )
    })
  ]

  #align(center)[
    At 128 workgroups: #text(fill: Green)[*2.9x fewer cycles*] with chiplet fence
  ]
]

// #body-frame("Chiplet Reduction Benchmark", subtitle: "Microbenchmarks")[
  
//   ```
//   float blockPartials[MAX_BLOCKS]
  
  
//   for (i = blockIdx.x; i < N; i += totalBlockCount)
//     sum += input[i]

//   blockPartials[]
    
//   ```
// ]

// #body-frame("Reduction Throughput", subtitle: "Microbenchmark Results")[
//   #set text(size: 20pt)
//   hipEvent timing, 256 workgroups

//   #align(center)[
//     #canvas(length: 1cm, {
//       plot.plot(
//         size: (24, 9),
//         x-label: [Elements],
//         y-label: [Time (µs)],
//         x-tick-step: none,
//         x-ticks: ((0, [1K]), (1, [4K]), (2, [16K]), (3, [64K]), (4, [256K]), (5, [1M])),
//         y-min: 0,
//         y-max: 400,
//         y-tick-step: 50,
//         legend: "north-west",
//         {
//           plot.add(
//             ((0, 5.89), (1, 6.45), (2, 10.03), (3, 25.96), (4, 85.89), (5, 328.75)),
//             mark: "o",
//             mark-size: 0.2,
//             style: (stroke: (paint: Accent, thickness: 3pt)),
//             label: [Chiplet Fence],
//           )
//           plot.add(
//             ((0, 5.77), (1, 6.97), (2, 11.26), (3, 30.04), (4, 97.71), (5, 367.20)),
//             mark: "triangle",
//             mark-size: 0.2,
//             style: (stroke: (paint: Red, thickness: 3pt)),
//             label: [Device Fence],
//           )
//           plot.add(
//             ((0, 5.85), (1, 6.15), (2, 9.84), (3, 25.52), (4, 85.51), (5, 327.93)),
//             mark: "square",
//             mark-size: 0.2,
//             style: (stroke: (paint: gray, thickness: 2pt, dash: "dashed")),
//             label: [No Fence],
//           )
//         },
//       )
//     })
//   ]

//   #align(center)[
//     Chiplet fence tracks close to no-fence baseline; device fence diverges at scale
//   ]
// ]


// #body-frame("Hierarchical Reduction Benchmark", subtitle: "Hierarchical Reduce Results")[
//   #set text(size: 20pt)
//   2-level (chiplet + device fence) vs 1-level (device-only fence)

//   #align(center)[
//     #canvas(length: 1cm, {
//       plot.plot(
//         size: (24, 9),
//         x-label: [Configuration (Elements / Blocks)],
//         y-label: [Time (µs)],
//         x-tick-step: none,
//         x-ticks: ((0, [1K/4]), (1, [4K/16]), (2, [16K/64]), (3, [64K/256]), (4, [256K/256]), (5, [1M/256])),
//         y-min: 0,
//         y-max: 500,
//         y-tick-step: 100,
//         legend: "north-west",
//         {
//           plot.add(
//             ((0, 20.39), (1, 32.53), (2, 33.25), (3, 36.67), (4, 114.09), (5, 423.59)),
//             mark: "o",
//             mark-size: 0.2,
//             style: (stroke: (paint: Accent, thickness: 3pt)),
//             label: [2-level (chiplet+device)],
//           )
//           plot.add(
//             ((0, 18.91), (1, 32.25), (2, 39.22), (3, 66.70), (4, 145.33), (5, 453.79)),
//             mark: "triangle",
//             mark-size: 0.2,
//             style: (stroke: (paint: Red, thickness: 3pt)),
//             label: [1-level (device-only)],
//           )
//         },
//       )
//     })
//   ]

//   #align(center)[
//     Peak: #text(fill: Green)[*1.82x speedup*] at 64K / 256 blocks (4M excluded for scale)
//   ]
// ]

// #body-frame("Hierarchical Reduction: Speedup", subtitle: "Hierarchical Reduce Results")[
//   #set text(size: 20pt)
//   Speedup of 2-level over 1-level (>1.0 = chiplet fence wins)

//   #align(center)[
//     #canvas(length: 1cm, {
//       plot.plot(
//         size: (24, 9),
//         x-label: [Configuration],
//         y-label: [Speedup],
//         x-tick-step: none,
//         x-ticks: ((0, [1K/4]), (1, [4K/16]), (2, [16K/64]), (3, [64K/256]), (4, [256K/256]), (5, [1M/256]), (6, [4M/256])),
//         y-min: 0.8,
//         y-max: 2.0,
//         y-tick-step: 0.2,
//         {
//           // Baseline at 1.0
//           plot.add(
//             ((-0.5, 1.0), (6.5, 1.0)),
//             style: (stroke: (paint: gray, thickness: 1.5pt, dash: "dashed")),
//           )
//           // Speedup line
//           plot.add(
//             ((0, 0.93), (1, 0.99), (2, 1.18), (3, 1.82), (4, 1.27), (5, 1.07), (6, 1.01)),
//             mark: "o",
//             mark-size: 0.25,
//             style: (stroke: (paint: Green, thickness: 3pt)),
//             fill: false,
//           )
//         },
//       )
//     })
//   ]

//   #align(center)[
//     Sweet spot at medium sizes where cross-XCD synchronization dominates
//   ]
// ]

// #body-frame("Block Count Sweep", subtitle: "Hierarchical Reduce Results")[
//   #set text(size: 20pt)
//   N = 262,144 elements fixed, varying block count

//   #grid(
//     columns: (1fr, 1fr),
//     column-gutter: 1em,
//     [
//       #align(center)[
//         *Execution Time*
//         #canvas(length: 1cm, {
//           plot.plot(
//             size: (11, 8),
//             x-label: [Blocks],
//             y-label: [Time (µs)],
//             x-tick-step: none,
//             x-ticks: ((0, [8]), (1, [16]), (2, [32]), (3, [64]), (4, [128]), (5, [256])),
//             y-min: 0,
//             y-max: 2500,
//             y-tick-step: 500,
//             legend: "north-east",
//             {
//               plot.add(
//                 ((0, 2284.31), (1, 1661.06), (2, 834.24), (3, 420.88), (4, 215.16), (5, 114.03)),
//                 mark: "o",
//                 mark-size: 0.2,
//                 style: (stroke: (paint: Accent, thickness: 3pt)),
//                 label: [2-level],
//               )
//               plot.add(
//                 ((0, 2282.00), (1, 1660.96), (2, 835.86), (3, 426.71), (4, 229.60), (5, 145.30)),
//                 mark: "triangle",
//                 mark-size: 0.2,
//                 style: (stroke: (paint: Red, thickness: 3pt)),
//                 label: [1-level],
//               )
//             },
//           )
//         })
//       ]
//     ],
//     [
//       #align(center)[
//         *Speedup*
//         #canvas(length: 1cm, {
//           plot.plot(
//             size: (11, 8),
//             x-label: [Blocks],
//             y-label: [Speedup],
//             x-tick-step: none,
//             x-ticks: ((0, [8]), (1, [16]), (2, [32]), (3, [64]), (4, [128]), (5, [256])),
//             y-min: 0.95,
//             y-max: 1.35,
//             y-tick-step: 0.05,
//             {
//               // Baseline
//               plot.add(
//                 ((-0.5, 1.0), (5.5, 1.0)),
//                 style: (stroke: (paint: gray, thickness: 1.5pt, dash: "dashed")),
//               )
//               plot.add(
//                 ((0, 1.00), (1, 1.00), (2, 1.00), (3, 1.01), (4, 1.07), (5, 1.27)),
//                 mark: "o",
//                 mark-size: 0.25,
//                 style: (stroke: (paint: Green, thickness: 3pt)),
//                 fill: false,
//               )
//             },
//           )
//         })
//       ]
//     ],
//   )

//   #align(center)[
//     Speedup scales with block count --- more blocks $arrow$ more cross-XCD traffic saved
//   ]
// ]

#slide([
  #align(center + horizon)[
    #text(fill: Accent)[#heading("Questions & Discussion", depth: 1)]
    Presented By: David Yue & Leo Sciortino
  ]
])
