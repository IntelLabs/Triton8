# VTune Profiler Integration — Example Output

This document shows real output from `skills/xpu_profiler.py` running Intel VTune on
Battlemage G21 (128 XVEs) hardware. The profiling tool collects OA (Observation
Architecture) hardware counters via `gpu-offload` collection and maps bottlenecks to
optimization recommendations from the knowledge base.

## Usage

```bash
python skills/xpu_profiler.py <triton_file> [--warmup 5] [--iters 20]
```

**Prerequisite**: OA counters require `observation_paranoid=0`:
```bash
echo 0 | sudo tee /proc/sys/dev/xe/observation_paranoid
```

## Two-Stage Collection

The profiler runs VTune in three stages:

1. **`gpu-offload` collection + `summary` report**: Captures Level Zero API tasks
   (host overhead) and basic GPU computing task list.
2. **`hotspots -group-by computing-task` report (Pass 1)**: Extracts per-kernel OA
   hardware counter columns — XVE execution, peak occupancy limiters, GPU memory
   bandwidth, L3 cache (busy/stalled/miss/BW), LSC miss ratios and pipeline stalls,
   instruction cache, SLM, TLB.
3. **`hotspots` report (Pass 2)**: Extracts columns that conflict with Pass 1 OA
   counter groups — measured XVE Threads Occupancy and LSC bandwidth. Results are
   merged by kernel name.

**VTune OA counter group conflicts**: Some metrics share hardware counters and cannot
be collected in the same report pass. Known conflicts:
- `XVE Threads Occupancy` vs `Peak XVE Threads Occupancy` (and its sub-limiters)
- `GPU L3:Average Bandwidth` vs `GPU Load Store Cache:Average Bandwidth`

If `observation_paranoid != 0`, stages 2-3 return no OA data and the profiler falls
back to summary-only metrics (less useful).

## Output Sections

### 1. Platform Info
GPU name, XVE count, max frequency — for verifying you're on the right hardware.

### 2. Host Tasks
CPU-side Level Zero API calls. Key items:
- `zeModuleCreate`: JIT compilation time (high = many autotune configs)
- `zeEventHostSynchronize`: Host waiting for GPU (high = unnecessary sync)
- `zeCommandListAppendMemoryCopy`: Data transfers

### 3. GPU Computing Tasks Table
Per-kernel summary with OA metrics. Overhead kernels (PyTorch Fill/Copy/Cast)
are marked with `*` and filtered from primary kernel selection.

Columns (when OA available):
| Column | Meaning |
|--------|---------|
| Time | Total execution time across all instances |
| Cnt | Number of kernel launches |
| Active | % of time XVE was executing instructions |
| Stall | % of time XVE was waiting (memory/dependency) |
| Idle | % of time XVE had no work scheduled |
| Occ% | Measured XVE thread occupancy |
| MemR | GPU memory read bandwidth (GB/s) |
| MemW | GPU memory write bandwidth (GB/s) |

### 4. Primary Kernel Detail
Full OA breakdown for the highest-time user compute kernel:

**Execution characterization**:
- XVE Active/Stalled/Idle: Where time is spent. Stalled > Active = memory bound.
- XVE Threads Occupancy: Measured thread count vs hardware max.

**Occupancy limiters** (tells WHY occupancy is low):
- Work Size Limit: Grid too small to fill the GPU
- SLM Use Limit: Kernel uses too much shared local memory per work group
- Barriers Use Limit: Too many synchronization points

**Cache hierarchy**:
- L3 Busy/Stalled: L3 cache utilization
- L3 Miss Ratio: Data streaming from VRAM (poor reuse)
- L3 BW Read/Write: L3 average bandwidth in GB/s
- L3 Input Available / Output Ready: L3 pipeline stall indicators (low = backpressure)
- LSC Miss Ratio: L1 load/store cache misses
- LSC→L3 Miss Ratio: L1 misses that also miss L3
- LSC BW Read/Write: LSC average bandwidth in GB/s (from Pass 2)
- LSC Input Available / Output Ready: LSC pipeline stall indicators
- LSC Partial Writes: Incomplete write coalescing

**Other**:
- Instruction Cache L3 Miss: High = compiled kernel binary too large for icache
- Spill Memory Size: Register file overflow (bytes spilled to memory)
- SLM Bank Conflicts: Shared local memory contention
- TLB Misses: Address translation overhead

### 5. Optimization Recommendations
Each recommendation is grounded in a specific KB pattern:

| Symptom | Diagnosis | KB Reference |
|---------|-----------|-------------|
| XVE Stalled > Active | Memory/dependency bound | `xpu_optimizations.yaml (xpu_descriptor_gemm_pattern)` + `optimization_levels.yaml (level_2)` |
| Low occupancy + Work Size limiter | Grid too small | `xpu_optimizations.yaml (xpu_tile_swizzling)` + `persistent_kernel_patterns.yaml` |
| Low occupancy + SLM limiter | Tile too large for SLM | `xpu_optimizations.yaml (xpu_grf_mode)` |
| Low occupancy + Barriers limiter | Too many syncs | `xpu_optimizations.yaml (xpu_warp_count)` |
| High L3 Miss Ratio | Poor cache reuse | `xpu_optimizations.yaml (xpu_descriptor_gemm_pattern, xpu_tile_swizzling)` |
| High LSC→L3 Miss | L1 thrashing | `xpu_optimizations.yaml (xpu_descriptor_gemm_pattern)` + `memory_patterns.yaml (mem_block_pointers)` |
| Spill > 0 bytes | Register pressure | `memory_patterns.yaml (reduce_liveness_sink_load_and_prefetch)` + `xpu_optimizations.yaml (xpu_grf_mode)` |
| Instruction cache L3 miss > 30% | Kernel binary too large | `xpu_optimizations.yaml (xpu_descriptor_gemm_pattern)` |
| Overhead kernels > 30% | Runtime type conversion | `optimization_levels.yaml (level_2_bandwidth_reduction)` |
| Host time >> GPU time | Hot path sync | `memory_patterns.yaml (no_device_to_host_scalar_sync)` |
| XVE Idle high, Active low | GPU underutilized | `persistent_kernel_patterns.yaml` |

## How the Agent Uses This Information

The profiling output feeds directly into the iterative trial loop
(see `CLAUDE.md` section 4f). Decision flow:

1. **Overhead kernels > 30%** → Apply Level 2: pre-pack weights and inputs to bf16
   at init time to eliminate PyTorch Fill/Copy/Cast ops.

2. **XVE Stalled > Active** → Memory bound. Apply Level 2 bandwidth reduction
   (bf16 pre-pack, tensor descriptors for better address codegen, tile swizzling for L3 locality).

3. **Low occupancy** → Read the limiter breakdown:
   - Work Size Limit low → increase grid (more tiles, persistent kernel)
   - SLM Use Limit low → reduce tile size or use `grf_mode='large'`
   - Barriers Use Limit low → reduce `num_warps` or restructure kernel

4. **High L3/LSC miss** → Data streaming from VRAM. Use tensor descriptors and
   tile swizzling to improve cache reuse.

5. **Register spill > 0** → Reduce variable liveness: sink `tl.load()` closer to
   `tl.dot()`, use `tl.prefetch()` to warm cache without holding registers.
   Also try `grf_mode='large'` for 256-register budget.

6. **No bottlenecks** → Kernel is well-optimized at hardware level. Remaining
   gains come from algorithmic changes (Level 3: algebraic fusion) or
   Level 4 techniques (Stream K, persistent kernels).

## Notes on VTune Overhead

VTune's Pin-based instrumentation adds significant overhead to elapsed time
(typically 10-20x for Python workloads). The `Elapsed Time` metric reflects this
instrumented runtime, not the real application runtime. Use the absolute GPU
compute and host task times for analysis, not the elapsed time ratio.
