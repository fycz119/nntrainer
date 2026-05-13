# CausalLM Qwen3-30B-A3B mobile optimization plan

This plan replaces the generic NEON roadmap with a phone-first plan for running
**Qwen3-30B-A3B** in CausalLM. The model is MoE, so the largest bottlenecks are
usually different from dense Qwen3 models: expert residency, expert loading from
flash, and single-token expert GEMV dominate before attention kernels do.

## Decision summary

**Best first optimization for phones:** use the cached-slim MoE implementation and
tune expert caching. For Qwen3-30B-A3B, this has the biggest practical TPS impact
because it reduces repeated expert loads from flash while keeping peak RAM below a
full 30B load.

**Best NEON kernel optimization:** implement `Q4_0 x FP16` single-token GEMV for
MoE expert projections and route expert `gate`, `up`, and `down` projections to
it. After expert-cache misses are controlled, active expert FC compute becomes the
main decode bottleneck.

**Lower priority for this model:** Q/K/V fusion and decode-attention fast paths
are still useful, but they should follow MoE cache and expert GEMV work. They are
more likely to be the top priority on dense 4B/7B models than on 30B-A3B MoE.

## Target scenario

- Device: Android phone, ARMv8-A NEON, big.LITTLE CPU, limited sustained thermal
  budget.
- Model: Qwen3-30B-A3B MoE, batch size 1, KV cache enabled, single-token decode.
- Precision: FP16 activations, Q4_0 FC/expert weights where accuracy is accepted.
- Primary metric: warmed steady-state decode TPS.
- Secondary metrics: first-token latency, prefill tokens/s, expert-cache hit rate,
  expert miss latency, peak RSS, storage read volume, and thermal throttling.

## Phase 0: make bottlenecks measurable first

Before changing kernels, add per-token and per-layer counters. Without these
counters, a phone run can look slow for many different reasons: expert-cache
misses, flash I/O, quantized dot performance, temporary tensor allocation, or CPU
thermal throttling.

Required counters:

| Counter | Purpose |
| --- | --- |
| `moe_cache_hit`, `moe_cache_miss`, `moe_cache_evict` | Shows whether TPS is I/O-bound. |
| `expert_activate_us` | Measures FSU/mmap/load latency on misses. |
| `router_us`, `topk_us` | Measures routing overhead. |
| `expert_gate_gemv_us`, `expert_up_gemv_us`, `expert_down_gemv_us` | Identifies FC/GEMV bottlenecks. |
| `swiglu_us`, `accumulate_us` | Finds elementwise and output accumulation overhead. |
| `attention_us`, `lm_head_us` | Confirms whether attention or LM head has become dominant. |
| `peak_rss_kb`, `cache_resident_experts` | Keeps cache tuning within phone memory budget. |

Implementation notes:

1. Keep counters behind a runtime flag or a lightweight profiling build flag.
2. Export one JSON line per generation run so Android benchmark scripts can
   compare before/after changes.
3. Report cold run and warmed average separately; cached-slim behavior differs
   greatly between the first tokens and warmed decode.

## Phase 1: cached-slim MoE as the default phone path

### 1.1 Use the cached-slim model variant

Use `qwen3_cached_slim_moe_causallm` for phone runs. The standard MoE model is
not the right default for memory-constrained phones because it assumes all expert
weights can be resident. The slim variant saves memory by loading experts on
demand; the cached-slim variant keeps recently activated experts mapped to reduce
repetitive storage I/O.

Recommended starting config:

```json
{
  "batch_size": 1,
  "model_tensor_type": "FP16-FP16",
  "embedding_dtype": "FP16",
  "fc_layer_dtype": "Q4_0",
  "fsu": true,
  "fsu_lookahead": 2,
  "moe_expert_cache_size": 32
}
```

### 1.2 Make expert-cache size configurable

The current cached path should not hard-code one cache size for all phones.
Expose a config/property such as `moe_expert_cache_size`, then sweep values by
available RAM and storage speed.

Suggested sweep:

| Phone memory class | Initial cache sweep |
| --- | --- |
| 8 GB RAM | 8, 16, 24, 32 experts |
| 12 GB RAM | 16, 24, 32, 48 experts |
| 16 GB+ RAM | 32, 48, 64 experts |

Acceptance target:

- warmed cache hit rate improves substantially;
- peak RSS stays below the phone-specific safety limit;
- sustained TPS improves after 3-5 minutes, not only in a cold burst.

### 1.3 Improve eviction and prefetch policy

Use router information to keep likely upcoming experts resident:

1. Keep true top-k experts for the current token.
2. Use extra router candidates as prefetch hints, not as guaranteed residents.
3. Update LRU order on every hit.
4. Add async prefetch only after counters show expert misses dominate; otherwise
   prefetch can steal CPU and memory bandwidth from GEMV.

Expected impact: this is the biggest phone-specific improvement if the model is
currently miss-heavy or repeatedly loading the same experts from flash.

## Phase 2: NEON `Q4_0 x FP16` expert GEMV

Once cache misses are controlled, the selected experts execute three projections:
`gate`, `up`, and `down`. These are matrix-vector operations in single-token
decode and should not dequantize full matrices before computing.

### 2.1 Add a direct quantized GEMV backend

Add a backend entry point similar to:

```cpp
void gemv_q4_0_fp16_neon(const void *packed_w,
                         const _FP16 *x,
                         _FP16 *y,
                         int rows,
                         int cols,
                         int row_begin,
                         int row_end);
```

Implementation requirements:

1. Start with Q4_0 because it is already the default FC quantization type.
2. Support the ARM-packed Q4_0 layout used by the quantizer.
3. Parallelize by output rows; each worker owns a contiguous row range.
4. Process 32-column quantization blocks, unpack nibbles with NEON, apply block
   scale, multiply by FP16 input, and accumulate in FP32 first for correctness.
5. Add an FP16-accumulate variant only after accuracy and speed are measured.
6. Keep a scalar reference path and unit tests for tails and non-ideal shapes.

### 2.2 Route MoE expert projections first

Do not wait for all `fully_connected` layers to use the new kernel. Add the first
call sites in the cached-slim MoE expert path:

- `expert_gate_proj`: `hidden_size -> moe_intermediate_size`
- `expert_up_proj`: `hidden_size -> moe_intermediate_size`
- `expert_down_proj`: `moe_intermediate_size -> hidden_size`

This gives the largest Qwen3-30B-A3B return because these projections run for the
active experts on every generated token.

Acceptance target:

- expert GEMV time decreases by at least 25% on a representative phone;
- end-to-end warmed TPS improves, not just microbenchmark GEMV;
- greedy decode quality stays within tolerance against the existing Q4_0 path.

## Phase 3: fuse the MoE expert hot path

After quantized GEMV works, reduce intermediate tensors and memory traffic.

### 3.1 Fuse gate and up projection scheduling

Gate and up projections consume the same token input. Compute both in one expert
hot path so the input vector is read once and outputs are produced in a layout
ready for SwiGLU.

### 3.2 Fuse SwiGLU and down input preparation

Current expert execution materializes gate output, activation output, up output,
and then runs down projection. A faster decode path should:

1. compute gate/up outputs;
2. apply `silu(gate) * up` in-place or into a single scratch buffer;
3. immediately feed that scratch into down GEMV;
4. accumulate weighted expert output directly into the final MoE output.

### 3.3 Remove per-token heap allocation in expert execution

Allocate scratch buffers once per layer/thread and reuse them across tokens.
Avoid creating new intermediate tensors for each active expert during generation.

Expected impact: medium to high after GEMV is fast, because allocation and memory
traffic become more visible.

## Phase 4: FP16 fast path cleanup

Keep Android decode on native FP16 tensors end-to-end. Any fallback that converts
Q/K/V/O between FP32 and FP16 inside attention should be treated as a benchmark
failure for the fast configuration.

Tasks:

1. Add a one-time warning or counter when the MHA FP32-to-FP16 temporary path is
   used.
2. Make the recommended benchmark config use FP16 activations everywhere.
3. If mixed precision is required, convert at the producer boundary rather than
   allocating temporary attention tensors every token.

Expected impact: low if the config is already pure FP16; medium/high if current
phone runs accidentally hit FP32 fallback copies.

## Phase 5: dense-model optimizations after MoE bottlenecks

Only after Phases 1-4 should Qwen3-30B-A3B prioritize dense-transformer style
work:

1. Q/K/V projection fusion for the non-expert attention block.
2. Decode attention fast path for `step_size == 1`.
3. RMSNorm + GEMV input packing.
4. LM head greedy/top-k fusion.
5. Thread affinity and big-core policy tuning.

These are still useful, but they are unlikely to beat expert cache tuning and
expert GEMV on a phone MoE workload.

## Benchmark protocol

Run every candidate with the same prompt and generated-token count.

Minimum runs:

1. cold run after process start;
2. one warmup run discarded;
3. three measured warmed runs;
4. one 3-5 minute sustained run for thermal stability.

Report:

- model variant and dtypes;
- `moe_expert_cache_size` and cache hit/miss/evict counts;
- prefill tokens/s and decode TPS;
- expert activation time and expert GEMV time;
- peak RSS;
- device model, CPU core count, Android version, and thermal note.

## Final landing order

1. Add profiling counters and benchmark JSON output.
2. Make `qwen3_cached_slim_moe_causallm` the recommended phone path.
3. Expose and tune `moe_expert_cache_size`; improve eviction/prefetch only when
   counters show misses dominate.
4. Implement NEON `Q4_0 x FP16` single-token GEMV.
5. Route cached-slim MoE expert `gate/up/down` projections to that GEMV.
6. Fuse gate/up + SwiGLU + down scratch usage.
7. Enforce native FP16 fast path and remove temporary attention conversions.
8. Then implement Q/K/V fusion, decode attention fast path, RMSNorm packing, and
   LM-head sampling optimizations.
