# CausalLM mobile NEON TPS optimization plan

This document proposes an implementation-oriented plan to raise CausalLM tokens per second (TPS) on Android phones with ARM NEON. It favors changes that are easy to land first while still targeting the kernels that dominate decode-time latency.

## Goals and assumptions

- **Target path:** batch size 1, causal generation with KV cache, Android ARMv8-A NEON, FP16 enabled where available.
- **Primary metric:** steady-state decode TPS after prefill. Track prefill tokens/s separately because it has different bottlenecks.
- **Accuracy guardrail:** each optimized kernel must compare against the existing FP32/FP16 implementation with per-layer tolerances and end-to-end greedy output checks.
- **Current useful hooks:** CausalLM already records prefill/generation timing and peak memory, supports quantized FC weights, uses persistent KV cache tensors, and parallelizes decode attention over KV heads.

## Priority matrix

| Priority | Optimization | Ease | Expected TPS impact | Why first |
| --- | --- | --- | --- | --- |
| P0 | Make the default Android recipe use native FP16 activations plus Q4_0 FC weights | Very easy | High on memory-bound phones | Mostly config/build guidance; avoids avoidable FP32 traffic and uses existing quantization. |
| P0 | Add a repeatable on-device TPS benchmark preset | Easy | Indirect but critical | Prevents regressions and shows which kernel wins on real phones. |
| P1 | Specialize single-token quantized GEMV for NEON and route FC layers to it | Medium | Very high | Decode is dominated by matrix-vector projections/MLP; direct int4/FP16 dot avoids dequantizing whole rows. |
| P1 | Fuse Q/K/V projection for decode | Medium | High | Reads the hidden state once and writes contiguous QKV once for every layer. |
| P1 | Remove transient FP32→FP16 copies in MHA decode | Easy/Medium | Medium to high if configs still feed FP32 | Current Android path may allocate/copy Q/K/V/O temporary FP16 tensors when inputs arrive as FP32. |
| P2 | Fused RMSNorm + linear input packing | Medium | Medium | RMSNorm is small but sits before every projection; packing once improves following GEMV. |
| P2 | Decode attention fast path for `step_size == 1` | Medium | Medium | Compute `QK`, softmax, and `AV` without triangle tensor setup and with cache-friendly K/V traversal. |
| P2 | Fused SwiGLU MLP decode | Medium/Hard | High | Avoids materializing intermediate gate/up tensors and improves cache reuse. |
| P3 | LM head top-k/top-p partial reduction | Medium | Medium | For large vocabulary, final logits and sampling can be non-trivial. |
| P3 | Threading and CPU-affinity policy | Medium | Device-dependent | Big-core pinning helps, but must not overheat or hurt latency consistency. |

## P0: easiest high-return changes

### 1. Ship a recommended Android inference configuration

Recommended defaults for phone decode:

```json
{
  "batch_size": 1,
  "model_tensor_type": "FP16-FP16",
  "embedding_dtype": "FP16",
  "fc_layer_dtype": "Q4_0",
  "fsu": false,
  "fsu_lookahead": 2
}
```

Implementation tasks:

1. Add a sample `nntr_config.android_neon_fast.json` for one small CausalLM model directory or document these settings in the model README.
2. Ensure the Android build command includes `-Denable-fp16=true` and compiles with NEON flags.
3. Quantize on the target ARM platform with `nntr_quantize --fc_dtype Q4_0 --embd_dtype FP16 --lmhead_dtype FP16` to avoid the known platform-specific Q4_0 mismatch.
4. Validate with greedy decoding on a short fixed prompt and record prefill/generation metrics.

Why this is first: it mostly reuses existing code, reduces weight bandwidth immediately, and avoids accidental FP32 activation paths on phones.

### 2. Add a reproducible TPS benchmark preset

Minimum benchmark output per run:

- device model, Android version, CPU governor if available;
- model path and key dtypes;
- prompt token count, generated token count;
- prefill ms/tokens/s, generation ms/TPS, peak memory;
- thermals note: cold run and 3-run warmed average.

Implementation tasks:

1. Extend the existing Android benchmark script or add a preset wrapper that always runs the same prompt and `num_to_generate`.
2. Store results as JSON/CSV so kernel changes can be compared before/after.
3. Add a `--warmup` option and discard the first generation run.

## P1: largest decode TPS wins

### 3. NEON single-token quantized GEMV for FC layers

Decode-time FC layers are matrix-vector operations. The highest leverage kernel is a direct `Q4_0/Q4_K weight x FP16 activation -> FP16/FP32 output` NEON GEMV that accumulates without expanding an entire matrix to FP16/FP32.

Implementation sketch:

1. Add a CPU backend entry point such as `gemv_q4_0_fp16_neon(const void *packed_w, const _FP16 *x, _FP16 *y, int rows, int cols)`.
2. Start with Q4_0 because it is already the default FC quantization type and has ARM-specific packing.
3. Use row-blocked output parallelism: each worker owns a contiguous output row range; inside each row, process 32 quantized columns per block.
4. Use NEON to unpack nibbles, convert to signed int4, multiply by block scale and FP16 activations, then accumulate in FP32 or FP16 depending on measured accuracy/TPS.
5. Add a scalar reference and unit tests for dimensions not divisible by the vector tile; keep a generic fallback for unsupported CPUs.
6. Route `fully_connected` inference to this kernel when `batch == 1`, activation dtype is FP16, and weight dtype is Q4_0.

Expected impact: very high, because every decoder block has Q/K/V/O projections and MLP projections. Even a 1.3x FC speedup often becomes a visible end-to-end TPS gain.

### 4. Fuse Q/K/V projection in decoder blocks

Current model builders add separate fully connected layers for Q, K, and V. For decode, all three consume the same input vector. A fused projection avoids reading the hidden state three times and reduces layer scheduling overhead.

Implementation sketch:

1. Introduce an optional `qkv_fused` layer used only in inference configs or model builders.
2. Concatenate packed Q/K/V weights in output-row order and compute all output rows in one GEMV call.
3. Preserve existing separate-layer path for training and fallback.
4. For GQA/MQA, keep K/V row counts smaller than Q but still emit the existing Q/K/V tensor views.

Expected impact: high for small/medium models where per-layer overhead and hidden-state reloads are significant.

### 5. Remove FP32-to-FP16 temporary copies in MHA

The MHA decode path currently has an Android FP16 branch that creates temporary FP16 tensors when `query_step` is FP32, copies Q/K/V into them, runs attention, then copies the output back. That is safe but expensive in per-token decode.

Implementation sketch:

1. Treat native FP16 activations as the fast-path requirement for Android NEON.
2. Audit model configs so Q/K/V outputs and MHA input/output tensors are already FP16.
3. Add a debug counter or log once when the fallback copy path is used, so benchmarks catch accidental FP32 configs.
4. If mixed precision is still required, add a fused conversion at the FC output boundary rather than allocating per-MHA temporary tensors.

Expected impact: medium to high when a model accidentally runs the fallback path; small when configs are already pure FP16.

## P2: attention and elementwise fusion

### 6. Dedicated decode attention fast path

For generation, `step_size == 1` is the common path. Avoid generic triangle tensor setup and process one query head at a time:

1. write rotated K/V to cache;
2. compute one row of logits for visible context only;
3. apply scaling, optional sink, local window, and softmax in-place;
4. immediately compute weighted V into the output head.

Implementation notes:

- Keep K/V cache contiguous by timestep as it is today, but benchmark an alternative head-major cache view for `QK` and `AV` locality.
- Parallelize across KV heads or query-head groups, matching the existing ThreadManager approach.
- Add a small fixed-size stack/scratch allocator for logits to avoid per-token heap allocations.

### 7. Fuse RMSNorm with GEMV input preparation

RMSNorm is required before attention and MLP projections. For decode:

1. compute mean square with NEON reduction;
2. produce a normalized FP16 vector in the exact layout expected by GEMV;
3. optionally combine residual add + RMSNorm when graph ordering allows.

This reduces memory traffic and avoids an additional pass over the hidden vector.

### 8. Fused SwiGLU MLP decode

SwiGLU commonly computes gate and up projections, applies activation/multiply, then down projection. For decode:

1. compute gate/up projections in one fused quantized GEMV pass;
2. apply SiLU and multiply in-place in FP16;
3. feed the result directly into the down GEMV without materializing unnecessary graph tensors.

This is larger work but can be one of the biggest wins after quantized GEMV.

## P3: sampling, LM head, and scheduling

### 9. Optimize LM head and token selection

For greedy or small top-k decoding, avoid full-vocabulary post-processing where possible:

1. make the LM head use the same quantized GEMV fast path if accuracy permits;
2. fuse bias/temperature/repetition penalty with top-k partial reduction;
3. for greedy mode, compute argmax while producing logits and skip storing the full logits tensor unless requested.

### 10. Threading and phone policy

1. Expose `num_threads`, `big_core_only`, and `spin_wait` knobs in the benchmark first, not as hard defaults.
2. For GEMV, parallelize by output rows; for attention, parallelize by KV head/query group.
3. Avoid using all cores for small models if thread overhead exceeds work; benchmark 1, 2, 4, and big-core-count threads.
4. Monitor thermal throttling: a slightly lower cold TPS can produce higher sustained TPS.

## Validation checklist

For every kernel optimization:

- Unit test against scalar reference for aligned and tail dimensions.
- Compare one decoder layer output with fixed random tensors.
- Run end-to-end greedy generation on at least two prompts and verify token IDs match or remain within accepted tolerance.
- Benchmark cold and warmed generation TPS on one low-end and one high-end Android phone.
- Confirm peak memory does not increase; for fusion work it should usually decrease.

## Suggested landing order

1. Document Android NEON fast config and benchmark preset.
2. Add logging/counters for MHA FP32 fallback copies and make FP16 config the default benchmark path.
3. Implement Q4_0 x FP16 NEON GEMV and route single-token FC decode to it.
4. Fuse Q/K/V projection using the same GEMV backend.
5. Add decode-only attention fast path with scratch reuse.
6. Fuse RMSNorm/GEMV preparation and then SwiGLU MLP.
7. Optimize LM head/top-k and tune thread policy per device class.
