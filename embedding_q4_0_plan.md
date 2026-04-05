# Embedding Layer Q4_0 Quantization Plan

## Goal
Switch `EmbeddingLayer` Q4_0 quantization from vocab-axis blocking to hidden-axis blocking, while preserving existing GEMM-optimized behavior for FC/projection layers.

## Background (Current Behavior)
- Embedding weight shape is `[height=in_dim(vocab), width=out_dim(hidden)]`.
- Current Q4_0 save path transposes weights before quantization, so Q4_0 blocks are formed along original `height` (vocab) for embedding matrices.
- This convention is inherited from GEMM-oriented GGML quantization paths and is likely mismatched with embedding row-gather semantics.

## Scope
- In scope:
  - Save-time quantization policy for embedding weights.
  - Runtime dequant/gather behavior for quantized embedding weights.
  - Tests and docs for axis policy and constraints.
- Out of scope:
  - Reworking all quantization schemes beyond Q4_0.
  - Changing FC/GEMM packing strategy.

## Design Principles
1. **Layer-aware policy**: Quantization axis must be selected per use case, not globally.
2. **Performance isolation**: Keep FC/projection GEMM-packed Q4_0 behavior unchanged.
3. **Correctness first**: Embedding lookup output should numerically match FP32 reference within expected Q4_0 error.
4. **Transparent constraints**: Validation errors should mention logical axes (`vocab`, `hidden`) and required block sizes.

## Implementation Plan

### Phase 1 — Introduce Quantization Axis Policy
1. Add a small policy enum/helper in quantization path (e.g., `QuantAxisPolicy::{GemmDefault, EmbeddingRowWise}`).
2. Thread policy through save-time Q4_0 code path (`layer_devel` and/or quantizer entrypoint).
3. Keep existing transpose+repack as the default policy for GEMM layers.

**Deliverable:** Policy plumbing merged without behavior change for existing layers.

---

### Phase 2 — Embedding Save Path: Hidden-Axis Q4_0
1. For embedding weights, quantize rows as embedding vectors (block over `out_dim` / hidden axis).
2. Update divisibility checks to validate blocked axis requirements for embedding path.
3. Improve exception text with actual semantic dimensions (`vocab`, `hidden`) and required multiple (32).

**Deliverable:** Embedding Q4_0 serialization follows hidden-axis convention.

---

### Phase 3 — Runtime Embedding Lookup for Q4_0
1. Add/choose a gather-friendly dequantization path for embedding rows (avoid forcing GEMM-only repacked assumptions).
2. Update `EmbeddingLayer::forwarding` and `incremental_forwarding` to correctly handle Q4_0 weights.
3. Ensure mixed precision and batch slicing behavior remain correct.

**Deliverable:** Quantized embedding lookup produces correct dense output tensors.

---

### Phase 4 — Tests
1. **Unit tests** for axis policy selection and dimension validation.
2. **Layer tests** for `EmbeddingLayer`:
   - FP32 vs Q4_0 output closeness on random IDs.
   - Boundary ID behavior and invalid index error paths.
   - Incremental forwarding parity with full forwarding.
3. **Regression tests** to confirm FC/GEMM Q4_0 outputs are unaffected.

**Deliverable:** Automated coverage for correctness and compatibility.

---

### Phase 5 — Documentation & Migration Notes
1. Document per-layer quantization-axis policy for Q4_0.
2. Clarify constraints and recommended embedding dimensions.
3. Add migration note: old checkpoints quantized with vocab-axis embedding may need re-export.

**Deliverable:** User-facing and developer-facing docs updated.

## Risk Assessment
- **Compatibility risk:** Old quantized embedding checkpoints may decode differently.
  - Mitigation: explicit checkpoint versioning or migration warning.
- **Performance risk:** Row-gather dequant may increase latency.
  - Mitigation: benchmark lookup path; cache/dequant strategy if needed.
- **Behavior drift risk in FC path:**
  - Mitigation: lock existing GEMM policy as default + regression tests.

## Acceptance Criteria
- Embedding Q4_0 quantization operates on hidden axis by design.
- `EmbeddingLayer` forward/incremental outputs are numerically stable vs FP32 baseline.
- FC/GEMM Q4_0 behavior unchanged in accuracy and expected performance envelope.
- Clear docs and actionable error messages are in place.

## Suggested Execution Order
1. Phase 1 policy plumbing
2. Phase 2 embedding save path
3. Phase 3 runtime lookup path
4. Phase 4 tests
5. Phase 5 docs/migration
