# CausalLM Performance Analysis Report

## Test Summary

This report is based on the runtime profile printed by `nntrainer_causallm`.

Overall runtime:

| Metric | Value |
| --- | ---: |
| Prefill tokens | 18 |
| Prefill time | 5622 ms |
| Prefill throughput | 3.20171 TPS |
| Generation tokens | 142 |
| Generation time | 81593 ms |
| Generation throughput | 1.74035 TPS |
| Total time | 87217 ms |
| Peak memory | 1331068 KB |

## Layer Type Breakdown

Percentages are calculated against the total runtime, `87217 ms`.

| Rank | Layer type | Total time (ms) | Calls | Avg time (ms) | Total ratio |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | moe_slim | 65010.665 | 6864 | 9.471 | 74.54% |
| 2 | fully_connected | 11101.403 | 27456 | 0.404 | 12.73% |
| 3 | mha_core | 4613.941 | 6864 | 0.672 | 5.29% |
| 4 | lm_head | 3028.674 | 143 | 21.180 | 3.47% |
| 5 | rms_norm | 716.933 | 13871 | 0.052 | 0.82% |
| 6 | addition | 339.105 | 13728 | 0.025 | 0.39% |
| 7 | reshaped_rms_norm | 315.830 | 13728 | 0.023 | 0.36% |
| 8 | multiout | 83.425 | 20592 | 0.004 | 0.10% |
| 9 | embedding_layer | 71.510 | 143 | 0.500 | 0.08% |
| 10 | input | 2.869 | 143 | 0.020 | 0.00% |

Layer-profiled time:

| Item | Time (ms) | Ratio |
| --- | ---: | ---: |
| Profiled layer total | 85284.355 | 97.78% |
| Unprofiled overhead | 1932.645 | 2.22% |

## MHA Core Breakdown

MHA core total is `4613.941 ms`, which is `5.29%` of total runtime.

Detailed MHA core stages:

| Stage | Total time (ms) | Calls | Avg time (ms) | Ratio of total runtime |
| --- | ---: | ---: | ---: | ---: |
| incremental_forwarding | 4567.276 | 6864 | 0.665 | 5.24% |
| one_batch_forwarding | 4328.117 | 6864 | 0.631 | 4.96% |
| qk_compute | 1988.683 | 6864 | 0.290 | 2.28% |
| av_compute | 1639.898 | 6864 | 0.239 | 1.88% |
| rope_query | 322.935 | 6864 | 0.047 | 0.37% |
| softmax | 194.451 | 6864 | 0.028 | 0.22% |
| rope_key | 40.250 | 6864 | 0.006 | 0.05% |
| qk_out_alloc | 19.388 | 6864 | 0.003 | 0.02% |
| tensor_slice | 18.799 | 6864 | 0.003 | 0.02% |
| value_cache_write | 15.718 | 6864 | 0.002 | 0.02% |

MHA is not the primary bottleneck in this run.

## Top Layer Names

The slowest individual layer is the output LM head, but the dominant aggregate cost comes from `moe_slim` layers repeated across the model.

| Rank | Layer name | Type | Total time (ms) | Calls | Avg time (ms) |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | output_of_causallm | lm_head | 3028.674 | 143 | 21.180 |
| 2 | layer0_ffn_down | moe_slim | 1589.899 | 143 | 11.118 |
| 3 | layer1_ffn_down | moe_slim | 1544.671 | 143 | 10.802 |
| 4 | layer2_ffn_down | moe_slim | 1517.220 | 143 | 10.610 |
| 5 | layer47_ffn_down | moe_slim | 1482.997 | 143 | 10.371 |
| 6 | layer37_ffn_down | moe_slim | 1476.865 | 143 | 10.328 |
| 7 | layer3_ffn_down | moe_slim | 1452.277 | 143 | 10.156 |
| 8 | layer34_ffn_down | moe_slim | 1448.187 | 143 | 10.127 |
| 9 | layer4_ffn_down | moe_slim | 1443.693 | 143 | 10.096 |
| 10 | layer10_ffn_down | moe_slim | 1442.606 | 143 | 10.088 |

## Main Findings

1. `moe_slim` is the dominant bottleneck.
   It consumes `65010.665 ms`, or `74.54%` of the total runtime.

2. `fully_connected` is the second largest cost.
   It consumes `11101.403 ms`, or `12.73%` of total runtime.

3. `mha_core` is relatively small.
   It consumes only `4613.941 ms`, or `5.29%` of total runtime.

4. `lm_head` is expensive per call.
   It runs only `143` times, but costs `21.180 ms` per call on average.

5. The layer profiler covers most runtime.
   Profiled layers account for `97.78%` of total time, so the current profile is representative.

## Optimization Priority

Recommended order:

1. Optimize `moe_slim`.
2. Optimize `fully_connected`.
3. Optimize `lm_head`.
4. Revisit `mha_core` only after the above costs are reduced.

## Suggested Next Profiling

For `moe_slim`, add internal stage timing around:

| Candidate stage | Why |
| --- | --- |
| router / gate computation | Determines selected experts and may include top-k logic |
| expert weight access / selection | May expose memory layout or cache issues |
| expert up projection | Likely GEMV-heavy |
| activation | Usually smaller, but worth confirming |
| expert down projection | Likely GEMV-heavy |
| output accumulation | Can reveal copy/add overhead |

For `fully_connected` and `lm_head`, split timing around:

| Candidate stage | Why |
| --- | --- |
| tensor slicing | Checks per-token view overhead |
| `Tensor::dot` | Expected dominant compute |
| bias add | Usually small |
| quantization/dequantization path | Important if quantized weights are used |

## Current Conclusion

The current performance issue is not primarily caused by attention. The model spends most of its time in MoE FFN execution, especially `moe_slim` layers. The next optimization pass should focus on `moe_slim` internals and matrix-vector kernels used by MoE and fully connected layers.
