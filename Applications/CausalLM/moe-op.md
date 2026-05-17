可以，针对当前 `moe_slim`，我建议按 **“先量化、再快路径、再缓存、最后融合算子”** 的顺序设计。这样风险低，而且每一步都能单独验证收益。

**目标**
当前 `moe_slim`：

```text
65010.665 ms / 87217 ms = 74.54%
avg 9.471 ms * 6864 calls
```

优化目标建议先定为：

```text
moe_slim avg: 9.47 ms -> 5~6 ms
generation TPS: 1.74 -> 2.3~3.0+
```

第一阶段不追求重写架构，先把 generation 单 token 路径打通。

**方案 1：增加 MoE 内部 profiler**
先在 [qwen_moe_layer_fsu.cpp](/home/czm/workspace/ai/ondevice/nntrainer/Applications/CausalLM/models/qwen3_slim_moe/qwen_moe_layer_fsu.cpp:379) 内部拆阶段统计：

```text
moe_total
input_slice
router_dot
router_softmax
router_topk
topk_norm
assignment_build
expert_output_alloc
weight_activate
expert_compute_total
expert_gate_dot
expert_activation
expert_up_dot
expert_mul
expert_down_dot
expert_scale_add
weight_deactivate
output_combine
```

这样可以确认到底是：

```text
权重 activate/deactivate 慢？
GEMV 慢？
Tensor 分配慢？
OpenMP 调度慢？
```

这个阶段只加统计，不改变行为，风险最低。

**方案 2：single-token fast path**
generation 阶段大多数情况下 `to - from == 1`，当前代码仍走通用 batch MoE 逻辑：

```cpp
std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(num_experts);
std::vector<nntrainer::Tensor> expert_outputs(num_experts);

#pragma omp parallel for schedule(dynamic)
for (expert_idx = 0; expert_idx < num_experts; ++expert_idx)
```

单 token 其实不需要扫描所有 expert，也不需要构造二维 assignments。

新增路径：

```cpp
if ((to - from) == 1 && input_.batch() == 1) {
  incremental_forwarding_single_token(context, from, to, training);
  return;
}
```

single-token 路径逻辑：

```text
1. slice current token input/output
2. router dot
3. softmax + topK
4. 只遍历 topK 个 expert
5. 每个 expert 计算 gate/up/down
6. 直接累加到 output
```

预期收益：

```text
减少 num_experts 全量扫描
减少 vector<vector<pair>> 构造
减少 expert_outputs(num_experts) 分配
减少 output.add_i(expert_outputs[i]) 全量合并
降低 OpenMP dynamic 调度成本
```

这是我认为最应该先做的行为优化。

**方案 3：复用临时 Tensor / buffer**
当前每个 expert 每个 token 都创建：

```cpp
nntrainer::Tensor gate_out(intermediate_dim);
nntrainer::Tensor acti_out(intermediate_dim);
nntrainer::Tensor up_out(intermediate_dim);
nntrainer::Tensor token_expert_output(token_output_dim);
```

single-token 下可以改为：

```text
每个 MoE layer 申请或缓存一组临时 Tensor
每次 expert 计算复用这些 Tensor
```

例如：

```text
gate_out: [1,1,1,intermediate_size]
up_out: [1,1,1,intermediate_size]
acti_out: [1,1,1,intermediate_size]
expert_out: [1,1,1,hidden_size]
```

如果要并行 topK expert，则需要 per-thread buffer；如果串行 topK，则一组 buffer 就够。

建议先做串行版本测基线，因为 topK 数量有限，避免线程调度反而拖慢。

**方案 4：expert 权重缓存**
当前 FSU 路径每次 active expert 都会：

```cpp
activate gate/up/down
compute
deactivate gate/up/down
```

如果 `activate()` 触发 mmap/load，这会非常贵。已有 cached 版本目录：

```text
Applications/CausalLM/models/qwen3_cached_slim_moe/
```

可以有两条路：

```text
A. 直接切模型使用 qwen3_cached_slim_moe 验证收益
B. 给 qwen_moe_layer_fsu.cpp 增加轻量 LRU expert cache
```

我建议先做 A，快速验证方向。如果 cached 版本明显快，再把缓存机制移植/简化到当前层。

缓存策略：

```text
cache_capacity: N 个 expert
命中：直接 compute
未命中：activate，加入 LRU
淘汰：deactivate 最久未使用 expert
```

更进一步可以利用 router topK 的局部性：

```text
当前 token topK experts 往往和前后 token 有重复
保持最近 active experts 不卸载
```

**方案 5：只遍历 active experts**
即使不做完整 single-token fast path，也应该把：

```cpp
for expert_idx in [0, num_experts)
```

改成：

```cpp
for expert_idx in active_experts
```

构造 active list：

```cpp
std::vector<unsigned> active_experts;
active_experts.reserve(topk);

for each topk index:
  if not seen:
    active_experts.push_back(expert_idx);
```

这样 OpenMP 也可以改成：

```cpp
#pragma omp parallel for schedule(static)
for (int i = 0; i < active_experts.size(); ++i)
```

对于 generation，`active_experts.size()` 大概率就是 `topk`，比扫 `num_experts` 更合理。

**方案 6：融合 gate/up projection**
当前每个 expert 做两次同输入投影：

```cpp
token_input.dot(gate_proj, gate_out);
token_input.dot(up_proj, up_out);
```

可以将权重布局变成：

```text
gate_up_proj = concat(gate_proj, up_proj)
```

一次 dot 得到：

```text
[gate_out, up_out]
```

然后：

```text
silu(gate_out) * up_out
down_proj
```

收益：

```text
减少一次 dot 调用
减少一次读取 input
可能改善 cache locality
```

风险：

```text
需要权重加载/转换配合
影响模型权重命名或导入逻辑
```

所以建议放在后面做。

**推荐落地顺序**
1. 加 `moe_slim` 内部 profiler，先定位 activate、dot、alloc 的占比。
2. 实现 `incremental_forwarding_single_token()`。
3. single-token 路径只遍历 topK active experts。
4. 复用临时 Tensor，减少循环内分配。
5. 对比 `qwen3_cached_slim_moe`，决定是否引入 LRU expert cache。
6. 最后考虑 gate/up fusion。

**我建议下一步**
先实现第 1 + 第 2 步：  
给 `qwen_moe_layer_fsu.cpp` 加内部耗时统计，并加一个不改变数学结果的 single-token fast path。这样很快能看到 `moe_slim avg 9.47 ms` 具体掉到哪里。