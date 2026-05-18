# MoE Slim 层专家计算优化计划

## 背景与问题分析

### 当前性能数据
根据性能分析，`expert_compute_total` 是 MoE 层的主要开销（占总时间的 40%+）：
- `expert_gate_dot`: 0.617-1.083 ms/次
- `expert_up_dot`: 0.643-0.745 ms/次  
- `expert_down_dot`: 0.660-0.684 ms/次
- `expert_activation`: 0.026-0.027 ms/次（silu/swish）

### 当前实现瓶颈

1. **逐 token 串行处理** (`compute_expert_forward_no_critical`)
   - 每个 token 单独创建临时张量（gate_out, acti_out, up_out）
   - 每个 token 执行 3 次独立的矩阵乘法（gate/up/down）
   - 内存分配开销累积

2. **临时张量重复分配**
   ```cpp
   nntrainer::Tensor gate_out(intermediate_dim);  // 每次循环都新建
   nntrainer::Tensor acti_out(intermediate_dim);  // 每次循环都新建
   nntrainer::Tensor up_out(intermediate_dim);   // 每次循环都新建
   ```

3. **激活函数未融合**
   - gate_dot → silu → up_dot → mul → down_dot 分步执行
   - 中间结果写入内存再读取

## 优化策略

### 方案 1: 批量处理 tokens（最大收益）

**核心思想**: 将同一专家的多个 token 合并为批次处理，利用矩阵乘法的并行性。

**当前** (N 个 tokens 串行):
```
for each token:
  [1, H] x [H, I] -> [1, I]   // gate
  [1, H] x [H, I] -> [1, I]   // up
```

**优化后** (N 个 tokens 并行):
```
[N, H] x [H, I] -> [N, I]   // gate (批量)
[N, H] x [H, I] -> [N, I]   // up (批量)
```

**预期收益**: 2-3x 加速（取决于 batch size）

**修改位置**: 
- `Applications/CausalLM/models/qwen3_slim_moe/qwen_moe_layer_fsu.cpp`
- 修改 `compute_expert_forward_no_critical` 函数

---

### 方案 2: 内存池预分配临时张量

**核心思想**: 避免在循环中重复分配临时张量。

**当前**:
```cpp
for (size_t i = 0; i < num_tokens; ++i) {
    nntrainer::Tensor gate_out(intermediate_dim);  // 每次分配
    nntrainer::Tensor acti_out(intermediate_dim);
    nntrainer::Tensor up_out(intermediate_dim);
    ...
}
```

**优化后**:
```cpp
// 预分配最大可能大小的张量
nntrainer::Tensor gate_out({max_tokens_per_expert, 1, 1, intermediate_size});
nntrainer::Tensor acti_out({max_tokens_per_expert, 1, 1, intermediate_size});
nntrainer::Tensor up_out({max_tokens_per_expert, 1, 1, intermediate_size});

for (size_t i = 0; i < num_tokens; ++i) {
    // 使用 slice 视图，不重新分配
    auto gate_out_slice = gate_out.getSharedDataTensor(...);
    ...
}
```

**预期收益**: 10-20% 开销减少

---

### 方案 3: SwiGLU 融合（gate + up + mul）

**核心思想**: 将 gate projection → silu → up projection → elementwise_mul 融合为单个内核。

**当前流程** (4 个步骤，3 次内存写入):
1. `token_input.dot(gate_proj, gate_out)`      // [1,H]x[H,I]->[1,I]
2. `acti_func.run_fn(gate_out, acti_out)`      // silu([1,I])
3. `token_input.dot(up_proj, up_out)`         // [1,H]x[H,I]->[1,I]
4. `acti_out.multiply_i(up_out)`              // [1,I]*[1,I]

**融合后** (1 个步骤，0 次中间内存写入):
```cpp
// 伪代码
for i in range(intermediate_size):
    gate_val = dot(input, gate_proj[:, i])
    up_val = dot(input, up_proj[:, i])
    output[i] = silu(gate_val) * up_val
```

**实现方式**:
- 使用 OpenMP SIMD 指令实现融合内核
- 或利用现有 NEON/x86 SIMD 优化

**预期收益**: 1.5-2x 加速（减少内存带宽占用）

**修改位置**:
- 在 `cpu_backend.h` 中新增 `swiglu_fused` 接口
- 在 `qwen_moe_layer_fsu.cpp` 中调用融合函数

---

### 方案 4: 使用 FP16/量化权重

**核心思想**: 利用现有的 FP16 计算后端减少内存带宽。

**现状**:
- `expert_compute_total` 时间包含权重读取
- FP32 权重占用大量内存带宽

**优化**:
- 将专家权重转换为 FP16 或 Q4_0/Q6_K 量化格式
- 使用 `dotFloat32Float16` 或 `dotQnK` 函数
- 利用 ARM NEON/x86 AVX2 的半精度计算能力

**预期收益**: 2x 加速（FP16）或 4x 加速（INT4）

**依赖**:
- 权重量化工具（已有 `quantize.cpp`）
- FP16 计算后端（已有 `shgemv`/`shgemm`）

---

## 实施优先级

| 优先级 | 方案 | 预期加速 | 工作量 | 风险 |
|--------|------|----------|--------|------|
| P0 | 方案1: 批量处理 | 2-3x | 中 | 低 |
| P1 | 方案3: SwiGLU融合 | 1.5-2x | 高 | 中 |
| P2 | 方案2: 内存池 | 10-20% | 低 | 低 |
| P3 | 方案4: FP16 | 2-4x | 中 | 中（精度） |

## 关键修改文件

1. **主要实现**:
   - `Applications/CausalLM/models/qwen3_slim_moe/qwen_moe_layer_fsu.cpp`
     - `compute_expert_forward` (line 388)
     - `compute_expert_forward_no_critical` (line 477)
     - `compute_single_token_expert` (line 557)

2. **SIMD 优化** (可选):
   - `nntrainer/tensor/cpu_backend/arm/arm_compute_backend.h`
   - `nntrainer/tensor/cpu_backend/x86/x86_compute_backend.h`

3. **激活函数**:
   - `nntrainer/layers/acti_func.h`

## 验证方案

1. **功能验证**:
   - 使用相同输入对比优化前后的输出（数值误差 < 1e-5）

2. **性能验证**:
   ```bash
   # 运行性能测试
   ./nntr_inference --model=qwen3_slim_moe --profile=true
   
   # 对比指标
   # - expert_compute_total 时间
   # - expert_gate_dot/up_dot/down_dot 单次耗时
   # - 总体 generation TPS
   ```

3. **内存验证**:
   - 监控 peak_memory 变化
   - 确保无内存泄漏

## 参考代码

### 批量处理示例
```cpp
// 收集同一专家的所有 token
std::vector<unsigned> token_indices;
for (auto &assignment : token_assignments) {
    token_indices.push_back(assignment.first);
}

// 构建 batch 输入
unsigned batch_size = token_indices.size();
nntrainer::Tensor batch_input({batch_size, 1, 1, hidden_size}, input.getTensorType());
for (size_t i = 0; i < batch_size; ++i) {
    // copy input slice
}

// 批量矩阵乘法
batch_input.dot(gate_proj, batch_gate_out);  // [B,H]x[H,I]->[B,I]
batch_input.dot(up_proj, batch_up_out);      // [B,H]x[H,I]->[B,I]
// ... 激活和融合
```

### SwiGLU 融合内核示例
```cpp
void swiglu_fused(const float* input, const float* gate_proj, 
                  const float* up_proj, float* output,
                  unsigned int hidden_size, unsigned int intermediate_size) {
    #pragma omp parallel for
    for (unsigned int i = 0; i < intermediate_size; ++i) {
        float gate_val = 0.0f, up_val = 0.0f;
        
        // 计算 gate projection
        for (unsigned int j = 0; j < hidden_size; ++j) {
            gate_val += input[j] * gate_proj[j * intermediate_size + i];
        }
        
        // 计算 up projection
        for (unsigned int j = 0; j < hidden_size; ++j) {
            up_val += input[j] * up_proj[j * intermediate_size + i];
        }
        
        // silu(gate) * up
        output[i] = gate_val / (1.0f + expf(-gate_val)) * up_val;
    }
}
```
