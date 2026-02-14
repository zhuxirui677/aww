# AWQ + W4A4/W4A8 加速实现路线图

本文档描述如何实现 **AWQ 权重量化 + 激活量化** 的真实推理加速，供有 CUDA/Triton 经验的开发者参考。

---

## 一、目标与现状

| 目标 | 当前状态 |
|------|----------|
| W4A8（4-bit 权重 + 8-bit 激活） | 仅有 fake 量化，无加速 |
| W4A4（4-bit 权重 + 4-bit 激活） | 同上 |
| 真实加速 | 需实现 INT4×INT8/INT4 GEMM 及融合 kernel |

---

## 二、AWQ 权重格式（需对接）

参考 vLLM 的 `awq.py` 和 AutoAWQ 实现，AWQ 存储结构如下：

### 2.1 核心参数

```
qweight:  INT32  packed，每 8 个 4-bit 权重打包成 1 个 int32
         shape: (in_features, out_features // 8)
scales:   FP16/BF16，per-group 缩放因子
         shape: (num_groups, out_features)，num_groups = in_features // group_size
qzeros:   INT32 packed，per-group 零点（zero_point=True 时）
         shape: (num_groups, out_features // 8)
group_size: 通常 128
pack_factor: 32 // 4 = 8
```

### 2.2 反量化公式

```
w_fp16 = (w_int4 - zero) * scale
# 其中 w_int4 从 qweight 解包，zero 从 qzeros 解包
```

### 2.3 vLLM 现有实现

```python
# vLLM 两种路径：
# 1. 大 batch：先 awq_dequantize 到 FP16，再 matmul（内存带宽受限）
# 2. 小 batch：awq_gemm 融合 kernel（减少中间结果）
out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros, pack_factor)
```

要支持 W4A8，需在 `awq_gemm` 前对激活 `x` 做 8-bit 量化，并实现 **INT4×INT8 GEMM** 或 **INT4×INT8 → FP16 融合**。

---

## 三、实现路线图

### 阶段 1：理解与验证（1–2 天）

1. **阅读 vLLM AWQ 源码**
   - `vllm/model_executor/layers/quantization/awq.py`
   - `vllm/csrc/quantization/awq/`（CUDA kernel）

2. **阅读 AutoAWQ 权重加载**
   - 确认 `qweight`、`scales`、`qzeros` 的布局与字节序

3. **验证现有 AWQ 推理**
   - 用 vLLM 或 AutoAWQ 跑通 W4A16，确认数值正确

### 阶段 2：激活量化 kernel（2–3 天）

**目标**：对 Linear 输入做 per-tensor 8-bit 量化，输出 INT8。

```python
# 伪代码
def quantize_activation_fp16_to_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """x: FP16, shape (batch, seq, hidden)
    Returns: (x_int8, scale)
    """
    scale = 127.0 / x.abs().max().clamp(min=1e-8)
    x_int8 = (x * scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale
```

**可选实现**：
- PyTorch 原生（先验证逻辑）
- Triton kernel（融合到后续 matmul）
- 参考 [INT8_Triton_Kernels](https://github.com/chinmaydk99/INT8_Triton_Kernels) 的 `int8_rowwise_quant.py`

### 阶段 3：INT4×INT8 GEMM kernel（5–10 天）

**目标**：实现 `(INT8 激活) × (INT4 权重) → FP16 输出`，并在 kernel 内完成权重的反量化。

**方案 A：Triton（推荐起步）**

- Triton 语法比 CUDA 简单，便于迭代
- 参考：
  - [PyTorch Triton GPTQ 加速](https://pytorch.org/blog/accelerating-triton/)
  - [INT8_Triton_Kernels](https://github.com/chinmaydk99/INT8_Triton_Kernels) 的 `int8_fused_dequant_matmul_rowwise.py`

**方案 B：CUDA**

- 性能上限更高，但开发周期更长
- 可参考 vLLM 的 `awq_gemm`、QServe 的 W4A8 kernel

**核心逻辑**：

```
输入: A_int8 (M, K), qweight (K, N/8), scales, qzeros
1. 解包 qweight → W_int4 (K, N)
2. 反量化: W_fp16 = (W_int4 - zeros) * scales
3. 反量化 A: A_fp16 = A_int8 / scale_a
4. C = A_fp16 @ W_fp16  (或直接在 INT 域做部分和再反量化)
```

为减少内存带宽，更优做法是 **在 kernel 内边解包边计算**，避免完整反量化后的 FP16 矩阵。

### 阶段 4：融合 kernel（3–5 天）

**目标**：将「激活量化 + INT4×INT8 GEMM + 输出反量化」融合为单个 kernel。

```
输入(FP16) → [Kernel] 量化 → INT8 matmul with INT4 weight → 输出(FP16)
```

这样可避免：
- 激活量化的中间 INT8 写回全局内存
- 权重的完整反量化

### 阶段 5：与 AWQ 模型集成（2–3 天）

1. **替换 Linear 层**
   - 继承 vLLM 的 `AWQLinearMethod` 或自定义 `LinearBase`
   - 在 `apply()` 中调用新的 W4A8 kernel

2. **处理 group_size 与 pack_factor**
   - 正确解析 `qweight`、`scales`、`qzeros` 的 group 划分
   - 处理 `group_size=-1`（per-channel）情况

3. **数值对齐**
   - 与 FP16 或现有 AWQ W4A16 实现逐层对比，误差在可接受范围

---

## 四、技术要点

### 4.1 AWQ 权重解包

```python
# 4-bit 打包：8 个 4-bit 存在 1 个 int32 里
def unpack_int4(qweight: torch.Tensor) -> torch.Tensor:
    # qweight: (K, N/8), dtype=int32
    # 每个 int32 含 8 个 4-bit，范围 0–15
    w = torch.zeros(K, N, dtype=torch.int32)
    for i in range(8):
        w[:, i::8] = (qweight >> (i * 4)) & 0xF
    return w
```

### 4.2 激活量化策略

| 策略 | 精度 | 速度 | 实现难度 |
|------|------|------|----------|
| Per-tensor 动态 | 中 | 快 | 低 |
| Per-token 动态 | 较高 | 中 | 中 |
| Per-channel 静态 | 高 | 快 | 高（需校准） |

建议先用 **per-tensor 动态** 验证流程，再考虑 per-token 或静态 scale。

### 4.3 参考实现

| 项目 | 用途 |
|------|------|
| [vLLM awq.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/awq.py) | AWQ 格式与调用方式 |
| [INT8_Triton_Kernels](https://github.com/chinmaydk99/INT8_Triton_Kernels) | Triton INT8 量化与 GEMM |
| [PyTorch Triton GPTQ](https://pytorch.org/blog/accelerating-triton/) | INT4 反量化 + GEMM 融合 |
| [QServe](https://github.com/mit-han-lab/omniserve) | W4A8 系统设计与 kernel 思路 |

---

## 五、预期工作量

| 阶段 | 工作量 | 前置条件 |
|------|--------|----------|
| 阶段 1 | 1–2 天 | 熟悉 PyTorch、AWQ |
| 阶段 2 | 2–3 天 | 阶段 1 |
| 阶段 3 | 5–10 天 | Triton 或 CUDA 经验 |
| 阶段 4 | 3–5 天 | 阶段 3 |
| 阶段 5 | 2–3 天 | 阶段 4 |

**合计**：约 2–4 周（视经验而定）。

---

## 六、最小可行验证（MVP）

已提供 **`scripts/mvp_awq_w4a8_verify.py`**，用纯 PyTorch 验证：

1. **AWQ 解包与反量化**：`pack_int4` / `unpack_int4` / `awq_dequantize`
2. **激活量化**：`quantize_activation_fp16_to_int8` / `quantize_activation_fp16_to_int4`
3. **W4A16 / W4A8 / W4A4 输出误差**：与 FP16 基准对比

```bash
# 合成数据验证（无需模型）
python scripts/mvp_awq_w4a8_verify.py

# 真实 AWQ 模型验证（需先量化或下载 AWQ 模型）
python scripts/mvp_awq_w4a8_verify.py --awq_model_path ./medgamma-awq-4bit
```

依赖：`torch`、`safetensors`（真实模型验证时）。

验证通过后，可继续：

1. **单层 Triton kernel**
   - 只实现一个 Linear 的 W4A8 版本
   - 与 PyTorch 实现对比，确保数值一致

2. **性能测试**
   - 若 Triton 版本明显快于「反量化 + FP16 matmul」，再推进全模型集成。

---

## 七、风险与备选

| 风险 | 缓解 |
|------|------|
| Triton 性能不足 | 改用 CUDA 或 cutlass |
| 精度损失过大 | 尝试 per-token 量化、静态 scale 或混合精度 |
| 与 vLLM 集成复杂 | 可先做独立推理脚本，再考虑贡献到 vLLM |

**备选**：若工程成本过高，可改用 QServe（QoQ 非 AWQ）或 TensorRT-LLM 的 W4A8，它们已有完整实现。
