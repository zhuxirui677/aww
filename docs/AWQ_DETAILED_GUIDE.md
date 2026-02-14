# AWQ 详细介绍与运作原理

## 一、AWQ 概述

**AWQ (Activation-aware Weight Quantization)** 是一种面向硬件的 LLM 低比特**仅权重**量化方法，由 MIT、上海交大、清华联合提出，发表于 arXiv:2306.00978，并获 **MLSys 2024 Best Paper Award**。

### 核心思想

> **权重并非同等重要**：保护仅 1% 的显著权重即可大幅降低量化误差。

与传统方法（如 GPTQ）基于权重大小或二阶信息不同，AWQ 通过**观察激活分布**识别重要权重通道，并对这些通道进行 per-channel 缩放保护，同时保持全权重量化，无需混合精度，硬件友好。

---

## 二、工作原理详解

### 2.1 数学原理

#### 量化误差分析

线性层运算：$y = Wx$，量化后：$y = Q(W) \cdot x$

量化函数：
$$Q(\mathbf{w}) = \Delta \cdot \text{Round}(\frac{\mathbf{w}}{\Delta}), \quad \Delta = \frac{\max(|\mathbf{w}|)}{2^{N-1}}$$

其中 $N$ 为量化比特数，$\Delta$ 为量化步长。

#### 缩放保护显著权重

对显著权重 $w$ 乘以 $s > 1$，对输入 $x$ 除以 $s$：
$$Q(w \cdot s) \cdot \frac{x}{s} = \Delta' \cdot \text{Round}(\frac{ws}{\Delta}) \cdot x \cdot \frac{1}{s}$$

**关键发现**：
- Round 误差近似均匀分布，期望不变
- 缩放单元素通常不改变组内极值，故 $\Delta' \approx \Delta$
- 相对误差：$\text{Err}' = \Delta' \cdot \text{RoundErr} \cdot \frac{1}{s}$，即 $s > 1$ 时显著权重误差减小

#### 最优缩放搜索

目标：最小化量化后输出与原始输出的差异
$$\mathbf{s}^* = \arg\min_{\mathbf{s}} \mathcal{L}(\mathbf{s}), \quad \mathcal{L}(\mathbf{s}) = \|Q(\mathbf{W} \cdot \mathbf{s})(\mathbf{s}^{-1} \cdot \mathbf{X}) - \mathbf{W}\mathbf{X}\|$$

简化搜索空间（激活感知）：
$$\mathbf{s} = \mathbf{s_X}^\alpha, \quad \alpha^* = \arg\min_{\alpha} \mathcal{L}(\mathbf{s_X}^\alpha)$$

其中 $\mathbf{s_X}$ 为激活幅度的统计量，$\alpha \in [0,1]$ 通过网格搜索确定。

### 2.2 算法流程

```
1. 分析激活分布：用校准数据前向传播，收集每通道激活统计 X
2. 计算通道重要性：s_X = f(X)  # 基于激活幅度
3. 搜索最优 α：在 [0,1] 上网格搜索，最小化 L(s_X^α)
4. 缩放权重：W' = W * s
5. 量化：Q(W') → INT4
6. 推理时：Q(W')/s · X ≈ W · X  （s^{-1} 可融合到前一层）
```

### 2.3 优势

| 特性 | 说明 |
|------|------|
| **无反向传播** | 不依赖梯度，不依赖重建损失 |
| **泛化性强** | 不过拟合校准集，跨域/多模态表现好 |
| **数据高效** | 校准数据需求约为 GPTQ 的 1/10 |
| **硬件友好** | 全 INT4 量化，无混合精度，易部署 |

---

## 三、TinyChat 推理框架

AWQ 团队开发的 **TinyChat** 是专为 4-bit 量化 LLM 设计的推理框架。

### 系统架构

```
┌─────────────────────────────────────┐
│         TinyChat System             │
├─────────────────────────────────────┤
│  算法层                              │
│  ├── AWQ 量化算法                    │
│  └── 4-bit weight search             │
├─────────────────────────────────────┤
│  系统层                              │
│  ├── Kernel fusion (算子融合)       │
│  ├── Platform-aware weight packing  │
│  └── Memory optimization             │
├─────────────────────────────────────┤
│  硬件层                              │
│  ├── Desktop GPUs (RTX 4090 等)     │
│  ├── Mobile GPUs (Jetson Orin)      │
│  └── 支持 4-bit INT 运算            │
└─────────────────────────────────────┘
```

### 关键优化

1. **Kernel Fusion**：将 quantize → matmul → dequantize 融合为单 kernel，减少内存访问
2. **Platform-aware Weight Packing**：8 个 4-bit 权重打包为 32-bit 整数，优化内存带宽
3. **性能**：相比 HuggingFace FP16 实现约 **3–4× 加速**

---

## 四、W4A4 / W4A8 与 AWQ 的关系

### 4.1 术语说明

| 表示 | 含义 |
|------|------|
| **W4A16** | 4-bit 权重 + 16-bit 激活（AWQ 默认） |
| **W4A8** | 4-bit 权重 + 8-bit 激活 |
| **W4A4** | 4-bit 权重 + 4-bit 激活 |

### 4.2 AWQ 的定位

- **AWQ 本身是 weight-only 量化**，即 W4A16
- W4A4 / W4A8 需要在 AWQ 权重量化的基础上，**额外对激活做量化**
- 激活量化由其他方法实现，如 PrefixQuant、QServe、QoQ 等

### 4.3 W4A4 的挑战（来自 ATOM、BCQ、PrefixQuant 等）

```
核心问题：
├── 激活 outliers 严重影响量化精度
├── Per-group 量化需要大量元数据
├── 反量化开销高（20–90% 性能损失）
└── 精度损失显著（可达 9% 以上）
```

### 4.4 相关方案对比

| 方案 | 特点 | 链接 |
|------|------|------|
| **PrefixQuant** | 静态量化超越动态，前缀 token 消除 outliers，支持 W4A4/W4A8 | [GitHub](https://github.com/ChenMnZ/PrefixQuant) |
| **QServe (OmniServe)** | W4A8KV4，MIT Han Lab，工业级，精度几乎无损 | [GitHub](https://github.com/mit-han-lab/omniserve) |
| **QAD (NVIDIA)** | 量化感知蒸馏，恢复接近 FP16 精度 | [arXiv:2601.20088](https://arxiv.org/abs/2601.20088) |

---

## 五、为什么选择 AWQ

1. **学术认可**：MLSys 2024 Best Paper，被广泛引用和集成
2. **生态成熟**：集成于 TensorRT-LLM、vLLM、HuggingFace TGI、LMDeploy 等
3. **易用性**：AutoAWQ、llm-awq 提供开箱即用的量化与推理
4. **泛化性**：对指令微调模型、多模态模型表现稳定
5. **工程友好**：无需训练，校准数据少，部署简单

---

## 六、参考链接汇总

### 论文与官方

| 资源 | 链接 |
|------|------|
| AWQ 论文 | https://arxiv.org/abs/2306.00978 |
| AWQ 官网 | https://hanlab.mit.edu/projects/awq |
| llm-awq 代码 | https://github.com/mit-han-lab/llm-awq |
| AutoAWQ | https://github.com/casper-hansen/AutoAWQ |

### Hugging Face

| 资源 | 链接 |
|------|------|
| Transformers AWQ 文档 | https://huggingface.co/docs/transformers/main/en/quantization/awq |
| AWQ 模型搜索 | https://huggingface.co/models?other=autoawq |

### W4A4 / W4A8 相关

| 资源 | 链接 |
|------|------|
| PrefixQuant 论文 | https://arxiv.org/abs/2410.05265 |
| PrefixQuant 代码 | https://github.com/ChenMnZ/PrefixQuant |
| QServe 论文 | https://arxiv.org/abs/2405.04532 |
| OmniServe (QServe) | https://github.com/mit-han-lab/omniserve |

### 蒸馏与优化

| 资源 | 链接 |
|------|------|
| QAD (NVIDIA) 论文 | https://arxiv.org/abs/2601.20088 |
| TensorRT-Model-Optimizer | https://github.com/NVIDIA/TensorRT-Model-Optimizer |

### Ashley 数据（示例）

| 资源 | 链接 |
|------|------|
| 171 份样本 | https://drive.google.com/drive/folders/15abez2hGSUg3ogRnhP7PDSAC1EUu5Y4U |
| 233 份样本 | https://drive.google.com/file/d/1KCTaGtf8F5RyAwBkOeyWrle7UoM3RmtO/view |

---

## 七、本项目中的 W4A4/W4A8 使用方式

### 7.1 量化命令

在项目根目录执行：

```bash
# W4A16（默认，与原有 quantize_medgamma_awq.py 一致）
python quantize_medgamma_awq_w4a4_w4a8.py \
  --model_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --output_path "./medgamma-awq-4bit" \
  --calibration_data "./mimic_train_cleaned.csv" \
  --num_samples 500 \
  --precision w4a16 \
  --mode quantize

# W4A8
python quantize_medgamma_awq_w4a4_w4a8.py \
  --model_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --output_path "./medgamma-awq-4bit" \
  --calibration_data "./mimic_train_cleaned.csv" \
  --precision w4a8 \
  --mode quantize

# W4A4
python quantize_medgamma_awq_w4a4_w4a8.py \
  --model_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --output_path "./medgamma-awq-4bit" \
  --calibration_data "./mimic_train_cleaned.csv" \
  --precision w4a4 \
  --mode quantize
```

### 7.2 输出说明

- **W4A16**：输出到 `--output_path`（如 `./medgamma-awq-4bit`）
- **W4A8**：输出到 `{output_path}-w4a8`（如 `./medgamma-awq-4bit-w4a8`）
- **W4A4**：输出到 `{output_path}-w4a4`（如 `./medgamma-awq-4bit-w4a4`）

权重量化流程完全一致，均使用 Hugging Face AutoAWQ。差异仅体现在输出目录和 `quantization_info.json` 中的 `activation_bits` 元数据。

### 7.3 激活量化说明

**激活(Activation)**：神经网络每层 Linear 的输入，即上一层的输出。W4A8 表示对 Linear 的输入做 8-bit 量化，W4A4 表示 4-bit。

**实现方式**（参考 [PrefixQuant](https://github.com/ChenMnZ/PrefixQuant) 的 `init_input_quantizer` + `QuantLinear.forward`）：
- 用 `forward_pre_hook` 在每层 Linear 的 forward 前对输入做 per-tensor 对称 fake 量化
- 8-bit：scale = max/127，量化到 [-128, 127]
- 4-bit：scale = max/7，量化到 [-8, 7]

### 7.4 推理说明

- **W4A16**：直接使用 `AutoAWQForCausalLM.from_quantized()` 加载
- **W4A8/W4A4**：使用 `load_with_activation_quant(model_path, activation_bits=8 或 4)` 加载，会自动注册激活量化 hooks；或 `--mode test --precision w4a8/w4a4` 测试

---

## 八、引用格式

```bibtex
@inproceedings{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Chen, Wei-Ming and Wang, Wei-Chen and Xiao, Guangxuan and Dang, Xingyu and Gan, Chuang and Han, Song},
  booktitle={MLSys},
  year={2024}
}
```
