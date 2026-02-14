# Hugging Face 上的 AWQ 医疗模型资源

## 🔍 如何搜索 AWQ 模型

### 方法 1：直接搜索
访问：https://huggingface.co/models?other=autoawq

在搜索框输入：
- `medical AWQ`
- `clinical AWQ`
- `radiology AWQ`
- `biomedical AWQ`

### 方法 2：按用户/组织筛选

```
https://huggingface.co/TheBloke?search=AWQ
https://huggingface.co/models?author=TheBloke&other=autoawq
```

---

## 📦 常见医疗/通用 AWQ 模型

### 1. 通用医疗模型（可能适用）

虽然 Hugging Face 上专门的医疗 AWQ 模型不多，但以下通用模型可能对医疗任务有帮助：

| 模型 | 大小 | 说明 | HF 链接 |
|------|------|------|---------|
| Mistral-7B-Instruct-AWQ | 7B | 通用指令模型，适合微调 | TheBloke/Mistral-7B-Instruct-v0.2-AWQ |
| Llama-2-7B-AWQ | 7B | Meta 的基础模型 | TheBloke/Llama-2-7B-AWQ |
| Gemma-7B-AWQ | 7B | Google 的开源模型 | 搜索 "gemma awq" |
| Qwen-7B-AWQ | 7B | 阿里的多语言模型 | 搜索 "qwen awq" |

### 2. 如果找不到医疗专用的 AWQ 版本

**方案 A：自己量化（推荐）**
```bash
# 使用我们提供的脚本
python quantize_medgamma_awq.py \
    --model_path "你的医疗模型" \
    --output_path "./model-awq"
```

**方案 B：找原始医疗模型 + AWQ 脚本**
```python
# 示例：量化 BioGPT
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained("microsoft/BioGPT")
model.quantize(...)
model.save_quantized("./BioGPT-AWQ")
```

---

## 🎯 医疗领域常见模型（需自己量化）

以下是 Hugging Face 上流行的医疗模型，你可以用我们的工具量化它们：

### 医疗文本模型

| 模型名称 | 专长 | HF 链接 |
|---------|------|---------|
| **BioGPT** | 生物医学文本生成 | microsoft/BioGPT |
| **PubMedBERT** | 医学文献理解（BERT 架构，不适合生成） | microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract |
| **ClinicalBERT** | 临床笔记分析 | emilyalsentzer/Bio_ClinicalBERT |
| **BioBERT** | 生物医学 NER/QA | dmis-lab/biobert-v1.1 |

### 医疗多模态模型

| 模型名称 | 专长 | HF 链接 |
|---------|------|---------|
| **MedGamma** | 医疗影像报告生成 | google/medgamma-* |
| **LLaVA-Med** | 医疗视觉问答 | microsoft/llava-med |
| **MAIRA** | X-ray 分析 | microsoft/maira |

---

## 🛠️ 量化这些模型的步骤

### 示例：量化 BioGPT

```bash
# 1. 下载模型
git lfs install
git clone https://huggingface.co/microsoft/BioGPT

# 2. 准备医疗文本校准数据（从你的 MIMIC 数据）
python prepare_calibration.py \
    --input mimic_train_cleaned.csv \
    --output biogpt_calib.json \
    --num_samples 500

# 3. 量化
python quantize_medgamma_awq.py \
    --model_path "microsoft/BioGPT" \
    --output_path "./BioGPT-AWQ" \
    --calibration_data "biogpt_calib.json" \
    --mode quantize

# 4. 测试
python quantize_medgamma_awq.py \
    --model_path "./BioGPT-AWQ" \
    --mode test
```

---

## 📊 AWQ 模型识别方法

### 如何确认一个模型是 AWQ 量化的？

#### 方法 1：检查模型名称
- 包含 `AWQ` 或 `awq` 关键词
- 示例：`TheBloke/Mistral-7B-AWQ`

#### 方法 2：检查 config.json
```json
{
  "quantization_config": {
    "quant_method": "awq",
    "zero_point": true,
    "group_size": 128,
    "bits": 4,
    "version": "gemm"
  }
}
```

#### 方法 3：检查文件大小
- 原始 7B 模型：~14 GB
- AWQ 4-bit：~3.5 GB
- 如果是 ~3-4 GB，很可能是量化过的

---

## 🔗 有用的链接

### 官方资源
- **AutoAWQ GitHub**: https://github.com/casper-hansen/AutoAWQ
- **HuggingFace AWQ 文档**: https://huggingface.co/docs/transformers/quantization/awq
- **AWQ 论文**: https://arxiv.org/abs/2306.00978

### 社区资源
- **TheBloke 的所有 AWQ 模型**: https://huggingface.co/TheBloke?search=AWQ
- **HF 论坛 - 量化讨论**: https://discuss.huggingface.co/c/quantization
- **Reddit r/LocalLLaMA**: https://reddit.com/r/LocalLLaMA（量化经验分享）

### 医疗 AI 资源
- **PhysioNet MIMIC**: https://physionet.org/
- **Stanford AIMI**: https://stanfordaimi.azurewebsites.net/
- **RadGraph**: https://github.com/jbdel/radgraph

---

## 💡 实用技巧

### 技巧 1：搜索特定用户的 AWQ 模型
```
site:huggingface.co TheBloke AWQ medical
```

### 技巧 2：按模型大小筛选
```
在 HuggingFace 搜索页面：
1. 输入 "AWQ"
2. 在左侧 Filters 选择 Model size
3. 选择适合你 GPU 的大小（如 < 10GB）
```

### 技巧 3：查看模型卡片的量化信息
```
访问模型页面 → README → 查找:
- Quantization method
- Bits per weight
- Group size
- 性能基准
```

---

## 📝 如果你想分享你的 AWQ 模型

### 上传到 Hugging Face

```bash
# 1. 登录
huggingface-cli login

# 2. 创建 repo
huggingface-cli repo create your-model-awq --type model

# 3. 上传
cd medgamma-awq-4bit
git lfs install
git init
git remote add origin https://huggingface.co/your-username/your-model-awq
git add .
git commit -m "Add AWQ quantized model"
git push origin main
```

### 模型卡片模板

```markdown
---
tags:
- medical
- radiology
- awq
- quantized
- 4-bit
license: apache-2.0
---

# MedGamma-3B-AWQ

AWQ 4-bit quantized version of google/medgamma-3b for medical report generation.

## Model Details
- **Original Model**: google/medgamma-3b
- **Quantization**: 4-bit AWQ
- **Group Size**: 128
- **Model Size**: 3.5 GB (75% reduction)

## Performance
- **F1 Score**: 0.845 (vs 0.850 original, -0.6%)
- **Inference Speed**: 2.5x faster
- **VRAM**: 5 GB (vs 16 GB original)

## Usage
\```python
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_quantized("your-username/medgamma-awq")
\```
```

---

## 🎓 学习资源

### 视频教程
- YouTube: "AWQ Quantization Explained"
- YouTube: "Quantizing LLMs for Production"

### 博客文章
- HuggingFace Blog: "AWQ Quantization"
- Medium: "Guide to LLM Quantization"

### 研究论文
- AWQ (2023): Activation-aware Weight Quantization
- GPTQ (2022): Post-Training Quantization
- SmoothQuant (2023): Mixed Precision

---

**祝你找到合适的模型！🚀**

如果有问题，可以：
1. 在 AutoAWQ GitHub 提 Issue
2. 在 HuggingFace 论坛发帖
3. 找你的队友 Lili 和 Ashley 讨论
