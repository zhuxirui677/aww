# MedGamma AWQ 量化完整指南

## 📋 目录
1. [快速开始](#快速开始)
2. [Hugging Face 上的 AWQ 资源](#hugging-face-资源)
3. [自己量化模型](#自己量化)
4. [性能对比评估](#性能评估)
5. [常见问题](#常见问题)

---

## 🚀 快速开始

### 方案一：使用 Hugging Face 上已量化的模型（最简单）

如果 Hugging Face 上已经有你想用的医疗模型的 AWQ 版本，直接下载即可：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 直接加载 AWQ 量化版本
model_id = "TheBloke/Mistral-7B-Instruct-AWQ"  # 示例
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 生成报告
prompt = "Generate a chest X-ray report..."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

**优点：**
- ✅ 无需量化过程，开箱即用
- ✅ 速度快，显存占用少
- ✅ 官方验证过质量

**缺点：**
- ❌ 不一定有你需要的医疗模型

---

## 🔍 Hugging Face 资源

### 1. 搜索已量化的医疗模型

访问：https://huggingface.co/models?other=autoawq

搜索关键词：
- `medical AWQ`
- `radiology AWQ`
- `clinical AWQ`

### 2. 常见 AWQ 模型库

| 用户/组织 | 说明 |
|----------|------|
| TheBloke | 最多 AWQ 模型，质量高 |
| gaunernst | Gemma 系列 AWQ 版本 |
| unsloth | 针对训练优化的量化模型 |

### 3. 如果没有医疗模型的 AWQ 版本怎么办？

**自己量化！** 👇

---

## 🛠️ 自己量化模型

### 安装依赖

```bash
# 安装 AutoAWQ
pip install autoawq transformers accelerate

# 如果需要 Intel CPU 优化
pip install intel-extension-for-pytorch
```

### 使用我们的脚本量化

```bash
# 1. 量化你的 medgamma 模型
python scripts/quantize_medgamma_awq.py \
    --model_path "mistralai/Mistral-7B-Instruct-v0.2" \
    --output_path "./medgamma-awq-4bit" \
    --calibration_data "./mimic_train_cleaned.csv" \
    --num_samples 500 \
    --prompt_file "./prompts/example_prompt.txt" \
    --mode quantize
```

**参数说明：**
- `--model_path`: 原始模型路径（HuggingFace ID 或本地路径）
- `--output_path`: 保存量化模型的位置
- `--calibration_data`: 你的 MIMIC-CXR 清洗数据（CSV/XLSX）
- `--num_samples`: 用于校准的样本数（建议 200-500）

### 量化过程

```
⏳ 预计时间: 10-30 分钟（取决于模型大小）
📊 显存需求: 原始模型大小 × 1.5

过程：
1. 加载原始模型 (2-5 分钟)
2. 准备校准数据 (1 分钟)
3. 计算激活统计 (5-15 分钟)
4. 量化权重 (3-8 分钟)
5. 保存模型 (1-2 分钟)
```

### 量化后的文件结构

```
medgamma-awq-4bit/
├── config.json
├── model.safetensors  # 量化后的权重
├── tokenizer.json
├── tokenizer_config.json
└── quantization_info.json  # 量化配置信息
```

---

## 📊 性能评估

### 1. 测试量化模型

```bash
# 生成一个示例报告
python scripts/quantize_medgamma_awq.py \
    --model_path "./medgamma-awq-4bit" \
    --mode test
```

### 2. 批量生成并计算 F1 Score

使用配套的评估脚本：

```bash
python scripts/evaluate_awq_model.py \
    --original_model "mistralai/Mistral-7B-Instruct-v0.2" \
    --quantized_model "./medgamma-awq-4bit" \
    --eval_data "./mimic_eval_cleaned.csv" \
    --output_dir "./evaluation_results"
```

这会生成：
- 原始模型的报告（Ground Truth）
- 量化模型的报告
- F1 Score 对比
- 速度对比
- 显存对比

---

## 📈 预期性能对比

| 指标 | 原始 FP16 | AWQ 4-bit | 提升 |
|------|-----------|-----------|------|
| 模型大小 | 14 GB | 3.5 GB | **75% ↓** |
| 显存占用 | 16 GB | 4.5 GB | **72% ↓** |
| 推理速度 | 45 tok/s | 90 tok/s | **2x ↑** |
| F1 Score | 0.850 | 0.845 | -0.5% |
| 成本 | A100 $3/h | 4090 $0.5/h | **83% ↓** |

---

## 🔧 集成到你的项目

### Ashley 的报告生成流程

**原来（FP16）：**
```python
# 1. 加载模型
model = AutoModel.from_pretrained("medgamma-fp16")

# 2. 生成报告
for patient in patients:
    report = model.generate(patient_images)  # 5秒
    save_report(report)
```

**现在（AWQ）：**
```python
from awq import AutoAWQForCausalLM

# 1. 加载量化模型
model = AutoAWQForCausalLM.from_quantized(
    "medgamma-awq-4bit",
    fuse_layers=True  # 🚀 启用加速
)

# 2. 生成报告（更快！）
for patient in patients:
    report = model.generate(patient_images)  # 2秒 ⚡
    save_report(report)
```

### 与 RadGraph F1 计算集成

```python
from radgraph import F1RadGraph

# 加载量化模型
awq_model = AutoAWQForCausalLM.from_quantized("medgamma-awq-4bit")

# 生成报告
generated_reports = []
for patient in eval_dataset:
    report = awq_model.generate(patient.images)
    generated_reports.append(report)

# 计算 F1
f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl")
mean_reward, _, _, _ = f1radgraph(
    hyps=generated_reports,
    refs=ground_truth_reports
)

print(f"AWQ 模型 F1 Score: {mean_reward}")
```

---

## ❓ 常见问题

### Q1: 我的模型是多模态的（图像+文本），可以量化吗？

**A:** 可以！但有注意事项：
- ✅ 文本生成部分可以量化
- ❌ 视觉编码器通常不量化（影响太大）
- 💡 只量化 LLM backbone 部分

### Q2: 量化会掉多少精度？

**A:** 经验值：
- 4-bit AWQ: F1 下降 0.5-1%
- 3-bit AWQ: F1 下降 1-2%
- 如果掉太多，可能是校准数据不够好

### Q3: 我的 GPU 显存不够量化怎么办？

**A:** 几种方案：
1. 用 CPU 量化（很慢，但可行）
2. 用更小的校准数据（100 samples）
3. 租用云 GPU（Colab, RunPod）
4. 分层量化（高级技巧）

### Q4: AWQ vs GPTQ vs GGUF，我该选哪个？

**A:** 快速决策树：
```
有 GPU 吗？
├─ 是 → 需要最快推理吗？
│   ├─ 是 → AWQ ✅
│   └─ 否 → GPTQ 或 AWQ 都可以
└─ 否 → GGUF（CPU 运行）
```

### Q5: 量化后模型还能微调吗？

**A:** 
- ❌ AWQ 量化后不能直接微调
- ✅ 可以用 QLoRA（量化+LoRA）继续训练
- 💡 建议：先微调，再量化

### Q6: 如何验证量化模型是否工作正常？

**A:** 3步验证：
```python
# 1. 加载测试
model = AutoAWQForCausalLM.from_quantized("model-awq")
print("✅ 模型加载成功")

# 2. 生成测试
output = model.generate(...)
print("✅ 能够生成文本")

# 3. 质量测试
f1_score = calculate_f1(...)
if f1_score > 0.80:
    print("✅ 质量合格")
```

---

## 📚 参考资源

### 官方文档
- AutoAWQ GitHub: https://github.com/casper-hansen/AutoAWQ
- HuggingFace AWQ Guide: https://huggingface.co/docs/transformers/main/en/quantization/awq

### 论文
- AWQ 原始论文: https://arxiv.org/abs/2306.00978
- RadGraph 论文: https://arxiv.org/abs/2106.14463

### 你们的 Repo
- F1 Score Calculator: https://github.com/sx2660-png/Redgraph-F1score-calculator

---

## 🎯 Siri 的任务清单

- [ ] 1. 安装 AutoAWQ 和依赖
- [ ] 2. 运行量化脚本
- [ ] 3. 验证量化模型能正常加载
- [ ] 4. 在 100 个样本上测试生成质量
- [ ] 5. 计算 F1 Score 对比（原始 vs 量化）
- [ ] 6. 测量推理速度和显存占用
- [ ] 7. 写实验报告（量化效果）
- [ ] 8. （可选）调整量化参数优化性能

---

**祝量化顺利！有问题随时问 🚀**
