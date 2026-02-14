# MedGamma AWQ + RadGraph F1 复现工程

这个仓库用于：
- 用清洗后的 MIMIC-CXR 文本做 AWQ 量化校准（AutoAWQ 仅支持部分模型）
- 用 MedGemma1.5-4B-it 生成报告
- 用 RadGraph F1 做质量评估

## 目录结构

```
.
├── scripts/                     # 主脚本（推荐从这里运行）
│   ├── quantize_medgamma_awq.py  # AWQ 量化
│   ├── evaluate_awq_model.py     # 原始 vs 量化评估
│   └── test_medgamma_clean.py    # MedGamma + RadGraph 一键评估
├── prompts/
│   └── example_prompt.txt        # 示例 prompt
├── config_example.json           # 示例配置
├── mimic_train_cleaned.csv       # 清洗数据（训练）
├── mimic_eval_cleaned.csv        # 清洗数据（评估）
├── README.md
└── quick_start.sh                # 兼容入口（会调用 scripts/）
```

根目录的 `quantize_medgamma_awq.py` / `evaluate_awq_model.py` / `test_medgamma_clean.py`
是兼容入口，实际逻辑在 `scripts/` 中。

## 工作流

### 1) 安装依赖
```
pip install torch transformers accelerate autoawq radgraph
```

### 2) 用 clean 数据做 AWQ 量化（你负责）
注意：AutoAWQ 目前不支持 `google/medgemma-1.5-4b-it`，请换成支持的模型（例如 Mistral/Llama/Qwen）。
```
python scripts/quantize_medgamma_awq.py \
  --model_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --output_path "./medgamma-awq-4bit" \
  --calibration_data "./mimic_train_cleaned.csv" \
  --num_samples 500 \
  --text_column "text" \
  --prompt_file "./prompts/example_prompt.txt" \
  --mode quantize
```

如果你更喜欢配置文件：
```
python scripts/quantize_medgamma_awq.py --config ./config_example.json --mode quantize
```

### 3) 评估 AWQ 量化效果（F1 + 速度 + 显存）
```
python scripts/evaluate_awq_model.py \
  --original_model "mistralai/Mistral-7B-Instruct-v0.2" \
  --quantized_model "./medgamma-awq-4bit" \
  --eval_data "./mimic_eval_cleaned.csv" \
  --prompt_file "./prompts/example_prompt.txt" \
  --num_samples 100
```

### 4) 仅做 MedGamma + RadGraph F1（不含 AWQ）
```
python scripts/test_medgamma_clean.py \
  --data "./mimic_eval_cleaned.csv" \
  --num_samples 10 \
  --prompt_file "./prompts/example_prompt.txt"
```

## prompt 使用说明

`prompts/example_prompt.txt` 已写入你的示例 prompt。  
如果想临时改 prompt，可以直接传 `--prompt_text`：
```
python scripts/test_medgamma_clean.py --prompt_text "Your prompt..."
```

如果 prompt 里需要用到数据列，可以写成模板，例如：
```
Findings: {text}
```
脚本会自动把 `{text}` 替换为对应数据列（默认列名 `text`）。

## 量化脚本改动说明

- 支持 `--config` 配置文件
- 支持 `--prompt_file` / `--prompt_template`
- 校准数据支持自定义 `--text_column`
- 量化配置可通过 CLI 覆盖（w_bit/group_size/zero_point/version）

## 备注

- `mimic_train_cleaned.csv` 和 `mimic_eval_cleaned.csv` 默认放在根目录
- 如需迁移到 `data/clean/`，只要同步修改 `--calibration_data` / `--eval_data` 路径即可

## 常见问题

- 模型下载慢：可设置 `HF_ENDPOINT` 或使用镜像
- 显存不足：减少 `--num_samples` 或分批评估
- F1 下降过大：增加校准样本或调小 `group_size`

# MedGamma AWQ 量化与 RadGraph F1 复现工程

医疗报告生成模型的 AWQ 量化、W4A4 与 W4A8 探索，以及 RadGraph F1 质量评估。

---

## 一、AWQ 原理

AWQ 全称 Activation-aware Weight Quantization，即激活感知的权重量化，由 MIT、上海交大、清华联合提出，获 MLSys 2024 最佳论文奖。

核心思想是：权重并非同等重要，只需保护约 1% 的显著权重即可大幅降低量化误差。与传统方法基于权重大小或二阶信息不同，AWQ 通过观察激活分布识别重要权重通道。激活幅度大的通道对应的权重更重要，因为它们处理更重要的特征。AWQ 对这些通道做 per-channel 缩放保护，再对所有权重做 4-bit 量化，同时保持全 INT4 格式，无需混合精度，硬件友好。

具体流程：用校准数据前向传播，收集每通道激活统计；根据激活幅度计算通道重要性；在 0 到 1 之间网格搜索最优缩放因子，最小化量化后输出与原始输出的差异；对权重做缩放后量化到 INT4；推理时通过缩放因子的逆与前一层的输出融合，实现等效计算。AWQ 不依赖反向传播或重建损失，校准数据需求约为 GPTQ 的十分之一，泛化性强，对指令微调模型和多模态模型表现稳定。

---

## 二、W4A8 与 W4A4 原理

W4A8 表示 4-bit 权重加 8-bit 激活，W4A4 表示 4-bit 权重加 4-bit 激活。其中 W 指权重，A 指激活。激活是神经网络每层 Linear 的输入，即上一层的输出。

AWQ 本身只做权重量化，即 W4A16（4-bit 权重、16-bit 激活）。W4A8 和 W4A4 需要在权重量化的基础上，额外对激活做量化。做法是在每层 Linear 的 forward 前，对输入做 per-tensor 对称量化：8-bit 时用 scale 将激活映射到 -128 到 127 的整数，4-bit 时映射到 -8 到 7，计算完成后再反量化回浮点。这样可进一步减少内存带宽和计算量，但需要专门的 INT4 乘 INT8 或 INT4 乘 INT4 的 GEMM kernel 才能获得实际加速。

W4A4 的难点在于：激活中的异常值会严重影响精度，per-group 量化需要大量元数据，反量化开销高，精度损失可达 9% 以上。PrefixQuant 通过前缀 token 消除异常值，使静态量化超越动态量化；QServe 采用 W4A8 加 4-bit KV cache，在精度和速度之间取得较好平衡。

---

## 三、优势与核心技术特点

AWQ 的优势：无需训练，仅需少量校准数据；不依赖梯度，不过拟合校准集；对指令微调模型、多模态模型泛化好；全 INT4 权重量化，易集成到 vLLM、TensorRT-LLM 等推理框架；模型体积约减少 75%，显存占用显著下降。

本项目的核心技术特点：基于 Hugging Face AutoAWQ 实现 4-bit 权重量化；参考 PrefixQuant 实现 W4A8 和 W4A4 的激活量化，通过 forward_pre_hook 在每层 Linear 前对输入做 fake 量化；提供 RadGraph F1 评估，包括 entities、relations 和 combined 指标；提供 MVP 数值验证脚本，验证 AWQ 反量化与激活量化的正确性；整理 QServe 完整推理指南和 AWQ 加 W4A8 加速实现路线图。

---

## 四、当前评估结果

在 Mistral-7B-Instruct 与 MIMIC-CXR 数据上的评估结果如下。显存占用从 13.5 GB 降至 3.9 GB，约减少 71%。F1 entities 从 0.188 升至 0.199，F1 relations 从 0.166 升至 0.174，F1 combined 从 0.103 略降至 0.100，退化约 3.3%。推理速度视 batch 和硬件而定。评估脚本为 scripts/evaluate_awq_model.py，结果保存在 evaluation_results 目录。

---

## 五、实际加速如何操作

本项目当前的 W4A8 和 W4A4 实现为 PyTorch 层面的 fake 量化，用于精度评估，不带来实际加速。要获得真实加速，有以下几种方式。

方式一：使用 QServe。QServe 实现 W4A8 加 4-bit KV cache 的完整推理，有融合 CUDA kernel，相比 TensorRT-LLM 可获得约 1.2 到 3.5 倍吞吐提升。操作步骤：克隆 OmniServe 仓库，安装依赖并编译 CUDA kernels，下载 QServe 格式的量化模型（如 Mistral-7B-QServe、Llama-3-8B-Instruct-QServe），用 qserve_e2e_generation.py 或 qserve_benchmark.py 进行推理和测速。注意 QServe 使用 QoQ 算法，模型格式与 AWQ 不同，需使用官方提供的 QServe 模型或按 DeepCompressor 文档自行量化后转换。详细步骤见 docs/QSERVE_GUIDE.md。

方式二：使用 TensorRT-LLM。TensorRT-LLM 已集成 QServe 风格的 W4A8 支持，可按其文档配置 W4A8 推理。

方式三：坚持使用 AWQ 权重并实现 W4A8 加速。这需要自行实现 INT4 乘 INT8 的 GEMM kernel，在 kernel 内完成激活量化、权重复用与反量化的融合，并与 AWQ 的 qweight、scales、qzeros 格式对接。技术路线包括：用 Triton 或 CUDA 实现 INT4 乘 INT8 GEMM，将量化、矩阵乘、反量化融合为单 kernel，替换模型中的 Linear 层。需 CUDA 或 Triton 经验。详细路线图、AWQ 格式说明和 MVP 验证脚本见 docs/AWQ_W4A4_W4A8_ACCELERATION_ROADMAP.md 和 scripts/mvp_awq_w4a8_verify.py。

---

## 六、目录结构

scripts 目录包含 quantize_medgamma_awq.py（AWQ 4-bit 量化）、quantize_medgamma_awq_w4a4_w4a8.py（W4A16、W4A8、W4A4 精度模式）、evaluate_awq_model.py（原始与量化模型对比评估）、mvp_awq_w4a8_verify.py（AWQ 反量化与激活量化数值验证）、test_medgamma_clean.py（MedGamma 加 RadGraph 一键评估）等脚本。

docs 目录包含 AWQ_DETAILED_GUIDE.md（AWQ 原理与数学推导）、QSERVE_GUIDE.md（QServe 完整推理加速指南）、AWQ_W4A4_W4A8_ACCELERATION_ROADMAP.md（AWQ 加 W4A8 加速实现路线图）。

根目录有 config_example.json、mimic_train_cleaned.csv、mimic_eval_cleaned.csv，以及 evaluation_results 目录存放评估结果。

---

## 七、快速开始

安装依赖：pip install torch transformers accelerate autoawq radgraph safetensors。

AWQ 4-bit 量化：python scripts/quantize_medgamma_awq.py --model_path "mistralai/Mistral-7B-Instruct-v0.2" --output_path "./medgamma-awq-4bit" --calibration_data "./mimic_train_cleaned.csv" --num_samples 500 --mode quantize。

W4A8 或 W4A4 量化：python quantize_medgamma_awq_w4a4_w4a8.py --model_path "mistralai/Mistral-7B-Instruct-v0.2" --calibration_data "./mimic_train_cleaned.csv" --precision w4a8 或 w4a4 --mode quantize。

评估：python scripts/evaluate_awq_model.py --original_model "mistralai/Mistral-7B-Instruct-v0.2" --quantized_model "./medgamma-awq-4bit" --eval_data "./mimic_eval_cleaned.csv" --num_samples 100。

MVP 数值验证：python scripts/mvp_awq_w4a8_verify.py 用于合成数据验证，加 --awq_model_path 可指定真实 AWQ 模型路径。

---

## 八、数据与说明

校准和训练使用 mimic_train_cleaned.csv，评估使用 mimic_eval_cleaned.csv。Ashley 的 171 份和 233 份样本链接见 docs/AWQ_DETAILED_GUIDE.md 中的链接汇总。

AutoAWQ 目前不支持 google/medgemma-1.5-4b-it，建议使用 Mistral、Llama、Qwen 等支持的模型。更多文档见 AWQ_QUANTIZATION_GUIDE.md、HUGGINGFACE_AWQ_RESOURCES.md。参考链接：AWQ 论文 https://arxiv.org/abs/2306.00978，llm-awq https://github.com/mit-han-lab/llm-awq，AutoAWQ https://github.com/casper-hansen/AutoAWQ，PrefixQuant https://github.com/ChenMnZ/PrefixQuant，QServe 与 OmniServe https://github.com/mit-han-lab/omniserve。

