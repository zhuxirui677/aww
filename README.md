# MedGamma AWQ + RadGraph F1 复现工程

这个仓库用于：
- 用清洗后的 MIMIC-CXR 文本做 AWQ 量化校准（AutoAWQ 仅支持部分模型）
- 用 MedGemma1.5-4B-it 生成报告
- 用 RadGraph F1 做质量评估

你负责 AWQ 量化部分，我已把需要的脚本和数据路径整理好。

## 目录结构

```
.
├── scripts/                     # 主脚本（推荐从这里运行）
│   ├── quantize_medgamma_awq.py  # AWQ 量化
│   ├── evaluate_awq_model.py     # 原始 vs 量化评估
│   └── test_medgamma_clean.py    # MedGamma + RadGraph 一键评估
├── prompts/
│   └── example_prompt.txt        # 你的示例 prompt
├── config_example.json           # 示例配置
├── mimic_train_cleaned.csv       # 清洗数据（训练）
├── mimic_eval_cleaned.csv        # 清洗数据（评估）
├── README.md
└── quick_start.sh                # 兼容入口（会调用 scripts/）
```

根目录的 `quantize_medgamma_awq.py` / `evaluate_awq_model.py` / `test_medgamma_clean.py`
是兼容入口，实际逻辑在 `scripts/` 中。

## 你的工作流（推荐）

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
如果你想临时改 prompt，可以直接传 `--prompt_text`：
```
python scripts/test_medgamma_clean.py --prompt_text "Your prompt..."
```

如果 prompt 里需要用到数据列，可以写成模板，例如：
```
Findings: {text}
```
脚本会自动把 `{text}` 替换为对应数据列（默认列名 `text`）。

## 量化脚本改动说明（已检查并修正）

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
