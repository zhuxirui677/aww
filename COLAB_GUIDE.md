# 在 Colab 上运行的指导

本指南面向你当前的流程：  
clean CSV → MedGamma 生成报告 → RadGraph F1 评估（以及可选的 AWQ 量化）

## 0. 你需要准备什么

- 一个 Colab 账号
- 你的代码仓库（本项目）
- clean 数据 CSV（至少需要包含 `text` 参考报告列）
- 如果模型需要权限：提前 `huggingface-cli login` 或设置 `HF_TOKEN`

## 1. 在 Colab 里打开项目

方式 A：上传压缩包（简单）
1) 把本项目打包为 zip  
2) 在 Colab 左侧 Files 面板上传 zip  
3) 在 Colab 运行：
```
!unzip medgamma.zip -d /content/medgamma
%cd /content/medgamma
```

方式 B：挂载 Google Drive（适合反复使用）
```
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/medgamma
```

## 2. 安装依赖

```
!pip install -q torch transformers accelerate autoawq radgraph
```

如果在国内或下载慢，可以设置镜像：
```
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

## 3. 准备数据（重点）

你的 clean CSV 里只有图片路径，没有报告文本时：
- 不能做 RadGraph F1（因为 F1 需要参考报告文本）
- 只能先做“生成”或“推理速度”测试

要做 F1，请确保 CSV 至少有 `text` 列（参考报告）

## 4. 运行 MedGamma + RadGraph F1（文本评估）

```
!python scripts/test_medgamma_clean_en.py \
  --data "./mimic_eval_cleaned.csv" \
  --num_samples 10 \
  --prompt_file "./prompts/example_prompt.txt"
```

## 5. 运行 AWQ 量化（你负责）

```
!python scripts/quantize_medgamma_awq.py \
  --model_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --output_path "./medgamma-awq-4bit" \
  --calibration_data "./mimic_train_cleaned.csv" \
  --num_samples 500 \
  --text_column "text" \
  --prompt_file "./prompts/example_prompt.txt" \
  --mode quantize
```

## 6. 评估 AWQ 量化效果

```
!python scripts/evaluate_awq_model.py \
  --original_model "mistralai/Mistral-7B-Instruct-v0.2" \
  --quantized_model "./medgamma-awq-4bit" \
  --eval_data "./mimic_eval_cleaned.csv" \
  --prompt_file "./prompts/example_prompt.txt" \
  --num_samples 100
```

## 7. FAQ（关于“图片路径”和“对话”）

### Q1: “对话”是什么意思？
这里的“对话”就是模型的 **输入提示（prompt）**，  
用来告诉 MedGamma 按什么格式输出报告。  
你现在的 prompt 在 `prompts/example_prompt.txt`。

### Q2: CSV 里只有图片路径，没有下载图片，能跑吗？
可以分两种情况：

1) 只做 **文本评估**（当前脚本）
   - 需要 `text` 作为参考报告
   - 不需要图片
   - 如果没有 `text`，就不能算 RadGraph F1

2) 做 **图像输入的多模态生成**
   - 必须有图片文件
   - 图片路径要指向真实文件
   - 你需要在 Colab 下载/挂载图片数据

### Q3: 对话里提到的两种解决方法是啥？
你截图里说的两种方法是：

方法 A：本地下载后再上传/挂载  
- 先在本地把数据下载好  
- 再上传到 Colab 或挂载 Google Drive  
- 适合数据很大或需要重复使用

方法 B：在 Kaggle/Colab 里直接 link dataset  
- 在 Notebook 里直接添加 Kaggle dataset  
- 不需要自己手动下载  
- 速度通常更快，步骤也更少

如果你的 CSV 只有图片路径，**建议用方法 B**（直接 link dataset），
这样路径和实际文件更容易对齐。

