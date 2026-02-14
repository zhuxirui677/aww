# RadGraph F1 复现 - 快速开始指南

## 🎯 目标

使用 **MedGamma1.5-4B-it** 模型生成放射学报告，并用 **RadGraph F1** 评估报告质量。

复现：推荐使用方法1

---

## 📋 准备工作

### 检查系统要求

```bash
# Python 版本 >= 3.8
python3 --version

# 推荐使用 GPU（可选）
nvidia-smi
```

---

## 🚀 方法一：使用快速开始脚本（推荐）（已经测试完毕）

### 1. 运行快速开始脚本

```bash
cd /Users/senzu/Desktop/radgraph-master
bash 快速开始.sh
```

这个脚本会：

- ✅ 检查 Python 环境
- ✅ 安装必要依赖
- ✅ 运行演示示例
- ✅ 验证安装是否成功

### 2. 查看演示结果

脚本会自动运行演示，展示：

- RadGraph 实体提取功能
- F1-RadGraph 分数计算
- 详细的评估结果

---

## 🔧 方法二：手动安装和配置（推荐）

### 步骤 1: 安装 RadGraph

```bash
cd /Users/senzu/Desktop/radgraph-master

# 1. 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 开发模式安装
pip install -e .

# 或者手动安装依赖
pip install torch>=2.1.0 transformers>=4.39.0 appdirs jsonpickle filelock h5py nltk dotmap pytest
```

### 步骤 2: 测试 RadGraph

```bash
# 运行演示脚本
pip install requests

# 或运行测试
pytest tests/
```

### 步骤 3: 使用示例数据

```bash

    pip install requests
# 使用提供的示例数据计算 F1 分数
# 1. 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 2. 显式指定缓存目录（避免系统权限问题）
mkdir -p ./radgraph_cache
export HUGGINGFACE_HUB_CACHE=./radgraph_cache

# 3. 运行
./.venv/bin/python3 run_f1_radgraph_demo.py

#4. 如果你已经有txt数据
#可以做的数据检查-》对其data：
wc -l example_references.txt
wc -l example_hypotheses.txt
——————————————————
python3 calculate_f1_score.py \
    --refs example_references.txt \
    --hyps example_hypotheses.txt \
    --model modern-radgraph-xl
# 或使用交互模式（这个不太好使亲测）
python3 calculate_f1_score.py
```

---

## 🤖 使用 MedGamma1.5-4B-it 模型

### 方法 A: 先生成报告，再评估（推荐）

#### 1. 准备 MedGamma 环境

```bash
# 安装额外依赖
pip install accelerate bitsandbytes
```

#### 2. 生成报告

```bash
# 使用示例提示生成报告（没有测试过）
python generate_with_medgamma.py \
    --input example_prompts.txt \
    --output medgamma_generated.txt \
    --model axiong/MedGamma-1.5-4B-it

# 或使用交互模式测试
python generate_with_medgamma.py --interactive
```

⚠️ **注意**: 

- 请确认 `axiong/MedGamma-1.5-4B-it` 是正确的模型名称
- 如果模型需要认证，运行：`huggingface-cli login`
- 首次运行会下载模型（可能需要几 GB 空间）

#### 3. 使用 RadGraph 评估

```bash
# 计算 F1 分数
python calculate_f1_score.py \
    --refs example_references.txt \
    --hyps medgamma_generated.txt \
    --model modern-radgraph-xl \
    --output evaluation_results.json
```

### 方法 B: 使用 Python 脚本

创建你自己的评估脚本：

```python
#!/usr/bin/env python3
"""自定义评估脚本"""

from radgraph import F1RadGraph

# 1. 准备数据
# 参考报告（ground truth）
refs = [
    "no acute cardiopulmonary abnormality",
    "bilateral pulmonary infiltrates",
    # ... 更多参考报告
]

# 使用 MedGamma 生成的报告
# (你需要先用 MedGamma 生成这些)
hyps = [
    "no acute cardiopulmonary findings",
    "bilateral lung infiltrates are present",
    # ... 更多生成的报告
]

# 2. 计算 F1 分数
f1radgraph = F1RadGraph(
    reward_level="all",  # 返回所有三个指标
    model_type="modern-radgraph-xl"  # 推荐使用
)

mean_reward, reward_list, _, _ = f1radgraph(hyps=hyps, refs=refs)

# 3. 显示结果
rg_e, rg_er, rg_bar_er = mean_reward
print(f"RadGraph F1 分数:")
print(f"  RG_E:      {rg_e:.4f}")
print(f"  RG_ER:     {rg_er:.4f}")  # ← 论文中常报告这个
print(f"  RG_ER_bar: {rg_bar_er:.4f}")
```

---

## 📊 评估指标说明

RadGraph F1 提供三个评估级别：


| 指标            | 说明           | 用途         |
| ------------- | ------------ | ---------- |
| **RG_E**      | 仅评估实体匹配      | 基础评估       |
| **RG_ER**     | 评估实体 + 关系存在性 | **论文常用** ⭐ |
| **RG_ER_bar** | 评估实体 + 完整关系  | 严格评估       |


通常在论文中报告 **RG_ER** 分数。

---

## 📁 项目文件说明

### 主要脚本


| 文件                          | 用途               |
| --------------------------- | ---------------- |
| `快速开始.sh`                   | 一键安装和测试脚本        |
| `run_f1_radgraph_demo.py`   | RadGraph 功能演示    |
| `calculate_f1_score.py`     | 计算 F1 分数的工具      |
| `generate_with_medgamma.py` | 使用 MedGamma 生成报告 |


### 示例数据


| 文件                       | 说明          |
| ------------------------ | ----------- |
| `example_references.txt` | 示例参考报告      |
| `example_hypotheses.txt` | 示例假设报告      |
| `example_prompts.txt`    | 示例输入提示      |
| `example_data.json`      | JSON 格式示例数据 |


### 文档


| 文件                  | 内容            |
| ------------------- | ------------- |
| `使用说明_中文.md`        | 详细使用指南        |
| `使用medgamma模型指南.md` | MedGamma 集成指南 |
| `快速开始指南.md`         | 本文档           |
| `README.md`         | 原始项目文档（英文）    |


---

## 🔍 常见问题

### Q1: 模型下载失败？

**解决方案**：

```bash
# 设置国内镜像（如果在中国）
export HF_ENDPOINT=https://hf-mirror.com

# 或者手动下载后指定路径
python calculate_f1_score.py --model-cache-dir /path/to/models
```

### Q2: GPU 内存不足？

**解决方案**：

```bash
# 使用 CPU
python calculate_f1_score.py --refs refs.txt --hyps hyps.txt

# 或使用 8-bit 量化（MedGamma）
python generate_with_medgamma.py --load-8bit --input prompts.txt
```

### Q3: MedGamma 模型找不到？

**解决方案**：

1. 确认模型名称是否正确
2. 检查是否需要访问权限：`huggingface-cli login`
3. 尝试搜索正确的模型：[https://huggingface.co/models?search=medgamma](https://huggingface.co/models?search=medgamma)
4. 如果找不到，可以使用其他医疗模型替代

### Q4: 如何在大数据集上评估？

**解决方案**：

```bash
# 分批处理（每批 100 条）
python << EOF
from radgraph import F1RadGraph
import numpy as np

# 加载数据
with open('all_refs.txt') as f:
    refs = [line.strip() for line in f]
with open('all_hyps.txt') as f:
    hyps = [line.strip() for line in f]

# 分批评估
batch_size = 100
f1radgraph = F1RadGraph(reward_level="all", model_type="modern-radgraph-xl")

all_scores = []
for i in range(0, len(refs), batch_size):
    batch_refs = refs[i:i+batch_size]
    batch_hyps = hyps[i:i+batch_size]
    scores, _, _, _ = f1radgraph(hyps=batch_hyps, refs=batch_refs)
    all_scores.append(scores)
    print(f"批次 {i//batch_size + 1} 完成")

# 计算总体平均
avg = tuple(np.mean([s[j] for s in all_scores]) for j in range(3))
print(f"总体 RG_ER: {avg[1]:.4f}")
EOF
```

---

## 🎓 完整工作流程示例

```bash
# 1. 安装依赖
cd /Users/senzu/Desktop/radgraph-master
pip install -e .

# 2. 测试 RadGraph 基本功能
python run_f1_radgraph_demo.py

# 3. 使用 MedGamma 生成报告（可选）
python3 generate_with_medgamma.py \
    --input example_prompts.txt \
    --output generated_reports.txt

# 4. 计算 F1 分数
python3 calculate_f1_score.py \
    --refs example_references.txt \
    --hyps example_hypotheses.txt \
    --output results.json

# 5. 查看结果
cat results.json
```

---

## 📚 更多资源

- **详细文档**: 阅读 `使用说明_中文.md`
- **MedGamma 集成**: 阅读 `使用medgamma模型指南.md`
- **原始项目**: [https://github.com/Stanford-AIMI/radgraph](https://github.com/Stanford-AIMI/radgraph)
- **论文**: [https://aclanthology.org/2024.findings-acl.765/](https://aclanthology.org/2024.findings-acl.765/)

---

## 💡 提示

1. **首次运行**: 会自动下载模型（约 1-2 GB），请耐心等待
2. **推荐模型**: 使用 `modern-radgraph-xl` 以获得最佳性能
3. **评估指标**: 论文中通常报告 **RG_ER** 分数
4. **数据格式**: 每行一条报告，参考和假设报告数量必须相同

---

## ✅ 验证安装

运行以下命令验证安装：

```bash
# 测试 RadGraph
python -c "from radgraph import RadGraph; print('✓ RadGraph 安装成功')"

# 测试 F1RadGraph
python -c "from radgraph import F1RadGraph; print('✓ F1RadGraph 安装成功')"

# 运行完整测试
pytest tests/
```

---

如有问题，请查看详细文档或访问项目主页。

祝使用愉快！

Ashley xu-github：sx2660-png