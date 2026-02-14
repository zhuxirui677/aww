# QServe (OmniServe) W4A8 完整推理加速指南

QServe 实现 **W4A8KV4**（4-bit 权重 + 8-bit 激活 + 4-bit KV cache）的完整推理，带来**真实加速**（约 1.2–3.5× 相比 TensorRT-LLM）。

---

## 一、环境要求

- **GPU**：NVIDIA A100 / L40S / RTX 4090 等（需 CUDA）
- **系统**：Linux 推荐
- **依赖**：Python 3.10、PyTorch、FlashAttention-2、CUDA Toolkit

---

## 二、安装步骤

```bash
# 1. 克隆 OmniServe（注意仓库名为 OmniServe，不是 omniserve）
git clone https://github.com/mit-han-lab/OmniServe
cd OmniServe

# 2. 创建 conda 环境
conda create -n OmniServe python=3.10 -y
conda activate OmniServe
pip install --upgrade pip

# 3. 安装 OmniServe 包
pip install -e .

# 4. 安装 FlashAttention-2
pip install flash-attn --no-build-isolation
# 若失败，可从 https://github.com/Dao-AILab/flash-attention/releases 下载预编译 wheel

# 5. 编译 CUDA kernels（关键步骤，实现 W4A8 加速）
pip install ninja
cd kernels
python setup.py install
cd ..

# 6. 安装 git-lfs（用于下载模型）
# Ubuntu: sudo apt install git-lfs
# Mac: brew install git-lfs
git lfs install
```

---

## 三、下载 QServe 量化模型

QServe 使用 **QoQ 算法**量化，格式与 AWQ 不同，需使用官方提供的 QServe 模型。

```bash
mkdir -p qserve_checkpoints && cd qserve_checkpoints

# 示例：Llama-3-8B-Instruct（适合医疗对话）
git clone https://huggingface.co/mit-han-lab/Llama-3-8B-Instruct-QServe

# 或 Mistral-7B（与 MedGamma 常用基座相近）
git clone https://huggingface.co/mit-han-lab/Mistral-7B-v0.1-QServe
```

### 可用模型一览

| 模型 | Per-channel | Per-group (g128) |
|------|-------------|------------------|
| Llama-3-8B | [链接](https://huggingface.co/mit-han-lab/Llama-3-8B-QServe) | [链接](https://huggingface.co/mit-han-lab/Llama-3-8B-QServe-g128) |
| Llama-3-8B-Instruct | [链接](https://huggingface.co/mit-han-lab/Llama-3-8B-Instruct-QServe) | [链接](https://huggingface.co/mit-han-lab/Llama-3-8B-Instruct-QServe-g128) |
| Mistral-7B | [链接](https://huggingface.co/mit-han-lab/Mistral-7B-v0.1-QServe) | [链接](https://huggingface.co/mit-han-lab/Mistral-7B-v0.1-QServe-g128) |
| Llama-2-7B | [链接](https://huggingface.co/mit-han-lab/Llama-2-7B-QServe) | [链接](https://huggingface.co/mit-han-lab/Llama-2-7B-QServe-g128) |

- **A100**：建议 per-channel（`--group-size -1`）
- **L40S / 消费级 GPU**：建议 per-group（`--group-size 128`）

---

## 四、推理与 benchmark

### 4.1 端到端生成（在线推理）

```bash
# 在 OmniServe 项目根目录执行
export MODEL_PATH=./qserve_checkpoints/Llama-3-8B-Instruct-QServe

python qserve_e2e_generation.py \
  --model $MODEL_PATH \
  --ifb-mode \
  --precision w4a8kv4 \
  --quant-path $MODEL_PATH \
  --group-size -1
```

### 4.2 速度 benchmark

```bash
export MODEL_PATH=./qserve_checkpoints/Llama-3-8B-QServe

# 固定 context=1024, generation=512，测最大吞吐
GLOBAL_BATCH_SIZE=128 \
python qserve_benchmark.py \
  --model $MODEL_PATH \
  --benchmarking \
  --precision w4a8kv4 \
  --group-size -1
```

### 4.3 一键脚本

```bash
# 端到端生成
bash scripts/qserve_e2e.sh

# A100 benchmark
bash scripts/qserve_benchmark/benchmark_a100.sh

# L40S benchmark
bash scripts/qserve_benchmark/benchmark_l40s.sh
```

---

## 五、常用参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model` | 模型配置目录（含 config.json） | `./qserve_checkpoints/Llama-3-8B-QServe` |
| `--quant-path` | 量化权重目录 | 通常与 `--model` 相同 |
| `--precision` | 精度模式 | `w4a8kv4`（默认）、`w4a8kv8`、`w8a8kv4` |
| `--group-size` | 权重量化分组 | `-1`（per-channel）或 `128` |
| `--ifb-mode` | 启用 in-flight batching | 推荐在 e2e 生成时开启 |
| `--benchmarking` | 速度 profiling 模式 | 用于 benchmark |

### 环境变量

| 变量 | 说明 |
|------|------|
| `GLOBAL_BATCH_SIZE` | benchmark 时的 batch size（如 128、256） |
| `NUM_GPU_PAGE_BLOCKS` | GPU 页数，可设为 `25 * batch_size` |

---

## 六、与 MedGamma 的对接

### 当前限制

- QServe 模型需用 **QoQ 算法**量化，与 AWQ 格式不兼容
- 官方模型库暂无医疗专用模型

### 可选方案

1. **直接用通用 QServe 模型**  
   使用 Llama-3-8B-Instruct-QServe 或 Mistral-7B-QServe 做医疗报告生成，再按需微调 prompt。

2. **自量化 QServe 模型**  
   - 用 [DeepCompressor](https://github.com/mit-han-lab/deepcompressor/tree/lmquant-v0.0.0-deprecated) 做 QoQ 量化  
   - 用 OmniServe 的 `checkpoint_converter.py` 转成 QServe 格式：
     ```bash
     python checkpoint_converter.py --model-path <path> --quant-path <path> --group-size -1 --device cpu
     ```

3. **双轨使用**  
   - 精度评估：继续用本项目的 AWQ + fake 激活量化  
   - 生产推理：用 QServe 的 W4A8 模型做加速

---

## 七、预期加速效果

| GPU | Llama-3-8B | Mistral-7B | Llama-2-13B |
|-----|------------|------------|-------------|
| L40S | 1.4× vs TRT-LLM | 1.47× | 3.02× |
| A100 | 1.2× vs TRT-LLM | 1.22× | 1.36× |

QServe 在 L40S 上可达 A100 级吞吐，成本约降 3×。

---

## 八、参考链接

- **OmniServe 仓库**：https://github.com/mit-han-lab/OmniServe
- **QServe 论文**：https://arxiv.org/abs/2405.04532
- **QServe 官网**：https://hanlab.mit.edu/projects/qserve
- **模型 Zoo**：https://huggingface.co/mit-han-lab
