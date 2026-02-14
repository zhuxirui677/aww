#!/bin/bash
# MedGamma AWQ 量化 - 快速开始脚本

set -e

echo "=============================================="
echo "MedGamma AWQ 量化工具 - 快速开始"
echo "=============================================="

echo ""
echo "检查 Python..."
if ! command -v python3 &> /dev/null; then
    echo "未找到 Python3"
    exit 1
fi
echo "Python 已安装"

echo ""
echo "安装依赖..."
pip install -q autoawq transformers accelerate radgraph torch
echo "依赖安装完成"

echo ""
echo "检查数据文件..."
if [ ! -f "mimic_train_cleaned.csv" ]; then
    echo "未找到 mimic_train_cleaned.csv，请确认数据文件在当前目录"
fi

echo ""
echo "接下来的步骤："
echo ""
echo "1) 量化模型（预计 10-30 分钟）："
echo "   python scripts/quantize_medgamma_awq.py \\"
echo "       --model_path \"google/medgemma-1.5-4b-it\" \\"
echo "       --output_path \"./medgamma-awq-4bit\" \\"
echo "       --calibration_data \"./mimic_train_cleaned.csv\" \\"
echo "       --num_samples 500 \\"
echo "       --prompt_file \"./prompts/example_prompt.txt\" \\"
echo "       --mode quantize"
echo ""
echo "2) 测试量化模型："
echo "   python scripts/quantize_medgamma_awq.py \\"
echo "       --model_path \"./medgamma-awq-4bit\" \\"
echo "       --mode test"
echo ""
echo "3) 性能评估："
echo "   python scripts/evaluate_awq_model.py \\"
echo "       --original_model \"google/medgemma-1.5-4b-it\" \\"
echo "       --quantized_model \"./medgamma-awq-4bit\" \\"
echo "       --eval_data \"./mimic_eval_cleaned.csv\" \\"
echo "       --prompt_file \"./prompts/example_prompt.txt\" \\"
echo "       --num_samples 100"
echo ""
echo "详细说明请查看 README.md"
