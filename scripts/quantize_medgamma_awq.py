#!/usr/bin/env python3
"""
AWQ 量化脚本 - MedGamma 医疗报告生成模型
用途：将原始 FP16 模型量化为 4-bit AWQ 格式，减少显存占用和提升推理速度
"""

import os
import json
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

try:
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
except ImportError:
    print("缺少依赖，请安装：")
    print("pip install autoawq transformers accelerate")
    raise


def _get_nested(config: Dict, keys: List[str], default=None):
    cur = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _load_prompt(prompt_file: str = None, prompt_template: str = None) -> str:
    if prompt_template:
        return prompt_template.strip()
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "Generate a radiology report:\n{text}"


def _build_prompt(template: str, text: str) -> str:
    if "{text}" in template:
        return template.format_map(defaultdict(str, {"text": text}))
    return f"{template}\n{text}"


class MedGammaAWQQuantizer:
    """MedGamma 模型 AWQ 量化器"""

    def __init__(
        self,
        model_path: str,
        output_path: str,
        calibration_data_path: str = None,
        num_calibration_samples: int = 500,
        text_column: str = "text",
        quant_config: Dict = None,
        prompt_template: str = None,
        trust_remote_code: bool = True,
    ):
        self.model_path = model_path
        self.output_path = output_path
        self.calibration_data_path = calibration_data_path
        self.num_calibration_samples = num_calibration_samples
        self.text_column = text_column
        self.quant_config = quant_config or {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM",
        }
        self.prompt_template = prompt_template or "Generate a radiology report:\n{text}"
        self.trust_remote_code = trust_remote_code

        print("=" * 60)
        print("MedGamma AWQ 量化工具")
        print("=" * 60)

    def prepare_calibration_data(self) -> List[str]:
        """准备校准数据"""
        print("\n准备校准数据...")

        if self.calibration_data_path and os.path.exists(self.calibration_data_path):
            if self.calibration_data_path.endswith(".csv"):
                df = pd.read_csv(self.calibration_data_path)
            else:
                df = pd.read_excel(self.calibration_data_path)

            if self.text_column not in df.columns:
                raise ValueError(
                    f"找不到列 {self.text_column}，可用列：{list(df.columns)}"
                )

            texts = df[self.text_column].dropna().tolist()
            import random

            random.seed(42)
            if len(texts) > self.num_calibration_samples:
                texts = random.sample(texts, self.num_calibration_samples)

            print(f"已提取 {len(texts)} 条报告用于校准")
            return texts

        print("未提供校准数据路径，使用默认医疗报告样本")
        return self._get_default_calibration_data()

    def _get_default_calibration_data(self) -> List[str]:
        return [
            "No focal consolidation, pleural effusion or pneumothorax. Normal cardiomediastinal silhouette.",
            "Mild pulmonary edema with small bilateral pleural effusions.",
            "Right lower lobe consolidation consistent with pneumonia. No pleural effusion.",
            "Cardiomegaly with pulmonary vascular congestion. Small bilateral effusions.",
            "Left apical pneumothorax. No focal consolidation or effusion.",
        ] * 100

    def quantize(self):
        """执行 AWQ 量化"""

        print(f"\n开始量化模型: {self.model_path}")
        print(f"输出路径: {self.output_path}")

        print("\n加载原始模型...")
        try:
            model = AutoAWQForCausalLM.from_pretrained(
                self.model_path, device_map="auto", trust_remote_code=self.trust_remote_code
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=self.trust_remote_code
            )
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("提示：确认 model_path 为 HuggingFace ID 或本地路径")
            return

        calibration_texts = self.prepare_calibration_data()

        calibration_data = []
        for text in calibration_texts[: self.num_calibration_samples]:
            prompt = _build_prompt(self.prompt_template, text)
            calibration_data.append({"text": prompt})

        print("\n量化配置:")
        print(f"  位宽: {self.quant_config['w_bit']}-bit")
        print(f"  分组大小: {self.quant_config['q_group_size']}")
        print(f"  Zero-point: {self.quant_config['zero_point']}")
        print(f"  内核: {self.quant_config['version']}")

        print("\n开始量化（预计 10-30 分钟）...")
        try:
            model.quantize(
                tokenizer,
                quant_config=self.quant_config,
                calib_data=calibration_data,
            )
            print("量化完成")
        except Exception as e:
            print(f"量化失败: {e}")
            return

        print(f"\n保存量化模型到 {self.output_path}...")
        try:
            os.makedirs(self.output_path, exist_ok=True)
            model.save_quantized(self.output_path)
            tokenizer.save_pretrained(self.output_path)

            config_info = {
                "original_model": self.model_path,
                "quantization_method": "AWQ",
                "config": self.quant_config,
                "num_calibration_samples": len(calibration_data),
                "text_column": self.text_column,
                "prompt_template": self.prompt_template,
            }
            with open(
                os.path.join(self.output_path, "quantization_info.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(config_info, f, indent=2, ensure_ascii=False)

            print("保存成功")
        except Exception as e:
            print(f"保存失败: {e}")
            return

        self._print_stats()

    def _print_stats(self):
        print("\n" + "=" * 60)
        print("量化统计")
        print("=" * 60)

        def get_model_size(path):
            total_size = 0
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith((".bin", ".safetensors")):
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
            return total_size / (1024**3)

        quantized_size = get_model_size(self.output_path)
        print(f"量化后模型大小: {quantized_size:.2f} GB")
        print("预期推理速度提升: 2-3x")
        print(f"预期显存占用: ~{quantized_size * 1.2:.2f} GB")
        print("\n下一步:")
        print("  1. 用量化模型生成报告")
        print("  2. 用 RadGraph F1 评估质量")
        print("  3. 对比原始模型 vs 量化模型性能")
        print("=" * 60)


def load_quantized_model(model_path: str, trust_remote_code: bool = True):
    """加载已量化的 AWQ 模型（用于推理）"""
    print(f"加载量化模型: {model_path}")

    model = AutoAWQForCausalLM.from_quantized(
        model_path,
        fuse_layers=True,
        trust_remote_code=trust_remote_code,
        safetensors=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )

    print("模型加载成功")
    return model, tokenizer


def generate_report_example(model, tokenizer, prompt_text: str = None):
    if not prompt_text:
        prompt_text = "INDICATION: Chest pain and shortness of breath\n\nFINDINGS:"

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MedGamma AWQ 量化工具")
    parser.add_argument("--model_path", type=str, help="原始模型路径")
    parser.add_argument(
        "--output_path", type=str, default="./medgamma-awq-4bit", help="输出路径"
    )
    parser.add_argument(
        "--calibration_data", type=str, default=None, help="校准数据路径"
    )
    parser.add_argument("--num_samples", type=int, default=500, help="校准样本数量")
    parser.add_argument("--text_column", type=str, default="text", help="文本列名")
    parser.add_argument("--w_bit", type=int, default=4, help="量化位宽")
    parser.add_argument("--group_size", type=int, default=128, help="分组大小")
    parser.add_argument("--zero_point", type=str, default="true", help="是否启用 zero-point")
    parser.add_argument("--version", type=str, default="GEMM", help="量化内核版本")
    parser.add_argument("--prompt_file", type=str, default=None, help="prompt 文件路径")
    parser.add_argument("--prompt_template", type=str, default=None, help="prompt 模板")
    parser.add_argument("--config", type=str, default=None, help="JSON 配置文件路径")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["quantize", "test"],
        default="quantize",
        help="运行模式",
    )

    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

    model_path = args.model_path or _get_nested(config, ["model", "original_model_path"])
    output_path = args.output_path or _get_nested(config, ["model", "quantized_model_path"], "./medgamma-awq-4bit")
    calibration_data_path = args.calibration_data or _get_nested(
        config, ["calibration", "data_path"]
    )
    num_samples = args.num_samples or _get_nested(
        config, ["calibration", "num_samples"], 500
    )
    text_column = args.text_column or _get_nested(
        config, ["calibration", "text_column"], "text"
    )
    prompt_file = args.prompt_file or _get_nested(
        config, ["generation", "prompt_file"]
    )
    prompt_template = args.prompt_template or _get_nested(
        config, ["generation", "prompt_template"]
    )

    quant_config = {
        "zero_point": str(args.zero_point).lower() != "false"
        if args.zero_point is not None
        else _get_nested(config, ["quantization", "zero_point"], True),
        "q_group_size": args.group_size
        or _get_nested(config, ["quantization", "group_size"], 128),
        "w_bit": args.w_bit or _get_nested(config, ["quantization", "bits"], 4),
        "version": args.version or _get_nested(config, ["quantization", "version"], "GEMM"),
    }

    if not model_path:
        raise ValueError("未提供 model_path，请使用 --model_path 或配置文件")

    prompt_text = _load_prompt(prompt_file, prompt_template)

    if args.mode == "quantize":
        quantizer = MedGammaAWQQuantizer(
            model_path=model_path,
            output_path=output_path,
            calibration_data_path=calibration_data_path,
            num_calibration_samples=num_samples,
            text_column=text_column,
            quant_config=quant_config,
            prompt_template=prompt_text,
        )
        quantizer.quantize()
    else:
        model, tokenizer = load_quantized_model(output_path)
        output = generate_report_example(model, tokenizer)
        print(output)


if __name__ == "__main__":
    main()
