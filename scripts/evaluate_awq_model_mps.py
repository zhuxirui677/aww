#!/usr/bin/env python3
"""
AWQ 量化模型评估脚本（Apple GPU / MPS）
注意：AutoAWQ 量化模型需要 CUDA，不支持 MPS。
"""

import os
import json
import time
import torch
import pandas as pd
import numpy as np
from typing import Dict, List
from collections import defaultdict

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from awq import AutoAWQForCausalLM
    from radgraph import F1RadGraph
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请安装: pip install transformers autoawq radgraph")
    raise


def select_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_prompt(prompt_file: str = None, prompt_text: str = None) -> str:
    if prompt_text:
        return prompt_text.strip()
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "Generate a radiology report for this chest X-ray."


def _format_prompt(template: str, row: pd.Series = None) -> str:
    if row is None:
        return template
    data = defaultdict(
        str,
        {k: "" if pd.isna(v) else str(v) for k, v in row.to_dict().items()},
    )
    try:
        return template.format_map(data)
    except Exception:
        return template


class ModelEvaluator:
    def __init__(
        self,
        original_model_path: str,
        quantized_model_path: str,
        eval_data_path: str,
        output_dir: str = "./evaluation_results",
        prompt_template: str = None,
        text_column: str = "text",
        radgraph_model: str = "modern-radgraph-xl",
    ):
        self.original_model_path = original_model_path
        self.quantized_model_path = quantized_model_path
        self.eval_data_path = eval_data_path
        self.output_dir = output_dir
        self.prompt_template = prompt_template or "Generate a radiology report."
        self.text_column = text_column
        self.radgraph_model = radgraph_model

        os.makedirs(output_dir, exist_ok=True)

        print("=" * 70)
        print("MedGamma AWQ 量化评估 (MPS)")
        print("=" * 70)

    def load_models(self):
        print("\n加载模型...")

        device = select_device()
        torch_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        device_map = "auto" if device == "cuda" else None

        print("  加载原始模型...")
        start = time.time()
        self.original_model = AutoModelForCausalLM.from_pretrained(
            self.original_model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        if device != "cuda":
            self.original_model.to(device)
        self.original_tokenizer = AutoTokenizer.from_pretrained(
            self.original_model_path, trust_remote_code=True
        )
        original_load_time = time.time() - start
        print(f"  原始模型耗时: {original_load_time:.2f}秒")

        if device != "cuda":
            print(
                "  [ERROR] AutoAWQ 量化模型需要 CUDA，MPS/CPU 不支持。"
            )
            raise RuntimeError("AutoAWQ requires CUDA for quantized model inference.")

        print("  加载量化模型...")
        start = time.time()
        self.quantized_model = AutoAWQForCausalLM.from_quantized(
            self.quantized_model_path,
            fuse_layers=True,
            trust_remote_code=True,
            safetensors=True,
        )
        self.quantized_tokenizer = AutoTokenizer.from_pretrained(
            self.quantized_model_path, trust_remote_code=True
        )
        quantized_load_time = time.time() - start
        print(f"  量化模型耗时: {quantized_load_time:.2f}秒")

        return {
            "original_load_time": original_load_time,
            "quantized_load_time": quantized_load_time,
        }

    def load_eval_data(self, num_samples: int = None) -> pd.DataFrame:
        print(f"\n加载评估数据: {self.eval_data_path}")
        if self.eval_data_path.endswith(".csv"):
            df = pd.read_csv(self.eval_data_path)
        else:
            df = pd.read_excel(self.eval_data_path)

        if num_samples and num_samples < len(df):
            df = df.sample(n=num_samples, random_state=42)

        print(f"  加载了 {len(df)} 个样本")
        return df

    def generate_report(
        self, model, tokenizer, prompt: str, max_new_tokens: int = 256
    ) -> str:
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def calculate_f1_scores(
        self, generated_reports: List[str], ground_truth_reports: List[str]
    ) -> Dict:
        print("\n计算 RadGraph F1 Scores...")
        f1radgraph = F1RadGraph(reward_level="all", model_type=self.radgraph_model)
        mean_reward, reward_list, _, _ = f1radgraph(
            hyps=generated_reports, refs=ground_truth_reports
        )
        rg_e, rg_er, rg_bar_er = mean_reward
        return {
            "f1_entities": float(rg_e),
            "f1_relations": float(rg_er),
            "f1_combined": float(rg_bar_er),
            "individual_scores": [float(x) for x in reward_list[2]],
        }

    def run_evaluation(self, num_samples: int = 100):
        results = {}

        load_stats = self.load_models()
        results["load_times"] = load_stats

        df = self.load_eval_data(num_samples)
        if self.text_column not in df.columns:
            raise ValueError(
                f"找不到列 {self.text_column}，可用列：{list(df.columns)}"
            )
        ground_truth = df[self.text_column].tolist()

        test_prompts = [
            _format_prompt(self.prompt_template, row) for _, row in df.iterrows()
        ]

        print(f"\n生成报告（{num_samples} 个样本）...")
        original_reports = []
        for prompt in test_prompts:
            report = self.generate_report(
                self.original_model, self.original_tokenizer, prompt
            )
            original_reports.append(report)

        original_f1 = self.calculate_f1_scores(original_reports, ground_truth)
        results["f1_scores"] = {"original": original_f1}

        self.save_results(results, original_reports, ground_truth)
        return results

    def save_results(
        self,
        results: Dict,
        original_reports: List[str],
        ground_truth: List[str],
    ):
        print(f"\n保存结果到 {self.output_dir}...")
        with open(
            os.path.join(self.output_dir, "evaluation_results.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        reports_df = pd.DataFrame(
            {
                "ground_truth": ground_truth,
                "original_model": original_reports,
            }
        )
        reports_df.to_csv(
            os.path.join(self.output_dir, "generated_reports.csv"), index=False
        )
        print("结果已保存")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AWQ 模型评估工具 (MPS)")
    parser.add_argument("--original_model", type=str, required=True, help="原始模型路径")
    parser.add_argument(
        "--quantized_model", type=str, required=True, help="量化模型路径"
    )
    parser.add_argument("--eval_data", type=str, required=True, help="评估数据路径")
    parser.add_argument(
        "--output_dir", type=str, default="./evaluation_results", help="结果保存目录"
    )
    parser.add_argument("--num_samples", type=int, default=100, help="评估样本数量")
    parser.add_argument("--prompt_file", type=str, default=None, help="prompt 文件")
    parser.add_argument("--prompt_text", type=str, default=None, help="prompt 文本")
    parser.add_argument("--text_column", type=str, default="text", help="参考报告列")
    parser.add_argument(
        "--radgraph_model",
        type=str,
        default="modern-radgraph-xl",
        help="RadGraph 模型类型",
    )

    args = parser.parse_args()

    prompt = _load_prompt(args.prompt_file, args.prompt_text)

    evaluator = ModelEvaluator(
        original_model_path=args.original_model,
        quantized_model_path=args.quantized_model,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir,
        prompt_template=prompt,
        text_column=args.text_column,
        radgraph_model=args.radgraph_model,
    )

    evaluator.run_evaluation(num_samples=args.num_samples)


if __name__ == "__main__":
    main()
