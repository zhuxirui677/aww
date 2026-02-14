#!/usr/bin/env python3
"""
AWQ 量化模型评估脚本
对比原始模型 vs 量化模型的性能（F1, 速度, 显存）
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
        print("MedGamma AWQ 量化评估")
        print("=" * 70)

    def _get_device(self, model):
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_models(self):
        print("\n加载模型...")

        print("  加载原始模型...")
        start = time.time()
        self.original_model = AutoModelForCausalLM.from_pretrained(
            self.original_model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.original_tokenizer = AutoTokenizer.from_pretrained(
            self.original_model_path, trust_remote_code=True
        )
        original_load_time = time.time() - start
        print(f"  原始模型耗时: {original_load_time:.2f}秒")

        return {"original_load_time": original_load_time}

    def load_quantized_model(self):
        print("  加载量化模型...")
        try:
            import awq_ext  # noqa: F401
        except Exception:
            print("  [WARNING] awq_ext 未安装，融合加速不可用")
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
        return {"quantized_load_time": quantized_load_time}

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
        ).to(next(model.parameters()).device)

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

    def benchmark_inference_speed(
        self, model, tokenizer, test_prompts: List[str], num_runs: int = 10
    ) -> Dict:
        print(f"\n测试推理速度（{num_runs} 次运行）...")
        times = []
        tokens_generated = []

        for prompt in test_prompts[:num_runs]:
            start = time.time()
            inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256)
            elapsed = time.time() - start
            times.append(elapsed)
            tokens_generated.append(len(outputs[0]) - len(inputs.input_ids[0]))

        avg_time = np.mean(times)
        avg_tokens = np.mean(tokens_generated)
        tokens_per_sec = avg_tokens / avg_time
        return {
            "avg_time_sec": float(avg_time),
            "avg_tokens": float(avg_tokens),
            "tokens_per_sec": float(tokens_per_sec),
        }

    def measure_memory(self, model) -> Dict:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            dummy_input = torch.randint(0, 1000, (1, 50)).to(next(model.parameters()).device)
            with torch.no_grad():
                _ = model.generate(dummy_input, max_new_tokens=10)
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            return {
                "allocated_gb": float(allocated),
                "reserved_gb": float(reserved),
                "peak_gb": float(peak),
            }
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "peak_gb": 0.0}

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

        print("\n测量显存占用（原始模型）...")
        original_mem = self.measure_memory(self.original_model)

        print("\n原始模型速度测试...")
        original_speed = self.benchmark_inference_speed(
            self.original_model,
            self.original_tokenizer,
            test_prompts,
            num_runs=min(10, num_samples),
        )

        print(f"\n生成报告（原始模型，{num_samples} 个样本）...")
        original_reports = []
        for prompt in test_prompts:
            report = self.generate_report(
                self.original_model, self.original_tokenizer, prompt
            )
            original_reports.append(report)

        del self.original_model
        del self.original_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        quant_load = self.load_quantized_model()
        results["load_times"].update(quant_load)

        print("\n测量显存占用（量化模型）...")
        quantized_mem = self.measure_memory(self.quantized_model)

        print("量化模型速度测试...")
        quantized_speed = self.benchmark_inference_speed(
            self.quantized_model,
            self.quantized_tokenizer,
            test_prompts,
            num_runs=min(10, num_samples),
        )

        print(f"\n生成报告（量化模型，{num_samples} 个样本）...")
        quantized_reports = []
        for prompt in test_prompts:
            report = self.generate_report(
                self.quantized_model, self.quantized_tokenizer, prompt
            )
            quantized_reports.append(report)

        results["memory"] = {
            "original": original_mem,
            "quantized": quantized_mem,
            "reduction_percent": (
                (original_mem["peak_gb"] - quantized_mem["peak_gb"])
                / original_mem["peak_gb"]
                * 100
            )
            if original_mem["peak_gb"] > 0
            else 0,
        }

        results["inference_speed"] = {
            "original": original_speed,
            "quantized": quantized_speed,
            "speedup": quantized_speed["tokens_per_sec"]
            / original_speed["tokens_per_sec"],
        }

        original_f1 = self.calculate_f1_scores(original_reports, ground_truth)
        quantized_f1 = self.calculate_f1_scores(quantized_reports, ground_truth)
        results["f1_scores"] = {
            "original": original_f1,
            "quantized": quantized_f1,
            "degradation_percent": (
                (original_f1["f1_combined"] - quantized_f1["f1_combined"])
                / original_f1["f1_combined"]
                * 100
            ),
        }

        self.save_results(results, original_reports, quantized_reports, ground_truth)
        self.print_summary(results)
        return results

    def save_results(
        self,
        results: Dict,
        original_reports: List[str],
        quantized_reports: List[str],
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
                "quantized_model": quantized_reports,
            }
        )
        reports_df.to_csv(
            os.path.join(self.output_dir, "generated_reports.csv"), index=False
        )
        print("结果已保存")

    def print_summary(self, results: Dict):
        print("\n" + "=" * 70)
        print("评估摘要")
        print("=" * 70)

        orig_f1 = results["f1_scores"]["original"]["f1_combined"]
        quant_f1 = results["f1_scores"]["quantized"]["f1_combined"]
        print("\nF1 Score:")
        print(f"  原始模型:  {orig_f1:.4f}")
        print(f"  量化模型:  {quant_f1:.4f}")
        print(f"  下降:      {results['f1_scores']['degradation_percent']:.2f}%")

        orig_speed = results["inference_speed"]["original"]["tokens_per_sec"]
        quant_speed = results["inference_speed"]["quantized"]["tokens_per_sec"]
        print("\n推理速度:")
        print(f"  原始模型:  {orig_speed:.2f} tokens/s")
        print(f"  量化模型:  {quant_speed:.2f} tokens/s")
        print(f"  加速:      {results['inference_speed']['speedup']:.2f}x")

        orig_mem = results["memory"]["original"]["peak_gb"]
        quant_mem = results["memory"]["quantized"]["peak_gb"]
        print("\n显存占用:")
        print(f"  原始模型:  {orig_mem:.2f} GB")
        print(f"  量化模型:  {quant_mem:.2f} GB")
        print(f"  减少:      {results['memory']['reduction_percent']:.2f}%")

        print("\n" + "=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AWQ 模型评估工具")
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
