#!/usr/bin/env python3
"""
MedGamma1.5-4B-it Testing Script
English-only version (no icons)
"""

import os
import sys
import json
import pandas as pd
import torch
from typing import List
from collections import defaultdict


print("=" * 70)
print("MedGamma1.5-4B-it Testing Script (English)")
print("=" * 70)


def _load_prompt(prompt_file: str = None, prompt_text: str = None) -> str:
    if prompt_text:
        return prompt_text.strip()
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "Generate a chest X-ray radiology report."


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


def check_environment():
    print("\nStep 1: Checking environment...")
    required = ["torch", "transformers", "pandas", "radgraph"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
            print(f"   [OK] {pkg}")
        except ImportError:
            print(f"   [MISSING] {pkg}")
            missing.append(pkg)

    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install command:")
        print("   pip install torch transformers accelerate radgraph")
        return False

    if torch.cuda.is_available():
        print(f"   [OK] GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"   [OK] VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
        )
    else:
        print("   [WARNING] No GPU detected, will use CPU (very slow)")

    return True


class MedGammaReportGenerator:
    def __init__(self, model_name="google/medgemma-1.5-4b-it"):
        print("\nStep 2: Loading MedGamma model...")
        print(f"   Model: {model_name}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Device: {self.device}")

        try:
            print("   Downloading/loading model (may take a few minutes)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print("   [OK] Model loaded successfully")
        except Exception as e:
            print(f"   [ERROR] Model loading failed: {e}")
            print("\n   Possible causes:")
            print("   1. Network issue (model is on HuggingFace)")
            print("   2. Insufficient VRAM (model needs ~8GB)")
            print("   3. Need to login: huggingface-cli login")
            raise

    def generate_report(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if prompt in generated_text:
            report = generated_text.replace(prompt, "").strip()
        else:
            report = generated_text.strip()

        return report

    def generate_batch(self, prompts: List[str]) -> List[str]:
        reports = []
        print(f"\nGenerating {len(prompts)} reports...")
        for i, prompt in enumerate(prompts):
            try:
                report = self.generate_report(prompt)
                reports.append(report)
            except Exception as e:
                print(f"\n   [WARNING] Sample {i} failed: {e}")
                reports.append("")
        return reports


def run_test(
    eval_data_path="mimic_eval_cleaned.csv",
    num_samples=10,
    save_results=True,
    prompt_template=None,
    text_column="text",
):
    print("\n" + "=" * 70)
    print("Step 3: Loading data")
    print("=" * 70)

    if not os.path.exists(eval_data_path):
        print(f"   [ERROR] Data file not found: {eval_data_path}")
        print("   Please ensure file is in current directory")
        return None

    df = pd.read_csv(eval_data_path)
    print(f"   [OK] Loaded {len(df)} records")
    print(f"   Columns: {df.columns.tolist()}")

    if len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=42)
        print(f"   Randomly selected {num_samples} samples for testing")

    if text_column not in df.columns:
        print(f"   [ERROR] Text column not found: {text_column}")
        return None

    reference_reports = df[text_column].tolist()
    print("\n   Sample ground truth:")
    print(f"   {reference_reports[0][:150]}...")

    print("\n" + "=" * 70)
    print("Step 4: Generating reports with MedGamma")
    print("=" * 70)

    try:
        generator = MedGammaReportGenerator()
        prompts = [_format_prompt(prompt_template, row) for _, row in df.iterrows()]
        generated_reports = generator.generate_batch(prompts)

        print("\n   [OK] Generation completed")
        print("\n   Sample generated report:")
        print(f"   {generated_reports[0][:150]}...")

    except Exception as e:
        print(f"\n   [ERROR] Generation failed: {e}")
        return None

    print("\n" + "=" * 70)
    print("Step 5: RadGraph F1 Evaluation")
    print("=" * 70)

    try:
        from radgraph import F1RadGraph

        print("\n   Initializing RadGraph...")
        print("   (First run will download model, takes a few minutes)")

        f1radgraph = F1RadGraph(reward_level="all", model_type="modern-radgraph-xl")

        print("   Computing F1 scores...")
        mean_reward, reward_list, _, _ = f1radgraph(
            hyps=generated_reports, refs=reference_reports
        )

        rg_e, rg_er, rg_bar_er = mean_reward
        print("   [OK] Evaluation completed")

    except Exception as e:
        print(f"   [ERROR] RadGraph evaluation failed: {e}")
        print("\n   Possible causes:")
        print("   1. RadGraph not properly installed")
        print("   2. Network issue (needs to download RadGraph model)")
        return None

    print("\n" + "=" * 70)
    print("Test Results")
    print("=" * 70)

    print("\nDataset Information:")
    print(f"  Test samples: {len(generated_reports)}")
    print(f"  Data source: {eval_data_path}")

    print("\nRadGraph F1 Scores:")
    print(f"  RG_E (entities only):     {rg_e:.4f}")
    print(f"  RG_ER (entities+relations): {rg_er:.4f}")
    print(f"  RG_ER_bar (complete):     {rg_bar_er:.4f}")

    print("\nQuality Assessment:")
    if rg_bar_er > 0.70:
        print("  [EXCELLENT] F1 > 0.70")
    elif rg_bar_er > 0.60:
        print("  [GOOD] F1 > 0.60")
    elif rg_bar_er > 0.50:
        print("  [FAIR] F1 > 0.50")
    else:
        print("  [NEEDS IMPROVEMENT] F1 < 0.50")

    print("=" * 70)

    results = {
        "model": "google/medgemma-1.5-4b-it",
        "dataset": eval_data_path,
        "num_samples": len(generated_reports),
        "prompt_template": prompt_template,
        "f1_scores": {
            "rg_e": float(rg_e),
            "rg_er": float(rg_er),
            "rg_bar_er": float(rg_bar_er),
        },
        "sample_outputs": [
            {"reference": reference_reports[i], "generated": generated_reports[i]}
            for i in range(min(3, len(generated_reports)))
        ],
    }

    if save_results:
        output_file = "medgamma_test_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nComplete results saved to: {output_file}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MedGamma One-Click Test (English)")
    parser.add_argument("--data", type=str, default="mimic_eval_cleaned.csv")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--prompt_file", type=str, default="prompts/example_prompt.txt")
    parser.add_argument("--prompt_text", type=str, default=None)
    parser.add_argument("--text_column", type=str, default="text")

    args = parser.parse_args()

    print("\nStarting test...")
    if not check_environment():
        print("\n[ERROR] Environment check failed, please install dependencies")
        print("\nInstall command:")
        print("pip install torch transformers accelerate radgraph")
        sys.exit(1)

    prompt_template = _load_prompt(args.prompt_file, args.prompt_text)
    results = run_test(
        eval_data_path=args.data,
        num_samples=args.num_samples,
        save_results=True,
        prompt_template=prompt_template,
        text_column=args.text_column,
    )

    if results:
        print("\n[SUCCESS] Test completed")
        print("\nNext steps:")
        print("1. Check medgamma_test_results.json for detailed results")
        print("2. If results are good, increase sample size and retest")
        print("3. If results are poor, adjust model parameters")
    else:
        print("\n[ERROR] Test failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
