#!/usr/bin/env python3
"""
AWQ W4A4 / W4A8 量化脚本 - 基于 Hugging Face AutoAWQ

说明：
- AWQ 本身是 weight-only 量化（W4A16：4-bit 权重 + 16-bit 激活）
- W4A8：4-bit 权重 + 8-bit 激活（权重量化用 AWQ + 激活量化，参考 PrefixQuant）
- W4A4：4-bit 权重 + 4-bit 激活（同上）

激活(Activation)：神经网络每层 Linear 的输入/输出。W4A8 表示对 Linear 的输入做 8-bit 量化，
W4A4 表示 4-bit。参考 PrefixQuant (https://github.com/ChenMnZ/PrefixQuant) 的 init_input_quantizer。

本脚本复用原有 quantize_medgamma_awq.py 的 AWQ 权重量化逻辑，通过 --precision 指定
输出格式和元数据。W4A8/W4A4 推理时使用 forward_pre_hook 对激活做 fake quantization。
"""

import os
import json
import torch
import torch.nn as nn
from typing import Optional, List, Tuple

# 复用原有量化器，不修改其实现
import sys
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from quantize_medgamma_awq import (
    MedGammaAWQQuantizer,
    _get_nested,
    _load_prompt,
    load_quantized_model,
    generate_report_example,
)


PRECISION_CONFIGS = {
    "w4a16": {"activation_bits": 16, "suffix": "4bit", "description": "AWQ 默认，4-bit 权重 + 16-bit 激活"},
    "w4a8": {"activation_bits": 8, "suffix": "w4a8", "description": "4-bit 权重 + 8-bit 激活"},
    "w4a4": {"activation_bits": 4, "suffix": "w4a4", "description": "4-bit 权重 + 4-bit 激活"},
}


# ---------------------------------------------------------------------------
# 激活量化实现（参考 PrefixQuant: https://github.com/ChenMnZ/PrefixQuant）
# PrefixQuant 用 init_input_quantizer 对 Linear 输入做量化，QuantLinear.forward 中:
#   if self.use_act_quant and self.input_bits < 16:
#       input = self.input_quantizer(input)
# 此处用 forward_pre_hook 实现等效效果，不修改 AWQ 模型结构
# ---------------------------------------------------------------------------

def _fake_quant_activation_tensor(x: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Per-tensor 对称 fake 量化（参考 PrefixQuant UniformAffineQuantizer 的量化逻辑）
    - 8-bit: 范围 [-128, 127]，scale = max/127
    - 4-bit: 范围 [-8, 7]，scale = max/7
    """
    if not x.is_floating_point():
        return x
    max_val = x.abs().max().clamp(min=1e-8)
    if bits == 8:
        scale = 127.0 / max_val
        q = (x * scale).round().clamp(-128, 127)
        return q / scale
    elif bits == 4:
        scale = 7.0 / max_val
        q = (x * scale).round().clamp(-8, 7)
        return q / scale
    return x


def _is_linear_like(module: nn.Module) -> bool:
    """判断是否为 Linear 类层（含 AWQ 自定义 Linear）"""
    if hasattr(module, "in_features") and hasattr(module, "out_features") and hasattr(module, "weight"):
        return True
    # AWQ 可能用其他名称
    if hasattr(module, "weight") and callable(getattr(module, "forward", None)):
        w = getattr(module, "weight", None)
        if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
            return True
    return False


def _should_quantize_layer(name: str) -> bool:
    """
    参考 PrefixQuant init_input_quantizer：跳过 q/k/v/up/gate，只对 o_proj、down_proj 等做输入量化。
    为更全面，此处对除 lm_head 外的所有 linear 做激活量化。
    """
    if "lm_head" in name:
        return False
    return True


def _register_activation_quant_hooks(model: nn.Module, activation_bits: int) -> List:
    """
    为所有 Linear 层注册 forward_pre_hook，在 forward 前对输入（激活）做量化。
    参考：PrefixQuant quant_utils.init_input_quantizer + int_linear_fake.QuantLinear.forward
    """
    hooks = []

    def make_hook(bits: int):
        def hook(module, input):
            if not isinstance(input, (tuple, list)) or len(input) == 0:
                return input
            inp = input[0]
            if isinstance(inp, torch.Tensor) and inp.is_floating_point():
                q_inp = _fake_quant_activation_tensor(inp, bits)
                return (q_inp,) + input[1:]
            return input
        return hook

    for name, module in model.named_modules():
        if not _is_linear_like(module) or not _should_quantize_layer(name):
            continue
        h = module.register_forward_pre_hook(make_hook(activation_bits))
        hooks.append((name, h))

    return hooks


def _quantize_with_precision(
    model_path: str,
    output_path: str,
    precision: str = "w4a16",
    calibration_data_path: Optional[str] = None,
    num_samples: int = 500,
    text_column: str = "text",
    w_bit: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
    version: str = "GEMM",
    prompt_file: Optional[str] = None,
    prompt_template: Optional[str] = None,
    config_path: Optional[str] = None,
):
    """执行 AWQ 量化，并根据 precision 写入对应元数据"""
    if precision not in PRECISION_CONFIGS:
        raise ValueError(f"precision 必须是 {list(PRECISION_CONFIGS.keys())} 之一，当前: {precision}")

    cfg = PRECISION_CONFIGS[precision]
    print(f"\n精度模式: {precision} ({cfg['description']})")
    print(f"权重量化: AWQ 4-bit（与原有逻辑一致）")
    if precision != "w4a16":
        print(f"激活量化: {cfg['activation_bits']}-bit（推理时由框架或 load_with_activation_quant 处理）")

    quant_config = {
        "zero_point": zero_point,
        "q_group_size": group_size,
        "w_bit": w_bit,
        "version": version,
    }

    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    prompt_text = _load_prompt(prompt_file, prompt_template)

    quantizer = MedGammaAWQQuantizer(
        model_path=model_path,
        output_path=output_path,
        calibration_data_path=calibration_data_path or _get_nested(config, ["calibration", "data_path"]),
        num_calibration_samples=num_samples or _get_nested(config, ["calibration", "num_samples"], 500),
        text_column=text_column or _get_nested(config, ["calibration", "text_column"], "text"),
        quant_config=quant_config,
        prompt_template=prompt_text,
    )

    quantizer.quantize()

    # 在 quantization_info.json 中记录 precision 和 activation_bits
    info_path = os.path.join(output_path, "quantization_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        info["precision"] = precision
        info["activation_bits"] = cfg["activation_bits"]
        info["description"] = cfg["description"]
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    return quantizer


def load_with_activation_quant(
    model_path: str,
    activation_bits: int = 8,
    trust_remote_code: bool = True,
):
    """
    加载 AWQ 量化模型，并应用激活量化（参考 PrefixQuant 的 init_input_quantizer）。

    激活：每层 Linear 的输入，即上一层的输出。在 forward 前对输入做 per-tensor 量化。

    注意：此为 PyTorch 层面的 fake quantization，用于精度评估，不提供实际加速。
    生产环境 W4A8/W4A4 推理建议使用 QServe 或 PrefixQuant。
    """
    model, tokenizer = load_quantized_model(model_path, trust_remote_code)

    if activation_bits == 16:
        return model, tokenizer

    if activation_bits not in (4, 8):
        raise ValueError("activation_bits 仅支持 4 或 8")

    # 注册 forward_pre_hook，在每层 Linear 前对输入（激活）做量化
    # 参考 PrefixQuant: quant_utils.init_input_quantizer + int_linear_fake.QuantLinear
    hooks = _register_activation_quant_hooks(model, activation_bits)
    model._activation_quant_hooks = hooks  # 保留引用便于移除
    print(f"[激活量化] 已对 {len(hooks)} 个 Linear 层注册 {activation_bits}-bit 输入量化（参考 PrefixQuant）")
    print(f"[说明] 此为 fake quantization，用于精度评估。生产加速请用 QServe/PrefixQuant。")

    return model, tokenizer


def remove_activation_quant_hooks(model: nn.Module) -> None:
    """移除激活量化 hooks"""
    if hasattr(model, "_activation_quant_hooks"):
        for _, h in model._activation_quant_hooks:
            h.remove()
        del model._activation_quant_hooks


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="MedGamma AWQ 量化（支持 W4A16/W4A8/W4A4 精度模式）"
    )
    parser.add_argument("--model_path", type=str, required=True, help="原始模型路径")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="输出路径，默认根据 precision 自动生成",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["w4a16", "w4a8", "w4a4"],
        default="w4a16",
        help="精度模式: w4a16(默认) | w4a8 | w4a4",
    )
    parser.add_argument("--calibration_data", type=str, default=None, help="校准数据路径")
    parser.add_argument("--num_samples", type=int, default=500, help="校准样本数量")
    parser.add_argument("--text_column", type=str, default="text", help="文本列名")
    parser.add_argument("--w_bit", type=int, default=4, help="权重量化位宽")
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
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

    model_path = args.model_path or _get_nested(config, ["model", "original_model_path"])
    if not model_path:
        raise ValueError("未提供 model_path，请使用 --model_path 或配置文件")

    base_output = args.output_path or _get_nested(
        config, ["model", "quantized_model_path"], "./medgamma-awq-4bit"
    )
    suffix = PRECISION_CONFIGS[args.precision]["suffix"]
    if args.precision == "w4a16":
        output_path = base_output
    else:
        output_path = base_output.rstrip("/").rstrip("\\")
        if not output_path.endswith(f"-{suffix}"):
            output_path = f"{output_path}-{suffix}"

    if args.mode == "quantize":
        _quantize_with_precision(
            model_path=model_path,
            output_path=output_path,
            precision=args.precision,
            calibration_data_path=args.calibration_data,
            num_samples=args.num_samples,
            text_column=args.text_column,
            w_bit=args.w_bit,
            group_size=args.group_size,
            zero_point=str(args.zero_point).lower() != "false",
            version=args.version,
            prompt_file=args.prompt_file,
            prompt_template=args.prompt_template,
            config_path=args.config,
        )
    else:
        # W4A8/W4A4 需加载并启用激活量化
        if args.precision in ("w4a8", "w4a4"):
            activation_bits = PRECISION_CONFIGS[args.precision]["activation_bits"]
            model, tokenizer = load_with_activation_quant(
                output_path, activation_bits=activation_bits
            )
        else:
            model, tokenizer = load_quantized_model(output_path)
        output = generate_report_example(model, tokenizer)
        print(output)


if __name__ == "__main__":
    main()
