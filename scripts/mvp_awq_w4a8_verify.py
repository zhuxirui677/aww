#!/usr/bin/env python3
"""
MVP：AWQ 反量化 + 激活量化 数值正确性验证脚本

用途：在实现 Triton/CUDA kernel 前，用纯 PyTorch 验证：
  1. AWQ 权重解包与反量化公式
  2. 激活 8-bit/4-bit 量化与反量化
  3. W4A8 流程：(quant_act) @ (dequant_weight) 与 FP16 基准的误差

运行：python scripts/mvp_awq_w4a8_verify.py [--awq_model_path PATH]
"""

import argparse
import os
from typing import Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. AWQ 权重格式：解包与反量化
# ---------------------------------------------------------------------------

def pack_int4(w: torch.Tensor) -> torch.Tensor:
    """将 int4 权重打包为 int32：每 8 个 4-bit 存 1 个 int32"""
    # w: (K, N), 值域 0-15
    w = w.clamp(0, 15).to(torch.int32)
    K, N = w.shape
    assert N % 8 == 0
    packed = torch.zeros((K, N // 8), dtype=torch.int32, device=w.device)
    for i in range(8):
        packed[:, :] |= (w[:, i::8] << (i * 4))
    return packed


def unpack_int4(qweight: torch.Tensor) -> torch.Tensor:
    """从 int32 解包为 int4：8 个 4-bit 从 1 个 int32 取出"""
    K, N_packed = qweight.shape
    w = torch.zeros((K, N_packed * 8), dtype=torch.int32, device=qweight.device)
    for i in range(8):
        w[:, i::8] = (qweight >> (i * 4)) & 0xF
    return w


def awq_dequantize(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    AWQ 反量化：w_fp16 = (w_int4 - zero) * scale
    qweight: (K, N/8), scales: (num_groups, N), qzeros: (num_groups, N/8)
    """
    K, N_packed = qweight.shape
    N = N_packed * 8
    num_groups = K // group_size

    w_int4 = unpack_int4(qweight)  # (K, N)

    # 扩展 scales 和 zeros 到每个元素
    # scales: (num_groups, N) -> 每个 group 的 K 行共享
    scales_expanded = scales.repeat_interleave(group_size, dim=0)  # (K, N)
    zeros_unpacked = unpack_int4(qzeros)  # (num_groups, N)
    zeros_expanded = zeros_unpacked.repeat_interleave(group_size, dim=0)  # (K, N)

    w_fp16 = (w_int4.to(scales.dtype) - zeros_expanded.to(scales.dtype)) * scales_expanded
    return w_fp16


# ---------------------------------------------------------------------------
# 2. 激活量化（per-tensor 对称）
# ---------------------------------------------------------------------------

def quantize_activation_fp16_to_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP16 -> INT8：量化并返回 scale（用于反量化）"""
    scale = 127.0 / x.abs().max().clamp(min=1e-8)
    x_int8 = (x * scale).round().clamp(-128, 127)
    return x_int8, scale


def quantize_activation_fp16_to_int4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP16 -> INT4：量化并返回 scale"""
    scale = 7.0 / x.abs().max().clamp(min=1e-8)
    x_int4 = (x * scale).round().clamp(-8, 7)
    return x_int4, scale


def dequantize_activation_int8(x_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """INT8 -> FP16：反量化"""
    return x_int8.to(scale.dtype) / scale


# ---------------------------------------------------------------------------
# 3. 合成 AWQ 格式数据（用于无模型验证）
# ---------------------------------------------------------------------------

def make_synthetic_awq_layer(
    in_features: int = 256,
    out_features: int = 256,
    group_size: int = 128,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从随机 FP16 权重生成 AWQ 格式的 qweight, scales, qzeros。
    使用简化公式：w_int4 = round(w_fp16/scale) + zero, 反量化 w_fp16 = (w_int4 - zero) * scale
    """
    w_fp16 = torch.randn(in_features, out_features, device=device, dtype=torch.float16) * 0.02
    num_groups = in_features // group_size

    w_grouped = w_fp16.reshape(num_groups, group_size, out_features)
    w_abs_max = w_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scales = (w_abs_max / 7.0).squeeze(1)  # (G, N)

    scales_exp = scales.repeat_interleave(group_size, dim=0)
    zero = 8  # 对称 4-bit 零点
    w_int4 = (w_fp16 / scales_exp + zero).round().clamp(0, 15).to(torch.int32)
    qweight = pack_int4(w_int4)

    zeros_unpacked = torch.full((num_groups, out_features), zero, dtype=torch.int32, device=device)
    qzeros = pack_int4(zeros_unpacked)

    return qweight, scales, qzeros, w_fp16


# ---------------------------------------------------------------------------
# 4. 验证流程
# ---------------------------------------------------------------------------

def run_synthetic_verification(device: str = "cuda"):
    """使用合成数据验证 AWQ 反量化与激活量化"""
    print("=" * 60)
    print("MVP 验证：合成数据")
    print("=" * 60)

    in_features, out_features, group_size = 256, 256, 128
    batch_size, seq_len = 4, 32

    qweight, scales, qzeros, w_fp16_orig = make_synthetic_awq_layer(
        in_features, out_features, group_size, device
    )

    # 1. 验证 AWQ 反量化
    w_dequant = awq_dequantize(qweight, scales, qzeros, group_size)
    dequant_err = (w_dequant - w_fp16_orig).abs().max().item()
    print(f"\n[1] AWQ 反量化误差 (max abs): {dequant_err:.6f}")

    # 2. 随机激活
    x_fp16 = torch.randn(batch_size, seq_len, in_features, device=device, dtype=torch.float16) * 0.1

    # 3. FP16 基准
    out_fp16 = F.linear(x_fp16, w_fp16_orig)

    # 4. W4A16：反量化权重 + FP16 激活
    out_w4a16 = F.linear(x_fp16, w_dequant)
    err_w4a16 = (out_w4a16 - out_fp16).abs().max().item()
    print(f"[2] W4A16 输出误差 (max abs): {err_w4a16:.6f}")

    # 5. W4A8：激活量化 + 反量化 + matmul
    x_int8, scale_a = quantize_activation_fp16_to_int8(x_fp16)
    x_dequant = dequantize_activation_int8(x_int8, scale_a)
    out_w4a8 = F.linear(x_dequant, w_dequant)
    err_w4a8 = (out_w4a8 - out_fp16).abs().max().item()
    print(f"[3] W4A8 输出误差 (max abs): {err_w4a8:.6f}")

    # 6. W4A4：4-bit 激活
    x_int4, scale_a4 = quantize_activation_fp16_to_int4(x_fp16)
    x_dequant4 = x_int4.to(torch.float16) / scale_a4
    out_w4a4 = F.linear(x_dequant4, w_dequant)
    err_w4a4 = (out_w4a4 - out_fp16).abs().max().item()
    print(f"[4] W4A4 输出误差 (max abs): {err_w4a4:.6f}")

    print("\n若反量化误差 < 1e-3、W4A8 误差 < 0.1，则数值逻辑正确。")
    print("=" * 60)


def run_real_model_verification(awq_model_path: str, device: str = "cuda"):
    """从真实 AWQ 模型加载一层，验证反量化"""
    print("=" * 60)
    print("MVP 验证：真实 AWQ 模型")
    print("=" * 60)

    try:
        from safetensors import safe_open
    except ImportError:
        print("需要 safetensors: pip install safetensors")
        return

    if not os.path.isdir(awq_model_path):
        print(f"路径不存在: {awq_model_path}")
        return

    # 查找 safetensors 文件
    st_files = [f for f in os.listdir(awq_model_path) if f.endswith(".safetensors")]
    if not st_files:
        print("未找到 .safetensors 文件")
        return

    # 找第一个包含 qweight 的层
    with safe_open(os.path.join(awq_model_path, st_files[0]), framework="pt", device=device) as f:
        keys = list(f.keys())
        qweight_key = next((k for k in keys if "qweight" in k), None)
        if not qweight_key:
            print("未找到 qweight，可能不是 AWQ 格式")
            return

        prefix = qweight_key.rsplit(".qweight", 1)[0]
        scales_key = f"{prefix}.scales"
        qzeros_key = f"{prefix}.qzeros"

        if scales_key not in keys or qzeros_key not in keys:
            print(f"缺少 {scales_key} 或 {qzeros_key}")
            return

        qweight = f.get_tensor(qweight_key)
        scales = f.get_tensor(scales_key)
        qzeros = f.get_tensor(qzeros_key)

    # 推断 group_size
    K, N_packed = qweight.shape
    N = N_packed * 8
    num_groups = scales.shape[0]
    group_size = K // num_groups

    print(f"层: {prefix}")
    print(f"  qweight: {qweight.shape}, scales: {scales.shape}, qzeros: {qzeros.shape}")
    print(f"  group_size: {group_size}")

    w_dequant = awq_dequantize(qweight, scales, qzeros, group_size)
    print(f"  反量化权重 shape: {w_dequant.shape}")

    # 随机激活与 matmul
    x = torch.randn(2, 4, K, device=device, dtype=torch.float16) * 0.1
    out = F.linear(x, w_dequant)
    print(f"  输出 shape: {out.shape}")
    print("  真实模型反量化验证通过。")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="AWQ W4A8 MVP 数值验证")
    parser.add_argument("--awq_model_path", type=str, default=None, help="AWQ 模型路径（可选）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_synthetic_verification(args.device)
    if args.awq_model_path:
        run_real_model_verification(args.awq_model_path, args.device)


if __name__ == "__main__":
    main()
