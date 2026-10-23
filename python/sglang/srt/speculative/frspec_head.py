"""Reduced-vocabulary (FR-Spec) draft head for checkpoints with a quantized lm_head.

Slicing rows of a quantized target head is wrong once its weights have been
repacked for the GEMM kernel (e.g. Marlin NVFP4), so the hot rows are read from
the checkpoint and dequantized to a BF16 [num_hot, hidden] head instead.
"""

from __future__ import annotations

import json
import os

import torch
from safetensors import safe_open

from sglang.srt.layers.quantization.dequantization import dequantize_nvfp4

_LM_HEAD = "lm_head.weight"


def _checkpoint_file(model_path: str, tensor_name: str) -> str:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        return os.path.join(model_path, weight_map[tensor_name])
    return os.path.join(model_path, "model.safetensors")


def _load(model_path: str, tensor_name: str) -> torch.Tensor:
    with safe_open(_checkpoint_file(model_path, tensor_name), framework="pt") as f:
        return f.get_tensor(tensor_name)


def load_hot_lm_head(
    model_path: str, hot_token_id: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Return the BF16 lm_head rows for ``hot_token_id``, dequantizing NVFP4."""
    if not os.path.isdir(model_path):
        raise ValueError(
            f"FR-Spec with a quantized lm_head needs a local checkpoint, got {model_path}"
        )
    hot = hot_token_id.to("cpu")
    weight = _load(model_path, _LM_HEAD)[hot]
    if weight.dtype != torch.uint8:
        return weight.to(device=device, dtype=torch.bfloat16)
    scale = _load(model_path, "lm_head.weight_scale")[hot]
    scale_2 = _load(model_path, "lm_head.weight_scale_2")
    return dequantize_nvfp4(
        weight.to(device), scale.to(device), scale_2.to(device), torch.bfloat16
    ).contiguous()
