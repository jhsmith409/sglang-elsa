"""Reduced-vocabulary (FR-Spec) draft head for checkpoints with a quantized lm_head.

Slicing rows of a quantized target head is wrong once its weights have been
repacked for the GEMM kernel (e.g. Marlin NVFP4), so the hot rows are read from
the checkpoint instead: either into a hot-vocab head with the target's own
quantization (NVFP4 W4A16), or dequantized to a BF16 [num_hot, hidden] tensor.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import torch
from safetensors import safe_open

from sglang.srt.layers.quantization.dequantization import dequantize_nvfp4
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4A16LinearMethod
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

logger = logging.getLogger(__name__)

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


def _require_local(model_path: str) -> None:
    if not os.path.isdir(model_path):
        raise ValueError(
            f"FR-Spec with a quantized lm_head needs a local checkpoint, got {model_path}"
        )


def load_hot_lm_head(
    model_path: str, hot_token_id: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Return the BF16 lm_head rows for ``hot_token_id``, dequantizing NVFP4."""
    _require_local(model_path)
    hot = hot_token_id.to("cpu")
    weight = _load(model_path, _LM_HEAD)[hot]
    if weight.dtype != torch.uint8:
        return weight.to(device=device, dtype=torch.bfloat16)
    scale = _load(model_path, "lm_head.weight_scale")[hot]
    scale_2 = _load(model_path, "lm_head.weight_scale_2")
    return dequantize_nvfp4(
        weight.to(device), scale.to(device), scale_2.to(device), torch.bfloat16
    ).contiguous()


class _PinnedQuantConfig:
    """Gives the hot head the target lm_head's quant method class and config.

    Mixed-precision ModelOpt configs resolve per-layer methods by prefix at the
    parent level; asking the NVFP4 sub-config directly would not reproduce it.
    """

    def __init__(self, target_quant_method):
        self._method_cls = type(target_quant_method)
        self._config = target_quant_method.quant_config

    def get_quant_method(self, layer, prefix):
        return self._method_cls(self._config)


def build_quantized_hot_lm_head(
    target_lm_head: torch.nn.Module,
    model_path: str,
    hot_token_id: torch.Tensor,
    device: torch.device,
) -> Optional[ParallelLMHead]:
    """A hot-vocab ParallelLMHead quantized like the target's NVFP4 W4A16 head.

    Returns None when the target head is not NVFP4 W4A16, so the caller can fall
    back to the BF16 rows of ``load_hot_lm_head``.
    """
    quant_method = target_lm_head.quant_method
    if type(quant_method) is not ModelOptNvFp4A16LinearMethod:
        logger.info(
            "FR-Spec draft head: target lm_head uses %s; using BF16 hot rows",
            type(quant_method).__name__,
        )
        return None
    _require_local(model_path)
    hot = hot_token_id.to("cpu")
    hidden_size = target_lm_head.embedding_dim
    with torch.device(device):
        head = ParallelLMHead(
            hot.numel(),
            hidden_size,
            params_dtype=torch.bfloat16,
            quant_config=_PinnedQuantConfig(quant_method),
            prefix="lm_head",
        )
    rows = head.num_embeddings_per_partition
    params = dict(head.named_parameters())
    for name in ("weight", "weight_scale"):
        source = _load(model_path, f"lm_head.{name}")[hot]
        params[name].data.zero_()
        params[name].data[: source.shape[0]].copy_(source.to(device))
    params["weight_scale_2"].data.fill_(_load(model_path, "lm_head.weight_scale_2").item())
    params["input_scale"].data.fill_(1.0)
    assert rows >= hot.numel()
    head.quant_method.process_weights_after_loading(head)
    return head
