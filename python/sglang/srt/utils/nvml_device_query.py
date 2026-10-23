"""Answer CUDA device queries from NVML in the launcher process.

The HTTP/tokenizer process imports modules (FlashInfer, sgl_kernel) that ask for
the device capability at import time. Through torch that creates a full CUDA
context (~600 MB on RTX 5090) in a process that never runs a kernel, which on a
32 GB card is memory the scheduler's pools and activations need.

Only the launcher's own process is patched; spawned scheduler and detokenizer
processes import this module afresh and keep the real torch queries.
"""

from __future__ import annotations

import functools
import os

import torch


@functools.cache
def _nvml_handles() -> tuple:
    import pynvml

    pynvml.nvmlInit()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        indices = list(range(pynvml.nvmlDeviceGetCount()))
        if visible is not None:
            indices = []
    else:
        indices = []
        for token in visible.split(","):
            token = token.strip()
            if token.isdigit():
                indices.append(int(token))
            else:
                # UUID / MIG identifiers: let torch handle these setups.
                raise ValueError(f"unsupported CUDA_VISIBLE_DEVICES entry {token}")
    return tuple(pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices)


def install_nvml_device_queries() -> bool:
    """Route capability/count/current-device queries through NVML until CUDA is
    initialized for real. Returns False (and patches nothing) when NVML cannot
    describe the visible devices."""
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        # NVML enumerates by PCI bus id; CUDA's default order may differ.
        return False
    try:
        handles = _nvml_handles()
    except Exception:
        return False
    if not handles:
        return False

    import pynvml

    real_capability = torch.cuda.get_device_capability
    real_count = torch.cuda.device_count
    real_current = torch.cuda.current_device

    def _index(device) -> int:
        if device is None:
            return 0
        if isinstance(device, int):
            return device
        if isinstance(device, str):
            device = torch.device(device)
        return device.index or 0

    def get_device_capability(device=None):
        if torch.cuda.is_initialized():
            return real_capability(device)
        return tuple(pynvml.nvmlDeviceGetCudaComputeCapability(handles[_index(device)]))

    def device_count():
        if torch.cuda.is_initialized():
            return real_count()
        return len(handles)

    def current_device():
        if torch.cuda.is_initialized():
            return real_current()
        return 0

    real_properties = torch.cuda.get_device_properties

    class _NvmlDeviceProperties:
        """The fields NVML can answer; anything else falls back to torch."""

        def __init__(self, index: int):
            handle = handles[index]
            self._index = index
            self.major, self.minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            self.name = name.decode() if isinstance(name, bytes) else name
            self.total_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total

        def __getattr__(self, attr):
            return getattr(real_properties(self._index), attr)

    def get_device_properties(device=None):
        if torch.cuda.is_initialized():
            return real_properties(device)
        return _NvmlDeviceProperties(_index(device))

    torch.cuda.get_device_capability = get_device_capability
    torch.cuda.get_device_properties = get_device_properties
    torch.cuda.device_count = device_count
    torch.cuda.current_device = current_device
    return True
