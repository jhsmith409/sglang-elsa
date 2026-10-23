#!/usr/bin/env python3
"""Remove binary kernel payloads an RTX Blackwell (SM120) server never loads.

Verified on RTX 5090 by reading /proc/<pid>/maps of the launcher, scheduler and
detokenizer with images, tools, MTP and long prefill exercised: sgl_kernel loads
sm100/common_ops (which carries the sm_120a code), flashmla_ops, deep_gemm and
Triton; it never loads the FlashAttention-3/4 library or the SM90 common ops.

    python prune_sm120.py --venv .venv          # prune (idempotent)
    python prune_sm120.py --venv .venv --check  # report only
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

# Relative to site-packages. Each entry is SM90/SM100-only code.
PRUNE = [
    "sgl_kernel/flash_ops.abi3.so",  # FlashAttention 3/4 (Hopper / datacenter Blackwell)
    "sgl_kernel/sm90",  # SM90 build of common_ops; SM120 loads sm100/common_ops
]


def _site_packages(venv: str) -> str:
    matches = glob.glob(os.path.join(venv, "lib", "python3*", "site-packages"))
    if len(matches) != 1:
        sys.exit(f"cannot find a unique site-packages under {venv}")
    return matches[0]


def _gpu_capability(venv: str) -> str:
    out = subprocess.run(
        [os.path.join(venv, "bin", "python"), "-c",
         "import pynvml; pynvml.nvmlInit(); "
         "print('%d.%d' % pynvml.nvmlDeviceGetCudaComputeCapability(pynvml.nvmlDeviceGetHandleByIndex(0)))"],
        capture_output=True, text=True,
    )
    return out.stdout.strip()


def _size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true", help="prune even if GPU 0 is not SM120")
    args = ap.parse_args()
    site = _site_packages(args.venv)
    capability = _gpu_capability(args.venv)
    if capability != "12.0" and not args.force:
        sys.exit(f"GPU 0 reports compute capability {capability!r}, not 12.0; refusing to prune")
    total = 0
    for rel in PRUNE:
        path = os.path.join(site, rel)
        if not os.path.exists(path):
            print(f"absent   {rel}")
            continue
        size = _size(path)
        total += size
        if args.check:
            print(f"present  {rel} ({size / 1e6:.0f} MB)")
            continue
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
        print(f"removed  {rel} ({size / 1e6:.0f} MB)")
    verb = "prunable" if args.check else "removed"
    print(f"{verb}: {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
