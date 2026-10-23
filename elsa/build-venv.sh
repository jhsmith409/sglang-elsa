#!/usr/bin/env bash
# Build the elsa runtime environment with uv and prune non-SM120 kernel payloads.
#   elsa/build-venv.sh [VENV_DIR]   (default: <repo>/.venv)
# Needs: uv, Python 3.12 (uv can install it), Rust/cargo (radix-tree core),
# a CUDA 13.x toolkit for kernel JIT, and an RTX Blackwell (SM120) GPU at index 0.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
VENV="${1:-$REPO/.venv}"

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
command -v cargo >/dev/null || [[ -x "$HOME/.cargo/bin/cargo" ]] || {
  echo "cargo is required (curl -sSf https://sh.rustup.rs | sh)" >&2; exit 1; }
export PATH="$HOME/.cargo/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0 FLASHINFER_CUDA_ARCH_LIST=12.0f
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-900}"

uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" \
  --prerelease=allow --index-strategy unsafe-best-match \
  -e "$REPO/python" ninja
"$VENV/bin/python" "$HERE/prune_sm120.py" --venv "$VENV"
"$VENV/bin/python" -c "import sglang, torch, flashinfer; print('sglang', sglang.__version__, 'torch', torch.__version__, 'flashinfer', flashinfer.__version__)"
