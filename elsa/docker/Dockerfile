# Elsa: SGLang fork tuned for Qwen3.6-35B-A3B NVFP4 on one RTX 5090 (SM120).
# Build from the repository root:
#   docker build -f elsa/docker/Dockerfile -t elsa-sglang .
# A CUDA *devel* base is required: SGLang and FlashInfer JIT-compile SM120 kernels at startup.
ARG CUDA_IMAGE=nvidia/cuda:13.3.1-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG UV_DEFAULT_INDEX=https://pypi.org/simple
ARG UV_VERSION=0.9.14

ENV DEBIAN_FRONTEND=noninteractive \
    UV_HTTP_TIMEOUT=900 \
    UV_LINK_MODE=copy \
    TORCH_CUDA_ARCH_LIST=12.0 \
    FLASHINFER_CUDA_ARCH_LIST=12.0f

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential pkg-config libssl-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
        | tar -xz --strip-components=1 -C /usr/local/bin \
    && curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.92  # matches rust/rust-toolchain.toml

WORKDIR /opt/elsa
COPY rust rust
COPY python python
COPY elsa elsa
COPY examples/assets/example_image.png examples/assets/example_image.png
COPY README.md LICENSE ./

# Non-editable install; the Rust toolchain only builds the radix-tree extension and is
# removed afterwards. No GPU at build time, so the SM120 prune is forced.
RUN export PATH="/root/.cargo/bin:$PATH" \
    && uv venv --python 3.12 /opt/elsa/.venv \
    && uv pip install --python /opt/elsa/.venv/bin/python \
        --default-index "$UV_DEFAULT_INDEX" --prerelease=allow --index-strategy unsafe-best-match \
        ./python ninja \
    && /opt/elsa/.venv/bin/python elsa/prune_sm120.py --venv /opt/elsa/.venv --force \
    && rm -rf /root/.cargo /root/.rustup /root/.cache/uv rust/target python/build

ENV VENV=/opt/elsa/.venv \
    PATH=/opt/elsa/.venv/bin:/usr/local/cuda/bin:$PATH \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000
ENTRYPOINT ["/opt/elsa/elsa/serve-elsa.sh"]
