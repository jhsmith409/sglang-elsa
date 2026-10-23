"""Split-K Triton GEMM for decode-sized BF16 linears with a narrow output.

cuBLAS picks 16x16 wmma tiles for shapes like the Qwen3.5 GDN ``in_proj_ba``
(N=64, K=2048) at M<=16, which launches only a handful of CTAs and runs the
whole K loop serially (~25 us on RTX 5090 for 256 KB of weights). Splitting K
across CTAs and reducing with fp32 atomics brings it to the launch floor.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_MAX_M = 16
_MAX_N = 2048


@triton.jit
def _skinny_bf16_gemm_kernel(
    x_ptr,
    w_ptr,
    acc_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_start = pid_k * K_PER_SPLIT
    for k0 in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = k_start + k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
            eviction_policy="evict_first",
        )
        acc += tl.dot(x, tl.trans(w))
    tl.atomic_add(
        acc_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        sem="relaxed",
    )


def use_skinny_bf16_gemm(m: int, n: int, k: int) -> bool:
    if not (0 < m <= _MAX_M and n <= _MAX_N and k % 256 == 0 and k >= 1024):
        return False
    # Measured on RTX 5090 under CUDA graphs: cuBLAS GEMV wins at M=1 for N<512.
    return m > 1 or n >= 512


def skinny_bf16_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """y[M, N] = x[M, K] @ weight[N, K]^T for small M and N, bf16 in and out."""
    x2 = x.reshape(-1, x.shape[-1])
    m, k = x2.shape
    n = weight.shape[0]
    block_n = 16 if n <= 128 else 64
    block_k = 128
    # Target roughly 128 CTAs so the weight stream covers the SMs.
    n_tiles = triton.cdiv(n, block_n)
    split_k = max(1, min(k // block_k, 128 // n_tiles))
    k_per_split = triton.cdiv(triton.cdiv(k, split_k), block_k) * block_k
    split_k = triton.cdiv(k, k_per_split)
    acc = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    _skinny_bf16_gemm_kernel[(n_tiles, split_k)](
        x2,
        weight,
        acc,
        m,
        n,
        k,
        x2.stride(0),
        weight.stride(0),
        BLOCK_M=16,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        K_PER_SPLIT=k_per_split,
        num_warps=4,
    )
    return acc.to(x.dtype).view(*x.shape[:-1], n)
