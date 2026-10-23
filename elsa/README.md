# Elsa — SGLang for Qwen3.6-35B-A3B NVFP4 on one RTX 5090

Elsa is an SGLang fork (based on upstream **v0.5.22**) tuned for
[`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
on a single **RTX 5090** (Blackwell SM120, 32 GB):

- **256K context, vision and tool calling**, 8 concurrent requests, 358K-token KV pool
- **NEXTN MTP speculative decoding** with a 32K-token FR-Spec draft vocabulary
- **~2x stock SGLang and ~2.3x vLLM** for single-request decode on the same card

It was inspired by [Penny Royal](https://github.com/jpezzulli/sglang-rtxpro6000), John
Pezzulli's SGLang-derived runtime for Qwen3.8 on the RTX PRO 6000, which showed how much a
runtime tuned for one model on one GPU family can gain.

Why "Elsa": the grandkids were watching *Frozen* and singing along when it needed a name.

## Results

Same checkpoint and the same scripts (`elsa/bench/`) for every row. "Thinking off" requests
set `enable_thinking: false` and temperature 0; "thinking on" requests set
`enable_thinking: true` and temperature 0.7. 8/16/32 requests of ~1,000 output tokens at
C=1/4/8. C=1 is the per-request median; C=4 and C=8 are aggregate tok/s.

| Server (RTX 5090, 600 W) | Thinking off C=1 | C=4 | C=8 | Thinking on C=1 | C=4 | C=8 |
|---|---:|---:|---:|---:|---:|---:|
| **Elsa** (Docker, CUDA 13.3.1) | **626** | **1,343** | **1,956** | **511** | **1,252** | **1,771** |
| Stock SGLang 0.5.22, no MTP | 310 | 762 | 1,201 | 303 | 792 | 1,255 |
| vLLM (FlashInfer b12x MoE, 256K, prefix caching) | 267 | 744 | 1,145 | 259 | 784 | 1,234 |

Stock SGLang with its built-in MTP reaches 528 tok/s at C=1, but on 32 GB it can only run
**one request at a time** with a 181K-token pool, too small for 256K context.

**Long context**, single request, unique prompts (no prefix-cache hits). Prefill tok/s and
decode tok/s of a 300-token answer.

| Prompt | Elsa prefill | Stock prefill | vLLM prefill | Elsa decode | Stock decode | vLLM decode |
|---|---:|---:|---:|---:|---:|---:|
| 30K | 25.8K | 26.4K | 24.0K | **381** | 284 | 252 |
| 120K | 11.1K | 11.2K | 11.0K | **297** | 240 | 223 |
| 225K | 6.53K | 6.56K | 6.65K | **256** | 204 | 196 |

**Two RTX 5090s, same Elsa image:**

| | 600 W, PCIe 5.0 x16 | 575 W, PCIe 4.0 x4 |
|---|---:|---:|
| Thinking off, C=1 / C=4 / C=8 | 626 / 1,343 / 1,956 | 630 / 1,320 / 1,955 |
| Thinking on, C=1 / C=4 / C=8 | 511 / 1,252 / 1,771 | 494 / 1,212 / 1,768 |
| Prefill 120K / 225K | 11.0K / 6.51K | 10.9K / 6.45K |

**CUDA version of the image** (its `nvcc` JIT-compiles FlashInfer and SGLang kernels at
startup), aggregate tok/s, mean of 3 warm runs:

| Base image | C=1 | C=4 |
|---|---:|---:|
| `nvidia/cuda:13.0.2-devel` | 520 | 1,300 |
| `nvidia/cuda:13.3.1-devel` | **559** | **1,329** |
| `nvidia/cuda:13.4.2-devel` | 542 | 1,310 |

**Quality and function.** GSM8K (first 200, greedy, thinking off) 94–97.5% across runs for
Elsa, 95.0% stock SGLang, 94.5% vLLM. A needle placed in the long prompt was recalled at every
length. Vision, tool calling (streaming and non-streaming, thinking on and off) and two memory
stress loads (7 sampled thinking requests plus a 200K prefill; 4 requests plus
30K/30K/56K/112K prefills) pass with no OOM.

## How it got there

C=1 decode (thinking off, per-request median) and the GPU KV pool after each step:

| Step | C=1 tok/s | KV pool | Max requests |
|---|---:|---:|---:|
| Stock SGLang 0.5.22 (needs `--moe-runner-backend flashinfer_cutlass` to start) | 309 | 345K | 9 |
| + stock NEXTN MTP (3 steps) | 528 | 181K | 1 |
| Share the draft's embed/lm_head before sizing pools | 536 | 244K | 3 |
| BF16 SSM state, ReplaySSM verify, explicit Mamba slots | 520 | 303K | 8 |
| Skinny BF16 GEMM; shared experts dequantized to BF16 | 525 | 283K | 8 |
| 5 draft steps + 32K FR-Spec draft vocabulary | 582 | 268K | 8 |
| FP8 projections on the SM12x small-M CUTLASS path | 616 | 268K | 8 |
| FlashInfer fused MoE finalize | 628 | 268K | 8 |
| FR-Spec head kept in NVFP4 (Marlin) | 623 | 278K | 8 |
| HTTP process off the GPU (+630 MB headroom, not pool) | — | 278K | 8 |
| Online FP8 MTP draft layer | 655 | **358K** | 8 |
| 4K window on the draft's prompt pass (prefill fix) | 628 | 358K | 8 |
| Docker image on CUDA 13.3.1 | 626 | 358K | 8 |

## What changed versus upstream SGLang

| Change | Effect on RTX 5090 |
|---|---|
| Share the MTP draft's embedding/lm_head **before** KV pools are sized (upstream allocated private copies and freed them after sizing) | +2 GB to pools; MTP goes from 1 to 3+ concurrent requests |
| Split-K Triton "skinny" BF16 GEMM for decode-sized narrow linears (cuBLAS picks 16x16 wmma tiles with a serial K loop) | GDN `in_proj_ba` 8.7 → 2.8 us; exact BF16 copies of the small W4A16 shared experts replace slow Marlin calls |
| FP8 projections: the SM12x small-M FP8 path never engaged (scale expanded per channel before the fast path's scalar was computed); enable the KDA CUTLASS skinny GEMM by size rule | bit-identical to cuBLAS; ~1.4 vs ~0.85 TB/s at M=4–16 |
| FR-Spec for quantized lm_heads: upstream sliced rows of the Marlin-packed NVFP4 head (silently wrong); build a hot-vocab NVFP4 head from the checkpoint | draft head reads 37 MB/step instead of ~290 MB; 32K map built from this model's own output (`elsa/bench/build_frspec.py`) |
| Keep the HTTP/tokenizer process off the GPU (NVML device queries, CPU image decode and preprocessing, no `sgl_kernel` package init) | +630 MB headroom for the scheduler |
| `SGLANG_QWEN35_MTP_ONLINE_FP8`: build the BF16 MTP draft layer with online FP8 | same acceptance; KV pool 278K → 358K tokens |
| `SGLANG_QWEN35_MTP_DRAFT_WINDOW`: 4K-token window on the draft's **prompt** pass only | long-prompt prefill back to stock speed (the draft's full attention over the prompt cost ~8%) |
| Launcher: FlashInfer CUTLASS MoE, fused MoE finalize, BF16 SSM state, ReplaySSM spec verify, 5-step MTP, mem fraction 0.88, SM120-only JIT arch lists | see `serve-elsa.sh` |

### Tried and dropped

- Fusing the shared expert into the routed MoE: wrong output, and slower even so.
- Triton decode attention (removes FlashInfer's per-step host sync): slower at long context.
- FlashInfer GDN prefill/decode kernels: no gain on this model.
- Upstream adaptive draft length: ~0.55 GB of graphs per candidate length (OOM), little upside.
- `nvfp4_online` draft quantization: needs SM100-only MoE runners.
- Splitting decode/verify CUDA graphs into segments: the launch bubble shrinks in profiles,
  but A/B end-to-end throughput did not change.
- Draft-decode window (`--speculative-draft-window-size`): no prefill effect, decode a wash.
- `--mem-fraction-static 0.90`: OOM under the 200K-prefill stress load.

## Run it

### Docker (recommended)

```bash
cd elsa/docker
ELSA_GPU=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | grep 5090 | cut -d, -f1) \
MODEL_DIR=/path/to/Qwen3.6-35B-A3B-NVFP4 \
docker compose up -d --build
```

The checkpoint directory must include `hf_quant_config.json`. The first start JIT-compiles
SM120 kernels (~10 minutes); later starts take ~1 minute. The API is OpenAI-compatible on
port 8005 (`ELSA_PORT`), served model name `qwen3.6-moe-nvfp4`, thinking off unless a request
sets `chat_template_kwargs.enable_thinking`. Reasoning is returned in `reasoning_content`.

### Native (uv)

Requirements: Linux, an SM120 GPU, NVIDIA driver with CUDA 13, a CUDA 13.x toolkit (13.3
recommended), `uv`, Rust/cargo, and the checkpoint on local disk.

```bash
elsa/build-venv.sh                       # uv venv at .venv, prunes non-SM120 kernels (1.1 GB)
cp elsa/elsa.env.example elsa/elsa.env   # set MODEL_PATH; PORT, GPU, model name, thinking default
elsa/serve-elsa.sh
```

### Tests

```bash
P=.venv/bin/python; B=http://127.0.0.1:8005; M=qwen3.6-moe-nvfp4
$P elsa/bench/vision.py $B $M
$P elsa/bench/tools.py $B $M
$P elsa/bench/gsm8k.py --base $B --model $M -n 200
$P elsa/bench/bench.py --base $B --model $M --conc 1,4,8 [--temperature 0.7 --thinking]
ELSA_MODEL_NAME=$M $P elsa/bench/longctx.py --base $B --repeats 3
ELSA_MODEL_NAME=$M $P elsa/bench/stress.py --base $B --short 7 --long 200000
```

## Limits

- Tuned and tested only on this checkpoint and RTX 5090s (TP1). Other models fall back to
  upstream behavior unless they share the changed paths.
- Draft step count barely matters with sampling (acceptance-limited); 5 is the best mix.
- `elsa/frspec/qwen3.6-35b-a3b-32k.pt` matches this checkpoint's tokenizer
  (`tokenizer.json` SHA-256 `5f9e4d49...cb42`); rebuild it for other tokenizers.
