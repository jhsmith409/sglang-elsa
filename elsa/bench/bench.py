#!/usr/bin/env python3
"""Throughput benchmark for an OpenAI-compatible server using realistic prompts.

Measures per-request decode speed (output tokens / time after first token),
TTFT, and aggregate throughput at a set of concurrency levels.
"""
import argparse, asyncio, json, statistics, time
import httpx

PROMPTS = [
    "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H12M' into a datetime.timedelta. Include docstring, type hints, error handling and five pytest unit tests.",
    "Explain how a B-tree differs from a binary search tree, why databases use B-trees, and walk through an insertion that causes a node split. Be thorough.",
    "Write a bash script that watches a directory for new .csv files, validates that each has a header row with columns id,name,email, and moves valid files to ./ok and invalid ones to ./bad, logging every action with timestamps.",
    "A train leaves city A at 9:00 traveling 80 km/h toward city B, 400 km away. Another leaves B at 10:00 traveling 120 km/h toward A. When and where do they meet? Show the work step by step, then generalize the formula.",
    "Implement a thread-safe LRU cache in Rust with get, put and a capacity limit. Explain the design choices and ownership issues.",
    "Write a short story (about 600 words) about a lighthouse keeper who discovers that the light is being used to send messages to someone at sea.",
    "Compare REST, GraphQL and gRPC for a mobile app backend. Cover performance, tooling, caching, versioning and give a recommendation with justification.",
    "Write a TypeScript React component for a sortable, filterable data table with pagination. Use hooks, no external table libraries, and include prop types.",
]

async def one(client, url, model, prompt, max_tokens, temperature, thinking):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if temperature == 0:
        body["top_p"] = 1.0
    t0 = time.perf_counter(); tfirst = None; ntok = None; chunks = 0
    async with client.stream("POST", url, json=body, timeout=3600) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: "): continue
            d = line[6:]
            if d == "[DONE]": break
            j = json.loads(d)
            if j.get("usage"): ntok = j["usage"]["completion_tokens"]
            for c in j.get("choices", []):
                delta = c.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    chunks += 1
                    if tfirst is None: tfirst = time.perf_counter()
    t1 = time.perf_counter()
    ntok = ntok or chunks
    dec = (ntok - 1) / (t1 - tfirst) if tfirst and t1 > tfirst else 0.0
    return dict(ttft=(tfirst or t1) - t0, total=t1 - t0, tokens=ntok, decode_tps=dec)

async def run_level(base, model, conc, nreq, max_tokens, temperature, thinking):
    url = base.rstrip("/") + "/v1/chat/completions"
    limits = httpx.Limits(max_connections=conc + 4)
    async with httpx.AsyncClient(limits=limits) as client:
        sem = asyncio.Semaphore(conc)
        async def task(i):
            async with sem:
                return await one(client, url, model, PROMPTS[i % len(PROMPTS)], max_tokens, temperature, thinking)
        t0 = time.perf_counter()
        res = await asyncio.gather(*[task(i) for i in range(nreq)])
        wall = time.perf_counter() - t0
    toks = sum(r["tokens"] for r in res)
    return dict(concurrency=conc, requests=nreq, wall_s=round(wall, 2),
                agg_tps=round(toks / wall, 1),
                per_req_decode_tps_median=round(statistics.median(r["decode_tps"] for r in res), 1),
                per_req_decode_tps_mean=round(statistics.mean(r["decode_tps"] for r in res), 1),
                ttft_median_s=round(statistics.median(r["ttft"] for r in res), 3),
                mean_out_tokens=round(toks / nreq, 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default="qwen3.6")
    ap.add_argument("--conc", default="1,4,8")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    # warmup
    asyncio.run(run_level(a.base, a.model, 1, 1, 64, a.temperature, a.thinking))
    rows = []
    for c in [int(x) for x in a.conc.split(",")]:
        n = max(c * 4, 8) if c > 1 else 8
        r = asyncio.run(run_level(a.base, a.model, c, n, a.max_tokens, a.temperature, a.thinking))
        r["label"] = a.label; rows.append(r); print(json.dumps(r), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            for r in rows: f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()
