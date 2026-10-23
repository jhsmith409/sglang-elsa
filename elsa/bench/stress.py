#!/usr/bin/env python3
"""Memory-headroom stress: max-concurrency sampled decode mixed with long prefills.

Passes if every request completes and the server stays healthy.
"""
import os
import argparse, asyncio, json, random, time
import httpx

DATA = os.environ.get("ELSA_BENCH_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"


def gsm8k_path():
    path = os.path.join(DATA, "gsm8k_test.jsonl")
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(DATA, exist_ok=True)
        urllib.request.urlretrieve(GSM8K_URL, path)
    return path


def long_text(n_tokens, seed):
    rnd = random.Random(seed)
    qs = [json.loads(l)["question"] for l in open(gsm8k_path())]
    out, approx = [], 0
    while approx < n_tokens:
        q = rnd.choice(qs); out.append(q); approx += len(q) / 3.6
    return "\n".join(out)

async def one(c, base, content, max_tokens, temp):
    r = await c.post(base + "/v1/chat/completions", json={
        "model": os.environ.get("ELSA_MODEL_NAME", "qwen3.6"), "max_tokens": max_tokens, "temperature": temp, "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": True},
        "messages": [{"role": "user", "content": content}]})
    r.raise_for_status()
    return r.json()["usage"]

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--short", type=int, default=6)
    ap.add_argument("--long", default="60000,120000")
    a = ap.parse_args()
    t0 = time.time()
    async with httpx.AsyncClient(timeout=3600) as c:
        tasks = [one(c, a.base, f"Write a long, detailed technical essay (topic #{i}) about distributed databases.", 3000, 0.8) for i in range(a.short)]
        tasks += [one(c, a.base, long_text(int(n), i) + "\nSummarize the kinds of problems above.", 500, 0.7) for i, n in enumerate(a.long.split(","))]
        res = await asyncio.gather(*tasks, return_exceptions=True)
        health = (await c.get(a.base + "/health")).status_code
    errs = [r for r in res if isinstance(r, Exception)]
    print(json.dumps({"ok": not errs and health == 200, "errors": [repr(e)[:200] for e in errs],
                      "prompt_tokens": [r["prompt_tokens"] for r in res if not isinstance(r, Exception)],
                      "wall_s": round(time.time() - t0, 1), "health": health}))

asyncio.run(main())
