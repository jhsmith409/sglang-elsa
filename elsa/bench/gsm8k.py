#!/usr/bin/env python3
"""GSM8K accuracy (first N questions, greedy, thinking off) against an OpenAI-compatible server."""
import os
import argparse, asyncio, json, re
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


def extract(s):
    m = re.findall(r"####\s*(-?[\d,\.]+)", s) or re.findall(r"-?\d[\d,]*\.?\d*", s)
    return m[-1].replace(",", "").rstrip(".") if m else None

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default="qwen3.6")
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--conc", type=int, default=16)
    a = ap.parse_args()
    qs = [json.loads(l) for l in open(gsm8k_path())][: a.n]
    sem = asyncio.Semaphore(a.conc)
    async with httpx.AsyncClient(timeout=1800) as c:
        async def one(q):
            async with sem:
                r = await c.post(a.base + "/v1/chat/completions", json={
                    "model": a.model, "temperature": 0, "max_tokens": 1024,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": q["question"] + "\nSolve step by step, then give the final answer on its own line as '#### <number>'."}]})
                out = r.json()["choices"][0]["message"]["content"] or ""
                gold = q["answer"].split("####")[-1].strip().replace(",", "")
                p = extract(out)
                try: return float(p) == float(gold)
                except Exception: return False
        res = await asyncio.gather(*[one(q) for q in qs])
    print(json.dumps({"gsm8k_n": len(res), "accuracy": round(sum(res) / len(res), 4)}))

asyncio.run(main())
