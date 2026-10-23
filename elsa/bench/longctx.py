#!/usr/bin/env python3
"""Long-context test: TTFT (prefill speed), decode speed at depth, and needle recall."""
import os
import argparse, json, random, time
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


def build(target_tokens, seed):
    rnd = random.Random(seed)
    qs = [json.loads(l)["question"] for l in open(gsm8k_path())]
    parts, approx = [], 0
    while approx < target_tokens:
        q = rnd.choice(qs); parts.append(q); approx += len(q) / 3.6
    code = f"{rnd.randint(100000, 999999)}"
    parts.insert(rnd.randint(len(parts) // 4, 3 * len(parts) // 4), f"IMPORTANT: the secret vault code is {code}. Remember it.")
    return "\n".join(parts), code

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--tokens", default="32000,128000,240000")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--seed-offset", type=int, default=0)
    a = ap.parse_args()
    jobs = [(n, r) for r in range(a.repeats) for n in [int(x) for x in a.tokens.split(",")]]
    for n, rep in jobs:
        # Distinct seeds per repeat keep prefix caches from serving earlier prompts.
        text, code = build(n, n + 7919 * (rep + 1) + a.seed_offset)
        body = {"model": os.environ.get("ELSA_MODEL_NAME", "qwen3.6"), "temperature": 0, "max_tokens": a.max_tokens, "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": text + "\n\nWhat is the secret vault code mentioned above? Reply with the code, then write a 200-word summary of the kinds of problems in the text."}]}
        t0 = time.perf_counter(); tf = None; out = ""; usage = {}
        with httpx.stream("POST", a.base + "/v1/chat/completions", json=body, timeout=3600) as r:
            for line in r.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]": continue
                j = json.loads(line[6:])
                if j.get("usage"): usage = j["usage"]
                for c in j.get("choices", []):
                    d = c["delta"].get("content")
                    if d:
                        if tf is None: tf = time.perf_counter()
                        out += d
        t1 = time.perf_counter()
        ptok, ctok = usage.get("prompt_tokens"), usage.get("completion_tokens")
        row = dict(label=a.label, rep=rep, prompt_tokens=ptok, ttft_s=round(tf - t0, 2),
                   prefill_tps=round(ptok / (tf - t0)), decode_tps=round((ctok - 1) / (t1 - tf), 1),
                   needle_ok=code in out)
        print(json.dumps(row), flush=True)
        if a.out:
            open(a.out, "a").write(json.dumps(row) + "\n")

main()
