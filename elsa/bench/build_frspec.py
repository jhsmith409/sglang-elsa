#!/usr/bin/env python3
"""Build FR-Spec hot-token maps (top-K token ids by frequency) for the Qwen3.6 tokenizer.

Sources: model-generated outputs (weighted x3, the distribution the draft must
predict), plus local Python source and markdown docs for coding/agent text.
"""
import os
import collections, glob, json, random, sys
import torch
from transformers import AutoTokenizer

DATA = os.environ.get("ELSA_BENCH_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"


def gsm8k_path():
    path = os.path.join(DATA, "gsm8k_test.jsonl")
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(DATA, exist_ok=True)
        urllib.request.urlretrieve(GSM8K_URL, path)
    return path
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


tok = AutoTokenizer.from_pretrained(os.environ["ELSA_MODEL"])
cnt = collections.Counter()

def add(text, w=1):
    for t in tok.encode(text, add_special_tokens=False):
        cnt[t] += w

n_gen = 0
for line in open(os.path.join(DATA, "gen_corpus.jsonl")):
    j = json.loads(line)
    add(j["reasoning"] + "\n" + j["content"], 3)
    n_gen += 1

rnd = random.Random(0)
py = glob.glob(os.path.join(REPO, "python/sglang/**/*.py"), recursive=True)
md = glob.glob(os.path.join(REPO, "docs/**/*.md"), recursive=True)
for path in rnd.sample(py, min(1500, len(py))) + md:
    try:
        add(open(path, errors="ignore").read()[:200000])
    except OSError:
        pass
for line in open(gsm8k_path()):
    add(json.loads(line)["question"])

special = set(tok.all_special_ids) | {i for t, i in tok.get_added_vocab().items()}
print(f"gen docs={n_gen} distinct tokens seen={len(cnt)} total={sum(cnt.values())} special={len(special)}")
ranked = [t for t, _ in cnt.most_common()]
for k in [int(x) for x in sys.argv[1:]] or [32768, 65536]:
    hot = sorted(special | set(ranked[: k - len(special)]))[:k]
    covered = sum(cnt[t] for t in hot) / sum(cnt.values())
    torch.save(hot, os.path.join(DATA, f"frspec_{k // 1024}k.pt"))
    print(f"k={k}: coverage of corpus tokens {covered:.4%}")
