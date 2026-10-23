#!/usr/bin/env python3
"""Generate model outputs over diverse prompts to estimate the output token distribution (FR-Spec)."""
import os
import asyncio, itertools, json, random
import httpx

DATA = os.environ.get("ELSA_BENCH_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))


TASKS = [
    "Write a {lang} function that {code}. Include comments and tests.",
    "Explain {topic} to a {aud}. Use examples.",
    "Debug this approach: someone tried to {code} in {lang} but it is slow and buggy. Describe likely bugs and give a fixed version.",
    "Write a {genre} story about {topic}.",
    "Solve step by step: {math}",
    "Summarize the main debates around {topic} and give your own assessment.",
    "Produce a JSON object describing {topic} with fields name, summary, pros, cons, references.",
    "Write a markdown README for a project that helps with {topic}, including install, usage and FAQ sections.",
    "Translate into {natlang} and then explain the grammar: 'The committee will review the proposal next week before making a final decision.'",
    "Plan a {n}-step agent workflow using tools (search, read_file, write_file, run_tests) to {code}. Show the tool calls as JSON.",
]
FILL = dict(
    lang=["Python", "TypeScript", "Rust", "Go", "C++", "Java", "SQL", "Bash", "C#", "Kotlin"],
    code=["parses CSV files and aggregates totals by category", "implements a rate limiter", "merges overlapping intervals",
          "builds a REST API for a todo list", "computes shortest paths in a weighted graph", "streams a large file and counts words",
          "validates email addresses and phone numbers", "implements an LRU cache", "schedules jobs with retries and backoff",
          "renders a paginated table in the browser", "diffs two JSON documents", "resizes images in parallel"],
    topic=["construction project scheduling", "quantum computing", "the history of the Roman Empire", "climate adaptation in cities",
           "transformer neural networks", "personal finance for young adults", "building codes for concrete structures",
           "the French Revolution", "supply chain logistics", "photosynthesis", "database indexing", "medieval architecture",
           "remote team management", "electric vehicles", "the immune system", "jazz improvisation"],
    aud=["ten-year-old", "senior engineer", "business executive", "university student"],
    genre=["science fiction", "mystery", "humorous", "historical", "fantasy"],
    math=["A tank fills at 12 L/min and drains at 5 L/min; starting with 40 L, when does it hold 250 L?",
          "Find all integer solutions of x^2 - 5x + 6 = 0 and prove there are no others.",
          "A rectangle's perimeter is 64 and its area is 240. Find its sides.",
          "What is the probability of rolling at least one six in four rolls of a fair die?",
          "Compute the derivative of x^3 * ln(x) and find its critical points."],
    natlang=["Spanish", "French", "German", "Chinese", "Japanese"],
    n=["5", "8"],
)

def prompts(n, seed=0):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        t = rnd.choice(TASKS)
        out.append(t.format(**{k: rnd.choice(v) for k, v in FILL.items()}))
    return out

async def main():
    ps = prompts(480)
    sem = asyncio.Semaphore(8)
    os.makedirs(DATA, exist_ok=True)
    f = open(os.path.join(DATA, "gen_corpus.jsonl"), "w")
    async with httpx.AsyncClient(timeout=1800) as c:
        async def one(i, p):
            async with sem:
                r = await c.post("http://127.0.0.1:30000/v1/chat/completions", json={
                    "model": "qwen3.6", "max_tokens": 1500, "temperature": 0.7,
                    "chat_template_kwargs": {"enable_thinking": i % 3 == 0},
                    "messages": [{"role": "user", "content": p}]})
                m = r.json()["choices"][0]["message"]
                f.write(json.dumps({"prompt": p, "reasoning": m.get("reasoning_content") or "", "content": m.get("content") or ""}) + "\n")
                f.flush()
        await asyncio.gather(*[one(i, p) for i, p in enumerate(ps)])

asyncio.run(main())
