#!/usr/bin/env python3
"""Summarize GPU kernel time by name from a torch profiler trace (.json or .json.gz)."""
import collections, gzip, json, sys
p = sys.argv[1]
d = json.load(gzip.open(p) if p.endswith(".gz") else open(p))
ev = [e for e in d["traceEvents"] if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
tot = collections.Counter(); cnt = collections.Counter()
for e in ev:
    n = e["name"][:110]; tot[n] += e["dur"]; cnt[n] += 1
T = sum(tot.values()); span = (max(e["ts"] + e["dur"] for e in ev) - min(e["ts"] for e in ev))
print(f"kernels={len(ev)} gpu_busy_ms={T/1000:.1f} span_ms={span/1000:.1f} util={T/span:.2%}")
for n, t in tot.most_common(int(sys.argv[2]) if len(sys.argv) > 2 else 30):
    print(f"{t/1000:8.2f}ms {t/T:6.1%} n={cnt[n]:5d} {n}")
