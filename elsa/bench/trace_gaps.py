#!/usr/bin/env python3
"""Find GPU idle gaps in a torch profiler trace and attribute them to the surrounding kernels/CPU ops."""
import collections, gzip, json, sys

d = json.load(gzip.open(sys.argv[1]))
thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
ev = d["traceEvents"]
k = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")), key=lambda e: e["ts"])
cpu = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation", "python_function", "cuda_runtime")), key=lambda e: e["ts"])
gaps = collections.Counter(); cnt = collections.Counter(); total = 0.0
end = k[0]["ts"] + k[0]["dur"]
prev = k[0]
for e in k[1:]:
    g = e["ts"] - end
    if g > thresh:
        key = (prev["name"][:60], e["name"][:60])
        gaps[key] += g; cnt[key] += 1; total += g
    if e["ts"] + e["dur"] > end:
        end = e["ts"] + e["dur"]; prev = e
print(f"total gap time > {thresh}us: {total/1000:.2f} ms")
for key, g in gaps.most_common(15):
    print(f"{g/1000:7.2f}ms n={cnt[key]:4d}  after [{key[0]}]  before [{key[1]}]")
# CPU ops overlapping the largest gaps
anns = collections.Counter()
for e in cpu:
    if e.get("cat") == "user_annotation" or e.get("cat") == "python_function":
        anns[e["name"][:80]] += e["dur"]
print("\nTop CPU annotations/python functions by duration:")
for n, t in anns.most_common(25):
    print(f"{t/1000:8.2f}ms {n}")
