#!/usr/bin/env python3
"""Tool-calling smoke test: the model must call get_weather with a city argument,
then answer from the tool result. Checks streaming and non-streaming."""
import json, sys
import httpx

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30000"
model = sys.argv[2] if len(sys.argv) > 2 else "qwen3.6"
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}]
msgs = [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}]
ok = True
for think in (False, True):
    r = httpx.post(base + "/v1/chat/completions", timeout=600, json={
        "model": model, "messages": msgs, "tools": tools, "temperature": 0, "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": think}}).json()
    m = r["choices"][0]["message"]
    calls = m.get("tool_calls") or []
    good = len(calls) == 1 and calls[0]["function"]["name"] == "get_weather" and \
        "paris" in json.loads(calls[0]["function"]["arguments"]).get("city", "").lower()
    print(f"thinking={think} tool_call_ok={good} finish={r['choices'][0]['finish_reason']}")
    ok &= good
    if good:
        follow = msgs + [m, {"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps({"temp_c": 18, "sky": "light rain"})}]
        r2 = httpx.post(base + "/v1/chat/completions", timeout=600, json={
            "model": model, "messages": follow, "tools": tools, "temperature": 0, "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": think}}).json()
        text = (r2["choices"][0]["message"].get("content") or "").lower()
        good2 = "18" in text and "rain" in text
        print(f"thinking={think} answer_uses_tool_result={good2}")
        ok &= good2
# streaming tool call
chunks = []
with httpx.stream("POST", base + "/v1/chat/completions", timeout=600, json={
        "model": model, "messages": msgs, "tools": tools, "temperature": 0, "max_tokens": 512, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False}}) as s:
    for line in s.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            for c in json.loads(line[6:]).get("choices", []):
                for tc in c.get("delta", {}).get("tool_calls") or []:
                    chunks.append(tc.get("function", {}).get("arguments") or "")
args = "".join(chunks)
sgood = "paris" in args.lower()
print(f"streaming tool_call_ok={sgood} args={args!r}")
ok &= sgood
print("TOOLS_OK" if ok else "TOOLS_FAIL")
