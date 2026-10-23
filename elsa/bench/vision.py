#!/usr/bin/env python3
"""Vision smoke test: describe SGLang's example image (a man ironing on the back of a taxi)."""
import base64, os, sys
import httpx

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30000"
model = sys.argv[2] if len(sys.argv) > 2 else "qwen3.6"
image = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../examples/assets/example_image.png")
img = base64.b64encode(open(image, "rb").read()).decode()
r = httpx.post(base + "/v1/chat/completions", timeout=600, json={
    "model": model, "temperature": 0, "max_tokens": 100,
    "chat_template_kwargs": {"enable_thinking": False},
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
        {"type": "text", "text": "Describe this image in one sentence."}]}]})
out = r.json()["choices"][0]["message"]["content"]
ok = "taxi" in out.lower() or "iron" in out.lower()
print("VISION_OK" if ok else "VISION_FAIL", repr(out))
