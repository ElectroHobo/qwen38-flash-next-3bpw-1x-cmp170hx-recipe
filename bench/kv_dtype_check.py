#!/usr/bin/env python3
"""KV-cache dtype check against a vLLM server: pool size, long-context decode,
and temperature-0 answers to compare between two runs.

Written 2026-10-19 for the W4A16 Qwen3.8-Flash-Next on Morrowmake's engine
(plans/qwen38_exl3_recipe_on_gpubox.md), to answer "does FP8 KV work here, and
what does it cost": run once on the BF16 server, once on the FP8 server, diff the
two JSON files.

  * pool: vllm:cache_config_info's kv_cache_size_tokens from /metrics;
  * long context: one request whose prompt is about --ctx tokens (a word salad
    repeated, length verified through /tokenize), 400 tokens pinned
    (min_tokens + ignore_eos), temperature 0, thinking off; reports time to
    first token and the decode rate after it, with the token count from the
    server's usage rather than the chunk count (MTP packs several accepted
    tokens into one chunk); three runs, median;
  * answers: five fixed prompts at temperature 0, thinking off, 300 tokens;
    the content is saved so two runs can be compared for drift.

    python3 bench/kv_dtype_check.py TAG [--port 8000] [--model qwen3.8-flash-next] [--ctx 100000]

Writes /root/evals/out/kv-dtype-<TAG>.json and prints a summary.
"""

import argparse
import json
import re
import statistics
import time
import urllib.request
import uuid
from pathlib import Path

OUT = Path("/root/evals/out")
WORDS = (
    "river stone lamp orbit copper meadow signal harbor quartz lantern ember violet "
    "canyon thread marble falcon cedar prism tundra saffron"
).split()
ANSWER_PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Explain in one paragraph why a hash map lookup is O(1) on average but not in the worst case.",
    "Give a bash one-liner that lists the five largest files under the current directory.",
    "A train leaves at 09:40 and arrives at 13:05. How long is the journey? Show the arithmetic.",
    "Write a haiku about a cold server room.",
]


def post(base, path, body, timeout=3600):
    req = urllib.request.Request(
        base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def pool(base):
    text = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    m = re.search(r'kv_cache_size_tokens="(\d+)"', text)
    dt = re.search(r'cache_dtype="([^"]+)"', text)
    return int(m.group(1)) if m else None, dt.group(1) if dt else None


def long_prompt(base, model, target):
    words = []
    i = 0
    while True:
        chunk = " ".join(WORDS[(i + k) % len(WORDS)] for k in range(2000))
        words.append(chunk)
        i += 7
        text = "\n".join(words)
        n = post(base, "/tokenize", {"model": model, "prompt": text})["count"]
        if n >= target:
            return text, n


def stream_decode(base, model, prompt):
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"[{uuid.uuid4()}]\n{prompt}\n\nSummarise the text above in 400 words.",
            }
        ],
        "max_tokens": 400,
        "min_tokens": 400,
        "ignore_eos": True,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    first = last = usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data:") or b"[DONE]" in line:
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            if (d.get("choices") or [{}])[0].get("delta", {}).get("content"):
                now = time.perf_counter()
                if first is None:
                    first = now
                last = now
    if first is None or usage is None:
        raise SystemExit("no streamed token or no usage chunk; is the server healthy?")
    n = usage["completion_tokens"]
    return first - t0, (n - 1) / (last - first) if n > 1 and last > first else 0.0


def main():
    a = argparse.ArgumentParser()
    a.add_argument("tag")
    a.add_argument("--port", type=int, default=8000)
    a.add_argument("--model", default="qwen3.8-flash-next")
    a.add_argument("--ctx", type=int, default=100000)
    a.add_argument("--runs", type=int, default=3)
    args = a.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    out = {"tag": args.tag, "model": args.model}
    out["pool_tokens"], out["cache_dtype"] = pool(base)
    pool_s = f"{out['pool_tokens']:,}" if out["pool_tokens"] is not None else "unknown"
    print(f"[{args.tag}] pool {pool_s} tokens, cache_dtype {out['cache_dtype']}")

    text, n = long_prompt(base, args.model, args.ctx)
    ttft, rates = [], []
    for _ in range(args.runs):
        t, r = stream_decode(base, args.model, text)
        ttft.append(t)
        rates.append(r)
    out["long_ctx"] = {"prompt_tokens": n, "ttft_s": ttft, "decode_tok_s": rates}
    print(
        f"[{args.tag}] long context {n:,} tokens: TTFT median {statistics.median(ttft):.1f} s, "
        f"decode median {statistics.median(rates):.1f} tok/s (runs {', '.join(f'{x:.1f}' for x in rates)})"
    )

    answers = []
    for p in ANSWER_PROMPTS:
        r = post(
            base,
            "/v1/chat/completions",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": p}],
                "max_tokens": 300,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        answers.append(
            {
                "prompt": p,
                "content": r["choices"][0]["message"]["content"],
                "finish_reason": r["choices"][0]["finish_reason"],
                "completion_tokens": r["usage"]["completion_tokens"],
            }
        )
    out["answers"] = answers
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"kv-dtype-{args.tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[{args.tag}] {len(answers)} answers saved to {path}")


if __name__ == "__main__":
    main()
