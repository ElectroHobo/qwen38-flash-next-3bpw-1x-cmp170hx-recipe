#!/usr/bin/env python3
"""Aggregate decode throughput from 1 to many users, with the server's own step counters.

N requests at once, each pinned to --tokens tokens (min_tokens + ignore_eos), temperature 0,
thinking off, a salt per request so the prefix cache stays out of it; the request prompts
rotate through the structured, code and prose tasks of cmp_recipe_bench.py so each round is a
mix. One warm-up round, then --rounds rounds per user count; the table reports the median.
Aggregate = completion tokens / the round's wall time (prefill and client included); ms per
step and tokens per request-step come from vllm:iteration_tokens_total_count and the
spec-decode counters, as in vllm_conc_steps.py.

    python3 bench/concurrency_sweep.py --port 8000 --model qwen3.8-flash-next --tag mine
    python3 bench/concurrency_sweep.py --port 8000 --model qwen3.8-flash-next --tag mine \\
        --users 1,4,16,64,128

Writes a markdown table to stdout and the rounds to <--out>/concurrency-<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import threading
import time
import urllib.request
import uuid
from pathlib import Path

PROMPTS = (
    "Emit a markdown table of 40 rows with columns id, name, role, team, location. Fill every "
    "cell with plausible values. Table only.",
    "Write a complete Python module implementing an LRU cache with a doubly linked list and "
    "dict: get, put, delete, iteration, resize, clear, invariant validation, full docstrings and "
    "type hints. Code only.",
    "Write a precise technical essay explaining how memory bandwidth, arithmetic intensity and "
    "batching interact in modern LLM inference.",
)
COUNTERS = (
    "iteration_tokens_total_count",
    "spec_decode_num_drafts_total",
    "spec_decode_num_accepted_tokens_total",
    "generation_tokens_total",
)


def metrics(base: str) -> dict[str, float]:
    text = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    return {
        key: sum(float(v) for v in re.findall(rf"^vllm:{key}\{{[^}}]*\}} ([0-9.e+]+)$", text, re.M))
        for key in COUNTERS
    }


def request(base: str, model: str, prompt: str, tokens: int, out: list, i: int) -> None:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": f"[{uuid.uuid4()}] {prompt}"}],
        "max_tokens": tokens,
        "min_tokens": tokens,
        "ignore_eos": True,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    out[i] = r["usage"]["completion_tokens"]


def round_(base: str, model: str, users: int, tokens: int) -> dict:
    before = metrics(base)
    done = [0] * users
    threads = [threading.Thread(target=request,
                                args=(base, model, PROMPTS[i % len(PROMPTS)], tokens, done, i))
               for i in range(users)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    after = metrics(base)
    d = {k: after[k] - before[k] for k in COUNTERS}
    steps = max(d["iteration_tokens_total_count"], 1)
    drafts = d["spec_decode_num_drafts_total"]
    toks = sum(done)
    return {
        "users": users, "tokens": toks, "wall_s": round(wall, 3),
        "aggregate_tok_s": round(toks / wall, 1),
        "ms_per_step": round(wall * 1000 / steps, 2),
        "tokens_per_request_step": round(toks / drafts, 3) if drafts else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="8000")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--users", default="1,4,8,16,32,48,64,96,128")
    ap.add_argument("--tokens", type=int, default=700)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    users = [int(u) for u in a.users.split(",")]
    rows, runs = [], []
    for n in users:
        round_(base, a.model, n, a.tokens)  # warm-up, not counted
        rs = [round_(base, a.model, n, a.tokens) for _ in range(a.rounds)]
        runs += rs
        row = {k: statistics.median(r[k] for r in rs)
               for k in ("aggregate_tok_s", "ms_per_step", "tokens_per_request_step")
               if all(r[k] is not None for r in rs)}
        rows.append((n, row))
        print(f"{n:4} users: {row.get('aggregate_tok_s', 0):8.1f} tok/s, "
              f"{row.get('ms_per_step', 0):6.1f} ms/step, "
              f"{row.get('tokens_per_request_step', 0) or 0:.2f} tokens/request-step", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"concurrency-{a.tag}.json").write_text(json.dumps(
        {"tag": a.tag, "model": a.model, "tokens": a.tokens, "rounds": runs}, indent=1))
    print(f"\n| users | aggregate tok/s | ms / step | tokens / request-step |\n|---:|---:|---:|---:|")
    for n, row in rows:
        print(f"| {n} | {row.get('aggregate_tok_s', 0):.1f} | {row.get('ms_per_step', 0):.1f} | "
              f"{row.get('tokens_per_request_step', 0) or 0:.2f} |")


if __name__ == "__main__":
    main()
