#!/usr/bin/env python3
"""Engine steps, step time and draft acceptance at 1, 2 and 4 users, from a
vLLM server's own counters.

Written 2026-10-18 for our MiMo recipe on the same engine, to find
out why two users do not get twice one user's tokens. The step count is
vllm:iteration_tokens_total_count, one per engine step; the drafts counter is
per request per step, so dividing by it at two users halves the step time
(the mistake in the recipe's first results.md). Replies are pinned to 400
tokens (min_tokens + ignore_eos), temperature 0, thinking off, a salt per
request so the prefix cache stays out of it; two runs per task and user count.

    python3 bench/vllm_conc_steps.py 8000 qwen3.8-flash-next
"""

import json
import re
import sys
import threading
import time
import urllib.request
import uuid

port, model = sys.argv[1], sys.argv[2]
base = f"http://127.0.0.1:{port}"
PROMPTS = {
    "structured": "Emit a markdown table of 40 rows with columns id, name, role, team, "
    "location. Fill every cell with plausible values. Table only.",
    "code": "Write a complete Python module implementing an LRU cache with a doubly linked "
    "list and dict: get, put, delete, iteration, resize, clear, invariant validation, full "
    "docstrings and type hints. Code only.",
    "prose": "Write a precise technical essay explaining how memory bandwidth, arithmetic "
    "intensity and batching interact in modern LLM inference.",
}
COUNTERS = (
    "iteration_tokens_total_count",
    "spec_decode_num_drafts_total",
    "spec_decode_num_draft_tokens_total",
    "spec_decode_num_accepted_tokens_total",
    "generation_tokens_total",
)
POS_RE = re.compile(
    r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\} '
    r"([0-9.e+]+)$",
    re.M,
)


def metrics():
    text = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    out = {}
    for key in COUNTERS:
        out[key] = sum(
            float(v) for v in re.findall(rf"^vllm:{key}\{{[^}}]*\}} ([0-9.e+]+)$", text, re.M)
        )
    pos = {}
    for m in POS_RE.finditer(text):
        pos[int(m.group(1))] = pos.get(int(m.group(1)), 0) + float(m.group(2))
    out["pos"] = pos
    return out


def one(prompt, out, i):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": f"[{uuid.uuid4()}] {prompt}"}],
            "max_tokens": 400,
            "min_tokens": 400,
            "ignore_eos": True,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    t0 = time.perf_counter()
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    r = json.load(urllib.request.urlopen(req, timeout=600))
    out[i] = (time.perf_counter() - t0, r["usage"]["completion_tokens"])


def run(cat, c):
    before = metrics()
    res = [None] * c
    t0 = time.perf_counter()
    ths = [threading.Thread(target=one, args=(PROMPTS[cat], res, i)) for i in range(c)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    after = metrics()
    d = {k: after[k] - before[k] for k in COUNTERS}
    pos = {p: after["pos"].get(p, 0) - before["pos"].get(p, 0) for p in sorted(after["pos"])}
    toks = sum(r[1] for r in res)
    steps = d["iteration_tokens_total_count"]
    note = "" if steps > 0 else "  (no engine steps counted)"
    steps = max(steps, 1)
    drafts = max(d["spec_decode_num_drafts_total"], 1)
    drafted = max(d["spec_decode_num_draft_tokens_total"], 1)
    acc = d["spec_decode_num_accepted_tokens_total"] / drafted
    req_walls = " ".join(f"{r[0]:.2f}" for r in res)
    posrate = " ".join(f"{pos[p] / drafts:.2f}" for p in sorted(pos))
    print(
        f"{cat:10} c={c} tokens={toks} wall={wall:.2f}s agg={toks / wall:.0f} tok/s | "
        f"engine steps={steps:.0f} ms/step={wall * 1000 / steps:.1f} drafts={drafts:.0f} "
        f"(={drafts / steps:.2f}/step) accepted/draft={acc:.0%} "
        f"tok/req-step={toks / drafts:.2f} | req walls: {req_walls} | per-pos: {posrate}{note}",
        flush=True,
    )


one(PROMPTS["prose"], [None], 0)  # warm
for c in (1, 2, 4):
    for cat in PROMPTS:
        for _ in range(2):
            run(cat, c)
