#!/usr/bin/env python3
"""Does the KV offload tier save a long shared prefix from being re-read?

gpubox 2026-10-19: this is the MiMo recipe's bench/kv_offload_bench.py with its
requests moved from /v1/completions to /v1/chat/completions with thinking off
(the `complete` function below), because Qwen3.8-Flash-Next answers a raw
completion prompt with reasoning tokens only, so the original's hidden-code
check reads empty replies. Timings and counters are otherwise the same harness.
Used for the W4A16 Qwen KV offload rows in plans/qwen38_exl3_recipe_on_gpubox.md.

Original docstring follows.

Does the KV offload tier save the model from re-reading a long shared prefix?

Stdlib only. The question is the many-agents one: several requests start from
the same long context, and by the time a later one arrives other traffic has
pushed that context's KV out of the GPU pool. Without an offload tier vLLM
recomputes it (cold prefill); with one (KV_OFFLOAD_GB, KV_OFFLOAD_DIR) it
should copy it back from host RAM or disk. So the run is:

    1. cold     the shared prefix, first time             -> full prefill
    2. warm     same prefix, new question                 -> GPU prefix hit
    3. flood    unique filler totalling > the KV pool     -> evicts the prefix
    4. evicted  same prefix, new question                 -> recompute, or reload

Compare the `evicted` row's time to first token between a run with offload off
and one with it on. Near `warm` means the reload works; near `cold` means it
did nothing. The `evicted-again` row comes after `evicted` re-prefilled the
prefix, so it is a GPU hit either way, not a second eviction test.

The RAM tier must be larger than the flood, or the flood pushes the prefix out
of host RAM too and row 4 is a recompute either way (unless a disk tier is
behind it, which is the disk-tier test). The default flood is sized for the
recipe's 1.62 M-token pool.

Correctness: the prefix hides one code per question at different depths, and
each question asks for one back. A reloaded prefix with the wrong bytes in it
loses the codes.

Prompts are seeded per run, so no run is served from an earlier run's cache.
Sizes are measured with the server's own /tokenize, not guessed.

Usage:
    python3 bench/kv_offload_bench.py TAG [--port 8000] [--model qwen3.8-flash-next]
        [--prefix-tokens 100000] [--flood-tokens 1800000] [--flood-size 100000]
"""

import argparse
import json
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://127.0.0.1:8000"
MODEL = "qwen3.8-flash-next"
WORDS = ["river", "stone", "lamp", "orbit", "copper", "meadow", "signal", "harbor", "quartz", "lantern", "ember", "violet", "canyon", "thread", "marble", "falcon", "prism", "tundra", "cedar", "beacon", "saddle", "glacier", "pepper", "anchor", "velvet", "comet", "hollow", "timber", "summit", "fable", "crater", "willow"]


def post(path: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(
        BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def count_tokens(text: str) -> int:
    return post("/tokenize", {"model": MODEL, "prompt": text})["count"]


def salad(rng: random.Random, n_words: int) -> str:
    out = []
    for i in range(n_words):
        out.append(rng.choice(WORDS))
        if i % 9 == 8:
            out.append(str(rng.randrange(10_000)) + ".")
    return " ".join(out)


def sized_salad(rng: random.Random, target: int) -> str:
    """Word salad of about `target` tokens, measured, not guessed."""
    probe = salad(rng, 2000)
    per_word = count_tokens(probe) / 2000
    return salad(rng, int(target / per_word))


def complete(prompt: str, max_tokens: int) -> tuple[float, float, str]:
    """Stream one completion; return (ttft_s, total_s, text)."""
    # gpubox 2026-10-19: the chat endpoint with thinking off, because this
    # model answers a raw completion prompt with reasoning tokens only.
    body = {
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    ttft = None
    text = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            piece = json.loads(line[6:])["choices"][0].get("delta", {}).get("content") or ""
            if piece and ttft is None:
                ttft = time.monotonic() - t0
            text.append(piece)
    return ttft or float("nan"), time.monotonic() - t0, "".join(text)


def metrics() -> dict[str, float]:
    """Prefix-cache and offload counters from /metrics, summed over labels."""
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        body = r.read().decode()
    out: dict[str, float] = {}
    for line in body.splitlines():
        if line.startswith("#") or not re.search(r"prefix_cache|offload|preempt", line):
            continue
        name = line.split("{")[0].split(" ")[0]
        try:
            out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
        except ValueError:
            pass
    return out


def main() -> None:
    global BASE, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--port", default="8000")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prefix-tokens", type=int, default=100_000)
    ap.add_argument("--flood-tokens", type=int, default=1_800_000)
    ap.add_argument("--flood-size", type=int, default=100_000)
    a = ap.parse_args()
    BASE = f"http://127.0.0.1:{a.port}"
    MODEL = a.model

    seed = int(time.time())
    rng = random.Random(seed)
    codes = [f"{rng.randrange(10**7):07d}" for _ in range(3)]
    names = ["alpha", "bravo", "charlie"]
    parts = [sized_salad(rng, a.prefix_tokens // 4) for _ in range(4)]
    prefix = "Read the notes below; three access codes are hidden in them.\n\n"
    for i, part in enumerate(parts):
        prefix += part + "\n\n"
        if i < 3:
            prefix += f"The access code for vault {names[i]} is {codes[i]}.\n\n"
    print(f"[{a.tag}] seed {seed}, prefix {count_tokens(prefix):,} tokens, codes {codes}", flush=True)

    def ask(i: int) -> tuple[float, float, str, bool]:
        q = prefix + f"Question: what is the access code for vault {names[i]}?\nAnswer: "
        ttft, total, text = complete(q, 24)
        return ttft, total, text.strip(), codes[i] in text

    rows = []
    m0 = metrics()
    rows.append(("cold", 0, *ask(0)))
    rows.append(("warm", 1, *ask(1)))

    n = max(1, a.flood_tokens // a.flood_size)
    flood = [sized_salad(random.Random(seed + 1 + k), a.flood_size) for k in range(n)]
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(lambda p: complete(p, 1), flood))
    print(f"[{a.tag}] flood: {n} x ~{a.flood_size:,} tokens in {time.monotonic() - t0:.0f} s", flush=True)

    rows.append(("evicted", 2, *ask(2)))
    rows.append(("evicted-again", 0, *ask(0)))
    m1 = metrics()

    print(f"\n[{a.tag}] {'row':<14} {'TTFT s':>8} {'total s':>8}  code  reply")
    for name, _, ttft, total, text, ok in rows:
        mark = "ok " if ok else "BAD"
        print(f"[{a.tag}] {name:<14} {ttft:8.2f} {total:8.2f}  {mark}   {text[:50]!r}")
    # If the prefix survived on the GPU, `evicted` reads like `warm` and says
    # nothing about offload. A working reload reads the same way, so only the
    # offload-off run can tell them apart.
    warm_ttft, evicted_ttft = rows[1][2], rows[2][2]
    if evicted_ttft < 3 * warm_ttft:
        print(f"[{a.tag}] NOTE: evicted TTFT is near warm - either offload reloaded the "
              "prefix or the flood did not evict it. If the offload-off run reads near "
              "warm too, raise --flood-tokens")
    print(f"\n[{a.tag}] counter changes over the run:")
    for k in sorted(set(m0) | set(m1)):
        d = m1.get(k, 0.0) - m0.get(k, 0.0)
        # Hit counters print even at zero: zero hits is the finding, and a
        # missing line would read the same as a missing counter.
        if d or "hits" in k:
            print(f"[{a.tag}]   {k} {d:+,.0f}")


if __name__ == "__main__":
    main()
