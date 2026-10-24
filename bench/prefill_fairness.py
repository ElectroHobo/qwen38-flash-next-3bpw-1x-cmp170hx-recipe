#!/usr/bin/env python3
"""Does a long prompt arriving mid-answer stall the user who is decoding?

Stdlib only. One request streams a long answer; three seconds in, a long
prompt (about --prompt-tokens tokens) arrives and is read while the first
request keeps decoding. Reported for the decoding request: tokens per second
before the prompt arrived, while it was being read, and after; and the
longest pause between two streamed chunks. For the long prompt: its time to
first token.

Compare a run on the defaults (FAIR_CHUNK=384: while anyone is decoding,
prefill goes in 384-token slices) with one on FAIR_CHUNK=0 (off) or
FAIR_CHUNK=128.

Usage:
    python3 bench/prefill_fairness.py TAG [--port 8000] [--model qwen3.8-flash-next]
        [--prompt-tokens 30000] [--answer-tokens 1500]
"""

import argparse
import itertools
import json
import random
import threading
import time
import urllib.request
import uuid

BASE = "http://127.0.0.1:8000"
MODEL = "qwen3.8-flash-next"
WORDS = ["river", "stone", "lamp", "orbit", "copper", "meadow", "signal", "harbor", "quartz", "lantern", "ember", "violet", "canyon", "thread", "marble", "falcon", "prism", "tundra", "cedar", "beacon", "saddle", "glacier", "pepper", "anchor", "velvet", "comet", "hollow", "timber", "summit", "fable", "crater", "willow"]


def post(path: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(
        BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def salad(rng: random.Random, n_words: int) -> str:
    out = []
    for i in range(n_words):
        out.append(rng.choice(WORDS))
        if i % 9 == 8:
            out.append(str(rng.randrange(10_000)) + ".")
    return " ".join(out)


def sized_salad(rng: random.Random, target: int) -> str:
    probe = salad(rng, 2000)
    per_word = post("/tokenize", {"model": MODEL, "prompt": probe})["count"] / 2000
    return salad(rng, int(target / per_word))


def decoder(answer_tokens: int, out: dict) -> None:
    """Stream one long answer; record the arrival time and size of every chunk."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"[{uuid.uuid4()}] Write a long, detailed "
                      "technical essay on how memory bandwidth, arithmetic intensity and "
                      "batching interact in LLM inference."}],
        "max_tokens": answer_tokens, "min_tokens": answer_tokens, "ignore_eos": True,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    arrivals: list[tuple[float, int]] = []
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            delta = ((d.get("choices") or [{}])[0].get("delta")) or {}
            text = delta.get("content") or delta.get("reasoning_content") or ""
            if text:
                arrivals.append((time.monotonic(), len(text)))
    out["arrivals"] = arrivals
    out["tokens"] = usage["completion_tokens"] if usage else 0


def rate(arrivals, chars_per_token, t_from, t_to) -> float:
    chars = sum(n for t, n in arrivals if t_from <= t < t_to)
    return chars / chars_per_token / max(t_to - t_from, 1e-9)


def main() -> None:
    global BASE, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--port", default="8000")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prompt-tokens", type=int, default=30_000)
    ap.add_argument("--answer-tokens", type=int, default=1500)
    a = ap.parse_args()
    BASE = f"http://127.0.0.1:{a.port}"
    MODEL = a.model

    rng = random.Random(int(time.time()))
    prompt = sized_salad(rng, a.prompt_tokens)
    n_prompt = post("/tokenize", {"model": MODEL, "prompt": prompt})["count"]

    out: dict = {}
    t_start = time.monotonic()
    th = threading.Thread(target=decoder, args=(a.answer_tokens, out))
    th.start()
    time.sleep(3)
    t_send = time.monotonic()
    post("/v1/completions", {"model": MODEL, "prompt": prompt, "max_tokens": 1,
                             "temperature": 0})
    t_done = time.monotonic()
    th.join()
    t_end = time.monotonic()

    arrivals = out["arrivals"]
    total_chars = sum(n for _, n in arrivals)
    cpt = total_chars / max(out["tokens"], 1)
    gaps = [(nxt[0] - cur[0], cur[0]) for cur, nxt in itertools.pairwise(arrivals)]
    worst = max(gaps, default=(0.0, t_start))
    print(f"[{a.tag}] long prompt {n_prompt:,} tokens, sent 3 s in, "
          f"read in {t_done - t_send:.2f} s; "
          f"decoder {out['tokens']} tokens in {t_end - t_start:.1f} s")
    print(f"[{a.tag}] decoder tok/s  before {rate(arrivals, cpt, t_start, t_send):6.1f}   "
          f"during prefill {rate(arrivals, cpt, t_send, t_done):6.1f}   "
          f"after {rate(arrivals, cpt, t_done, t_end):6.1f}")
    print(f"[{a.tag}] longest pause between chunks {worst[0]:.2f} s, "
          f"{worst[1] - t_start:.1f} s into the run"
          f"{' (during the prefill)' if t_send <= worst[1] < t_done else ''}")


if __name__ == "__main__":
    main()
