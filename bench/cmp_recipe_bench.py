#!/usr/bin/env python3
"""Decode and cold-prefill numbers in the form Morrowmake's recipes publish.

Morrowmake's GLM and Qwen recipes report their tables "under the MiaAI-Lab
protocol" (MiaAI-Lab's DGX-Spark kits, bench/decode.py and bench/cold_prefill.py,
AGPL-3.0 — so this is our own harness, written to the same rules, with their
category prompts used as data with attribution):

  * temperature 0, thinking off, every reply pinned to exactly --tokens tokens
    (min_tokens + ignore_eos), a salt per request so the prefix cache is out;
  * decode rate = (completion_tokens - 1) / (last chunk - first content token);
  * one user: the median of five streaming decode rates;
  * 2, 4 and 8 users: the median of three runs of the aggregate rate, actual
    completion tokens divided by the batch's wall time, prefill and client
    overhead included;
  * cold prefill: the median rate over three real-text prompts, repeated three
    times each, prompt length set through /tokenize (one rescale), no cache hits.

    python3 bench/cmp_recipe_bench.py --port 8000 --model qwen3.8-flash-next --tag mine
    python3 bench/cmp_recipe_bench.py --port 8000 --model qwen3.8-flash-next --tag mine --prefill-only

Writes a markdown table to stdout and the raw runs to
/root/evals/out/cmp-recipe-bench-<tag>.json.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.request
import uuid
from pathlib import Path

# MiaAI-Lab's category prompts (bench/decode.py in their MiMo-V2.6-Flash-2x-DGX-Sparks
# kit), used as data. Morrowmake's tables use the structured, code and prose rows.
PROMPTS = {
    "structured": "Emit a markdown table of 40 rows with columns id, name, role, team, "
    "location. Fill every cell with plausible values. Table only.",
    "code": "Write a complete Python module implementing an LRU cache with a doubly linked "
    "list and dict: get, put, delete, iteration, resize, clear, invariant validation, full "
    "docstrings and type hints. Then ten usage examples. Code only.",
    "prose": "Write a precise technical essay explaining how memory bandwidth, arithmetic "
    "intensity and batching interact in modern LLM inference. Continue until cut off.",
}
OUT = Path("/root/evals/out")
# Real text for the cold-prefill prompts: Python sources, concatenated to length.
SOURCE_DIRS = ("/opt/qwen-flash-1card/vllm/src/vllm", "/usr/lib/python3.13", "/usr/lib/python3.12")


def post(base: str, path: str, body: dict, timeout: float = 1800):
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=timeout)


def stream_chat(base: str, model: str, content: str, tokens: int, pin: bool) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if pin:
        body["min_tokens"] = tokens
        body["ignore_eos"] = True
    t0 = time.perf_counter()
    first = last = None
    usage = None
    with post(base, "/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            delta = ((d.get("choices") or [{}])[0].get("delta")) or {}
            if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                now = time.perf_counter()
                if first is None:
                    first = now
                last = now
    if first is None or usage is None:
        raise RuntimeError("no streamed token or no usage chunk")
    n = usage["completion_tokens"]
    if pin:
        if not last > first:
            raise RuntimeError("fewer than two content chunks; the decode rate needs --tokens >= 2")
        if n != tokens:
            # the server ignored min_tokens / ignore_eos; the rows would mix lengths
            print(f"  warning: pinned reply came back {n} tokens, not {tokens}", file=sys.stderr)
    return {
        "completion_tokens": n,
        "prompt_tokens": usage["prompt_tokens"],
        "ttft_s": first - t0,
        "decode_tok_s": (n - 1) / (last - first) if last > first else None,
        "wall_s": time.perf_counter() - t0,
    }


def salted(prompt: str) -> str:
    return f"Benchmark {uuid.uuid4().hex[:16]}.\n{prompt}"


def decode_table(base: str, model: str, tokens: int, concs: list[int], runs: dict) -> list[str]:
    lines = []
    for c in concs:
        cells = []
        for cat, prompt in PROMPTS.items():
            if c == 1:
                rates = []
                for _ in range(5):
                    r = stream_chat(base, model, salted(prompt), tokens, pin=True)
                    rates.append(r["decode_tok_s"])
                    runs.setdefault("decode", []).append({"cat": cat, "c": 1, **r})
                cells.append(statistics.median(rates))
            else:
                aggs = []
                for _ in range(3):
                    wall0 = time.perf_counter()

                    def one(_i: int, p: str = prompt) -> dict:
                        return stream_chat(base, model, salted(p), tokens, pin=True)

                    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                        rs = list(pool.map(one, range(c)))
                    wall = time.perf_counter() - wall0
                    agg = sum(r["completion_tokens"] for r in rs) / wall
                    aggs.append(agg)
                    runs.setdefault("decode", []).append(
                        {
                            "cat": cat,
                            "c": c,
                            "aggregate_tok_s": agg,
                            "wall_s": wall,
                            "per_stream_median": statistics.median(r["decode_tok_s"] for r in rs),
                            "lengths": [r["completion_tokens"] for r in rs],
                        }
                    )
                cells.append(statistics.median(aggs))
            print(f"  {c} user(s) {cat:10} {cells[-1]:7.1f} tok/s", flush=True)
        label = "Streaming decode, 1 user" if c == 1 else f"Decode, {c} users, aggregate"
        lines.append(
            f"| {label}, structured / code / prose | "
            + " / ".join(f"{v:.1f}" for v in cells)
            + " tok/s |"
        )
    return lines


def real_text(min_chars: int) -> str:
    chunks, total = [], 0
    for d in SOURCE_DIRS:
        p = Path(d)
        if not p.is_dir():
            continue
        for f in sorted(p.rglob("*.py")):
            try:
                t = f.read_text(errors="replace")
            except OSError:
                continue
            if 8_000 < len(t) < 60_000:
                chunks.append(f"# {f}\n{t}")
                total += len(t)
                if total >= min_chars:
                    return "\n\n".join(chunks)
    raise SystemExit(f"not enough source text for the prefill prompts under {SOURCE_DIRS}")


def tokenize(base: str, model: str, text: str) -> int:
    with post(base, "/tokenize", {"model": model, "prompt": text}, timeout=300) as r:
        return json.load(r)["count"]


def prefill_rows(base: str, model: str, targets: list[int], runs: dict) -> list[str]:
    rates, sizes = [], []
    for target in targets:
        # over-long by characters first, then one rescale from the measured token count
        text = real_text(target * 5)[: int(target * 4.5)]
        text = text[: int(len(text) * target / tokenize(base, model, text))]
        for _ in range(3):
            prompt = f"[{uuid.uuid4().hex}]\n{text}\n\nReply with OK."
            r = stream_chat(base, model, prompt, 1, pin=False)
            rate = r["prompt_tokens"] / r["ttft_s"]
            rates.append(rate)
            runs.setdefault("prefill", []).append({**r, "tok_s": rate})
            print(
                f"  prefill {r['prompt_tokens']:6d} tokens  TTFT {r['ttft_s']:6.2f} s  "
                f"{rate:7.0f} tok/s",
                flush=True,
            )
        sizes.append(r["prompt_tokens"])  # the length the rate was measured on, salt and all
    return [
        f"| Cold prefill | {statistics.median(rates):,.0f} tok/s |",
        f"| Cold prefill prompts (tokens) | {', '.join(f'{s:,}' for s in sizes)} |",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--port", default="8000")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--prefill-targets", default="24000,34000,38000")
    ap.add_argument("--prefill-only", action="store_true")
    ap.add_argument("--decode-only", action="store_true")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    runs: dict = {
        "tag": args.tag,
        "model": args.model,
        "tokens": args.tokens,
        "when": time.strftime("%Y-%m-%d %H:%M"),
    }
    rows = ["| | |", "|---|---:|"]
    try:
        if not args.prefill_only:
            # warm the server once; the salt keeps the real runs cold
            stream_chat(base, args.model, salted(PROMPTS["prose"]), 64, pin=True)
            rows += decode_table(
                base, args.model, args.tokens, [int(x) for x in args.concurrency.split(",")], runs
            )
        if not args.decode_only:
            rows += prefill_rows(
                base, args.model, [int(x) for x in args.prefill_targets.split(",")], runs
            )
    finally:
        # written even if a request fails part-way, so an hour of runs is not lost
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"cmp-recipe-bench-{args.tag}.json").write_text(json.dumps(runs, indent=1))
    print()
    print("\n".join(rows))


if __name__ == "__main__":
    main()
