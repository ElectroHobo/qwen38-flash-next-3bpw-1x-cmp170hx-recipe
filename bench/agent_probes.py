#!/usr/bin/env python3
"""Two agent-shaped probes against a vLLM chat endpoint: code editing and follow-up turns.

Written 2026-10-19 for the W4A16 Qwen3.8-Flash-Next on gpubox
(plans/qwen38_exl3_recipe_on_gpubox.md), after reading inordic's build, which
gets +15 % on code editing from n-gram drafting and sub-second follow-up turns
from a smaller prefix-match unit. Both are things agents do all day and neither
shows in a generation bench.

  edit      a ~2k-token Python file in the prompt, "rename every process_item to
            handle_item and return the complete file" — the reply is mostly a copy
            of the prompt, which is what n-gram / suffix drafting is good at.
            Reports decode tok/s (from usage, not chunks), completion tokens, and
            whether the rename landed. Three runs, salted.
  followup  a ~15k-token conversation: a long document, a first question (answered
            by the server), then a second question in the same conversation.
            Reports time to first token for turn 1 (cold) and turn 2 (should be a
            prefix-cache hit). Three conversations.

    python3 agent_probes.py edit TAG [--port 8000] [--model qwen3.8-flash-next]
    python3 agent_probes.py followup TAG [--port 8000] [--model qwen3.8-flash-next]

Temperature 0, thinking off. Writes /root/evals/out/agent-<probe>-<TAG>.json.
"""

import argparse
import json
import statistics
import time
import urllib.request
import uuid

WORDS = (
    "river stone lamp orbit copper meadow signal harbor quartz lantern ember violet "
    "canyon thread marble falcon cedar prism tundra saffron"
).split()


def post(base, body, timeout=3600):
    req = urllib.request.Request(
        base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def stream(base, body):
    """Stream one chat completion; return (ttft_s, decode_tok_s, completion_tokens, text)."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    first = last = None
    text = []
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data:") or b"[DONE]" in line:
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                now = time.perf_counter()
                if first is None:
                    first = now
                last = now
                text.append(ch[0]["delta"]["content"])
    n = (usage or {}).get("completion_tokens", 0)
    rate = (n - 1) / (last - first) if first and last and last > first and n > 1 else 0.0
    return (first - t0) if first else float("nan"), rate, n, "".join(text)


def count_tokens(base, model, text):
    req = urllib.request.Request(
        base + "/tokenize", json.dumps({"model": model, "prompt": text}).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["count"]


def python_file(n_funcs=15):
    out = ["import json", "import time", "from pathlib import Path", "", ""]
    for i in range(n_funcs):
        out += [
            f"def process_item_{i}(item: dict, retries: int = {i % 5 + 1}) -> dict:",
            f'    """Process item kind {i}: validate, enrich, and stamp it."""',
            "    if not isinstance(item, dict):",
            f'        raise TypeError("item {i} must be a dict")',
            f"    result = dict(item)",
            f"    result['kind'] = {i}",
            f"    result['weight'] = item.get('weight', 0) * {i + 1} + {i * 7 % 13}",
            "    for attempt in range(retries):",
            "        if result['weight'] > attempt:",
            "            result['attempt'] = attempt",
            "            break",
            "    result['stamp'] = time.time()",
            "    return result",
            "",
            "",
        ]
    out += [
        "def process_item(item: dict) -> dict:",
        '    """Dispatch to the kind-specific processor."""',
        "    kind = item.get('kind', 0) % " + str(n_funcs),
        "    return globals()[f'process_item_{kind}'](item)",
        "",
        "",
        "if __name__ == '__main__':",
        "    print(json.dumps(process_item({'kind': 3, 'weight': 2})))",
        "",
    ]
    return "\n".join(out)


def salad(n_words, seed):
    return " ".join(WORDS[(seed * 7 + k * 3) % len(WORDS)] for k in range(n_words))


def run_edit(base, model, tag, runs):
    src = python_file()
    rows = []
    for i in range(runs):
        prompt = (
            f"[{uuid.uuid4()}] Here is a Python file:\n\n```python\n{src}\n```\n\n"
            "Rename every function `process_item_N` to `handle_item_N` and `process_item` to "
            "`handle_item`, including every call site, and return the complete file in one "
            "```python block. No commentary."
        )
        ttft, rate, n, text = stream(base, {
            "model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4000, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        ok = "def handle_item_14" in text and "def process_item" not in text
        rows.append({"ttft_s": ttft, "decode_tok_s": rate, "completion_tokens": n, "renamed": ok})
        print(f"[{tag}] edit run {i + 1}: TTFT {ttft:.2f} s, {n} tokens at {rate:.1f} tok/s, renamed={ok}", flush=True)
    med = statistics.median(r["decode_tok_s"] for r in rows)
    print(f"[{tag}] edit: prompt ~{count_tokens(base, model, src):,} tokens of code; decode median {med:.1f} tok/s")
    return {"prompt_code_tokens": count_tokens(base, model, src), "runs": rows, "decode_median": med}


def run_followup(base, model, tag, runs, doc_tokens):
    rows = []
    for i in range(runs):
        seed = int(uuid.uuid4().int % 10000)
        doc = salad(doc_tokens, seed)
        while count_tokens(base, model, doc) < doc_tokens:
            doc += " " + salad(2000, seed + 1)
        msgs = [{"role": "user", "content": f"[{uuid.uuid4()}] Read this document:\n\n{doc}\n\nIn one sentence, what is the seventh word?"}]
        t1, _, n1, a1 = stream(base, {"model": model, "messages": msgs, "max_tokens": 60, "temperature": 0,
                                      "chat_template_kwargs": {"enable_thinking": False}})
        msgs += [{"role": "assistant", "content": a1},
                 {"role": "user", "content": "And the twelfth word? One sentence."}]
        t2, _, n2, _ = stream(base, {"model": model, "messages": msgs, "max_tokens": 60, "temperature": 0,
                                     "chat_template_kwargs": {"enable_thinking": False}})
        rows.append({"doc_tokens": count_tokens(base, model, doc), "turn1_ttft_s": t1, "turn2_ttft_s": t2})
        print(f"[{tag}] conversation {i + 1}: {rows[-1]['doc_tokens']:,}-token document; turn 1 TTFT {t1:.2f} s, turn 2 TTFT {t2:.2f} s", flush=True)
    print(f"[{tag}] followup: turn 2 TTFT median {statistics.median(r['turn2_ttft_s'] for r in rows):.2f} s "
          f"(turn 1 {statistics.median(r['turn1_ttft_s'] for r in rows):.2f} s)")
    return {"runs": rows}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("probe", choices=["edit", "followup"])
    a.add_argument("tag")
    a.add_argument("--port", type=int, default=8000)
    a.add_argument("--model", default="qwen3.8-flash-next")
    a.add_argument("--runs", type=int, default=3)
    a.add_argument("--doc-tokens", type=int, default=15000)
    args = a.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    out = run_edit(base, args.model, args.tag, args.runs) if args.probe == "edit" else \
        run_followup(base, args.model, args.tag, args.runs, args.doc_tokens)
    path = f"/root/evals/out/agent-{args.probe}-{args.tag}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"[{args.tag}] saved {path}")


if __name__ == "__main__":
    main()
