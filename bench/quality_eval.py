#!/usr/bin/env python3
"""Compare answer quality of two OpenAI-compatible endpoints on checkable problems.

Deployed to gpubox as /root/qwen-quality-eval.py. Written 2026-09-23 to decide
between Qwen3.8-Flash-Next at 3 bpw on one card and Qwen3.8-27B AWQ on one card:
the Flash-Next build is faster per card, but its authors report wrong arithmetic,
and a throughput win is worth nothing if the answers are worse.

Three suites, all scored by a machine rather than by reading:

* ``humaneval`` -- all 164 problems. The model writes the function; the
  dataset's own tests run against it in a ``python:3.12-slim`` container with
  ``--network none``, one process per problem, 15 s timeout. Model output is
  untrusted code, which is why it never runs on the host.
* ``gsm8k`` -- the first 200 test problems. The answer is the number on the
  last ``Answer:`` line (else the last number in the reply), compared
  numerically with the gold after ``####``.
* ``livecodebench`` -- the 175 problems of release file ``test6.jsonl``
  (AtCoder and LeetCode, contests 2025-01 to 2025-04, 60 of them hard). Added
  because the two suites above sit at 97-99% for both models and cannot
  separate them. A pass needs every public and hidden test to pass, each in its
  own process with a 10 s limit, in the same sandbox. Scored by difficulty.
  Run it with a larger ``--max-tokens`` (32000): hard problems think long.

Both endpoints get identical requests: thinking on, ``reasoning_effort``
medium (the chat template defaults to xhigh, which on these builds can spend
the whole budget and answer nothing -- plans/qwen38_27b_on_gpubox.md 4b),
temperature 0.6 / top_p 0.95 / top_k 20, max_tokens 16000. A reply cut off by
the budget counts as a failure and is also reported on its own, so "wrong" and
"ran out of room" can be told apart.

Extraction reads the reply's ``content`` and tolerates reasoning leaking into
it (the 27B's reasoning parser does not separate it): the LAST code block, the
LAST ``Answer:`` line.

Stdlib only. Resumable: each reply is appended to a JSONL as it lands, and a
rerun skips ids already there.

Usage:
    python3 qwen-quality-eval.py run --base http://127.0.0.1:18441 --tag flash-next
    python3 qwen-quality-eval.py score --tag flash-next
    python3 qwen-quality-eval.py report flash-next qwen27b
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/root/evals")
DATA = ROOT / "data"
OUT = ROOT / "out"
ROWS_API = "https://datasets-server.huggingface.co/rows"
SUITES = {
    "humaneval": ("openai/openai_humaneval", "openai_humaneval", "test", 164),
    "gsm8k": ("openai/gsm8k", "main", "test", 200),
    # The newest release file: 175 problems, contest dates 2025-01-04 to
    # 2025-04-06. Not served by datasets-server (the repo carries a loading
    # script), so the JSONL is fetched directly.
    "livecodebench": ("livecodebench/code_generation_lite", "test6.jsonl", "test", 175),
}
LCB_URL = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
SANDBOX_IMAGE = "python:3.12-slim"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def load_suite(suite: str) -> list[dict]:
    """Fetch a suite once through the datasets-server rows API, then cache it."""
    path = DATA / f"{suite}.json"
    if path.exists():
        return json.loads(path.read_text())
    dataset, config, split, count = SUITES[suite]
    rows: list[dict] = []
    if suite == "livecodebench":
        raw = OPENER.open(LCB_URL + config, timeout=600).read().decode()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    while len(rows) < count:
        query = urllib.parse.urlencode({
            "dataset": dataset, "config": config, "split": split,
            "offset": len(rows), "length": min(100, count - len(rows)),
        })
        page = json.load(OPENER.open(f"{ROWS_API}?{query}", timeout=60))
        batch = [r["row"] for r in page["rows"]]
        if not batch:
            break
        rows.extend(batch)
    if len(rows) != count:
        raise SystemExit(f"{suite}: fetched {len(rows)} rows, expected {count}")
    DATA.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return rows


def item_id(suite: str, index: int, row: dict) -> str:
    if suite == "livecodebench":
        return f"lcb/{row['question_id']}"
    return row["task_id"] if suite == "humaneval" else f"gsm8k/{index}"


# LiveCodeBench's own prompt wording, so scores stay comparable with its board.
LCB_SYSTEM = (
    "You will be given a question (problem specification) and will generate a correct "
    "Python program that matches the specification and passes all tests.\n\n"
)
LCB_STARTER = (
    "You will use the following starter code to write the solution to the problem and "
    "enclose your code within delimiters.\n```python\n{starter}\n```"
)
LCB_STDIN = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not "
    "directly test on the sample inputs). Enclose your code within delimiters as follows. "
    "Ensure that when the python program runs, it reads the inputs, runs the algorithm "
    "and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```"
)


def build_prompt(suite: str, row: dict) -> str:
    if suite == "livecodebench":
        tail = (LCB_STARTER.format(starter=row["starter_code"]) if row["starter_code"]
                else LCB_STDIN)
        return LCB_SYSTEM + "Question: " + row["question_content"] + "\n\n" + tail
    if suite == "humaneval":
        return (
            "Complete the following Python function. Reply with the complete function, "
            "including its signature and any imports it needs, in a single ```python "
            "code block.\n\n```python\n" + row["prompt"] + "```"
        )
    return (
        row["question"]
        + "\n\nSolve the problem. End your reply with a final line of the form "
        "'Answer: <number>'."
    )


def ask(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "reasoning_effort": "medium",
        "chat_template_kwargs": {"enable_thinking": True},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    reply = json.load(OPENER.open(req, timeout=3600))
    choice = reply["choices"][0]
    return {
        "content": choice["message"].get("content") or "",
        "finish": choice.get("finish_reason"),
        "completion_tokens": reply["usage"]["completion_tokens"],
        "wall_s": round(time.perf_counter() - t0, 2),
    }


def cmd_run(args: argparse.Namespace) -> None:
    model = json.load(OPENER.open(args.base + "/v1/models", timeout=30))["data"][0]["id"]
    OUT.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    for suite in args.suites:
        rows = load_suite(suite)
        path = OUT / f"{args.tag}-{suite}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(line)["id"] for line in path.read_text().splitlines()}
        todo = [(i, r) for i, r in enumerate(rows) if item_id(suite, i, r) not in done]
        print(f"{args.tag} {suite}: {len(done)} done, {len(todo)} to go, model {model}",
              flush=True)

        def one(item: tuple[int, dict], suite: str = suite, path: Path = path) -> None:
            index, row = item
            try:
                result = ask(args.base, model, build_prompt(suite, row), args.max_tokens)
            except Exception as exc:  # recorded, not fatal: a rerun retries it
                print(f"  {item_id(suite, index, row)}: {exc}", file=sys.stderr, flush=True)
                return
            record = {"id": item_id(suite, index, row), "index": index, **result}
            with lock, path.open("a") as f:
                f.write(json.dumps(record) + "\n")

        with ThreadPoolExecutor(max_workers=args.conc) as pool:
            list(pool.map(one, todo))


CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def extract_code(content: str, row: dict) -> str:
    blocks = CODE_BLOCK.findall(content)
    code = blocks[-1] if blocks else content
    # The prompt always goes first. Its stub (a docstring, which is a valid
    # body) is replaced by the reply's own definition, and any imports and
    # helpers the prompt defines survive. Several problems ship a helper the
    # tests call (HumanEval/38 encode_cyclic, /50 encode_shift), and a correct
    # reply that returns only the asked-for function would otherwise fail on a
    # NameError. The first version prepended only when the function was
    # missing, and scored exactly those two correct replies as failures.
    if f"def {row['entry_point']}" not in code:
        return row["prompt"] + code
    return row["prompt"] + "\n    pass\n\n" + code


def _to_float(text: str) -> float | None:
    try:
        return float(text.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def extract_number(content: str) -> float | None:
    """First number on the last ``Answer:`` line, else the last number anywhere."""
    answers = re.findall(r"Answer:\s*(.+)", content)
    if answers:
        found = NUMBER.findall(answers[-1])
        if found:
            return _to_float(found[0])
    found = NUMBER.findall(content)
    return _to_float(found[-1]) if found else None


def score_humaneval(records: list[dict], rows: list[dict]) -> dict[str, bool]:
    by_id = {r["task_id"]: r for r in rows}
    with tempfile.TemporaryDirectory() as tmp:
        for rec in records:
            row = by_id[rec["id"]]
            program = (extract_code(rec["content"], row) + "\n\n" + row["test"]
                       + f"\n\ncheck({row['entry_point']})\n")
            (Path(tmp) / (rec["id"].replace("/", "_") + ".py")).write_text(program)
        runner = (
            "import subprocess,glob,json,os\n"
            "res={}\n"
            "for p in sorted(glob.glob('/w/*.py')):\n"
            "    try:\n"
            "        r=subprocess.run(['python',p],capture_output=True,timeout=15)\n"
            "        res[os.path.basename(p)]=r.returncode==0\n"
            "    except subprocess.TimeoutExpired:\n"
            "        res[os.path.basename(p)]=False\n"
            "print(json.dumps(res))\n"
        )
        out = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", "--memory", "2g",
             "--pids-limit", "256", "-v", f"{tmp}:/w:ro", SANDBOX_IMAGE,
             "python", "-c", runner],
            capture_output=True, text=True, check=True,
        )
        results = json.loads(out.stdout.strip().splitlines()[-1])
    return {rec["id"]: results.get(rec["id"].replace("/", "_") + ".py", False)
            for rec in records}


def score_gsm8k(records: list[dict], rows: list[dict]) -> dict[str, bool]:
    passed = {}
    for rec in records:
        gold = float(rows[rec["index"]]["answer"].split("####")[-1].strip().replace(",", ""))
        got = extract_number(rec["content"])
        passed[rec["id"]] = got is not None and abs(got - gold) < 1e-6
    return passed


# Runs INSIDE the sandbox container. Decoding the hidden tests happens in here
# too: they are base64(zlib(pickle(json))), and unpickling is code execution,
# so it gets the same isolation as the model's programs.
LCB_GRADER = r'''
import base64, json, multiprocessing, pickle, subprocess, sys, zlib

PRELUDE = """import sys, math, collections, itertools, heapq, bisect, functools, string, re
from typing import *
from collections import *
from itertools import *
from functools import *
from heapq import *
from bisect import *
from math import *
sys.setrecursionlimit(1 << 25)
"""
CALL = """
import json as _json, sys as _sys
_args = [_json.loads(_l) for _l in _sys.stdin.read().split("\\n") if _l.strip()]
_r = getattr(Solution(), {name!r})(*_args)
print(_json.dumps(_r))
"""

def tests_of(p):
    tests = json.loads(p["public"])
    priv = p["private"]
    try:
        tests += json.loads(priv)
    except ValueError:
        tests += json.loads(pickle.loads(zlib.decompress(base64.b64decode(priv.encode()))))
    return tests

def same_stdout(got, want):
    g, w = got.split(), want.split()
    if g == w:
        return True
    if len(g) != len(w):
        return False
    for a, b in zip(g, w):
        if a == b:
            continue
        try:
            fa, fb = float(a), float(b)
        except ValueError:
            return False
        if abs(fa - fb) > 1e-6 * max(1.0, abs(fb)):
            return False
    return True

def same_json(got, want):
    try:
        return json.loads(got) == json.loads(want)
    except ValueError:
        return got.strip() == want.strip()

def grade(p):
    path = "/tmp/" + p["id"].replace("/", "_") + ".py"
    functional = bool(p["func_name"])
    src = PRELUDE + p["code"] + (CALL.format(name=p["func_name"]) if functional else "")
    open(path, "w").write(src)
    try:
        tests = tests_of(p)
    except Exception:
        return p["id"], None
    for t in tests:
        try:
            r = subprocess.run(["python", path], input=t["input"], capture_output=True,
                               text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return p["id"], False
        if r.returncode != 0:
            return p["id"], False
        ok = same_json(r.stdout, t["output"]) if functional else same_stdout(r.stdout, t["output"])
        if not ok:
            return p["id"], False
    return p["id"], True

if __name__ == "__main__":
    problems = json.load(open("/w/problems.json"))
    with multiprocessing.Pool(16) as pool:
        print(json.dumps(dict(pool.map(grade, problems, chunksize=1))))
'''


def extract_last_block(content: str) -> str:
    blocks = CODE_BLOCK.findall(content)
    return blocks[-1] if blocks else ""


def score_livecodebench(records: list[dict], rows: list[dict]) -> dict[str, bool]:
    """Pass means every public and hidden test passes; the first failure stops it."""
    by_id = {f"lcb/{r['question_id']}": r for r in rows}
    problems = []
    for rec in records:
        row = by_id[rec["id"]]
        problems.append({
            "id": rec["id"],
            "code": extract_last_block(rec["content"]),
            "func_name": json.loads(row["metadata"]).get("func_name") if row["starter_code"]
            else None,
            "public": row["public_test_cases"],
            "private": row["private_test_cases"],
        })
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "problems.json").write_text(json.dumps(problems))
        (Path(tmp) / "grader.py").write_text(LCB_GRADER)
        out = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", "--memory", "8g",
             "--pids-limit", "1024", "-v", f"{tmp}:/w:ro", SANDBOX_IMAGE,
             "python", "/w/grader.py"],
            capture_output=True, text=True, check=True,
        )
        results = json.loads(out.stdout.strip().splitlines()[-1])
    undecodable = [k for k, v in results.items() if v is None]
    if undecodable:
        raise SystemExit(f"could not decode tests for {len(undecodable)}: {undecodable[:5]}")
    return {k: bool(v) for k, v in results.items()}


SCORERS = {
    "humaneval": score_humaneval,
    "gsm8k": score_gsm8k,
    "livecodebench": score_livecodebench,
}


def cmd_score(args: argparse.Namespace) -> None:
    for suite in args.suites:
        path = OUT / f"{args.tag}-{suite}.jsonl"
        if not path.exists():
            continue
        records = [json.loads(line) for line in path.read_text().splitlines()]
        rows = load_suite(suite)
        passed = SCORERS[suite](records, rows)
        (OUT / f"{args.tag}-{suite}.scores.json").write_text(json.dumps(passed, indent=1))
        # Out of the whole suite, not the replies on disk: a request that errored
        # is never written, and must count as a failure rather than vanish.
        total = SUITES[suite][3]
        missing = total - len(records)
        print(f"{args.tag} {suite}: {sum(passed.values())}/{total} passed"
              + (f" ({missing} with no reply; rerun `run` to retry them)" if missing else ""))


def cmd_report(args: argparse.Namespace) -> None:
    print("| suite | model | passed | truncated | median completion tokens | median wall s |")
    print("|---|---|---|---|---|---|")
    for suite in SUITES:
        for tag in args.tags:
            scores_path = OUT / f"{tag}-{suite}.scores.json"
            if not scores_path.exists():
                continue
            scores = json.loads(scores_path.read_text())
            records = [json.loads(line) for line in
                       (OUT / f"{tag}-{suite}.jsonl").read_text().splitlines()]
            truncated = sum(r["finish"] == "length" for r in records)
            tokens = sorted(r["completion_tokens"] for r in records)
            walls = sorted(r["wall_s"] for r in records)
            n = len(records)
            total = SUITES[suite][3]  # errored requests are unwritten; they count as failures
            print(f"| {suite} | {tag} | {sum(scores.values())}/{total} "
                  f"({100 * sum(scores.values()) / total:.1f}%) | {truncated} | "
                  f"{tokens[n // 2]} | {walls[n // 2]} |")
        if len(args.tags) == 2:
            a, b = (json.loads((OUT / f"{t}-{suite}.scores.json").read_text())
                    if (OUT / f"{t}-{suite}.scores.json").exists() else {}
                    for t in args.tags)
            both = set(a) & set(b)
            only_a = sum(a[i] and not b[i] for i in both)
            only_b = sum(b[i] and not a[i] for i in both)
            print(f"|  | solved only by {args.tags[0]}: {only_a}; "
                  f"only by {args.tags[1]}: {only_b} | | | | |")
    # LiveCodeBench by difficulty: an average over easy and hard problems hides
    # where two models actually differ.
    if not (DATA / "livecodebench.json").exists():
        return
    rows = load_suite("livecodebench")
    level = {f"lcb/{r['question_id']}": r["difficulty"] for r in rows}
    print("\n| livecodebench | " + " | ".join(args.tags) + " |")
    print("|---|" + "---|" * len(args.tags))
    for diff in ("easy", "medium", "hard"):
        ids = [i for i, d in level.items() if d == diff]
        cells = []
        for tag in args.tags:
            path = OUT / f"{tag}-livecodebench.scores.json"
            scores = json.loads(path.read_text()) if path.exists() else {}
            cells.append(f"{sum(scores.get(i, False) for i in ids)}/{len(ids)}")
        print(f"| {diff} | " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--base", required=True)
    run.add_argument("--tag", required=True)
    run.add_argument("--suites", nargs="+", default=list(SUITES), choices=list(SUITES))
    run.add_argument("--conc", type=int, default=8)
    run.add_argument("--max-tokens", type=int, default=16000)
    score = sub.add_parser("score")
    score.add_argument("--tag", required=True)
    score.add_argument("--suites", nargs="+", default=list(SUITES), choices=list(SUITES))
    report = sub.add_parser("report")
    report.add_argument("tags", nargs="+")
    args = parser.parse_args()
    {"run": cmd_run, "score": cmd_score, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
