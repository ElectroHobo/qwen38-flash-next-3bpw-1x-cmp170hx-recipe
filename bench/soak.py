#!/usr/bin/env python3
"""Soak: N concurrent streams of mixed requests for M minutes, plus a periodic
image request. Counts completions, failures and tokens; prints a summary.

    python3 bench/soak.py 8000 qwen3.8-flash-next 20 8
"""
import base64
import json
import random
import struct
import sys
import threading
import time
import urllib.request
import zlib

port, model, minutes, streams = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
base = f"http://127.0.0.1:{port}"
PROMPTS = [
    "Emit a markdown table of 40 rows with columns id, name, role, team, location. Fill every cell with plausible values. Table only.",
    "Write a complete Python module implementing an LRU cache with a doubly linked list and dict: get, put, delete, iteration, resize, clear, invariant validation, full docstrings and type hints. Code only.",
    "Write a precise technical essay explaining how memory bandwidth, arithmetic intensity and batching interact in modern LLM inference.",
    "What is 17 * 23 + 91 / 7? Show your working briefly, then give the number.",
    "Summarise the plot of a heist film you invent, in five sentences.",
    "Write a SQL query that returns the top 5 customers by total order value in 2025, with a short explanation.",
]
w, h = 448, 448


def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)


rows = b"".join(b"\x00" + bytes([255, 0, 0] * (w // 2) + [0, 0, 255] * (w // 2)) for _ in range(h))
png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
IMG = "data:image/png;base64," + base64.b64encode(png).decode()

lock = threading.Lock()
stats = {"ok": 0, "fail": 0, "tokens": 0, "img_ok": 0, "img_bad": 0, "img_fail": 0, "errors": []}
deadline = time.time() + minutes * 60


def post(body):
    r = urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                                      headers={"Content-Type": "application/json"}), timeout=600)
    return json.load(r)


def worker(i):
    rnd = random.Random(i)
    while time.time() < deadline:
        p = rnd.choice(PROMPTS)
        think = rnd.random() < 0.5
        try:
            r = post({"model": model, "messages": [{"role": "user", "content": f"[{rnd.random():.6f}] {p}"}],
                      "max_tokens": rnd.choice([200, 400, 800]), "temperature": 0.7,
                      "chat_template_kwargs": {"enable_thinking": think}})
            with lock:
                stats["ok"] += 1
                stats["tokens"] += r["usage"]["completion_tokens"]
        except Exception as e:  # noqa: BLE001
            with lock:
                stats["fail"] += 1
                if len(stats["errors"]) < 5:
                    stats["errors"].append(repr(e)[:160])


def imager():
    while time.time() < deadline:
        try:
            r = post({"model": model, "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": IMG}},
                {"type": "text", "text": "What colours are on the left half and the right half? One sentence."}]},
            ], "max_tokens": 200, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
            txt = (r["choices"][0]["message"]["content"] or "").lower()
            with lock:
                if "red" in txt and "blue" in txt:
                    stats["img_ok"] += 1
                else:
                    stats["img_bad"] += 1
        except Exception as e:  # noqa: BLE001
            with lock:
                stats["img_fail"] += 1
                if len(stats["errors"]) < 5:
                    stats["errors"].append("img " + repr(e)[:150])
        time.sleep(30)


t0 = time.time()
threads = [threading.Thread(target=worker, args=(i,)) for i in range(streams)] + [threading.Thread(target=imager)]
for t in threads:
    t.start()
last = 0
while any(t.is_alive() for t in threads):
    time.sleep(60)
    with lock:
        if stats["ok"] != last:
            print(f"  {(time.time()-t0)/60:4.1f} min: {stats['ok']} ok, {stats['fail']} failed, "
                  f"{stats['tokens']} tokens ({stats['tokens']/(time.time()-t0):.0f} tok/s), "
                  f"images {stats['img_ok']} right / {stats['img_bad']} wrong / {stats['img_fail']} failed", flush=True)
            last = stats["ok"]
el = time.time() - t0
print(f"soak done: {el/60:.1f} min, {streams} streams: {stats['ok']} requests ok, {stats['fail']} failed, "
      f"{stats['tokens']} completion tokens ({stats['tokens']/el:.0f} tok/s sustained); "
      f"images {stats['img_ok']} right / {stats['img_bad']} wrong / {stats['img_fail']} failed")
for e in stats["errors"]:
    print("  error:", e)
