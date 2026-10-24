#!/usr/bin/env python3
"""Build the served checkpoint: klee100's 3 bpw AutoRound weights with halt95's FP8 n-gram table.

    build_checkpoint.py SOURCE TABLE OUT

SOURCE is klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP downloaded without its BF16 n-gram
table shard (model-00001-of-00011.safetensors, ~95 GiB, which holds nothing else); TABLE is
halt95/Qwen3.8-Flash-Next-W4A16-Merlin, of which only model.safetensors.index.json and the FP8
table shards (model-plefp8-*) are needed (the full checkpoint works too). Same model, same tensor
names, same 128 table shards: the FP8 table with its one global scale replaces the BF16 one,
because the BF16 table does not fit beside one card's server (95 GiB of pinned host memory).

OUT gets hard links to every SOURCE file but the table shard and to the FP8 table shards (no
extra disk, so SOURCE, TABLE and OUT must be on one filesystem; with BUILD_COPY=1 files that
cannot be linked are copied instead, ~96 GiB), a merged model.safetensors.index.json, and
config.json with ple_embedding_dtype float8_e4m3fn. Runs again safely: existing links are kept,
the index and config are rewritten.
"""

import errno
import json
import os
import shutil
import sys

TABLE_SHARD = "model-00001-of-00011.safetensors"


def fail(msg: str) -> None:
    print(f"build_checkpoint: {msg}", file=sys.stderr)
    sys.exit(1)


def link(src: str, dst: str) -> str:
    if os.path.exists(dst):
        if os.path.samefile(src, dst) or os.path.getsize(src) == os.path.getsize(dst):
            return "kept"
        os.remove(dst)
    try:
        os.link(src, dst)
        return "linked"
    except OSError as e:
        if e.errno != errno.EXDEV or os.environ.get("BUILD_COPY") != "1":
            fail(f"cannot hard-link {src} to {dst} ({e.strerror}): keep the downloads and the "
                 "built checkpoint on one filesystem (one MODELS_DIR), or set BUILD_COPY=1 to copy")
        shutil.copy2(src, dst)
        return "copied"


def main() -> None:
    if len(sys.argv) != 4:
        fail("usage: build_checkpoint.py SOURCE TABLE OUT")
    src, table, out = sys.argv[1:]
    a = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    b = json.load(open(os.path.join(table, "model.safetensors.index.json")))
    wa, wb = a["weight_map"], b["weight_map"]
    table_a = {k for k in wa if ".ngram_embedding." in k}
    if {wa[k] for k in table_a} != {TABLE_SHARD}:
        fail(f"{src}: the n-gram table is not alone in {TABLE_SHARD}; wrong revision?")
    if any(v == TABLE_SHARD and k not in table_a for k, v in wa.items()):
        fail(f"{src}: {TABLE_SHARD} holds more than the n-gram table; wrong revision?")
    table_b = {k: v for k, v in wb.items() if ".ngram_embedding." in k}
    if {k for k in table_b if k.endswith(".weight")} != table_a:
        fail(f"{table}: its FP8 table shards do not match the 3 bpw checkpoint's names")
    needed_src = sorted({v for k, v in wa.items() if k not in table_a})
    needed_table = sorted(set(table_b.values()))
    missing = [f for f in needed_src if not os.path.exists(os.path.join(src, f))]
    missing += [f for f in needed_table if not os.path.exists(os.path.join(table, f))]
    if missing:
        fail(f"incomplete download, missing {len(missing)} files, e.g. {missing[:3]}")

    os.makedirs(out, exist_ok=True)
    counts: dict[str, int] = {}
    for f in sorted(os.listdir(src)):
        path = os.path.join(src, f)
        if f in (TABLE_SHARD, "model.safetensors.index.json", "config.json") or f.startswith(".") \
                or not os.path.isfile(path):
            continue
        r = link(path, os.path.join(out, f))
        counts[r] = counts.get(r, 0) + 1
    for f in needed_table:
        r = link(os.path.join(table, f), os.path.join(out, f))
        counts[r] = counts.get(r, 0) + 1
    wmap = {k: v for k, v in wa.items() if k not in table_a}
    wmap.update(table_b)
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": a.get("metadata", {}), "weight_map": wmap}, fh, indent=1)
    cfg = json.load(open(os.path.join(src, "config.json")))
    target = cfg["text_config"] if "text_config" in cfg else cfg
    target["ple_embedding_dtype"] = "float8_e4m3fn"
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"build_checkpoint: {out}: {len(wmap)} tensors in {len(set(wmap.values()))} files, "
          f"{len(table_b)} table tensors from FP8; "
          + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    if counts.get("copied"):
        print("build_checkpoint: files were copied, not linked: put MODELS_DIR's folders on one "
              "filesystem to save the disk")


if __name__ == "__main__":
    main()
