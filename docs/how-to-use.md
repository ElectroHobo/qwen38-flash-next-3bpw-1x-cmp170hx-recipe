# How to use this repo

## What you need

- One CMP 170HX card exposing 64 GiB, measured on an x16 link, in a Linux host
  with **96 GB of RAM or more** (the n-gram table takes 64 GiB of it as the engine
  allocates it, 48 GiB shared with `PLE_SHARED_DIR`) and NVIDIA driver 580 or
  newer. Two cards? See the two-card recipe,
  [qwen38-flash-next-w4a16-2x-cmp170hx-recipe](https://github.com/ElectroHobo/qwen38-flash-next-w4a16-2x-cmp170hx-recipe). Add `KV_OFFLOAD_GB` to the RAM figure if you turn the KV
  tier on.
- Docker with the NVIDIA Container Toolkit, usable by your user.
- About 100 GiB of free disk for the downloads (~97 GiB) and the engine's compile
  caches, on **one filesystem** (the served checkpoint is hard links to the
  downloads). Any disk: it is read once per start, not during decoding; a slow
  one (a USB drive) makes the first start long, see `READY_TIMEOUT` below.
- `git`, `curl`, `flock`, `setsid`, Python 3, `jq`.

## Step by step

1. `git clone https://github.com/ElectroHobo/qwen38-flash-next-3bpw-1x-cmp170hx-recipe && cd qwen38-flash-next-3bpw-1x-cmp170hx-recipe`
2. `cp .env.example .env` and edit it if the defaults are wrong for you: which
   card (`CUDA_VISIBLE_DEVICES`), where the checkpoint goes (`MODELS_DIR`),
   whether other machines may connect (`HOST=0.0.0.0` and an `API_KEY`). The
   first run copies the example for you if you skip this.
3. `./start.sh`. It checks the host (read-only), pulls the engine image by
   digest, downloads klee100's checkpoint without its BF16 n-gram table, the FP8
   table shards of halt95's and the DFlash 2 drafter at their pinned revisions
   (resumable), builds the served checkpoint from them with hard links, starts
   the server detached and waits for `/health`, then sends two requests to
   confirm it drafts. The first start compiles kernels and takes about
   12 minutes on top of reading the checkpoint (about 4 minutes from an
   SSD, 31 minutes from a USB disk); later starts mostly read it. Watch with
   `./start.sh logs`. `start.sh` waits `READY_TIMEOUT` seconds (1800) for
   `/health`; from a slow disk set `READY_TIMEOUT=7200` for the first start.
4. `./start.sh status` shows the server, `/health` and the KV pool line
   (`GPU KV cache size: 292,784 tokens` with the defaults).

`DRY=1 ./start.sh` prints every check, the environment the container would get
and the launch command, and changes nothing.

## Talk to it

An OpenAI-compatible API on `127.0.0.1:8000`, model `qwen3.8-flash-next`, no
key by default:

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Reasoning comes back in the message's `reasoning` field and the answer in
`content`; `max_tokens` covers both, so size it for the thinking. To turn
reasoning off for a request:
`"chat_template_kwargs": {"enable_thinking": false}`. Tool calls use the Qwen3
coder parser; see [`examples/tool_call.py`](../examples/tool_call.py). The
model also takes images.

## Few users or many: which drafter

The default drafter, DFlash 2, is the faster one for few users, and by far on
code (296.4 against 193.4 tok/s for one user). From about 16 to 24 users the
checkpoint's own MTP head with three drafts catches up: level on our sweep of
fixed tasks (1,403 against 1,384 tok/s at 48), 3 to 15 % ahead on a mixed load
in development:

```
SPEC_MODE=mtp ./start.sh restart     # or SPEC_MODE=mtp in .env
```

Both use RecoverSSM and fit 48 requests. Tables: [results.md](results.md).
Fewer requests at once (`MAX_SEQS`) keep each one faster.

## Several servers, one table

The n-gram table is 48 GiB of host memory that every server pins for itself
(64 GiB as the engine allocates it). Servers on one host can share one copy:

```
PLE_SHARED_DIR=/dev/shm/qwen38-ple   # in .env of every server that shares
```

The directory must be on tmpfs (`/dev/shm` is) and the same path for every
server; `start.sh` mounts it into the container. The first server to start
fills the table file; the others map it and only compare it with their
checkpoint. This recipe and the two-card
recipe read the same table, so a one-card and a two-card server on one host
share it too (48 GiB together instead of 128). A server whose table differs
refuses to start rather than touch the shared rows; give it another
`PLE_SHARED_NAME`. The file stays in RAM until the host reboots or you delete
it: `rm /dev/shm/qwen38-ple/*` once no server uses it. With Docker Compose,
uncomment the volume in `docker-compose.yml` and set `PLE_SHARED_DIR` in
`docker/container.env`.

## Keep long contexts around (KV offload)

Agents that share a long context — a repository, a document — pay for it once
per conversation while it sits on the cards, and again every time other
traffic pushes it out. `KV_OFFLOAD_GB` keeps evicted KV blocks in host RAM and
copies them back:

```
KV_OFFLOAD_GB=64      # in .env; then ./start.sh restart
```

Measured on the two-card recipe (the same engine and model; not re-measured on
one card): with 64 GiB a 100k-token context that other requests had pushed off
the cards came back in 0.68 s instead of being re-read in 22 s, and the engine
checked the content was the same. The model's KV is about 30 KB per token, so 64 GiB
holds roughly 2.2 million tokens of history. The tier is a shared mapping in
the container's `/dev/shm`; `start.sh` sizes that to `KV_OFFLOAD_GB + 32` GiB.
Budget the RAM: 64 GiB for the table (or its share), the tier, and 20 GB or so for the engine.

`KV_OFFLOAD_DIR=/path/on/a/fast/disk` adds a disk tier behind the RAM tier
(needs `KV_OFFLOAD_GB` > 0). A reload from an NVMe took 1.85 s. **The disk tier
is write-through and never pruned**: every KV block ever evicted is written
there, and one benchmark run wrote 58 GB. Give it a volume of its own with a
hard size, or do not use it.

## Serving other machines

`HOST=0.0.0.0` makes the server listen on every interface; set `API_KEY` first
if anyone else can reach the port. Then point clients (Open WebUI, LiteLLM, an
agent framework) at `http://<host>:8000/v1` with the model id
`qwen3.8-flash-next`.

## Day to day

- `./start.sh stop` stops the server this checkout started, gracefully (a hard
  kill on a CUDA context has produced GPU errors on these cards). `restart`
  stops and starts.
- `./start.sh update` pulls the new release, pulls the image if the pin moved,
  restarts; `.env` is preserved.
- `logs/serve.log` is the server's output; `logs/container.id` the container.
- The checkpoint and the compile caches survive restarts in `models/` and
  `cache/` (or where `MODELS_DIR` and `CACHE_DIR` point).

## Reproduce the numbers

The harnesses in [`bench/`](../bench/) take a port and a model id:

```bash
python3 bench/cmp_recipe_bench.py --port 8000 --model qwen3.8-flash-next --tag mine
python3 bench/concurrency_sweep.py --port 8000 --model qwen3.8-flash-next --tag mine --users 1,4,8,16,24,32,48
python3 bench/vllm_conc_steps.py 8000 qwen3.8-flash-next
python3 bench/kv_offload_bench.py mine --port 8000 --model qwen3.8-flash-next
python3 bench/kv_dtype_check.py mine --port 8000 --model qwen3.8-flash-next
```

Run them on an idle server; `cmp_recipe_bench.py` takes about five minutes,
`concurrency_sweep.py` (1 to 48 users) about ten, `kv_offload_bench.py` about
eight (it floods the pool on purpose).
[`docs/results.md`](results.md) says what each measured and how.

## Advanced

- `GRAPH_MODE=PIECEWISE` restores Morrowmake's recipe's graph mode. It is about
  3× slower per step on an old CPU; on a current CPU the gap is smaller. There
  is no reason to set it other than to see the difference.
- The drafter: `SPEC_MODE=dflash` (default) runs inco.ai's DFlash 2 drafter;
  `SPEC_MODE=mtp` runs the checkpoint's own MTP head (about 35 % slower on code
  for few users, level or ahead from about 16 to 24, the same quality); `SPEC_MODE=none`
  drafts nothing (set `BOOT_CHECK=0`).
- `SPEC_N` and `SPEC_SCHEDULE` set the draft count. With DFlash 2 the default is
  7, scheduled 7 / 5 / 3 by the number of requests (`[[1,2,7],[3,4,5],[5,48,3]]`);
  with MTP 3 at every count (fewer MTP drafts under load did not help on one
  card). `SPEC_SCHEDULE=` for a fixed count. The sweeps are in `results.md`.
- `RECOVER_SSM=0` turns RecoverSSM off (the engine's per-draft states: 12 to 16
  requests fit at three drafts, 6 at seven); `SSM_DTYPE=float32` keeps the GDN states at
  the checkpoint's precision (the block size then defaults to 1680). Neither is
  needed for quality; both are for comparison.
- `MAX_BATCHED` (3460) is the prefill slice per scheduler step; on the two-card
  recipe 8192 read prompts 6 % faster for a 6 % smaller pool, a smaller slice keeps a decoding
  user more responsive while a long prompt is read, for a slower read. The
  engine also has `--prefill-chunk-with-decodes` (via
  `EXTRA_ARGS`), not yet measured on this build.
- FP8 KV cache: edit `QWEN_DEFAULT_KV_CACHE_DTYPE` in `serve.sh` to `fp8`. On the
  two-card recipe: a 73 % larger pool, 8 to 10 % slower decode, answers shift. Not
  measured on one card, not the default.
- Path and preflight settings: [`.env.advanced.example`](../.env.advanced.example).
