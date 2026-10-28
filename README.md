# Qwen3.8-Flash-Next 3 bpw on 1× CMP 170HX

[![engine](https://img.shields.io/badge/engine-vllm--cmp170hx%2018305948ac-blue)](https://github.com/Morrowmake/vllm-cmp170hx/tree/18305948acba3f4c4d912cdd56f2202d2ebd94e8)
![release](https://img.shields.io/badge/release-0.2.0-2ea44f)

**Qwen3.8-Flash-Next (125B MoE) on one CMP 170HX card with the full
262,144-token context: 197.8 / 296.4 / 135.5 tok/s for one user (structured /
code / prose), 371 tok/s editing code, 799.5 / 791.3 / 586.4 tok/s across
eight, and up to 48 requests at once — 1,384 tok/s in total with the default
drafter, 1,403 with the MTP mode. 2- and 3-bit AutoRound experts,
the FP8 n-gram table in host RAM or in a file on disk (one copy can serve several
servers), the whole
decode step in one CUDA graph, inco.ai's DFlash 2 drafter, RecoverSSM for the
linear-attention layers, bf16 KV cache and recurrent states.**

**Two cards?** The sibling recipe
[qwen38-flash-next-w4a16-2x-cmp170hx-recipe](https://github.com/ElectroHobo/qwen38-flash-next-w4a16-2x-cmp170hx-recipe)
serves halt95's 4-bit W4A16 build on two CMP 170HX with the same engine, patches
and n-gram table: faster for one user (350 against 296.4 tok/s on code) and up
to 128 requests. Both can run on one host from one copy of the table.

**New in 0.2.0: the n-gram table on disk, for hosts with less RAM.** Set
`PLE_SHARED_DIR` to a folder on a disk filesystem instead of tmpfs and the 48 GiB
table is a file there: the card reads it through HMM, still inside the one CUDA
graph per step, and the page cache keeps in RAM as much of it as the host can spare.
Without that setting 0.2.0 runs as 0.1.0. From fresh clones, 2026-10-27:

| | table in RAM (default) | table on disk |
|---|---:|---:|
| 1 user, structured / code / prose | 200.0 / 295.0 / 146.2 | 226.7 / 277.7 / 131.2 |
| 8 users, total | 785.5 / 792.1 / 598.7 | 752.5 / 763.2 / 538.1 |
| 48 users, total | 1,384 (0.1.0's release run) | 1,316 |
| Cold prefill | 4,243 (0.1.0's release run) | 3,376 |
| HumanEval / GSM8K (first 200) | 161/164, 195/200 (0.1.0) | 161/164, 196/200 |
| Host memory the server holds | the 48 GiB table pinned (64 GiB as allocated) + 6 GiB | 6 GiB, plus what the page cache keeps of the table |
| Whole container capped at 14 GiB, 8 users | does not fit | 723.7 / 758.9 / 468.1 |

It needs NVIDIA's open kernel modules with HMM (`nvidia-smi -q` shows
`Addressing Mode: HMM`; `./start.sh` checks) and 48 GiB of disk for the file.
How to use it: [The table on disk](docs/how-to-use.md#the-table-on-disk); what it
costs and why: [results](docs/results.md#release-020).

Serve [klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP](https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP)
with the FP8 n-gram table of [halt95/Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin)
on one NVIDIA CMP 170HX with [Morrowmake's vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx)
fork of vLLM: an OpenAI-compatible API with the model's full context, DFlash 2
speculative decoding, vision, reasoning and tool calls. The engine runs only in a
container pinned by digest. One command from a fresh clone, after the
[one-time host setup](#one-time-host-setup):

```bash
git clone https://github.com/ElectroHobo/qwen38-flash-next-3bpw-1x-cmp170hx-recipe
cd qwen38-flash-next-3bpw-1x-cmp170hx-recipe
./start.sh          # pull the engine image, fetch and build the model, start, wait for /health
```

`./start.sh` skips anything already done, so running it twice is safe; an
interrupted download resumes. Both checkpoints are downloaded at the exact
Hugging Face revisions this release was validated with, and only the parts used:
klee100's checkpoint without its 95 GiB BF16 n-gram table, and only the FP8 table
shards of halt95's (~97 GiB in all, instead of ~255).

**Prefer Docker Compose?** The same container, also one command from a fresh
clone: `docker compose up` (or `docker compose up -d`). Use one or the other, not
both.

> **This is a community recipe.** It is made and maintained by its author, not by
> Morrowmake, who has not tested, endorsed or supported it, and not by klee100 or
> halt95, whose checkpoints it combines and serves on an engine they did not build
> them for. It runs on Morrowmake's engine and image and descends from their
> [Qwen3.8-Flash-Next EXL3 recipe](https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe)
> by way of our two-card recipe; what is ours is the checkpoint combination, the
> configuration, the engine patches in [`patches/`](patches/) mounted over the
> image's (AutoRound loading fixes, DFlash 2 support, RecoverSSM, the shared n-gram
> table, a copy-on-write fix), the benchmarks and the measurements. **Contact for
> issues: @ElectroHobo, in
> [this repository's issues](https://github.com/ElectroHobo/qwen38-flash-next-3bpw-1x-cmp170hx-recipe/issues)** —
> not Morrowmake's. Made to the shape of Morrowmake's
> [community recipes guidelines](https://github.com/Morrowmake/vllm-cmp170hx/blob/ampere/COMMUNITY_RECIPES.md).

## What it is

- **The weights** are klee100's mixed-precision AutoRound build: the routed experts
  at 3 bits in groups of 128 on layers 0–11 and 36–47 and 2 bits in groups of 64 on
  12–35 (2.998 effective bits per weight), the attention and linear-attention
  projections at 8 bits, the rest BF16; 47 GiB on the card, with an MTP head (4-
  and 8-bit). That is what lets a 125B MoE fit one 64 GiB card.
- **The n-gram (PLE) table** is not klee100's 95 GiB BF16 one but halt95's FP8 one
  (RadixArk's quantisation, one global scale), the same model's table in 48 GiB of
  host RAM. `start.sh` builds the served checkpoint from the two downloads with hard
  links (`docker/build_checkpoint.py`): no extra disk.
- **The step** is the two-card recipe's: the table in host RAM, so the engine
  captures the whole decode step as one CUDA graph, which is what makes it fast on
  an old CPU (measured on a 2016 Broadwell host; the two-card recipe's README says
  why).
- **The drafter** is [inco.ai's DFlash 2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2):
  7 drafts for one or two requests, 5 for three or four, 3 beyond. Against the
  checkpoint's own MTP head it is 53 % faster on code for one user;
  `SPEC_MODE=mtp` runs MTP, which catches up from about 16 to 24 users.
- **Many users**: RecoverSSM (one recurrent state per request instead of one per
  draft) and BF16 recurrent states let 48 requests fit on one card; with the
  engine's own per-draft states and float32 states 12 fitted at three drafts.

## Few users or many: which drafter

Measured from a fresh clone, total tok/s, 700-token replies at temperature 0
(`bench/concurrency_sweep.py`):

| Users | 1 | 4 | 8 | 16 | 24 | 32 | 48 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DFlash 2 (default) | 206 | 412 | 618 | 869 | 1,039 | 1,157 | 1,384 |
| MTP (`SPEC_MODE=mtp`) | 166 | 390 | 584 | 839 | 1,048 | 1,166 | 1,403 |

DFlash 2 leads up to 16 users; from 24 the two are level on this sweep, MTP 1 %
ahead. The sweep's three fixed tasks at temperature 0 suit DFlash 2; on a mixed
load in development (different prompts and lengths per request) MTP was 3 to 15 %
ahead from 16 users (48 users: 1,128 against 980). One server cannot switch
drafters by load. To have both, run a second server with the other drafter: it
needs its own card and some host RAM for its engine, but with the shared n-gram
table not a second copy of the 48 GiB table
([docs/how-to-use.md](docs/how-to-use.md), "Several servers, one table").

## One-time host setup

- **1× NVIDIA CMP 170HX exposing 64 GiB**, measured on a PCIe x16 link. The
  n-gram rows are read from host memory over PCIe every step, so a narrower link
  (a card without the capacitor modification trains x4) costs decode; untested.
- **96 GB of host RAM** with the n-gram table in RAM (it takes 48 GiB, 64 GiB as
  the engine allocates it; the engine and the host need the rest). **Much less
  with the table on disk** (`PLE_SHARED_DIR` on a disk filesystem, 0.2.0): the
  server holds about 6 GiB and the page cache keeps the rows in use; capped at
  14 GiB in all it ran within 3–4 % of uncapped on structured and code and 13 % on
  prose, on our prompts ([results](docs/results.md#how-much-ram-the-table-on-disk-needs)).
  Measured on 204 GiB (0.1.0) and 352 GB (0.2.0).
- Linux with NVIDIA driver **580 or newer** (measured on 610.57.04); for the
  table on disk, the open kernel modules with HMM addressing.
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  usable by your user: `docker run --rm --gpus all nvidia/cuda:13.3.1-base-ubuntu24.04 nvidia-smi`
  lists the card.
- For `./start.sh`: `git`, `curl`, `flock`, `setsid`, Python 3 for the boot check,
  and `jq` for the smoke test.
- About 100 GiB of free disk for the downloads (~97 GiB), **on one filesystem**
  (the served checkpoint is hard links to them), plus the engine image and the
  kernel compile caches. The disk only matters while loading.

## Hardware

| | |
|---|---|
| GPU | 1× NVIDIA CMP 170HX (sm_80), 65,536 MiB, power limit 180 W |
| Link | PCIe Gen2 x16 |
| Host | 2× Xeon E5-2697A v4 (Broadwell, 16 cores each), 204 GiB RAM |
| Driver | 610.57.04 (580 or newer) |
| Disk | ~100 GiB free on one filesystem; read once per start |
| Software | Docker with the NVIDIA Container Toolkit |

## Engine

| | |
|---|---|
| Engine | [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx/tree/18305948acba3f4c4d912cdd56f2202d2ebd94e8) at `18305948acba3f4c4d912cdd56f2202d2ebd94e8` |
| Image | `ghcr.io/morrowmake/vllm-cmp170hx@sha256:5b45c4d6feafefe2eba962375b0a9a0b64fd22b9e05abb4d1966f14a32a2db79`, Morrowmake's release image for their Qwen EXL3 recipe, used as released apart from the mounted files. The first start compiles kernels and RecoverSSM's verify (about 12 minutes); later starts reuse that, and most of their time is reading the checkpoint (about 4 minutes from an SSD, 31 minutes from a USB disk) |
| Checkpoint | `klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`, revision `ce0e0b94083895bd836b916f29bf105c40a8162a`, without `model-00001-of-00011.safetensors` (its BF16 n-gram table); the FP8 table shards (`model-plefp8-*`) of `halt95/Qwen3.8-Flash-Next-W4A16-Merlin`, revision `cec53e9f90e5995b81691604ec0b32191debf797` |
| Precision | routed experts 3-bit g128 / 2-bit g64 by layer (Intel AutoRound, SignRoundV2), QSA and GDN projections 8-bit, routers / shared experts / embeddings / lm_head / vision BF16, MTP head 4- and 8-bit, n-gram table FP8 in host RAM, **KV cache BF16**, indexer cache BF16, **GDN recurrent states BF16** |
| Drafter | [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2), revision `a22a1744e85c17364dcc54a6a420af7135d4c7f6` (Apache-2.0, 1.1 GiB): 7 drafts for one or two requests, 5 for three or four, 3 beyond. Alternative: the checkpoint's MTP head (`SPEC_MODE=mtp`), 3 drafts |
| Engine patches | files in [`patches/`](patches/), mounted read-only over the image's: AutoRound expert bits and the MTP layer for this engine's loader, the AutoRound n-gram table, the DFlash 2 capture hook and speculation-method check (dflash mode), RecoverSSM for the GDN layers, the shared n-gram table loader and chunked copy-on-write |
| Requests | up to 48 at once (`MAX_SEQS`) |
| CUDA graphs | `FULL_AND_PIECEWISE`: the decode step captured whole |

### Settings, exactly

What the container runs with the defaults (`DRY=1 ./start.sh` prints it):
`vllm serve /models/Qwen3.8-Flash-Next-3bpw-fp8ple --tensor-parallel-size 1
--pipeline-parallel-size 1 --dtype bfloat16 --kv-cache-dtype bfloat16
--load-format auto --mm-encoder-tp-mode data
--attention-config '{"indexer_kv_dtype":"bf16"}' --engram-config '{"cpu_offload":true}'
--max-model-len 262144 --gpu-memory-utilization 0.95 --max-num-seqs 48
--max-num-batched-tokens 3460 --long-prefill-token-threshold 0
--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
--prefix-match-unit 16 --block-size 912 --mamba-ssm-cache-dtype bfloat16
--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
--speculative-config '{"method":"dflash","model":"/drafter","num_speculative_tokens":7,"num_speculative_tokens_per_batch_size":[[1,2,7],[3,4,5],[5,48,3]]}'`,
with `KV_OFFLOAD_GB` adding `--kv-transfer-config` for the `OffloadingConnector`
and `"disable_eagle_block_drop":true` to the speculative config. The drafter is
mounted at `/drafter`, the files from `patches/` over their engine paths
([`patches/README.md`](patches/README.md)). Environment: `VLLM_GDN_RECOVER=1`
(RecoverSSM), `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, `VLLM_QWEN4_QSA_TOPK_CANONICAL=1`,
`VLLM_THIN_GEMM=1`, `VLLM_THIN_GEMM_GLUON=1`, `VLLM_FLA_AUTOTUNE_SKIP_SPILLING=1`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`, peer access off
(`NCCL_P2P_DISABLE=1`), `CUDA_DEVICE_ORDER=PCI_BUS_ID`; with `PLE_SHARED_DIR` also
`VLLM_QWEN_PLE_SHARED_DIR`, `VLLM_QWEN_PLE_SHARED_NAME` and
`VLLM_QWEN_PLE_SHARED_PIN` (`1` on tmpfs, `0` on disk).

## Headline numbers

Measured 2026-10-23 from a fresh clone of this repository (empty `cache/`, empty
model directory, bare `./start.sh`) on 1× CMP 170HX, 74 SMs, recipe defaults (no
KV tier), 262,144-token context, 180 W and a PCIe x16 link, under the protocol
Morrowmake's recipes use (MiaAI-Lab's: 400 tokens, temperature 0, thinking off;
one user is the median of five streaming runs, two to eight users the median
aggregate of three runs). Raw files:
[`bench/results/release-0.1.0/`](bench/results/release-0.1.0/).

| | DFlash 2 (default) | MTP (`SPEC_MODE=mtp`) |
|---|---:|---:|
| **1 user** · structured / code / prose | **197.8 / 296.4 / 135.5** | 176.7 / 193.4 / 139.0 |
| **2 users, total** | 307.5 / 403.9 / 218.2 | 269.0 / 298.8 / 222.9 |
| **4 users, total** | 515.6 / 587.1 / 370.9 | 492.1 / 476.0 / 364.2 |
| **8 users, total** | **799.5 / 791.3 / 586.4** | 764.0 / 724.3 / 542.6 |
| **48 users, total** (`concurrency_sweep.py`, 700 tokens) | 1,384 | 1,403 |
| **Prompt reading** (cold prefill) | 4,243 | 4,230 |
| **KV pool** (tokens) | 292,784 | 304,893 |
| Code editing, one user (`bench/agent_probes.py edit`) | **371** | 215 |
| Follow-up turn, time to first token | 0.42 | 0.24 |
| Quality: HumanEval / GSM8K (first 200), thinking on | 161/164, 195/200 | 163/164, 195/200 (from the SSD copy, [results](docs/results.md)) |

Morrowmake publishes no one-card Qwen3.8-Flash-Next recipe to compare with. On
the same host the two-card recipe does 241 / 350 / 198 for one user and 1,937 at
128 users.

## Using it

`./start.sh` runs preflight (read-only GPU, disk and port checks), installs the
engine, downloads and builds the checkpoint, launches the server detached (output
in `logs/serve.log`) and waits for `/health`, then runs a two-request startup
check. Commands:

| Command | What it does |
|---|---|
| `./start.sh` | preflight, install, download, launch (default) |
| `./start.sh install` | the engine image only |
| `./start.sh download` | the checkpoint parts and the drafter, then the built checkpoint |
| `./start.sh stop` / `restart` | stop or restart the server this checkout started |
| `./start.sh status` | server, `/health`, KV cache line |
| `./start.sh logs` | follow `logs/serve.log` |
| `./start.sh update` | `git pull`, pull the image if the pin moved, restart |
| `./start.sh smoke` | one chat request and one tool call |
| `DRY=1 ./start.sh` | print what would happen, change nothing |

The API listens on `127.0.0.1:8000` as model `qwen3.8-flash-next`, with no API
key. Reasoning comes back in the message's `reasoning` field and the answer in
`content`; size `max_tokens` for both; `chat_template_kwargs: {"enable_thinking": false}`
turns reasoning off. Examples: [`examples/`](examples/). Docker Compose:
[`docker-compose.yml`](docker-compose.yml). Step by step, serving other machines,
several servers on one table: [docs/how-to-use.md](docs/how-to-use.md).

## Options

Settings go in `.env` (copied from [`.env.example`](.env.example) on the first
run); a prefix assignment beats `.env` for one run
(`CUDA_VISIBLE_DEVICES=2 ./start.sh restart`). Every option:

| Setting | Default | Meaning |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0` | the card (exactly one) |
| `HOST` | `127.0.0.1` | listen address; `0.0.0.0` for every interface (set `API_KEY` first) |
| `PORT` | `8000` | API port |
| `API_KEY` | unset | require `Authorization: Bearer <key>` |
| `SERVED_MODEL_NAME` | `qwen3.8-flash-next` | model id clients send |
| `MODELS_DIR` | `./models` | where the downloads and the built checkpoint go, one filesystem |
| `KV_OFFLOAD_GB` | `0` | GiB of host RAM for KV blocks evicted from the card; `0` is off |
| `KV_OFFLOAD_DIR` | unset | a disk tier behind the RAM tier (needs `KV_OFFLOAD_GB`); write-through, never pruned |
| `GRAPH_MODE` | `FULL_AND_PIECEWISE` | `PIECEWISE` is the EXL3 recipe's mode and is ~3× slower per step on an old CPU |
| `SPEC_MODE` | `dflash` | `dflash` (inco.ai's DFlash 2 drafter; the faster for few users), `mtp` (the checkpoint's MTP head; level or ahead from about 16 to 24) or `none` (set `BOOT_CHECK=0`) |
| `SPEC_N` | `7` (dflash) / `3` (mtp) | draft tokens per step; at most 7 for dflash |
| `SPEC_SCHEDULE` | `[[1,2,7],[3,4,5],[5,48,3]]` (dflash) / empty (mtp) | draft count by number of requests; `SPEC_SCHEDULE=` for `SPEC_N` drafts at every count |
| `MAX_SEQS` | `48` | concurrent sequences; 48 fill the KV pool |
| `RECOVER_SSM` | `1` | RecoverSSM for the GDN layers; `0` = the engine's per-draft states (far fewer requests fit) |
| `SSM_DTYPE` | `bfloat16` | GDN recurrent state precision; `float32` is the checkpoint's |
| `BLOCK_SIZE` | `912` (`1680` with `float32` states) | attention block size in dflash mode: a multiple of 48 at least the GDN state page |
| `PLE_SHARED_DIR` | unset | where the n-gram table lives, shared by every server on this host given the same folder (this recipe's and the two-card recipe's): on tmpfs (e.g. `/dev/shm/qwen38-ple`) one 48 GiB copy in RAM; on a disk filesystem (e.g. `/srv/qwen38-ple`) a 48 GiB file read through HMM, for hosts with less RAM ([how-to](docs/how-to-use.md#the-table-on-disk)) |
| `PLE_SHARED_NAME` | `qwen38-flash-next-ple-fp8` | the table's file name there |
| `DRAFTER_REPO` / `DRAFTER_REVISION` | inco.ai's, pinned | the DFlash 2 drafter |
| `MAX_BATCHED` | `3460` | tokens per scheduler step |
| `REASONING_PARSER` / `TOOL_PARSER` | `qwen3` / `qwen3_coder` | output parsers |
| `EXTRA_ARGS` | empty | extra vLLM arguments; options that change the deployment are refused |
| `BOOT_CHECK` | `1` | two-request startup check after `/health` |
| `READY_TIMEOUT` / `STOP_TIMEOUT` | `1800` / `120` | seconds |

The context (262,144) and GPU memory fraction (0.95) are fixed: `serve.sh` refuses
other values. Checkpoint pins and paths are in
[`.env.advanced.example`](.env.advanced.example).

## Update

- `./start.sh update` pulls the new release and engine image and restarts; your
  `.env` is preserved.
- With Compose: `git pull`, then `docker compose up -d`.

## Licence

The recipe is MIT ([`LICENSE`](LICENSE)); the scripts descend from Morrowmake's
MIT-licensed Qwen3.8-Flash-Next recipe. The engine and the patches are
Apache-2.0. Both checkpoints are under the Qwen Community License 1.0 and carry
components with their own terms: [docs/model-licences.md](docs/model-licences.md).
The benchmark prompts come from MiaAI-Lab's repository, used as data with
attribution.

## Credits

- **Morrowmake** — the [vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx)
  engine and image this runs on, and the
  [Qwen3.8-Flash-Next recipe](https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe)
  (MIT) whose scripts this recipe descends from. Not involved in this recipe.
- **klee100** — the [3 bpw AutoRound checkpoint](https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP)
  with its MTP head, made for a 64 GiB card. Not involved in this recipe.
- **halt95** — the [W4A16 checkpoint](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin)
  whose FP8 n-gram table this recipe uses; **RadixArk** — that
  [FP8 table](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4). Not involved.
- **Intel** — AutoRound, the quantisation method of both checkpoints.
- **Qwen** — [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
  (Qwen Community License 1.0).
- **vLLM** — the [project](https://github.com/vllm-project/vllm) (Apache-2.0)
  Morrowmake's engine forks.
- **MiaAI-Lab** — the benchmark protocol and category prompts (used here as data,
  the harness is our own).
- **inco.ai** — the [DFlash 2 drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
  (Apache-2.0).

If this recipe helps you, credit the people above and link their work. Citation
metadata for this repository: [`CITATION.cff`](CITATION.cff).
