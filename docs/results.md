# Results

Every number here was measured on the same box: one CMP 170HX (74 SMs, 64 GiB,
180 W, PCIe Gen2 x16) in a dual Xeon E5-2697A v4 (Broadwell) host with 204 GiB
of RAM (352 GB at 1600 MT/s for 0.2.0), driver 610.57.04, Morrowmake's engine at
`18305948acba3f4c4d912cdd56f2202d2ebd94e8`. The two-card recipe served on two
other cards of the same host for most of it. Raw files are in
[`bench/results/release-0.1.0/`](../bench/results/release-0.1.0/), the
development runs in its `development/` folder; the harnesses are in
[`bench/`](../bench/). Dates are 2026-10-19 to 2026-10-23.

## Release 0.2.0

0.2.0 adds one thing, the n-gram table on disk; the defaults, engine image and
checkpoints are 0.1.0's. Measured 2026-10-27 on the same box with its memory changed:
352 GB since that day (0.1.0 ran on 204 GiB), all 24 slots filled, so the memory runs
at 1600 MT/s instead of 2133. Raw files:
[`bench/results/release-0.2.0/`](../bench/results/release-0.2.0/) (the run's log is
`fresh-clone-0.2.0-run.txt`).

### From fresh clones

A fresh clone of this release beside one of the two-card recipe's 0.3.0, this one on
card 4 with its checkpoint on the same NVMe volume as the table; first both with
their own pinned tables (the default), then both with
`PLE_SHARED_DIR=/srv/models/qwen38-ple`, a folder on an XFS volume striped over three
NVMe drives, which the first start filled. `cmp_recipe_bench.py` as before;
`concurrency_sweep.py` at 48 users only. Every start's startup check passed.

| | table in RAM (default) | table on disk, DFlash 2 | table on disk, MTP |
|---|---:|---:|---:|
| 1 user, structured / code / prose | 200.0 / 295.0 / 146.2 | 226.7 / 277.7 / 131.2 | 170.6 / 194.6 / 140.3 |
| 2 users, total | 326.6 / 405.4 / 217.9 | 340.3 / 361.3 / 208.6 | 282.3 / 303.4 / 217.2 |
| 4 users, total | 516.4 / 614.7 / 367.8 | 501.5 / 599.4 / 348.2 | 453.3 / 457.3 / 327.4 |
| 8 users, total | 785.5 / 792.1 / 598.7 | 752.5 / 763.2 / 538.1 | 710.0 / 696.0 / 513.1 |
| 48 users, total (`concurrency_sweep.py`) | 1,384 (0.1.0) | 1,315.7 | 1,327.5 |
| Cold prefill | 4,243 (0.1.0) | 3,376 | — |
| KV pool | 292,784 tokens | 298,742 tokens | 310,539 tokens |
| Start to healthy | 1,039 s (with the two-card server pinning its table at the same time) | 348 s (the file written in it) | 389 s |

With the whole table in the page cache the disk costs 4 % at eight users on
structured and code, 10 % on prose, 5 % at 48 users, and 20 % of prompt reading: a
long new prompt touches many table pages the card has not mapped yet, and each first
touch is a fault. That is more than on the two-card recipe (1–4 % at eight users):
there each card holds half the rows. One-user cells move by more than that between
runs (structured here has ranged 197.8–226.7), so they say little either way.

### How it works

The engine looks the table up with a Triton kernel that copies rows from pinned host
memory through a CUDA view of it, inside the step's CUDA graph. A file on disk cannot
be pinned (`cudaHostRegister` fails: the kernel refuses long-term pins of its pages),
but on this driver the card can read ordinary host memory directly: the cmpunlocker
driver is NVIDIA's open kernel modules, which report `Addressing Mode: HMM`. Torch
and Triton both refuse a tensor over memory CUDA does not know, so the disk path
passes the table's address to a copy of the kernel as an integer. A page in the page
cache is read over PCIe as pinned memory is; a page missing from it faults, and the
kernel reads it from the file. The probe numbers (one lookup: 42–47 µs pinned, about
227 µs from the mapped file once warm) and the check that both kernels return the same
bytes are in the two-card recipe's
[results](https://github.com/ElectroHobo/qwen38-flash-next-w4a16-2x-cmp170hx-recipe/blob/main/docs/results.md#how-it-works-and-what-one-lookup-costs).

### How much RAM the table on disk needs

The server alone with the table on disk, its container capped with
`docker update --memory` (the cap holds the server's own memory and every page of the
file it caches), stepping down; the page cache had been dropped before the start, so
the server cached only what it touched. The server's own memory was 6 GiB; of the
table, 6–9 GiB stayed cached through these runs.

| Container capped at | 1 user | 8 users, total |
|---|---:|---:|
| none (48 GiB of the table cached) | 226.7 / 277.7 / 131.2 | 752.5 / 763.2 / 538.1 |
| 54 GiB, just after the cap | 178.7 / 230.7 / 100.6 | 720.1 / 702.1 / 474.6 |
| 38 GiB | 212.9 / 257.4 / 123.4 | 729.8 / 771.2 / 468.4 |
| 30 GiB | 195.2 / 256.2 / 134.9 | 743.4 / 738.3 / 463.6 |
| 22 GiB | 218.1 / 266.9 / 138.4 | 767.0 / 767.3 / 466.6 |
| 14 GiB | 197.7 / 259.9 / 122.1 | 723.7 / 758.9 / 468.1 |

Our three benchmark prompts use little of the table; varied traffic uses more, so
read these as how small the floor can be, not as a promise. Prose, with the most
varied word combinations, loses most (13 % at eight users). The first capped row is
lower than the next ones because it was measured while the kernel was still evicting
what the start had cached. Under memory pressure from elsewhere on the host, see the
two-card recipe's results; this recipe on the same test lost 4–20 % at eight users with
the table whole (prose most), and 11–32 % with 60 % of it cached.

### Same output, and quality

Five fixed prompts (the bench's three, a word problem and a summary), 400 tokens,
temperature 0, thinking off, sent twice to the same server: with the table pinned in
RAM 1 of 5 replies came back byte-identical, with the table on disk 0 of 5 (the two-card recipe: 0 and 1).
The replies part after tens to hundreds of characters on a different, equally
plausible word. Why, we have not traced; it happens with the table in RAM too, so it
is not the disk path, and it means RAM against disk cannot be compared byte for byte
(0 of 5 there, as expected). What is checked exactly is the lookup: the same bytes
from both kernels ([above](#how-it-works)), and every start compares the
table file with the checkpoint byte for byte.

So quality is scored: HumanEval and GSM8K (first 200) with the table on disk,
DFlash 2, thinking on, **161/164, 196/200**, against 0.1.0's 161/164, 195/200 with the table in RAM. Raw:
`quality-1x-0.2.0-fresh-disk-*`, `same-output-1x-0.2.0-*.json`, `repeat-1x-0.2.0-*.json`.

## Release 0.1.0

### From a fresh clone

Recipe defaults, started 2026-10-23 from a fresh clone with an empty `cache/`, an
empty model directory and a bare `./start.sh` (download and build, then healthy
2,868 s after the container started, 1,871 s of it reading the checkpoint from a USB disk; startup check passed: 666 of 707 drafts accepted), no KV tier; then the same
checkout with `SPEC_MODE=mtp` (healthy after 2,720 s, reading the checkpoint twice, 569 of 594 drafts accepted).
`cmp_recipe_bench.py` (MiaAI-Lab's protocol: 400 tokens, temperature 0,
thinking off; one user the median of five streaming runs, more users the median
aggregate of three runs, prefill the median of three real-text prompts repeated
three times); `concurrency_sweep.py` 700-token replies at temperature 0, the
structured, code and prose prompts rotating across requests, median of two
rounds after a warm-up. Raw: `cmp-recipe-bench-1x-0.1.0-fresh-*.json`,
`concurrency-1x-0.1.0-fresh-*.json`, `agent-*-1x-0.1.0-fresh-*.json`,
`conc-steps-1x-0.1.0-fresh-*.txt`, `quality-1x-0.1.0-fresh-*`, and the run's
log `fresh-clone-0.1.0-run.txt`.

| | DFlash 2 (default) | MTP (`SPEC_MODE=mtp`) |
|---|---:|---:|
| 1 user, structured / code / prose | **197.8 / 296.4 / 135.5** | 176.7 / 193.4 / 139.0 |
| 2 users, total | 307.5 / 403.9 / 218.2 | 269.0 / 298.8 / 222.9 |
| 4 users, total | 515.6 / 587.1 / 370.9 | 492.1 / 476.0 / 364.2 |
| 8 users, total | 799.5 / 791.3 / 586.4 | 764.0 / 724.3 / 542.6 |
| Code editing, one user | 371 | 215 |
| Follow-up turn, time to first token | 0.42 s | 0.24 s |
| Cold prefill | 4,243 | 4,230 |
| KV pool | 292,784 tokens | 304,893 tokens |
| Quality: HumanEval / GSM8K (first 200) | 161/164, 195/200 | 163/164, 195/200 ¹ |

¹ The fresh clone's MTP quality run was stopped before it finished; this one ran
2026-10-24 on the development copy of this build with the checkpoint on an SSD: the
same engine arguments (apart from the served name), the same patched code (the
files differ only in their comment headers), the same startup check (569 of 594
drafts accepted). Its KV pool was 307,313 tokens; it ran on the shared table. Raw:
`quality-1x-0.1.0-ssd-mtp-*`, log `ssd-check-0.1.0-run.txt`.

`concurrency_sweep.py`, total tok/s (ms per engine step, tokens per request-step):

| Users | DFlash 2 | MTP |
|---:|---:|---:|
| 1 | **206.4** (20.1, 4.18) | 166.2 (18.4, 3.08) |
| 4 | **412.3** (28.9, 3.62) | 390.4 (28.6, 3.10) |
| 8 | **618.3** (34.0, 3.12) | 583.8 (36.9, 3.09) |
| 16 | **868.7** (47.2, 3.07) | 839.4 (49.2, 3.05) |
| 24 | 1,038.7 (59.7, 3.06) | **1,048.0** (60.4, 3.08) |
| 32 | 1,156.8 (71.1, 3.07) | **1,166.0** (71.3, 3.09) |
| 48 | 1,384.2 (89.1, 3.07) | **1,402.9** (88.9, 3.07) |

DFlash 2 leads up to 16 users; from 24 the two are level on this sweep, MTP 1 %
ahead. The sweep's three fixed tasks at temperature 0 suit DFlash 2; on a mixed
load in development (different prompts and lengths per request) MTP was 3 to 15 %
ahead from 16 users (48 users: 1,128 against 980).

### Loading the 3 bpw checkpoint on this engine

klee100's checkpoint was made for another engine. On Morrowmake's it needed three
fixes in the INC (AutoRound) loading code, all in [`patches/`](../patches/):

- **The n-gram table** was refused (`Qwen4Exp PLE embedding does not support
  quantization config INCConfig`): the checkpoint's `block_name_to_quantize`
  covers the table's layer. AutoRound never quantizes an embedding, so the table
  loads unquantized under INC, as under Quark.
- **The experts' bits.** The experts are 2-bit g64 by default with layers 0–11
  and 36–47 overridden to 3-bit g128 in `extra_config`. The INC reader applied
  overrides only to layers whose class name contains `fusedmoe`, and this
  engine's expert class is `RoutedExperts`, so every layer got 2 bits and the
  load failed (`The size of tensor a (40) must match the size of tensor b (60)`).
  Checked by resolving all 48 layers from the real config before and after.
- **The MTP head.** Its AutoRound config names the layer `mtp.layers.0` where the
  engine calls it `mtp.layers.48`, and its 4-bit experts are listed only in
  `mtp_pack_plan.json`. The parser maps the name and reads the plan.

The checkpoint's own n-gram table is BF16, 95 GiB of pinned host memory. Beside
another server that ran the host out of memory, so this recipe serves halt95's
FP8 table of the same model instead (the same 128 shards and tensor names, one
global scale). Quality with it on this engine, 2 drafts, float32 states:
HumanEval 162/164, GSM8K 196/200, against 161/164 and 197/200 for the same
checkpoint with its BF16 table on the engine it was made for.

### Why requests did not fit, and the fix

On one card the KV pool is only ~9.5 GiB (63.4 × 0.95 less 47.3 of weights,
graphs and activations), and every page in it has one size. The 36 Gated DeltaNet
layers keep a recurrent state per request, ~3 MiB a layer in float32, and to
verify drafts the engine kept one per draft position. It also sized the attention
block up to that state (1,680 tokens), so a request's cost did not depend on its
length: one short request took 15.8 % of the pool at 7 drafts, 8.0 % at 3 and
6.3 % at 2. At 7 drafts 6 requests ran at once.

Development runs with a mixed-load script (different prompts and lengths per
request, 700 tokens; not in this repository), total tok/s:

| Users | 7 / 5 / 3 drafts, 16 requests | 3 drafts | 2 drafts | BF16 states, 864-token blocks, 3 drafts |
|---:|---:|---:|---:|---:|
| 1 | 144.6 | 143.7 | 146.4 | 148.7 |
| 4 | 298.5 | 297.5 | 302.1 | 306.9 |
| 8 | 385.4 | 447.3 | 474.4 | 462.8 |
| 16 | 397.5 | 472.4 (12 fit) | 549.6 (15 fit) | **611.9** (16 fit) |

BF16 states alone changed nothing: the state page was still padded to the
1,680-token attention page. With the block brought down to 864 one request took
4.1 %. Quality: HumanEval 160/164, GSM8K 195/200.

RecoverSSM ([`patches/README.md`](../patches/README.md)) keeps one state per
request, records per token what the update needs during the verify, and replays
the accepted tokens after sampling, so drafts no longer cost state pages. First
with float32 states and 1,680-token blocks, 3 drafts:

| Users | without | RecoverSSM, 32 requests | RecoverSSM, BF16 states, 48 requests |
|---:|---:|---:|---:|
| 1 | 143.7 | 145.8 | 149.1 |
| 8 | 447.3 | 468.1 | 461.2 |
| 16 | 472.4 (12 fit) | 592.8 | 603.1 |
| 32 | | 766.3 (30 fit) | 806.6 |
| 48 | | | **973.7** (48 fit) |

One request took 3.0 % of the pool instead of 8.0 % with float32 states. At 64
requests only 53 fitted (other buffers grow with `MAX_SEQS`), so 48 is the
ceiling. Quality: HumanEval 161/164 and GSM8K 196/200 without prefix caching,
163/164 and 196/200 with it (float32), 163/164 and 193/200 with BF16 states (all
7 GSM8K misses also missed in other runs of this harness, which scores 193–196 on
this model). Needles at 117K and 139K tokens: all four right.

How RecoverSSM was checked is in the two-card recipe's
[results](https://github.com/ElectroHobo/qwen38-flash-next-w4a16-2x-cmp170hx-recipe/blob/main/docs/results.md)
(the kernel bitwise against the original op, every verify and commit against the
replaced path in the engine); the checks ran on this build first. One bug got past
them and was found by quality (HumanEval 155, then 148): eager steps updated only
part of the conv window. The quality numbers above are after the fix.

The faster replay (the verify's own `ftz` instructions, 8-row tiles) took 48
users from 973.7 to 1,025.7 tok/s.

### Drafts for one user

With RecoverSSM drafts no longer cost memory, so one user can have the drafter's
7 (with 912-token blocks: seven drafts lengthen the conv state, and 864 was
silently doubled to 1,728). `cmp_recipe_bench.py`, one user, two runs each,
structured / code / prose:

| Drafts | Run 1 | Run 2 |
|---|---:|---:|
| 3 | 177.9 / 216.9 / 156.2 | 192.6 / 218.0 / 149.4 |
| 5 / 4 / 3 | 192.9 / 265.7 / 162.9 | 187.6 / 258.3 / 150.5 |
| **7 / 5 / 3** (default) | **223.6 / 276.9 / 141.3** | **229.2 / 273.8 / 142.3** |

+20 / +26 / −7 % against 3 drafts. From five requests up the schedule runs 3
drafts and measures as before. Quality: HumanEval 161/164, GSM8K 196/200. Two
drafts from 33 users, which helps the two-card recipe, did not help here (48
users 970.0 against 979.5).

### DFlash 2 or MTP

Development runs, the mixed-load script, both with RecoverSSM and BF16 states:

| Users | 1 | 16 | 32 | 48 |
|---|---:|---:|---:|---:|
| DFlash 2, 7 / 5 / 3 | 151–163 | 600.5 | 804.8 | 979.5 |
| MTP, 3 drafts | 137.2 | **653.4** | **905.0** | **1,128.3** |

One user, `cmp_recipe_bench.py`: MTP 170.3 / 195.7 / 136.9 and 175.4 / 193.1 /
141.5 against DFlash 2's 223.6 / 276.9 / 141.3 and 229.2 / 273.8 / 142.3. So
DFlash 2 is the better drafter for one to a few users (+30–40 % on code) and MTP
from about 16 users (+3–15 %: more tokens accepted per step under load and
slightly shorter steps). Fewer MTP drafts under load, which helps the two-card
recipe, did not help here (3 / 2 / 1: 48 users 1,078.0 against 1,128.3). MTP
quality: HumanEval 163/164, GSM8K 197/200.

### One n-gram table for several servers

With `PLE_SHARED_DIR` the table is one tmpfs file registered with CUDA instead of
64 GiB of pinned memory per server. This recipe on it: host shared memory 64 →
48 GiB, step times as with its own table (16 users 55.15 against 55.77 ms per
step, 48 users 100.29 against 100.37). (This said "greedy outputs" too until 0.2.0;
repeated requests to one server do not match byte for byte on this engine, so that
comparison showed nothing: [Same output, and quality](#same-output-and-quality).) Then the two-card
recipe on the same file: both servers together in 48 GiB of shared memory.
Measured again 2026-10-24 with the release code: this recipe's development copy
(MTP mode, the checkpoint on an SSD; see the quality footnote above) and the
two-card recipe's fresh clone on one table file of 47.68 GiB. Host shared memory
was 48 GiB with this server and 49 GiB with both, against 64 GiB per server
without `PLE_SHARED_DIR`. One user on each with both up: here 175.8 / 193.9 /
142.8 tok/s (MTP; 176.7 / 193.4 / 139.0 alone), the two-card recipe 241.0 / 347.0 /
196.0 (241.2 / 349.7 / 198.4 alone). Raw: `cmp-recipe-bench-shared-c-*.json`,
`ssd-check-0.1.0-run.txt`.

### Copy-on-write in chunks

The first MTP load test died at 32 users: a burst of requests sharing a prompt
asked the engine's copy-on-write for 4.04 GiB in one gather with 3.86 GiB free
(`torch.OutOfMemoryError` in `copy_kv_cache_blocks_inplace`; MTP mode lets the
engine pick larger blocks). In chunks of 256 MiB the same load ran clean.

### Things that did not help

- **Records sized to the draft count.** RecoverSSM's records hold 8 token rows per
  request and 3 drafts use 4. Sizing them to the drafts grew the pool by 0.2 GiB
  (about 1.5 requests) and measured no faster.
- **Clock locks.** The card sits at the 180 W cap during decode (1,275–1,425 MHz)
  and a warmer card clocks lower, which is most of the run-to-run spread for one
  user. Locking the clock makes runs repeatable, not faster.
- **An 8-bit drafter** could save at most ~2 % at one user: the drafter's weights
  are ~1.2 GB read per step, under 1 ms of a 19 ms step.

## Notes on the method

- Development numbers come from a mixed-load script, not `concurrency_sweep.py`:
  its prompts and lengths differ per request, so DFlash 2 accepts less than on the
  sweep's three fixed tasks. Compare columns within a table, not across tables.
- One-user speed on this card moves several percent between runs with the clock
  (above); single numbers are medians of five runs, comparisons were repeated.
- Quality runs use `bench/quality_eval.py` at the model's recommended sampling
  (temperature 0.6, thinking on), so scores move by a few problems between runs of
  an unchanged server: 193–196 on GSM8K for this checkpoint.
