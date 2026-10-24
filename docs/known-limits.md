# Status and known limits

## Status

Release 0.1.0 (2026-10-24). Measured on one box: a dual Xeon E5-2697A v4 host
with CMP 170HX cards, this recipe on one of them. The configuration ran through
two days of development benchmarks, debug checks and quality runs beside the
two-card recipe, then the release benchmarks from a fresh clone. It has not
served real traffic for days yet.

## Known limits

- **Quality is scored on two checkable suites only.** HumanEval / GSM8K (first
  200 problems) 161/164, 195/200 with the DFlash 2 default and 163/164, 195/200 with MTP, thinking on
  ([results.md](results.md)). Nothing here measures long-form writing,
  tool-call adherence or hard competition code.
- **2- and 3-bit experts.** Half the layers' routed experts are 2-bit (layers
  12–35), the rest 3-bit; that is what fits a 125B MoE on one card. The two-card
  recipe's 4-bit experts score 162/164 and 198/200 on the same suites, which are
  not where such a difference would show first.
- **96 GB of host RAM is the floor.** The n-gram table is 48 GiB of pinned host
  memory (64 GiB as the engine allocates it), allocated at start; the engine
  refuses to start without it. `PLE_SHARED_DIR` shares one copy between servers,
  this recipe's and the two-card recipe's; its file stays in RAM until a reboot
  or until you delete it.
- **One card, so nothing to split, and nothing to grow into.** The checkpoint
  takes 47 GiB of the card's 64; what is left holds the KV pool and the
  recurrent states of up to 48 requests. More requests need the two-card recipe.
- **The served checkpoint is built, not downloaded.** `start.sh` downloads
  klee100's checkpoint without its BF16 n-gram table and halt95's FP8 table
  shards, and joins them with hard links (`docker/build_checkpoint.py`). Keep
  `MODELS_DIR` on one filesystem; `BUILD_COPY=1` copies across filesystems
  instead (~96 GiB more). The FP8 table is the same model's table quantised by
  RadixArk, not klee100's.
- **The first start compiles for about 12 minutes.** That includes
  RecoverSSM's CUDA verify (about 95 s, built with the image's compiler into
  `cache/`). Later starts reuse `cache/`; a change of `GRAPH_MODE`, the drafter,
  `MAX_SEQS` or the KV dtype recompiles part of it. Loading reads the ~97 GiB
  checkpoint once per start: from a slow disk (a USB drive took
  31 minutes) the first start can pass the 30-minute `READY_TIMEOUT`; set
  `READY_TIMEOUT=7200` in `.env` for it. `SPEC_MODE=mtp` reads the checkpoint
  twice, once for the model and once for its MTP layer, so its start takes about
  twice as long from the same disk.
- **Decode is power-bound at 180 W.** The card sits at the cap during decode, at
  1,275 to 1,425 MHz. Locking clocks does nothing; raising the cap is a decision
  about your power supply, not this recipe.
- **Many users get the total, not the speed.** At 48 requests the total is
  1,384 tok/s with DFlash 2 and 1,403 with MTP, about 29 tok/s per user; MTP
  catches up with DFlash 2 from about 16 to 24 users and is behind it below, and one server
  cannot switch between them.
- **Follow-up turns are a little slower than with MTP.** The second turn of an
  18.7k-token conversation starts in 0.42 s, against 0.24 s with
  MTP: four or more drafts need a prefix-match unit of 16 instead of 64.
- **Agent fairness is not tuned.** A long prompt arriving while someone is
  decoding slows their decode for as long as the prompt is read. The two-card
  recipe measured the engine's `--prefill-chunk-with-decodes` (via `EXTRA_ARGS`)
  as a trade of shorter pauses for a much slower read; not measured here.
- **The KV tier is untested on one card.** `KV_OFFLOAD_GB` and `KV_OFFLOAD_DIR`
  work as in the two-card recipe, but no release number here uses them.
  `KV_OFFLOAD_DIR` is write-through and never pruned: a dedicated volume with a
  hard size, or leave it off.
- **The engine runs with patched files.** Morrowmake's image is used as
  released, but `start.sh` and Compose mount source files from
  [`patches/`](../patches/) over the image's: the AutoRound loading fixes, the
  DFlash 2 capture (dflash mode), RecoverSSM, the n-gram table loader and the
  chunked copy-on-write. They are pinned to the engine commit this recipe pins; a
  new engine release needs them re-made against it.
- **RecoverSSM is ours, not the engine's.** It follows the engine's own design
  for GLM's KDA layers and was checked bitwise against the path it replaces
  ([`patches/README.md`](../patches/README.md)), but it is new code on a recurrent
  state the model depends on. `RECOVER_SSM=0` is the way back, with far fewer
  requests.
- **Seven drafts need a chosen block size.** Five to seven DFlash drafts need an
  attention block size the sparse-attention ring (12) and the prefix-match unit
  (16) divide, at least the GDN state page: 912 with BF16 states, 1680 with
  float32, set by `serve.sh`. A size too small for the state page is silently
  doubled by the engine, halving how many requests fit. The drafter allows at
  most 7.
- **The engine image is Morrowmake's and may move.** This recipe pins it by
  digest and runs it as released, with the files above mounted; if they
  retire the digest, `./start.sh update` cannot help until this recipe is
  re-pinned and re-measured.
- **`docker/` is Morrowmake's build tooling** with this recipe's checkpoint
  names and pins in `prepare.sh` and `container.env`, and
  `build_checkpoint.py` added. It still names the EXL3 checkpoint in the image
  build files; this recipe does not rebuild the image.
