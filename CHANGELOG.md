# Changelog

## 0.2.0 — 2026-10-28

The n-gram table on disk, for hosts with less RAM. Same engine image, checkpoints
and defaults as 0.1.0; without the new use of `PLE_SHARED_DIR` it runs as 0.1.0.
The same change as the two-card recipe's 0.3.0.

- **`PLE_SHARED_DIR` on a disk filesystem** keeps the 48 GiB table in a file
  there instead of pinned RAM (`patches/qwen4_exp_common_ngram_embedding.py`,
  `serve.sh`): the mapping is not registered with CUDA (a disk file cannot be
  pinned), and the lookup is a copy of the pinned kernel that takes the table's
  address as an integer, which the card reads through HMM inside the same CUDA
  graph. On tmpfs nothing changes. `serve.sh` picks by the folder's filesystem and
  sets `VLLM_QWEN_PLE_SHARED_PIN`; it and `./start.sh`'s preflight stop if the card
  does not report `Addressing Mode: HMM`, and the preflight checks the 48 GiB.
- From fresh clones, table on disk against RAM: one user 226.7 / 277.7 / 131.2
  against 200.0 / 295.0 / 146.2 tok/s, eight users 752.5 / 763.2 / 538.1 against
  785.5 / 792.1 / 598.7, 1,316 tok/s at 48 users (1,328 with MTP), cold prefill
  3,376 tok/s (4,243 in 0.1.0). With the whole container capped at 14 GiB: eight
  users 723.7 / 758.9 / 468.1. HumanEval / GSM8K
  with the table on disk 161/164, 196/200 (0.1.0 in RAM: 161/164, 195/200).
- `.env.advanced.example`: `PLE_PLACEMENT` is the EXL3 reader's switch and does
  not apply to this build; its comment said otherwise.

## 0.1.0 — 2026-10-24

First release. The one-card sibling of
[qwen38-flash-next-w4a16-2x-cmp170hx-recipe](https://github.com/ElectroHobo/qwen38-flash-next-w4a16-2x-cmp170hx-recipe)
0.2.0, from which its scripts, patches and benchmarks come; that recipe descends
from Morrowmake's Qwen3.8-Flash-Next EXL3 recipe. Runs Morrowmake's engine image
by digest with the files in `patches/` mounted over it.

- Checkpoint: `klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP` at `ce0e0b94`
  (2- and 3-bit AutoRound experts, 47 GiB on the card) without its 95 GiB BF16
  n-gram table, served with the FP8 table shards of
  `halt95/Qwen3.8-Flash-Next-W4A16-Merlin` at `cec53e9f`. `start.sh` downloads
  only those parts (~97 GiB) and builds the served checkpoint with hard links
  (`docker/build_checkpoint.py`; `BUILD_COPY=1` copies across filesystems).
- AutoRound loading fixes for this engine (`inc_config_parser.py`, the n-gram
  table loader): per-layer expert bits for `RoutedExperts`, the MTP layer's name
  and pack plan, the unquantized table under INC.
- One card: tensor parallel 1, no peer access, up to 48 requests.
- DFlash 2 by default (inco.ai's drafter, `a22a1744`), 7 drafts for one or two
  requests, 5 for three or four, 3 beyond, 912-token blocks: 197.8 / 296.4 /
  135.5 tok/s for one user, 799.5 / 791.3 / 586.4 across eight,
  1,384 at 48. `SPEC_MODE=mtp` runs the checkpoint's MTP head, 3 drafts:
  176.7 / 193.4 / 139.0 for one user, 1,403 at 48.
- RecoverSSM and BF16 GDN states, from the two-card recipe: 48 requests fit
  where 12 did at 3 drafts.
- The shared n-gram table (`PLE_SHARED_DIR`): one 48 GiB copy for this recipe's
  and the two-card recipe's servers on one host.
- Chunked copy-on-write, which fixes a CUDA out-of-memory under prefix-cache
  bursts in MTP mode.
- Quality, thinking on: HumanEval / GSM8K (first 200) 161/164, 195/200 with DFlash 2,
  163/164, 195/200 with MTP (run on the development copy of this build with
  the checkpoint on an SSD; docs/results.md says why that is the same build).
- Raw files of the release runs and the development runs in
  `bench/results/release-0.1.0/`.
