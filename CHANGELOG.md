# Changelog

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
  163/164, 195/200 with MTP.
- Raw files of the release runs and the development runs in
  `bench/results/release-0.1.0/`.
