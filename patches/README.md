# Engine patches

Files from Morrowmake's [vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx)
at `18305948acba3f4c4d912cdd56f2202d2ebd94e8`, the commit the recipe's image is
built from, changed for this recipe, and one new package. `start.sh` and
`docker-compose.yml` mount them read-only over the image's copies; the image itself
is used as released. Each file keeps its licence header and says what changed.

| File here | Mounted over (`/opt/vllm-src/vllm/…`) | When | Change |
|---|---|---|---|
| `qwen4_exp_nvidia_model.py` | `models/qwen4_exp/nvidia/model.py` | `SPEC_MODE=dflash` | the model becomes a drafter target and captures the drafter's layers |
| `models_config.py` | `model_executor/models/config.py` | `SPEC_MODE=dflash` | `dflash` accepted as a speculation method for this model |
| `gdn_recover/` (new) | `sm80/kernels/gdn/` | always | RecoverSSM: the record verify (CUDA, built at first start) and the commit |
| `mamba_gdn_qwen_gdn_linear_attn.py` | `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` | always | RecoverSSM in the GDN layer (inert with `RECOVER_SSM=0`) |
| `qwen4_exp_nvidia_model_state.py` | `models/qwen4_exp/nvidia/model_state.py` | always | RecoverSSM: the commit after sampling |
| `qwen4_exp_nvidia_ple_layer.py` | `models/qwen4_exp/nvidia/ple_layer.py` | always | RecoverSSM for the PLE short-conv layers |
| `qwen4_exp_common_ngram_embedding.py` | `models/qwen4_exp/common/ngram_embedding.py` | always | the AutoRound (INC) checkpoint's n-gram table loads; the table as one shared host copy (`PLE_SHARED_DIR`) |
| `inc_config_parser.py` | `model_executor/layers/quantization/inc/config_parser.py` | always | AutoRound's per-layer expert bits reach this engine's expert layer; the MTP layer's bits resolve |
| `v1_worker_utils.py` | `v1/worker/utils.py` | always | copy-on-write block copies in chunks |

The unified diffs against Morrowmake's files: [`qwen4exp-dflash2-capture.diff`](qwen4exp-dflash2-capture.diff),
[`recoverssm.diff`](recoverssm.diff), [`ple-table.diff`](ple-table.diff),
[`chunked-copy.diff`](chunked-copy.diff), [`inc.diff`](inc.diff).

## DFlash 2

So that inco.ai's
[DFlash 2 drafter for Qwen3.8-Flash-Next](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
can attach to the model (48 lines added, 1 changed).

### What the change does

The engine already has everything DFlash 2 needs on the drafter side: the
speculator, the draft model and the CUDA graphs for it, which Morrowmake's GLM
recipe uses. What the Qwen3.8-Flash-Next model lacked is the other half:

- **A drafter target.** `Qwen4ExpModel` gains vLLM's `EagleModelMixin`, and the
  two model classes declare `SupportsEagle3`, the interface through which the
  engine tells a model which layers' hidden states a drafter reads.
- **The capture.** For each of those layers the layer loop hands out the
  hidden state entering it: the previous layer's pending hyper-connection
  combine completed (with that layer's `mlp_hyper_connection`, the same way the
  model's own deepstack and pipeline paths materialise it, out of place), then
  averaged over the model's four hyper-connection streams. inco.ai's capture
  patch for SGLang, shipped with the drafter, describes the training input as
  the completed, zero-based outputs of layers 5, 15, 25, 35 and 45, averaged
  over the four streams. vLLM adds one to DFlash layer ids, so the capture is
  of the state entering layers 6, 16, 26, 36 and 46 — the same tensors. The
  server logs them at start: `Using Eagle3 auxiliary layers from config: (6,
  16, 26, 36, 46)`.
- **The guard.** The model's config check refused any speculation method but
  `mtp`, `ngram` and `ngram_gpu`, which was right while nothing could feed a
  drafter; `dflash` is added.

The main computation is untouched: the capture reads the stream and averages a
copy, and only runs for the layers a drafter asked for. With DFlash 2 drafting
the quality suites score as they do with MTP (HumanEval 163/164 and GSM8K
197/200 against 161/164 and 199/200; [docs/results.md](../docs/results.md),
"DFlash 2"). `SPEC_MODE=mtp` does not mount these two files.

## The 3 bpw AutoRound checkpoint

klee100's checkpoint is Intel AutoRound (INC) with per-layer bits: experts of layers
0–11 and 36–47 at 3 bits in groups of 128, the rest at 2 bits in groups of 64, and an
MTP head packed by its own plan. Two things in the engine kept it from loading:

- **Expert bits.** AutoRound's per-layer overrides are applied only to layers whose
  class name contains `fusedmoe`; this engine's expert layer is `RoutedExperts`, so
  every expert layer got the default 2 bits and the load failed (`The size of tensor
  a (40) must match the size of tensor b (60)`). `inc_config_parser.py` applies them
  to `RoutedExperts` too.
- **The MTP layer.** The config names it `mtp.layers.0.*`, the engine
  `mtp.layers.48`, and its 4-bit experts are listed only in the checkpoint's
  `mtp_pack_plan.json`. The parser maps the engine's index to the config's when no
  entry names the engine's, and reads the plan (4 bits, groups of 128, as the packed
  shapes say). `SPEC_MODE=mtp` needs it; DFlash 2 does not use the MTP head.
- **The n-gram table** (`qwen4_exp_common_ngram_embedding.py`): the engine refused
  an AutoRound checkpoint's table because `block_name_to_quantize` covers its layer,
  though AutoRound never quantises it. The served checkpoint carries halt95's FP8
  table in place of klee100's BF16 one (`docker/build_checkpoint.py`).

## RecoverSSM for the GDN layers

Qwen3.8-Flash-Next's linear-attention (Gated DeltaNet) layers keep a recurrent state
per request. To verify drafts the engine keeps one state per draft position per
request, so the state at the accepted position can be picked up next step: with 7
drafts each request reserves eight state pages in every GDN layer. On one card that
reservation, not the attention KV, decides how many requests fit: 16 with 3 drafts
and fp32 states, where with RecoverSSM and BF16 states 48 fit.

RecoverSSM (the design Morrowmake's engine already uses for GLM's KDA layers,
`vllm/sm80/kernels/kda/recover.py`) keeps one state per request:

- **Verify** (`gdn_recover/gdn_recover_verify.cu`): the engine's fused CUDA
  spec-decode kernel, followed line for line, starting from the request's single
  state, writing no state and recording per token the correction, the normalised
  key and the decay (fp32). Built at first start with `torch.utils.cpp_extension`
  (about 95 s, cached in `cache/`), with `--use_fast_math` as the engine builds the
  original, so it rounds the same way.
- **Commit** (`gdn_recover/recover.py`), after sampling: replays the accepted tokens
  from the records into the state (a Triton kernel using the same flush-to-zero
  multiply and FMA instructions the verify compiles to), and the engine's conv
  window compaction; with prefix caching it also writes the block-boundary states
  and the PLE layers' conv windows.
- **Pages**: the GDN and PLE layers ask for no draft-position pages.

Checked: on random inputs at TP1 and TP2 head counts, fp32 and BF16 states, 1 to 7
drafts and values near the denormal limit, the verify's outputs are bitwise the
original op's, it writes no state, and the replay reproduces the original's stored
state bit for bit at every accepted count. In the server, with a debug mode that
runs the replaced path beside it: verifies and commits match on eager steps, every
spec row reads its own request's count (16,000 rows with prefix caching, 0 wrong),
and states crossing a prefix-cache block boundary match. Quality with it is level
([docs/results.md](../docs/results.md)). `RECOVER_SSM=0` turns it off; the mounted
files then behave as the originals.

## One host copy of the n-gram table

The 48 GiB FP8 n-gram table lives in pinned host memory, allocated per server;
torch's pinned allocator rounds it to 64 GiB, and nothing shares it between servers.
With `PLE_SHARED_DIR` (a host tmpfs directory, mounted at the same path) the table
is a shared mapping of one file there, in global row order, registered with CUDA
instead of allocated: each card maps its own row range, so a one-card server and a
two-card one of the same table (this recipe and
[qwen38-flash-next-w4a16-2x-cmp170hx-recipe](https://github.com/ElectroHobo/qwen38-flash-next-w4a16-2x-cmp170hx-recipe))
share it. Loading fills empty rows and compares filled ones byte for byte under a
file lock; a server whose table differs stops at load instead of changing rows
another server reads. Measured: two servers (one card and two cards) in 48 GiB of
host shared memory together, against 64 GiB for one before; step times and outputs
as with a private table.

## Copy-on-write in chunks

For partial prefix-cache hits the engine copies cache blocks before the forward pass
with one gather of every copied block. A burst of 32 requests sharing a prompt asked
for 4 GiB at once with 3.9 GiB free on this build and stopped the engine
(`torch.OutOfMemoryError` in `copy_kv_cache_blocks_inplace`). The copy now runs in
chunks of at most 256 MiB.

## Licence

Every file is Apache-2.0, as in the engine; their licence headers are kept and each
carries a note saying what changed and where. If Morrowmake's engine gains the same
support, these mounts become unnecessary and the recipe will drop them.
