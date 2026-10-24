#!/bin/bash
# Launch Qwen3.8-Flash-Next (klee100 3 bpw AutoRound with halt95's FP8 n-gram table)
# on two CMP 170HX cards with whole-expert EP, on Morrowmake's vllm-cmp170hx engine.
#
# Usage: serve.sh [dflash|mtp|none|--help]
#   (no argument)  same as "dflash" -- the default.
#   dflash         inco.ai's DFlash 2 drafter (DFLASH_MODEL, mounted at /drafter by
#                  start.sh), SPEC_N drafts (default 7) by the SPEC_SCHEDULE below, on the
#                  two patched engine files in patches/ that start.sh mounts.
#   mtp            the checkpoint's native MTP head, SPEC_N (3) drafts at every batch
#                  size: faster than dflash from about 16 users up, slower for one
#                  to a few.
#   none           no speculative decoding.
# Set DRY=1 to print the environment and the command that would run instead
# of running them.
#
# Paths (all relative to this repo unless overridden):
#   VENV            venv from ./start.sh install    (default ./venv)
#   MODEL           checkpoint directory (default ./models/Qwen3.8-Flash-Next-3bpw-fp8ple,
#                   built by ./start.sh download)
#   CUDA_HOME       CUDA toolkit root               (default /usr/local/cuda-13.3)
# API:
#   HOST            address to listen on (default 127.0.0.1: this machine only).
#                   0.0.0.0 listens on every interface -- set API_KEY first.
#   PORT            default below
#   API_KEY         if set, every /v1 request must send "Authorization: Bearer
#                   <key>" (passed to vLLM as VLLM_API_KEY, never on the command line).
#   SERVED_MODEL_NAME  model id clients send (default below)
# Card: CUDA_VISIBLE_DEVICES selects exactly one physical card (nvidia-smi
#   numbering; default below).
# Env overrides: MAX_SEQS, MAX_BATCHED, SPEC_N, LOCAL_ARGMAX, SPEC_SCHEDULE,
#   SSM_DTYPE, BLOCK_SIZE, RECOVER_SSM, PLE_SHARED_DIR, PLE_SHARED_NAME,
#   GRAPH_MODE, KV_OFFLOAD_GB, KV_OFFLOAD_DIR, REASONING_PARSER, TOOL_PARSER,
#   EXTRA_ARGS, P2P, BOOT_CHECK.
#   MAX_LEN must stay 262144 and GPU_UTIL 0.95 (refused otherwise);
#   FIRST_LIGHT=1 is the 32K/eager/no-MTP diagnostic, not a serving mode.
#
# ===== sm_80 feature switches ===============================================
# Each has a one-variable kill switch: set it to 0 in the environment and
# restart, no rebuild.
#   VLLM_QWEN4_QSA_TOPK_CANONICAL=0  canonical sparse-attention top-k ties
#   VLLM_CUSTOM_ALLREDUCE_FLAGS=0    flags-in-data two-shot TP2 all-reduce
#   (the engine's VLLM_EXL3_* switches are still exported, as Morrowmake's recipe
#   does; a W4A16 checkpoint never reaches the EXL3 code paths, so they do nothing here)
#   VLLM_THIN_GEMM=0                 thin-M BF16 GEMM for the small-row unquantized projections
#   VLLM_THIN_GEMM_GLUON=0           Gluon thin GEMM for the decode shapes captured in CUDA graphs
#   VLLM_FLA_AUTOTUNE_SKIP_SPILLING=0  FLA autotune skips configs that spill local memory
#   P2P=auto                         verify peer content before enabling P2P
#   P2P=off                          disable all peer transports
#   P2P=force                        bypass the content check
# PLE placement (PLE_PLACEMENT) is the EXL3 reader's switch and does not apply
#   here: the FP8 table is loaded into host RAM by the engine's Engram cpu_offload.
# GRAPH_MODE: FULL_AND_PIECEWISE (default: the whole decode step in one CUDA graph,
#   which is what makes this build fast on an old CPU), PIECEWISE (the EXL3
#   recipe's mode; ~3x slower per step here), FULL, NONE (with FIRST_LIGHT=1).
# ============================================================================
set -euo pipefail
# generated:serving-defaults:begin
# Serving defaults for Qwen3.8-Flash-Next; recipe copies are generated.
# Constants are deliberately not exported. Launcher knobs retain their existing
# unset-versus-empty policy, diagnostics and deployment checks. An engine switch
# is listed only under its current name and only where its serving value differs
# from the code default (the PLE placement default, ssd, is the engine's own).
QWEN_DEFAULT_FIRST_LIGHT=0
QWEN_DEFAULT_MODE=dflash
QWEN_DEFAULT_DIAGNOSTIC_MODE=none
QWEN_DEFAULT_LAYOUT=tp1
QWEN_DEFAULT_CUDA_VISIBLE_DEVICES=0
QWEN_DEFAULT_VLLM_QWEN4_QSA_TOPK_CANONICAL=1
QWEN_DEFAULT_VLLM_CUSTOM_ALLREDUCE_FLAGS=1
QWEN_DEFAULT_VLLM_EXL3_PLE_OVERLAP=1
QWEN_DEFAULT_VLLM_EXL3_MOE_V2=1
QWEN_DEFAULT_VLLM_EXL3_DENSE_SPLITK=1
QWEN_DEFAULT_VLLM_FLA_AUTOTUNE_SKIP_SPILLING=1
QWEN_DEFAULT_VLLM_USE_BREAKABLE_CUDAGRAPH=1
QWEN_DEFAULT_VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1
QWEN_DEFAULT_NCCL_P2P_LEVEL=SYS
QWEN_DEFAULT_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
QWEN_DEFAULT_MAX_LEN=262144
QWEN_DEFAULT_DIAGNOSTIC_MAX_LEN=32768
QWEN_DEFAULT_GPU_UTIL=0.95
QWEN_DEFAULT_SPEC_N=3
QWEN_DEFAULT_LOCAL_ARGMAX=0
QWEN_DEFAULT_SPEC_SCHEDULE=
QWEN_DEFAULT_DFLASH_MODEL=/drafter
QWEN_DEFAULT_DFLASH_SPEC_N=7
QWEN_DEFAULT_DFLASH_SPEC_SCHEDULE=[[1,2,7],[3,4,5],[5,48,3]]
QWEN_DEFAULT_MTP_SPEC_SCHEDULE=
QWEN_DEFAULT_DFLASH_BLOCK_SIZE=912
QWEN_DEFAULT_DFLASH_BLOCK_SIZE_FP32_STATES=1680
QWEN_DEFAULT_SSM_DTYPE=bfloat16
QWEN_DEFAULT_RECOVER_SSM=1
QWEN_DEFAULT_DFLASH_PREFIX_MATCH_UNIT=16
QWEN_DEFAULT_MAX_SEQS=48
QWEN_DEFAULT_MAX_BATCHED=3460
QWEN_DEFAULT_REASONING_PARSER=qwen3
QWEN_DEFAULT_TOOL_PARSER=qwen3_coder
QWEN_DEFAULT_PORT=8000
QWEN_DEFAULT_SERVED_MODEL_NAME=qwen3.8-flash-next
QWEN_DEFAULT_TP=1
QWEN_DEFAULT_PP=1
QWEN_DEFAULT_DTYPE=bfloat16
QWEN_DEFAULT_KV_CACHE_DTYPE=bfloat16
QWEN_DEFAULT_ATTENTION_CONFIG='{"indexer_kv_dtype":"bf16"}'
QWEN_DEFAULT_ENGRAM_CONFIG='{"cpu_offload":true}'
QWEN_DEFAULT_GRAPH_CONFIG='{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
QWEN_DEFAULT_DIAGNOSTIC_GRAPH_CONFIG='{"cudagraph_mode":"NONE"}'
QWEN_DEFAULT_LONG_PREFILL_TOKEN_THRESHOLD=0
# generated:serving-defaults:end
FIRST_LIGHT=${FIRST_LIGHT:-$QWEN_DEFAULT_FIRST_LIGHT}
case "$FIRST_LIGHT" in 0|1) ;; *) echo 'FIRST_LIGHT must be 0 or 1' >&2; exit 2;; esac
if [ "$FIRST_LIGHT" = 1 ]; then MODE=${1:-$QWEN_DEFAULT_DIAGNOSTIC_MODE}; else MODE=${1:-$QWEN_DEFAULT_MODE}; fi
case "$MODE" in
  -h|--help) sed -n '2,48p' "$0"; exit 0 ;;
  dflash|mtp|none) ;;
  *) echo "serve.sh: unknown argument '$MODE' (expected dflash, mtp, none or --help)" >&2; exit 2 ;;
esac
# Container compilers use a private temporary directory on the cache mount.
# This runs for start.sh, Compose and direct launches before Python imports.
if [ -n "${RECIPE_CACHE_ROOT:-}" ] && [ "${DRY:-0}" != 1 ]; then
    [ "$RECIPE_CACHE_ROOT" = /cache ] || { echo '[boot-check] FAIL: container cache root must be /cache' >&2; exit 1; }
    temp_base="${TMPDIR:-/cache/tmp}"
    case "$temp_base/" in /cache/*) ;; *) echo '[boot-check] FAIL: container TMPDIR must be under /cache' >&2; exit 1 ;; esac
    mkdir -p "$temp_base"
    temp_base="$(realpath "$temp_base")"
    case "$temp_base/" in /cache/*) ;; *) echo '[boot-check] FAIL: container TMPDIR resolves outside /cache' >&2; exit 1 ;; esac
    TMPDIR="$(mktemp -d "$temp_base/run.XXXXXXXX")"
    export TMPDIR
    export TMP="$TMPDIR" TEMP="$TMPDIR"
    unset temp_base
fi
# Cache seeds are opt-in until the image's cold/warm acceptance passes.
if [ "${VLLM_IMAGE_CACHE_SEED:-0}" = 1 ] && [ "${DRY:-0}" = 1 ]; then
    echo "[image-cache] enabled (dry run)"
elif [ "${VLLM_IMAGE_CACHE_SEED:-0}" = 1 ]; then
    [ -f /opt/image-tools/cache_seed.py ] || { echo "[image-cache] enabled but image helper missing" >&2; exit 1; }
    cache_exports="$(/opt/venv/bin/python /opt/image-tools/cache_seed.py seed /opt/cache-seed)"
    eval "$cache_exports"
    unset cache_exports
else
    echo "[image-cache] disabled"
fi
# Explicit compiler destinations also cover launches without a seed.
if [ -n "${RECIPE_CACHE_ROOT:-}" ] && [ "${DRY:-0}" != 1 ]; then
    export HOME="${HOME:-/cache}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
    export HF_HOME="${HF_HOME:-/cache/huggingface}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
    export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"
    export VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR="${VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR:-$HOME/.cache/vllm/custom_allreduce_flags}"
    export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
    export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$HOME/.tilelang/cache}"
    export TILELANG_TMP_DIR="${TILELANG_TMP_DIR:-$HOME/.tilelang/cache/tmp}"
    export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$HOME/.nv/ComputeCache}"
    export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$HOME}"
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/torchinductor}"
    for cache_path in "$HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" \
        "$VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR" "$VLLM_CACHE_ROOT" "$TILELANG_CACHE_DIR" "$TILELANG_TMP_DIR" \
        "$CUDA_CACHE_PATH" "$FLASHINFER_WORKSPACE_BASE" "$TORCHINDUCTOR_CACHE_DIR"; do
        case "$cache_path/" in /cache/*) ;; *) echo "[boot-check] FAIL: compiler cache outside /cache: $cache_path" >&2; exit 1 ;; esac
        mkdir -p "$cache_path"
        cache_path="$(realpath "$cache_path")"
        case "$cache_path/" in /cache/*) ;; *) echo "[boot-check] FAIL: cache symlink outside /cache: $cache_path" >&2; exit 1 ;; esac
        [ -w "$cache_path" ] || { echo "[boot-check] FAIL: cache is not writable: $cache_path" >&2; exit 1; }
    done
    unset cache_path
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$REPO_ROOT/venv}"
MODEL="${MODEL:-$REPO_ROOT/models/Qwen3.8-Flash-Next-exl3}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
export CUDA_HOME
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
LAYOUT=${LAYOUT:-$QWEN_DEFAULT_LAYOUT}
if [ "$LAYOUT" != tp1 ]; then echo 'This recipe requires LAYOUT=tp1' >&2; exit 2; fi
TP=$QWEN_DEFAULT_TP PP=$QWEN_DEFAULT_PP
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$QWEN_DEFAULT_CUDA_VISIBLE_DEVICES}
# Card numbers are physical (nvidia-smi) numbers.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo 'CUDA_VISIBLE_DEVICES must select exactly one card' >&2; exit 2
fi
# Engine switches are exported only where the value differs from the engine default; the
# PLE placement is exported only when PLE_PLACEMENT or VLLM_EXL3_PLE_PLACEMENT asks for one.
PLE_PLACEMENT=${PLE_PLACEMENT:-${VLLM_EXL3_PLE_PLACEMENT:-}}
if [ -n "$PLE_PLACEMENT" ]; then
  case "$PLE_PLACEMENT" in
    ssd|ram|device) ;;
    *) echo 'PLE placement must be ssd, ram or device' >&2; exit 2 ;;
  esac
  export VLLM_EXL3_PLE_PLACEMENT=$PLE_PLACEMENT
fi
# Canonical QSA top-k ties (lowest block index wins), so a request repeats at exact score ties.
export VLLM_QWEN4_QSA_TOPK_CANONICAL=${VLLM_QWEN4_QSA_TOPK_CANONICAL:-$QWEN_DEFAULT_VLLM_QWEN4_QSA_TOPK_CANONICAL}
# Flags-in-data two-shot custom all-reduce kernel for the TP2 all-reduces it covers.
export VLLM_CUSTOM_ALLREDUCE_FLAGS=${VLLM_CUSTOM_ALLREDUCE_FLAGS:-${QWEN_DEFAULT_VLLM_CUSTOM_ALLREDUCE_FLAGS:-0}}
# SSD PLE row reads overlap the preceding decoder layer instead of stalling the step.
export VLLM_EXL3_PLE_OVERLAP=${VLLM_EXL3_PLE_OVERLAP:-${QWEN_DEFAULT_VLLM_EXL3_PLE_OVERLAP:-0}}
# SSD PLE rows of a step are read in one native call that releases the GIL, not a Python thread pool.
export VLLM_EXL3_PLE_NATIVE_READ=${VLLM_EXL3_PLE_NATIVE_READ:-1}
# EXL3 MoE kernel v2 for MoE calls of 1-64 rows (decode and short prefills).
export VLLM_EXL3_MOE_V2=${VLLM_EXL3_MOE_V2:-${QWEN_DEFAULT_VLLM_EXL3_MOE_V2:-0}}
# EXL3 split-K dense kernel for the small-row (tiny and small tile) launches.
export VLLM_EXL3_DENSE_SPLITK=${VLLM_EXL3_DENSE_SPLITK:-${QWEN_DEFAULT_VLLM_EXL3_DENSE_SPLITK:-0}}
# Tensor-core blocks for the split-K kernel's dense launches of 1-32 rows (needs VLLM_EXL3_DENSE_SPLITK).
export VLLM_EXL3_DENSE_MMA=${VLLM_EXL3_DENSE_MMA:-1}
# Tensor-core EXL3 dense kernel for prefill launches of more than 64 rows.
export VLLM_EXL3_DENSE_PREFILL_TC=${VLLM_EXL3_DENSE_PREFILL_TC:-1}
# Tensor-core EXL3 routed experts for prefill MoE calls of 65-3460 rows.
export VLLM_EXL3_MOE_PREFILL_TC=${VLLM_EXL3_MOE_PREFILL_TC:-1}
# Thin-M BF16 GEMM for the small-row calls of the unquantized projections; the Gluon
# kernel takes the decode shapes captured in CUDA graphs (needs VLLM_THIN_GEMM).
export VLLM_THIN_GEMM=${VLLM_THIN_GEMM:-1}
export VLLM_THIN_GEMM_GLUON=${VLLM_THIN_GEMM_GLUON:-1}
# FLA autotune never launches a config that spills on sm_80 (first-boot KV loss on a cold Triton cache).
export VLLM_FLA_AUTOTUNE_SKIP_SPILLING=${VLLM_FLA_AUTOTUNE_SKIP_SPILLING:-${QWEN_DEFAULT_VLLM_FLA_AUTOTUNE_SKIP_SPILLING:-0}}
export VLLM_USE_BREAKABLE_CUDAGRAPH=${VLLM_USE_BREAKABLE_CUDAGRAPH:-$QWEN_DEFAULT_VLLM_USE_BREAKABLE_CUDAGRAPH}
if [ "$FIRST_LIGHT" = 0 ] && [ "$VLLM_USE_BREAKABLE_CUDAGRAPH" != 1 ]; then
  echo 'Serving requires VLLM_USE_BREAKABLE_CUDAGRAPH=1' >&2; exit 2
fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-$QWEN_DEFAULT_PYTORCH_CUDA_ALLOC_CONF}
# One card: no peer traffic, so peer access stays off and no peer check runs,
# whatever P2P says (p2p_check.sh is the two-card recipe's).
P2P=off
export VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0
unset NCCL_P2P_LEVEL
export NCCL_P2P_DISABLE=1
echo "[p2p] P2P off: one card"
FL_ARGS=()
# The graph mode defaults come from the serving defaults; GRAPH_MODE overrides them.
if [ "$FIRST_LIGHT" = 1 ]; then DEFAULT_GRAPH_CONFIG=$QWEN_DEFAULT_DIAGNOSTIC_GRAPH_CONFIG; else DEFAULT_GRAPH_CONFIG=$QWEN_DEFAULT_GRAPH_CONFIG; fi
DEFAULT_GRAPH_MODE=${DEFAULT_GRAPH_CONFIG#*\"cudagraph_mode\":\"}
DEFAULT_GRAPH_MODE=${DEFAULT_GRAPH_MODE%%\"*}
GRAPH_MODE=${GRAPH_MODE:-$DEFAULT_GRAPH_MODE}
case "$GRAPH_MODE" in PIECEWISE|NONE|FULL_AND_PIECEWISE|FULL) ;; *) echo "GRAPH_MODE must be PIECEWISE, FULL_AND_PIECEWISE, FULL or NONE" >&2; exit 2;; esac
if [ "$GRAPH_MODE" = "$DEFAULT_GRAPH_MODE" ]; then
  GRAPH_ARGS=(--compilation-config "$DEFAULT_GRAPH_CONFIG")
else
  GRAPH_ARGS=(--compilation-config "{\"cudagraph_mode\":\"$GRAPH_MODE\"}")
fi
if [ "$FIRST_LIGHT" = 1 ]; then
  if [ "$MODE" != none ] || [ "${MAX_LEN-$QWEN_DEFAULT_DIAGNOSTIC_MAX_LEN}" != 32768 ] || [ "$GRAPH_MODE" != NONE ]; then
    echo 'FIRST_LIGHT=1 requires no MTP, MAX_LEN=32768 and GRAPH_MODE=NONE' >&2; exit 2
  fi
  FL_ARGS=(--enforce-eager)
  MAX_LEN=$QWEN_DEFAULT_DIAGNOSTIC_MAX_LEN
else
  if [ "$GRAPH_MODE" = NONE ]; then
    echo 'GRAPH_MODE=NONE requires FIRST_LIGHT=1' >&2; exit 2
  fi
  MAX_LEN=${MAX_LEN-$QWEN_DEFAULT_MAX_LEN}
  if [ "$MAX_LEN" != 262144 ]; then
    echo 'Serving requires MAX_LEN=262144; use FIRST_LIGHT=1 for the 32768 diagnostic' >&2; exit 2
  fi
fi
GPU_UTIL=${GPU_UTIL-$QWEN_DEFAULT_GPU_UTIL}
if [ "$GPU_UTIL" != 0.95 ]; then
  echo 'Serving requires GPU_UTIL=0.95, including FIRST_LIGHT=1' >&2; exit 2
fi
# GDN recurrent states: bfloat16 (default) or float32 (what the checkpoint asks for). BF16
# halves the state pages and shortens the step 1-3 % at load; quality measured level.
SSM_DTYPE=${SSM_DTYPE:-$QWEN_DEFAULT_SSM_DTYPE}
case "$SSM_DTYPE" in float32|bfloat16) ;; *) echo 'SSM_DTYPE must be float32 or bfloat16' >&2; exit 2 ;; esac
# RecoverSSM (patches/gdn_recover, mounted by start.sh and Compose): one GDN state per request,
# accepted tokens replayed after sampling, instead of one state per draft position.
RECOVER_SSM=${RECOVER_SSM:-$QWEN_DEFAULT_RECOVER_SSM}
case "$RECOVER_SSM" in 0|1) ;; *) echo 'RECOVER_SSM must be 0 or 1' >&2; exit 2 ;; esac
if [ "$RECOVER_SSM" = 1 ]; then export VLLM_GDN_RECOVER=1; else unset VLLM_GDN_RECOVER; fi
# PLE_SHARED_DIR: the n-gram table as one shared host copy (patches/qwen4_exp_common_ngram_embedding.py).
if [ -n "${PLE_SHARED_DIR:-}" ]; then
  export VLLM_QWEN_PLE_SHARED_DIR=$PLE_SHARED_DIR
  export VLLM_QWEN_PLE_SHARED_NAME=${PLE_SHARED_NAME:-qwen38-flash-next-ple-fp8}
fi
SPEC=()
if [ "$MODE" = mtp ]; then
  SPEC_N=${SPEC_N:-$QWEN_DEFAULT_SPEC_N}
  if [[ ! "$SPEC_N" =~ ^[1-9][0-9]*$ ]]; then echo 'SPEC_N must be positive' >&2; exit 2; fi
  # LOCAL_ARGMAX=1: greedy drafts take each rank's local argmax over its vocab shard and gather (value, index)
  # pairs instead of gathering full-vocab draft logits. Same draft tokens.
  LOCAL_ARGMAX=${LOCAL_ARGMAX:-${QWEN_DEFAULT_LOCAL_ARGMAX:-0}}
  case "$LOCAL_ARGMAX" in 0|1) ;; *) echo 'LOCAL_ARGMAX must be 0 or 1' >&2; exit 2;; esac
  SPEC_JSON="{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_N"
  if [ "$LOCAL_ARGMAX" = 1 ]; then SPEC_JSON+=",\"use_local_argmax_reduction\":true"; fi
  # SPEC_SCHEDULE (set empty: off): upstream dynamic speculative decoding, num_speculative_tokens_per_batch_size,
  # a JSON list of [first_batch, last_batch, drafts] ranges without spaces starting at batch 1, e.g.
  # [[1,2,4],[3,8,3]] with SPEC_N=4 (drafts above SPEC_N are clamped to it). Each MTP draft is a
  # pass of the MTP layer of its own, so under load fewer drafts win (docs/results.md, "MTP").
  SPEC_SCHEDULE=${SPEC_SCHEDULE-$QWEN_DEFAULT_MTP_SPEC_SCHEDULE}
  if [ -n "$SPEC_SCHEDULE" ]; then
    SCHEDULE_RANGE='\[[1-9][0-9]*,[1-9][0-9]*,[0-9]+\]'
    if [[ ! "$SPEC_SCHEDULE" =~ ^\[$SCHEDULE_RANGE(,$SCHEDULE_RANGE)*\]$ ]]; then
      echo 'SPEC_SCHEDULE must be [[first,last,drafts],...] without spaces' >&2; exit 2
    fi
    SPEC_JSON+=",\"num_speculative_tokens_per_batch_size\":$SPEC_SCHEDULE"
  fi
  SPEC_JSON+=",\"disable_eagle_block_drop\":true"
  SPEC=(--speculative-config "$SPEC_JSON}")
fi
# DFlash 2 (the default). The drafter drafts a whole block in one pass, so more drafts
# cost little; its block allows 7. 5-7 drafts need an attention block size that the
# sparse-attention ring (12 for 5-7 drafts) and the prefix-match unit 16 divide, so a
# multiple of 48, and at least the GDN state page, which the engine otherwise pads up to
# it: 912 with BF16 states (padding 5 %), 1,680 with float32 ones. The schedule keeps 7
# drafts where they pay (one or two requests), 3 where the step is the GPU's and 2 from 33
# requests. Measured in docs/results.md, "DFlash 2" and "0.2.0".
DFLASH_ARGS=()
PREFIX_MATCH_UNIT=64
if [ "$MODE" = dflash ]; then
  DFLASH_MODEL=${DFLASH_MODEL:-$QWEN_DEFAULT_DFLASH_MODEL}
  # DRY=1 runs this on the host, where /drafter is not mounted.
  [ -n "${DRY:-}" ] || [ -f "$DFLASH_MODEL/config.json" ] || { echo "serve.sh: no DFlash 2 drafter at $DFLASH_MODEL" >&2; exit 2; }
  SPEC_N=${SPEC_N:-$QWEN_DEFAULT_DFLASH_SPEC_N}
  if [[ ! "$SPEC_N" =~ ^[1-7]$ ]]; then echo 'SPEC_N for dflash must be 1 to 7 (the drafter block is 8)' >&2; exit 2; fi
  SPEC_SCHEDULE=${SPEC_SCHEDULE-$QWEN_DEFAULT_DFLASH_SPEC_SCHEDULE}
  SPEC_JSON="{\"method\":\"dflash\",\"model\":\"$DFLASH_MODEL\",\"num_speculative_tokens\":$SPEC_N"
  if [ -n "$SPEC_SCHEDULE" ]; then
    SCHEDULE_RANGE='\[[1-9][0-9]*,[1-9][0-9]*,[0-9]+\]'
    if [[ ! "$SPEC_SCHEDULE" =~ ^\[$SCHEDULE_RANGE(,$SCHEDULE_RANGE)*\]$ ]]; then
      echo 'SPEC_SCHEDULE must be [[first,last,drafts],...] without spaces' >&2; exit 2
    fi
    SPEC_JSON+=",\"num_speculative_tokens_per_batch_size\":$SPEC_SCHEDULE"
  fi
  # A drafter's own cache group would otherwise zero every KV-offload lookup (vLLM #52735).
  [ "${KV_OFFLOAD_GB:-0}" != 0 ] && SPEC_JSON+=",\"disable_eagle_block_drop\":true"
  SPEC=(--speculative-config "$SPEC_JSON}")
  if [ "$SSM_DTYPE" = float32 ]; then DEFAULT_BLOCK=$QWEN_DEFAULT_DFLASH_BLOCK_SIZE_FP32_STATES
  else DEFAULT_BLOCK=$QWEN_DEFAULT_DFLASH_BLOCK_SIZE; fi
  BLOCK_SIZE=${BLOCK_SIZE:-$DEFAULT_BLOCK}
  [[ "$BLOCK_SIZE" =~ ^[1-9][0-9]*$ ]] && [ $((BLOCK_SIZE % 48)) = 0 ] ||
    { echo 'BLOCK_SIZE must be a multiple of 48' >&2; exit 2; }
  DFLASH_ARGS=(--block-size "$BLOCK_SIZE")
  PREFIX_MATCH_UNIT=$QWEN_DEFAULT_DFLASH_PREFIX_MATCH_UNIT
fi
# KV offload (KV_OFFLOAD_GB, default 0: off): a tier of host RAM, in GiB, that keeps KV
# blocks evicted from the cards and copies them back instead of re-prefilling (vLLM's
# OffloadingConnector with TieringOffloadingSpec; a shared mmap in /dev/shm, so start.sh
# sizes shm to fit). KV_OFFLOAD_DIR (empty: none) adds a disk tier behind the RAM one;
# start.sh mounts the directory named in .env at /kvtier. Carried over from the MiMo
# recipe on the same engine (gpubox, 2026-10-19). The MTP config above already sets
# disable_eagle_block_drop, which offload needs (vLLM #52735).
KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-0}
KV_OFFLOAD_DIR=${KV_OFFLOAD_DIR:-}
[[ "$KV_OFFLOAD_GB" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'KV_OFFLOAD_GB must be a whole number of GiB' >&2; exit 2; }
if [ -n "$KV_OFFLOAD_DIR" ] && [ "$KV_OFFLOAD_GB" = 0 ]; then
  echo 'KV_OFFLOAD_DIR needs a RAM tier in front of it: set KV_OFFLOAD_GB' >&2; exit 2
fi
KV_OFFLOAD=()
if [ "$KV_OFFLOAD_GB" != 0 ]; then
  OFFLOAD_JSON="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":$((KV_OFFLOAD_GB * 1073741824))"
  [ -n "$KV_OFFLOAD_DIR" ] && OFFLOAD_JSON+=",\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$KV_OFFLOAD_DIR\"}]"
  KV_OFFLOAD=(--kv-transfer-config "$OFFLOAD_JSON}}")
fi
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$QWEN_DEFAULT_SERVED_MODEL_NAME}
PORT=${PORT:-$QWEN_DEFAULT_PORT}
# The key goes to vLLM through the environment rather than --api-key, so it
# does not show up in the process list.
if [ -n "${API_KEY:-}" ]; then export VLLM_API_KEY="$API_KEY"; fi
CMD=("$VENV/bin/vllm" serve "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "${HOST:-127.0.0.1}"
  --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP"
  --dtype "$QWEN_DEFAULT_DTYPE" --kv-cache-dtype "$QWEN_DEFAULT_KV_CACHE_DTYPE"
  --load-format auto --mm-encoder-tp-mode data
  --attention-config "$QWEN_DEFAULT_ATTENTION_CONFIG"
  --engram-config "$QWEN_DEFAULT_ENGRAM_CONFIG"
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL"
  --max-num-seqs "${MAX_SEQS:-$QWEN_DEFAULT_MAX_SEQS}" --max-num-batched-tokens "${MAX_BATCHED:-$QWEN_DEFAULT_MAX_BATCHED}"
  --long-prefill-token-threshold "$QWEN_DEFAULT_LONG_PREFILL_TOKEN_THRESHOLD"
  --reasoning-parser "${REASONING_PARSER:-$QWEN_DEFAULT_REASONING_PARSER}"
  --enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-$QWEN_DEFAULT_TOOL_PARSER}"
  --port "$PORT"
  --prefix-match-unit "$PREFIX_MATCH_UNIT" ${DFLASH_ARGS[@]+"${DFLASH_ARGS[@]}"}
  --mamba-ssm-cache-dtype "$SSM_DTYPE"
  ${FL_ARGS[@]+"${FL_ARGS[@]}"} "${GRAPH_ARGS[@]}" ${SPEC[@]+"${SPEC[@]}"} ${KV_OFFLOAD[@]+"${KV_OFFLOAD[@]}"})
# Additional arguments are whitespace-separated.
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
# FlexibleArgumentParser accepts abbreviations, underscores and dotted JSON keys.
# Config files and model overrides can change several controlled fields at once.
CONTROLLED_ARGS=(
  --config --model --hf-config-path --hf-overrides --model-class-overrides
  --model-impl --runner --convert --load-format --model-loader-extra-config
  --device-ids --distributed-executor-backend --nnodes --node-rank
  --tensor-parallel-size --pipeline-parallel-size
  --decode-context-parallel-size --prefill-context-parallel-size
  --data-parallel-size --data-parallel-size-local --data-parallel-rank
  --data-parallel-start-rank --data-parallel-address --data-parallel-rpc-port
  --data-parallel-backend --data-parallel-hybrid-lb --data-parallel-external-lb
  --data-parallel-multi-port-external-lb --enable-expert-parallel
  --enable-elastic-ep --elastic-ep-max-dp-size
  --dtype --quantization --quantization-config --kv-cache-dtype
  --kv-cache-dtype-skip-layers --mamba-cache-dtype --mamba-ssm-cache-dtype
  --attention-config --compilation-config --optimization-level --performance-mode
  --cudagraph-capture-sizes --max-cudagraph-capture-size
  --enforce-eager --engram-config --speculative-config
  --spec-method --spec-model --spec-tokens --diffusion-config
  --language-model-only --limit-mm-per-prompt --mm-encoder-only
  --mm-encoder-tp-mode --mm-encoder-attn-dtype
  --max-model-len --gpu-memory-utilization --device-memory-utilization --kv-cache-memory-bytes
  --num-gpu-blocks-override
  # No KV or weight offload; the n-gram table placement is PLE_PLACEMENT.
  --cpu-offload-gb --cpu-offload-params --offload-backend --offload-group-size --offload-num-in-group
  --offload-prefetch-step --offload-params --kv-offloading-size --kv-offloading-backend --kv-transfer-config
)
for argument in ${EXTRA[@]+"${EXTRA[@]}"}; do
  blocked=0
  case "$argument" in
    --) blocked=1 ;;
    --*)
      option=${argument%%=*}
      option=${option%%.*}
      option=${option//_/-}
      for controlled in "${CONTROLLED_ARGS[@]}"; do
        if [[ "$controlled" == "$option"* || "--no-${controlled#--}" == "$option"* ]]; then
          blocked=1; break
        fi
      done ;;
    -*)
      option=${argument%%=*}
      option=${option%%.*}
      for controlled in -tp -pp -dp -dcp -pcp -ep -n -r -q -sc -cc -ac -dc -O; do
        if [[ "$option" == "$controlled"* || "$controlled" == "$option"* ]]; then
          blocked=1; break
        fi
      done ;;
  esac
  if [ "$blocked" = 1 ]; then
    echo "EXTRA_ARGS cannot override the EXL3 deployment contract: $argument" >&2; exit 2
  fi
done
CMD+=(${EXTRA[@]+"${EXTRA[@]}"})
echo "serve.sh: 3 bpw enabled TP=1 PP=1 GPU=$CUDA_VISIBLE_DEVICES PLE=host-ram dtype=BF16 KV=$QWEN_DEFAULT_KV_CACHE_DTYPE indexer=BF16 vision=on MTP=$MODE graphs=$GRAPH_MODE kv_offload_gb=$KV_OFFLOAD_GB FIRST_LIGHT=$FIRST_LIGHT ssm_states=$SSM_DTYPE recover_ssm=$RECOVER_SSM ple_shared=${PLE_SHARED_DIR:-off}"
echo "serve.sh: Qwen4Exp CUDA graph mode=$GRAPH_MODE (GRAPH_MODE=$GRAPH_MODE)"
BOOT_CHECK=${BOOT_CHECK:-1}
case "$BOOT_CHECK" in 0|1) ;; *) echo "[boot-check] FAIL: BOOT_CHECK must be 0 or 1" >&2; exit 2 ;; esac
if [ "${DRY:-0}" = 1 ]; then
  echo "[boot-check] BOOT_CHECK=$BOOT_CHECK (dry run)"
  echo "serve.sh: DRY=1, layout $LAYOUT (PP=$PP TP=$TP), environment the server would get:"
  env | LC_ALL=C sort | grep -E '^(VLLM_|NCCL_|PYTORCH_CUDA_ALLOC_CONF=|CUDA_VISIBLE_DEVICES=|CUDA_DEVICE_ORDER=)' \
    | grep -v '^VLLM_API_KEY=' | sed 's/^/  /' || true
  if [ -n "${VLLM_API_KEY:-}" ]; then echo "  VLLM_API_KEY=(set, not shown)"; else echo "  (no API key: /v1 is open to anyone who can reach HOST:PORT)"; fi
  echo "serve.sh: DRY=1, command:"
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi
if [ "$BOOT_CHECK" = 0 ]; then
  echo "[boot-check] disabled (BOOT_CHECK=0)"
elif [ "${RECIPE_BOOT_CHECK_OWNER:-}" = start ]; then
  echo "[boot-check] delegated to start.sh after health"
else
  CLIENT_HOST=${HOST:-127.0.0.1}
  case "$CLIENT_HOST" in 0.0.0.0|::|'[::]'|'') CLIENT_HOST=127.0.0.1 ;; esac
  boot_args=()
  [ -z "${RECIPE_CACHE_ROOT:-}" ] || boot_args+=(--container)
  exec "$VENV/bin/python" "$REPO_ROOT/boot_check.py" ${boot_args[@]+"${boot_args[@]}"} \
    --base "http://$CLIENT_HOST:$PORT" \
    --model "$SERVED_MODEL_NAME" \
    --ready-timeout "${READY_TIMEOUT:-1800}" --serve "${CMD[@]}"
fi
exec "${CMD[@]}"
