#!/usr/bin/env bash
# ============================================================================
# start.sh — Qwen3.8-Flash-Next 3 bpw on 1 CUDA GPU (sm_80 Ampere patches)
# ============================================================================
#
# We serve klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP (AutoRound 2- and 3-bit
# experts) with halt95's FP8 n-gram table in place of its BF16 one, on one GPU
# with Morrowmake's patched vLLM: OpenAI API on 127.0.0.1:8000 as
# "qwen3.8-flash-next" (local only unless you set HOST; no API key unless you set
# API_KEY), inco.ai's DFlash 2 drafter (7 drafts for one or two users, 3 from
# five), up to 48 requests at once, 262,144-token context, vision on, the whole
# decode step in one CUDA graph. Every weight is on the card except the n-gram
# (PLE) table, which lives in host RAM (48 GiB; PLE_SHARED_DIR lets several
# servers share one copy, this recipe's and the two-card one's alike). bf16 KV
# cache and GDN states, RecoverSSM for the GDN layers (one state per request
# instead of one per draft); KV_OFFLOAD_GB adds a host-RAM tier for KV blocks
# evicted from the card.
#
# The engine runs in a container: Docker with the NVIDIA Container Toolkit,
# image pinned by digest. docker-compose.yml runs the same container.
#
# Every step is idempotent and skipped when it is already done, so running
# ./start.sh twice is safe and the second run just launches.
#
# What we do:
#   1. preflight — docker, git, a GPU of >=60 GiB (read only), disk, port
#   2. install   — the pinned engine image, if missing or the pin moved
#   3. download  — the 3 bpw checkpoint without its BF16 table, the FP8 table
#                  shards, the drafter (resumed), then the served checkpoint
#                  built from them with hard links
#   4. launch    — detached, output to logs/serve.log
#   5. wait      — poll /health up to READY_TIMEOUT, then print KV + model id
#
# Usage:
#   ./start.sh                 preflight, install, download, launch — default
#   ./start.sh install         pull the engine image only
#   ./start.sh download        fetch and build the checkpoint only
#   ./start.sh stop            stop the server this checkout started
#   ./start.sh restart         stop + start
#   ./start.sh status          server + /health + KV cache line
#   ./start.sh logs            follow logs/serve.log
#   ./start.sh update          git pull, pull the image if the pin moved, restart
#   ./start.sh smoke           one chat request and one tool call
#   ./start.sh help            this text
#
# The image is Morrowmake's release image for their Qwen EXL3 recipe, used as
# released (its EXL3 extension is simply unused by a W4A16 checkpoint).
#
# Config lives in .env, copied from .env.example on first run. A prefix env
# assignment beats .env for every key:
#
#   CUDA_VISIBLE_DEVICES=2 ./start.sh restart
#   KV_OFFLOAD_GB=64 ./start.sh restart
#
# DRY=1 ./start.sh runs the preflight, says whether it would install or
# download, and prints the server environment and launch command. It installs,
# downloads and launches nothing, and works before the first install. With
# stop, restart or update, DRY=1 also takes the checkout's lock and says what
# it would stop, without stopping, pulling or installing anything.
#
# Lifecycle commands on this checkout are serialised by a flock on
# logs/lifecycle.lock. start/restart/install/download refuse immediately when
# another one holds it; stop waits LOCK_WAIT seconds and then exits 1 WITHOUT
# stopping anything. The lock is per checkout: another clone is not covered.
#
# stop only ever stops the container in logs/container.id, and only after
# confirming it is the server this checkout launched. It never searches by
# container name, so it cannot touch an unrelated vLLM on the same machine.
# ============================================================================
set -euo pipefail

_say() { printf '\033[%sm%s\033[0m %s\n' "$1" "$2" "${*:3}"; }
log()  { _say '1;36' 'qwen38 ·' "$@"; }
warn() { _say '1;33' 'qwen38 !' "$@" >&2; }
die()  { _say '1;31' 'qwen38 ✗' "$@" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

usage() {
    sed -n '3,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" \
        | sed -e '/^set -euo pipefail/d' -e 's/^# \{0,1\}//'
}

# ---------------------------- .env handling --------------------------------
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    [ -f "$SCRIPT_DIR/.env.example" ] || die "missing .env.example"
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    log "wrote .env from .env.example — edit it, or override any key inline"
fi
# .env supplies defaults for keys the caller has not already set. It is parsed
# as KEY=VALUE, never sourced, so a config file cannot run code and no shell
# expansion happens inside a value. A key already present in the environment is
# left alone even when it is explicitly empty, which is what makes
# `KEY= ./start.sh` mean "unset this knob" rather than "use the .env value".
#
# Values: a value in matching quotes ("..." or '...') is taken exactly as
# written between them; a note after the closing quote is ignored. An unquoted
# value ends at the first whitespace-then-# (`KEY=v  # note` gives `v`), then a
# trailing note in parentheses after two or more spaces is dropped
# (`KEY=v   (default: x)` gives `v`), then surrounding whitespace is trimmed.
# A # or ( with no whitespace before it (`a#b`, `f(x)`) or after a single
# space (`a (b)`) stays in the value; quote the value to keep anything else.
load_env_defaults() {
    local file="$1" line key val raw
    local re_dq='^"([^"]*)"[[:space:]]*(#.*|\(.*\))?$'
    local re_sq="^'([^']*)'[[:space:]]*(#.*|\\(.*\\))?\$"
    local re_paren='^(.*)[[:space:]][[:space:]]+\([^()]*\)[[:space:]]*$'
    [ -r "$file" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        line="${line#"${line%%[![:space:]]*}"}"          # ltrim
        case "$line" in ''|'#'*) continue ;; esac
        case "$line" in export\ *) line="${line#export }" ;; esac
        case "$line" in *=*) ;; *) continue ;; esac
        key="${line%%=*}"; val="${line#*=}"
        key="${key%"${key##*[![:space:]]}"}"             # rtrim
        case "$key" in ''|*[!A-Za-z0-9_]*) continue ;; esac
        [ -n "${!key+x}" ] && continue                   # caller already set it
        raw="$val"
        val="${val#"${val%%[![:space:]]*}"}"             # ltrim
        if [[ $val =~ $re_dq || $val =~ $re_sq ]]; then  # quoted: verbatim
            val="${BASH_REMATCH[1]}"
        else
            case "$val" in                               # quotes around a value
                \"*\") val="${val#\"}"; val="${val%\"}" ;;   # that holds quotes
                \'*\') val="${val#\'}"; val="${val%\'}" ;;
                *)                                       # unquoted: drop notes
                    val="${raw%%[[:space:]]#*}"          # (KEY= # x is empty)
                    [[ $val =~ $re_paren ]] && val="${BASH_REMATCH[1]}"
                    val="${val#"${val%%[![:space:]]*}"}"
                    val="${val%"${val##*[![:space:]]}"}" ;;      # trim
            esac
        fi
        export "$key=$val"
    done <"$file"
}
load_env_defaults "$SCRIPT_DIR/.env"

# ---------------------------- configuration --------------------------------
# This release's engine: the fork commit the container image is built from,
# and that image, pinned by digest (docker-compose.yml runs the same image).
# Both are filled in at release (tools/set-pin.sh moves the commit pin
# everywhere at once). The commit pin is read by the release tools, not here.
# shellcheck disable=SC2034
RELEASE_VLLM_COMMIT=18305948acba3f4c4d912cdd56f2202d2ebd94e8
RELEASE_IMAGE=ghcr.io/morrowmake/vllm-cmp170hx@sha256:5b45c4d6feafefe2eba962375b0a9a0b64fd22b9e05abb4d1966f14a32a2db79
IMAGE="${IMAGE:-$RELEASE_IMAGE}"
MODELS_DIR="${MODELS_DIR:-$SCRIPT_DIR/models}"
# The checkpoint, pinned to the revision the release was validated with: klee100's
# 3 bpw AutoRound build (its MTP head is inside it) without its BF16 n-gram table
# shard, halt95's FP8 table shards from the W4A16 build (the same model's table, one
# global scale), and the served checkpoint docker/build_checkpoint.py makes of them.
TARGET_REPO="${TARGET_REPO:-klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP}"
TARGET_REVISION="${TARGET_REVISION:-ce0e0b94083895bd836b916f29bf105c40a8162a}"
SOURCE_MODEL="${SOURCE_MODEL:-$MODELS_DIR/${TARGET_REPO##*/}}"
TABLE_REPO="${TABLE_REPO:-halt95/Qwen3.8-Flash-Next-W4A16-Merlin}"
TABLE_REVISION="${TABLE_REVISION:-cec53e9f90e5995b81691604ec0b32191debf797}"
TABLE_MODEL="${TABLE_MODEL:-$MODELS_DIR/${TABLE_REPO##*/}}"
MODEL="${MODEL:-$MODELS_DIR/Qwen3.8-Flash-Next-3bpw-fp8ple}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next}"
# One layout: tp1 (the whole model on one card).
LAYOUT="${LAYOUT:-tp1}"
[ "$LAYOUT" = tp1 ] || die "LAYOUT must be tp1, not '$LAYOUT'"
PP=1; TP=1
export LAYOUT
# The card, as nvidia-smi numbers it.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
# Native MTP (mtp, the default) or no speculation (none).
SPEC_MODE="${SPEC_MODE:-dflash}"
case "$SPEC_MODE" in dflash|mtp|none) ;; *) die "SPEC_MODE must be dflash, mtp or none, not '$SPEC_MODE'" ;; esac
# The DFlash 2 drafter (SPEC_MODE=dflash, the default), pinned like the checkpoint,
# and the two engine files it needs (patches/, mounted over the image's copies;
# the other patched files there are mounted in every mode, patches/README.md).
DRAFTER_REPO="${DRAFTER_REPO:-incoai/Qwen3.8-Flash-Next-DFlash2}"
DRAFTER_REVISION="${DRAFTER_REVISION:-a22a1744e85c17364dcc54a6a420af7135d4c7f6}"
DFLASH_MODEL="${DFLASH_MODEL:-$MODELS_DIR/${DRAFTER_REPO##*/}}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next-1x}"
# RecoverSSM for the GDN layers (patches/gdn_recover, 1 = on, the default): each
# request keeps one recurrent state instead of one per draft position, which is
# what lets 128 requests fit. 0 = the engine's own per-draft states.
RECOVER_SSM="${RECOVER_SSM:-1}"
case "$RECOVER_SSM" in 0|1) ;; *) die "RECOVER_SSM must be 0 or 1" ;; esac
# One host copy of the n-gram table for several servers: a host directory on tmpfs
# (e.g. /dev/shm/qwen38-ple), mounted read-write at the same path. Every server
# given the same PLE_SHARED_DIR and PLE_SHARED_NAME maps the same 48 GiB instead of
# pinning its own (64 GiB each); empty = off, the table is the server's own.
PLE_SHARED_DIR="${PLE_SHARED_DIR:-}"
if [ -n "$PLE_SHARED_DIR" ]; then
    [[ "$PLE_SHARED_DIR" = /* ]] || die "PLE_SHARED_DIR must be an absolute path"
fi
PLE_SHARED_NAME="${PLE_SHARED_NAME:-qwen38-flash-next-ple-fp8}"
# KV offload: a RAM tier of KV_OFFLOAD_GB GiB lives in the container's /dev/shm, so
# the default shm grows to hold it (the n-gram table is pinned host memory, not shm);
# KV_OFFLOAD_DIR (a host directory) is mounted at /kvtier for the disk tier behind it.
# serve.sh turns both into engine flags. Carried over from the MiMo recipe (gpubox,
# 2026-10-19).
KV_OFFLOAD_GB="${KV_OFFLOAD_GB:-0}"
[[ "$KV_OFFLOAD_GB" =~ ^(0|[1-9][0-9]*)$ ]] || die "KV_OFFLOAD_GB must be a whole number of GiB"
KV_OFFLOAD_DIR="${KV_OFFLOAD_DIR:-}"
if [ -n "$KV_OFFLOAD_DIR" ]; then
    [ "$KV_OFFLOAD_GB" != 0 ] || die "KV_OFFLOAD_DIR needs a RAM tier in front of it: set KV_OFFLOAD_GB"
    [[ "$KV_OFFLOAD_DIR" = /* ]] || die "KV_OFFLOAD_DIR must be an absolute path"
fi
if [ "$KV_OFFLOAD_GB" != 0 ]; then
    SHM_SIZE="${SHM_SIZE:-$((KV_OFFLOAD_GB + 32))g}"
else
    SHM_SIZE="${SHM_SIZE:-16g}"
fi
CACHE_DIR="${CACHE_DIR:-$SCRIPT_DIR/cache}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
BOOT_CHECK="${BOOT_CHECK:-1}"
case "$BOOT_CHECK" in 0|1) ;; *) die "BOOT_CHECK must be 0 or 1" ;; esac
LAUNCHED=0
STOP_TIMEOUT="${STOP_TIMEOUT:-120}"
MIN_GPUS="${MIN_GPUS:-1}"
MIN_GPU_MIB="${MIN_GPU_MIB:-61440}"      # 60 GiB; the cards we run report 65,536 MiB
NEED_DISK_GB="${NEED_DISK_GB:-100}"      # GiB, as df -BG counts it: 3 bpw 48 GiB + FP8 table 48 GiB + drafter 1.1 GiB
LOCK_WAIT="${LOCK_WAIT:-30}"

# The 3 bpw checkpoint's BF16 n-gram table, alone in this shard (~95 GiB), is not downloaded.
SOURCE_TABLE_SHARD="model-00001-of-00011.safetensors"

LOGDIR="$SCRIPT_DIR/logs"
SERVE_LOG="${SERVE_LOG:-$LOGDIR/serve.log}"
LOCKFILE="$LOGDIR/lifecycle.lock"
LOCKPID="$LOGDIR/lifecycle.lock.pid"
CIDFILE="$LOGDIR/container.id"
LABEL="qwen38-recipe.checkout"

# Where our own health checks connect. A wildcard bind address is not a
# destination, so talk to the server over loopback in that case.
case "$HOST" in
    0.0.0.0|::|'[::]'|'') CLIENT_HOST=127.0.0.1 ;;
    *) CLIENT_HOST="$HOST" ;;
esac

export MODEL HOST PORT SERVED_MODEL_NAME
export SERVE_LOG READY_TIMEOUT BOOT_CHECK
export MODEL_ID="${MODEL_ID:-$SERVED_MODEL_NAME}"

banner() {
    printf '\n  \033[1;36m┌──────────────────────────────────────────────┐\033[0m\n'
    printf '  \033[1;36m│\033[0m  \033[1mQwen3.8-Flash-Next \033[0m  \033[2m·  %-14s\033[0m  \033[1;36m│\033[0m\n' "${1:-start.sh}"
    printf '  \033[1;36m└──────────────────────────────────────────────┘\033[0m\n\n'
}

# ------------------------------- locking -----------------------------------
# The flock on fd 9 is the authoritative owner; $LOCKPID is advisory
# diagnostics only and is never signalled.

# Lock $LOCKFILE on fd 9. Returns 0 when locked, 1 when another command holds it.
acquire_lock() {
    mkdir -p "$LOGDIR"
    exec 9>"$LOCKFILE"
    flock -n 9 && return 0
    exec 9>&-
    return 1
}

take_lock() {
    if ! acquire_lock; then
        local holder; holder="$(tr -d '[:space:]' 2>/dev/null <"$LOCKPID" || true)"
        die "this checkout is busy${holder:+ with pid $holder}. Wait for that command to finish, then try again."
    fi
    echo $$ >"$LOCKPID" 2>/dev/null || true
}
take_lock_for_stop() {
    mkdir -p "$LOGDIR"
    exec 9>"$LOCKFILE"
    if ! flock -w "$LOCK_WAIT" 9; then
        local holder; holder="$(tr -d '[:space:]' 2>/dev/null <"$LOCKPID" || true)"
        warn "gave up waiting ${LOCK_WAIT}s for this checkout${holder:+ (pid $holder has it)}."
        die "The server was left running. Try stop again once that command finishes."
    fi
    echo $$ >"$LOCKPID" 2>/dev/null || true
}

# ------------------------------ preflight ----------------------------------
need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "$1 not found — $2"
}

# preflight [--with-port]   -- the port is only checked when we are about to
# bind it, so `install` and `download` do not fail on a busy port.
preflight() {
    local check_port=0
    [ "${1:-}" = "--with-port" ] && check_port=1
    log "preflight (layout: $LAYOUT)"
    need_cmd git  "install it: apt-get install -y git"
    need_cmd curl "install it: apt-get install -y curl"
    [ "$BOOT_CHECK" = 0 ] || need_cmd python3 "install Python 3 for the boot check"
    preflight_docker

    if command -v nvidia-smi >/dev/null 2>&1; then
        # Read only. We never set power, persistence, clocks or fan state.
        local mems n small
        mems="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
        if [ -z "$mems" ]; then
            warn "nvidia-smi present but returned no GPUs — check the driver"
        else
            n="$(printf '%s\n' "$mems" | grep -c .)"
            small="$(printf '%s\n' "$mems" | awk -v m="$MIN_GPU_MIB" '$1+0 < m' | grep -c . || true)"
            [ "$n" -ge "$MIN_GPUS" ] || die "found $n GPU(s), need $MIN_GPUS"
            if [ "${small:-0}" -gt 0 ]; then
                die "$small GPU(s) report under $((MIN_GPU_MIB/1024)) GiB. This recipe needs a card of about 64 GB (the 3 bpw weights take ~48 GiB, plus the drafter and the KV cache). Nothing was downloaded. Set MIN_GPU_MIB to override this check."
            fi
            log "  GPUs: $n visible, smallest $(printf '%s\n' "$mems" | sort -n | head -1) MiB"
        fi
        # Read only. Tensor parallelism wants x16 links.
        local widths narrow
        widths="$(nvidia-smi --query-gpu=pcie.link.width.current --format=csv,noheader,nounits 2>/dev/null || true)"
        if [ -n "$widths" ]; then
            log "  PCIe link width: $(printf '%s' "$widths" | tr -d ' ' | paste -sd, -)"
            narrow="$(printf '%s\n' "$widths" | awk '$1+0 > 0 && $1+0 < 16' | grep -c . || true)"
            if [ "${narrow:-0}" -gt 0 ] && [ "$TP" -gt 1 ]; then
                warn "  $narrow card(s) are below x16 and TP=$TP. Every decode step runs small"
                warn "  collectives between the two cards, so expect less than the published numbers"
                warn "  on narrow links (they were measured on x16). Continuing anyway."
            fi
        fi
    else
        warn "nvidia-smi not found — skipping the GPU check"
    fi

    mkdir -p "$MODELS_DIR"
    local avail_gb; avail_gb="$(df -BG --output=avail "$MODELS_DIR" 2>/dev/null | tail -1 | tr -dc '0-9' || true)"
    if [ -n "${avail_gb:-}" ] && [ "$avail_gb" -gt 0 ]; then
        if [ "$avail_gb" -lt "$NEED_DISK_GB" ] && ! models_present; then
            die "only ${avail_gb} GiB free at $MODELS_DIR, need ~${NEED_DISK_GB} GiB for the checkpoints — free space or set MODELS_DIR"
        fi
        log "  disk: ${avail_gb} GiB free at $MODELS_DIR"
    fi

    if [ "$check_port" = 1 ]; then
        if ! port_busy; then
            log "  port $PORT: free"
        elif server_is_ours; then
            log "  port $PORT: in use by our server"
        elif [ -n "${DRY:-}" ]; then
            warn "  port $PORT is in use by something else (not fatal under DRY)"
        else
            die "port $PORT is already taken by another process. Set PORT to a free one, or stop whatever holds it."
        fi
    fi
}

# Docker must be installed and usable by this user, with the NVIDIA Container
# Toolkit for --gpus. Under DRY a missing or unusable docker is only a warning.
preflight_docker() {
    local how="install Docker Engine and the NVIDIA Container Toolkit (see \"One-time host setup\" in the README)"
    if ! command -v docker >/dev/null 2>&1; then
        [ -n "${DRY:-}" ] && { warn "docker not found — $how"; return 0; }
        die "docker not found — $how"
    fi
    if ! docker info >/dev/null 2>&1; then
        [ -n "${DRY:-}" ] && { warn "docker is installed but this user cannot reach the Docker daemon"; return 0; }
        die "docker is installed but this user cannot reach the Docker daemon (permission denied, or the daemon is not running). Fix that: $how."
    fi
    if docker info --format '{{range $k, $v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null | grep -qw nvidia; then
        log "  docker: ok, NVIDIA runtime present"
    else
        warn "  docker: no 'nvidia' runtime registered — without the NVIDIA Container Toolkit, --gpus fails"
    fi
}

port_busy() {
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "sport = :$PORT" 2>/dev/null | tail -n +2 | grep -q .
    else
        curl -fsS -m 2 "http://$CLIENT_HOST:$PORT/health" >/dev/null 2>&1
    fi
}

# ------------------------------- install -----------------------------------
# Container: the image is the whole engine, so "installed" means the image
# named by IMAGE (a digest) is present locally. A release that moves the pin
# names a new image, which is pulled on the next start or update.
container_image_present() { docker image inspect "$IMAGE" >/dev/null 2>&1; }

install_done() { container_image_present; }

do_install() {
    if container_image_present && [ "${FORCE_INSTALL:-0}" != "1" ]; then
        log "install: image present — skipping (FORCE_INSTALL=1 to pull again)"
        log "  $IMAGE"
        return 0
    fi
    log "install: docker pull (the engine image, several GB)"
    log "  $IMAGE"
    docker pull "$IMAGE"
    log "install: done"
}

# ------------------------------ download -----------------------------------
# A checkpoint is complete when docker/prepare.sh says so: the config, every
# shard its index names, and no partial download left behind. The same check
# decides for Docker Compose.
model_present() { bash "$SCRIPT_DIR/docker/prepare.sh" check "$1"; }
models_present() { model_present "$MODEL" && { [ "$SPEC_MODE" != dflash ] || model_present "$DFLASH_MODEL"; }; }

# hf_fetch_args <repo> <revision> <local dir> [hf download filters...]: as hf_download,
# with --include / --exclude patterns for the checkpoint parts this recipe uses.

# hf_download <repo> <revision> <local dir>: with the hf client in the engine
# image, run as this user so the files are yours. An interrupted download
# resumes.
hf_download() {
    local repo="$1" revision="$2" dir="$3"; shift 3
    mkdir -p "$dir"
    container_image_present || die "the engine image is not here yet — run ./start.sh install first"
    docker run --rm --user "$(id -u):$(id -g)" -e HF_HOME=/tmp/hf \
        ${HF_TOKEN:+-e HF_TOKEN} -v "$dir:/download" \
        -v "$SCRIPT_DIR/docker/prepare.sh:/recipe/prepare.sh:ro" \
        --entrypoint /bin/bash "$IMAGE" /recipe/prepare.sh fetch "$repo" /download "$revision" "$@"
}

# build_model: the served checkpoint from the two downloads (docker/build_checkpoint.py,
# hard links, so the three folders must share MODELS_DIR's filesystem).
build_model() {
    local models d; models="$(cd "$MODELS_DIR" && pwd -P)"
    for d in "$SOURCE_MODEL" "$TABLE_MODEL" "$MODEL"; do
        [ "$(cd "$(dirname "$d")" 2>/dev/null && pwd -P)" = "$models" ] ||
            die "SOURCE_MODEL, TABLE_MODEL and MODEL must be folders directly in MODELS_DIR (hard links): $d"
    done
    docker run --rm --user "$(id -u):$(id -g)" ${BUILD_COPY:+-e BUILD_COPY} -v "$models:/models" \
        -v "$SCRIPT_DIR/docker/build_checkpoint.py:/recipe/build_checkpoint.py:ro" \
        --entrypoint /opt/venv/bin/python "$IMAGE" /recipe/build_checkpoint.py \
        "/models/$(basename "$SOURCE_MODEL")" "/models/$(basename "$TABLE_MODEL")" "/models/$(basename "$MODEL")"
}

do_download() {
    mkdir -p "$MODELS_DIR"

    if model_present "$MODEL" && [ "${REFRESH_WEIGHTS:-0}" != "1" ]; then
        log "download: $(basename "$MODEL") complete — skipping"
    else
        log "download: $TARGET_REPO @ $TARGET_REVISION without its BF16 n-gram table (~48 GiB)"
        hf_download "$TARGET_REPO" "$TARGET_REVISION" "$SOURCE_MODEL" --exclude "$SOURCE_TABLE_SHARD"
        log "download: the FP8 n-gram table of $TABLE_REPO @ $TABLE_REVISION (~48 GiB)"
        hf_download "$TABLE_REPO" "$TABLE_REVISION" "$TABLE_MODEL" \
            --include "model-plefp8-*.safetensors" --include "model.safetensors.index.json"
        log "download: building $(basename "$MODEL") from them (hard links)"
        build_model
    fi
    if [ "$SPEC_MODE" = dflash ]; then
        if model_present "$DFLASH_MODEL" && [ "${REFRESH_WEIGHTS:-0}" != "1" ]; then
            log "download: $(basename "$DFLASH_MODEL") complete — skipping"
        else
            log "download: $DRAFTER_REPO @ $DRAFTER_REVISION (~1.1 GiB)"
            hf_download "$DRAFTER_REPO" "$DRAFTER_REVISION" "$DFLASH_MODEL"
        fi
    fi
    log "download: done"
}

# ------------------------------- process -----------------------------------
server_is_ours() { container_is_ours; }

# ------------------------------ container ----------------------------------
# The container this checkout started is recorded by id in $CIDFILE and carries
# the label $LABEL=<this checkout>. Only a container with both is ever stopped
# or removed; nothing is matched by name alone.
read_cid() { [ -f "$CIDFILE" ] && tr -dc '0-9a-f' <"$CIDFILE" || true; }

# container_state <id>: "<running> <label>" (true/false), or nothing.
container_state() {
    docker container inspect -f "{{.State.Running}} {{index .Config.Labels \"$LABEL\"}}" "$1" 2>/dev/null || true
}
container_is_ours() {
    local id; id="$(read_cid)"
    [ -n "$id" ] || return 1
    [ "$(container_state "$id")" = "true $SCRIPT_DIR" ]
}

# Liveness while waiting for a launch: the container counts as gone only when
# two ownership checks in a row fail, so one failed docker query (an empty
# answer) does not stop a server that is still loading.
container_alive() {
    container_is_ours && return 0
    sleep 2
    container_is_ours
}

# Address:port on the host the API is published on (IPv6 in brackets).
publish_addr() {
    case "$HOST" in
        \[*) printf '%s:%s' "$HOST" "$PORT" ;;
        *:*) printf '[%s]:%s' "$HOST" "$PORT" ;;
        *) printf '%s:%s' "$HOST" "$PORT" ;;
    esac
}

# The environment the container's serve.sh gets: the in-container paths, then
# every setting of yours that serve.sh reads. serve.sh (mounted from this
# checkout) turns them into the server's flags.
CONTAINER_KEYS="P2P P2P_PREBUILT P2P_CHECK_CACHE LAYOUT CUDA_VISIBLE_DEVICES MAX_LEN MAX_SEQS MAX_BATCHED GPU_UTIL
SPEC_N SSM_DTYPE BLOCK_SIZE RECOVER_SSM PLE_SHARED_DIR PLE_SHARED_NAME LOCAL_ARGMAX SPEC_SCHEDULE GRAPH_MODE PLE_PLACEMENT FIRST_LIGHT REASONING_PARSER TOOL_PARSER EXTRA_ARGS KV_OFFLOAD_GB
SERVED_MODEL_NAME API_KEY"
container_env() {
    local k
    echo "VENV=/opt/venv"
    echo "MODEL=/models/$(basename "$MODEL")"
    echo "CUDA_HOME=/usr/local/cuda"
    echo "HOST=0.0.0.0"
    echo "PORT=8000"
    echo "BOOT_CHECK=$BOOT_CHECK"
    echo "RECIPE_BOOT_CHECK_OWNER=start"
    # The server runs as you, not root: every cache it writes (Triton,
    # FlashInfer, TileLang, torch, vLLM, Hugging Face) goes under the mounted
    # cache directory, owned by you. USER and LOGNAME give Python a user name
    # for a uid the image does not know.
    echo "PYTHONDONTWRITEBYTECODE=1"
    echo "RECIPE_CACHE_ROOT=/cache"
    echo "TMPDIR=/cache/tmp"
    echo "HOME=/cache"
    echo "USER=$(id -un)"
    echo "LOGNAME=$(id -un)"
    echo "XDG_CACHE_HOME=/cache/.cache"
    echo "HF_HOME=/cache/huggingface"
    echo "FLASHINFER_WORKSPACE_BASE=/cache"
    echo "TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor"
    echo "TRITON_CACHE_DIR=/cache/.triton/cache"
    echo "TORCH_EXTENSIONS_DIR=/cache/.cache/torch_extensions"
    echo "VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR=/cache/.cache/vllm/custom_all_reduce_flags"
    echo "VLLM_CACHE_ROOT=/cache/.cache/vllm"
    echo "TILELANG_CACHE_DIR=/cache/.tilelang/cache"
    echo "TILELANG_TMP_DIR=/cache/.tilelang/cache/tmp"
    echo "CUDA_CACHE_PATH=/cache/.nv/ComputeCache"
    echo "VLLM_IMAGE_CACHE_SEED=${VLLM_IMAGE_CACHE_SEED:-0}"
    # The disk tier directory is mounted at /kvtier and the drafter at /drafter (container_args).
    [ -n "$KV_OFFLOAD_DIR" ] && echo "KV_OFFLOAD_DIR=/kvtier"
    [ "$SPEC_MODE" = dflash ] && echo "DFLASH_MODEL=/drafter"
    for k in $CONTAINER_KEYS; do
        [ -n "${!k+x}" ] && printf '%s=%s\n' "$k" "${!k}"
    done
    env | LC_ALL=C sort | grep -E '^(VLLM_|NCCL_|PYTORCH_CUDA_ALLOC_CONF=)' \
        | grep -vE '^VLLM_(COMMIT|REPO|BRANCH|SRC|USE_PRECOMPILED|PRECOMPILED_[A-Z_]*|API_KEY)=' || true
}

# The docker run arguments, into the array CRUN.
container_args() {
    local envfile="$1"
    CRUN=(docker run -d --name "$CONTAINER_NAME" --label "$LABEL=$SCRIPT_DIR"
        --user "$(id -u):$(id -g)" --workdir /cache
        --gpus all --shm-size "$SHM_SIZE"
        -p "$(publish_addr):8000"
        --env-file "$envfile"
        -v "$SCRIPT_DIR/serve.sh:/recipe/serve.sh:ro"
        -v "$SCRIPT_DIR/boot_check.py:/recipe/boot_check.py:ro"
        -v "$SCRIPT_DIR/p2p_check.py:/recipe/p2p_check.py:ro"
        -v "$SCRIPT_DIR/p2p_check.sh:/recipe/p2p_check.sh:ro"
        -v "$SCRIPT_DIR/p2p_probe.cu:/recipe/p2p_probe.cu:ro"
        -v "$MODEL:/models/$(basename "$MODEL"):ro")
    if [ -n "$KV_OFFLOAD_DIR" ]; then
        [ -n "${DRY:-}" ] || mkdir -p "$KV_OFFLOAD_DIR"
        CRUN+=(-v "$KV_OFFLOAD_DIR:/kvtier")
    fi
    if [ "$SPEC_MODE" = dflash ]; then
        CRUN+=(-v "$DFLASH_MODEL:/drafter:ro"
            -v "$SCRIPT_DIR/patches/qwen4_exp_nvidia_model.py:/opt/vllm-src/vllm/models/qwen4_exp/nvidia/model.py:ro"
            -v "$SCRIPT_DIR/patches/models_config.py:/opt/vllm-src/vllm/model_executor/models/config.py:ro")
    fi
    # Every mode: RecoverSSM (inert with RECOVER_SSM=0), the n-gram table loader
    # (PLE_SHARED_DIR), the AutoRound bits fix and the chunked copy-on-write (patches/README.md).
    CRUN+=(-v "$SCRIPT_DIR/patches/gdn_recover:/opt/vllm-src/vllm/sm80/kernels/gdn:ro"
        -v "$SCRIPT_DIR/patches/mamba_gdn_qwen_gdn_linear_attn.py:/opt/vllm-src/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:ro"
        -v "$SCRIPT_DIR/patches/qwen4_exp_nvidia_model_state.py:/opt/vllm-src/vllm/models/qwen4_exp/nvidia/model_state.py:ro"
        -v "$SCRIPT_DIR/patches/qwen4_exp_nvidia_ple_layer.py:/opt/vllm-src/vllm/models/qwen4_exp/nvidia/ple_layer.py:ro"
        -v "$SCRIPT_DIR/patches/qwen4_exp_common_ngram_embedding.py:/opt/vllm-src/vllm/models/qwen4_exp/common/ngram_embedding.py:ro"
        -v "$SCRIPT_DIR/patches/inc_config_parser.py:/opt/vllm-src/vllm/model_executor/layers/quantization/inc/config_parser.py:ro"
        -v "$SCRIPT_DIR/patches/v1_worker_utils.py:/opt/vllm-src/vllm/v1/worker/utils.py:ro")
    if [ -n "$PLE_SHARED_DIR" ]; then
        [ -n "${DRY:-}" ] || mkdir -p "$PLE_SHARED_DIR"
        CRUN+=(-v "$PLE_SHARED_DIR:$PLE_SHARED_DIR")
    fi
    CRUN+=(-v "$CACHE_DIR:/cache"
        --entrypoint /bin/bash "$IMAGE" /recipe/serve.sh "$SPEC_MODE")
}

container_launch() {
    if [ -n "${DRY:-}" ]; then
        container_args "<env file>"
        log "launch: DRY=1, a real start would run:"
        printf '  '; printf '%q ' "${CRUN[@]}"; printf '\n'
        log "  with this environment in the container:"
        container_env | sed -e 's/^\(API_KEY\)=.*/\1=(set, not shown)/' -e 's/^/    /'
        log "  and serve.sh inside it would set:"
        env $(container_env | grep -E '^(VENV|MODEL|CUDA_HOME|HOST|PORT)=') \
            DRY=1 "$SCRIPT_DIR/serve.sh" "$SPEC_MODE" | sed 's/^/    /'
        return 0
    fi
    if container_is_ours; then
        log "launch: already running (container $(read_cid | cut -c1-12)) — skipping"
        return 0
    fi
    # A stopped container of ours is removed; any other container with the
    # same name is left alone and we stop.
    local id st
    id="$(read_cid)"
    if [ -n "$id" ]; then
        st="$(container_state "$id")"
        [ "${st#* }" = "$SCRIPT_DIR" ] && docker rm -f "$id" >/dev/null 2>&1 || true
        rm -f "$CIDFILE"
    fi
    st="$(container_state "$CONTAINER_NAME")"
    if [ -n "$st" ]; then
        if [ "$st" = "false $SCRIPT_DIR" ]; then
            docker rm "$CONTAINER_NAME" >/dev/null
        elif [ "$st" = "true $SCRIPT_DIR" ]; then
            docker container inspect -f '{{.Id}}' "$CONTAINER_NAME" >"$CIDFILE"
            log "launch: already running (container $CONTAINER_NAME) — skipping"
            return 0
        else
            die "a container named $CONTAINER_NAME exists and was not started from this checkout. Set CONTAINER_NAME to another name."
        fi
    fi
    mkdir -p "$LOGDIR" "$CACHE_DIR"
    [ -w "$CACHE_DIR" ] || die "$CACHE_DIR is not writable by $(id -un); the server runs as you and keeps its compile caches there. Fix its ownership or set CACHE_DIR."
    log "launch: container, $SPEC_MODE, layout $LAYOUT (TP=$TP PP=$PP), ctx ${MAX_LEN:-262144}, $(publish_addr)"
    log "  image: $IMAGE"
    log "  log: $SERVE_LOG"
    BOOT_LOG_OFFSET=0
    [ ! -f "$SERVE_LOG" ] || BOOT_LOG_OFFSET="$(wc -c < "$SERVE_LOG")"
    local envfile="$LOGDIR/container.env.run"
    ( umask 077; container_env >"$envfile" )
    container_args "$envfile"
    id="$("${CRUN[@]}")" || { rm -f "$envfile"; die "docker run failed"; }
    rm -f "$envfile"
    echo "$id" >"$CIDFILE"
    LAUNCHED=1
    log "  container: ${id:0:12} ($CONTAINER_NAME)"
    # Copy the container's output into $SERVE_LOG; the follower ends when the
    # container does. 9>&- : the follower must not inherit the lifecycle lock
    # (fd 9), or it would hold the lock for the container's whole life and
    # block every later stop, restart and update.
    setsid nohup docker logs -f "$id" 9>&- >>"$SERVE_LOG" 2>&1 </dev/null &
}

container_stop() {
    local id st; id="$(read_cid)"
    if [ -z "$id" ]; then
        log "stop: no container started from this checkout"
        return 0
    fi
    st="$(container_state "$id")"
    if [ -z "$st" ]; then
        log "stop: container ${id:0:12} is gone — clearing the id file"
        rm -f "$CIDFILE"; return 0
    fi
    if [ "${st#* }" != "$SCRIPT_DIR" ]; then
        warn "stop: container ${id:0:12} was not started from this checkout — leaving it alone and clearing the id file"
        rm -f "$CIDFILE"; return 0
    fi
    if [ "${st%% *}" = true ]; then
        log "stop: docker stop ${id:0:12} (up to ${STOP_TIMEOUT}s)"
        docker stop -t "$STOP_TIMEOUT" "$id" >/dev/null || true
    fi
    docker rm "$id" >/dev/null 2>&1 || true
    rm -f "$CIDFILE"
    log "stop: done"
}

health_ok() { curl -fsS -m "${1:-5}" "http://$CLIENT_HOST:$PORT/health" >/dev/null 2>&1; }

# First "id" in /v1/models. A plain greedy sed would pick up the permission id
# further down the same line, so match the field and take the first one.
# /v1 needs the key when API_KEY is set; /health never does.
served_id() {
    local auth=()
    [ -n "${API_KEY:-}" ] && auth=(-H "Authorization: Bearer $API_KEY")
    curl -fsS -m 5 ${auth[@]+"${auth[@]}"} "http://$CLIENT_HOST:$PORT/v1/models" 2>/dev/null \
        | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4
}

kv_line() {
    [ -r "$SERVE_LOG" ] || return 1
    local l; l="$(grep -h 'GPU KV cache size' "$SERVE_LOG" 2>/dev/null | tail -1 || true)"
    [ -n "$l" ] || return 1
    printf '%s\n' "${l#*] }"
}

wait_ready() {
    [ -z "${DRY:-}" ] || return 0
    local url="http://$CLIENT_HOST:$PORT/health" elapsed=0 started=$SECONDS checker status
    log "waiting for $url (weight load and graph capture on a 125B MoE are slow; timeout ${READY_TIMEOUT}s)"
    local log_args=(--log "$SERVE_LOG" --log-offset "${BOOT_LOG_OFFSET:-0}" --container)
    while true; do
        elapsed=$((SECONDS - started))
        [ "$elapsed" -lt "$READY_TIMEOUT" ] || break
        if [ "$LAUNCHED" = 1 ] && [ "$BOOT_CHECK" != 0 ] && [ -f "$SERVE_LOG" ]; then
            if ! python3 "$SCRIPT_DIR/boot_check.py" "${log_args[@]}" --log-only; then
                warn "[boot-check] FAIL: stopping this checkout's failed launch"
                do_stop
                return 1
            fi
        fi
        if health_ok "$((READY_TIMEOUT - elapsed < 5 ? READY_TIMEOUT - elapsed : 5))"; then
            log "healthy after ${elapsed}s"
            if [ "$LAUNCHED" = 1 ]; then
                if [ "$BOOT_CHECK" = 0 ]; then
                    log "[boot-check] disabled (BOOT_CHECK=0)"
                else
                    # Health can briefly precede the metrics/API frontend. Keep
                    # connection readiness inside the original launch budget.
                    python3 "$SCRIPT_DIR/boot_check.py" --wait-ready \
                        --ready-timeout "$((READY_TIMEOUT - (SECONDS - started)))" \
                        --base "http://$CLIENT_HOST:$PORT" --model "$SERVED_MODEL_NAME" "${log_args[@]}" &
                    checker=$!
                    trap 'kill "$checker" 2>/dev/null || true; wait "$checker" 2>/dev/null || true; do_stop; exit 1' INT TERM
                    while kill -0 "$checker" 2>/dev/null; do
                        if ! container_alive; then
                            kill "$checker" 2>/dev/null || true
                            break
                        fi
                        sleep 1
                    done
                    status=0
                    wait "$checker" || status=$?
                    trap - INT TERM
                    if [ "$status" != 0 ]; then
                        warn "[boot-check] FAIL: stopping this checkout's failed launch"
                        do_stop
                        return 1
                    fi
                fi
            else
                log "[boot-check] existing server: no new requests"
            fi
            local kv; kv="$(kv_line || true)"
            [ -n "$kv" ] && log "  $kv"
            log "  model: $(served_id)"
            log "  API:   http://$HOST:$PORT/v1${API_KEY:+ (API key required)}"
            return 0
        fi
        if ! container_alive; then
            warn "the server exited after ${elapsed}s — last 40 lines:"
            sleep 1
            tail -40 "$SERVE_LOG" >&2 || true
            return 1
        fi
        sleep 5; elapsed=$((SECONDS - started))
    done
    warn "not healthy after ${READY_TIMEOUT}s; it may still be loading — ./start.sh logs"
    return 1
}

# --------------------------------- stop ------------------------------------
# Stops the container this checkout started, and nothing else.
do_stop() { container_stop; }

# -------------------------------- status -----------------------------------
do_status() {
    log "layout $LAYOUT (TP=$TP PP=$PP)"
    local id st; id="$(read_cid)"
    if [ -z "$id" ]; then
        log "process: no container started from this checkout"
    else
        st="$(container_state "$id")"
        case "$st" in
            "true $SCRIPT_DIR") log "process: container running, ${id:0:12} ($CONTAINER_NAME)" ;;
            "false $SCRIPT_DIR") log "process: container ${id:0:12} has stopped — ./start.sh logs" ;;
            "") log "process: container ${id:0:12} is gone (stale id file)" ;;
            *) log "process: container ${id:0:12} is not ours (stale id file)" ;;
        esac
    fi
    if health_ok; then
        log "API:     healthy — http://$HOST:$PORT/v1${API_KEY:+ (API key required)}"
        log "model:   $(served_id)"
    else
        log "API:     not responding on :$PORT"
    fi
    local kv; kv="$(kv_line || true)"
    if [ -n "$kv" ]; then log "KV:      $kv"; else log "KV:      (no KV line in $SERVE_LOG)"; fi
    log "install: $(install_done && echo "image present" || echo 'image missing or pin moved — ./start.sh install'), $IMAGE"
    log "models:  $(models_present && echo ok || echo 'missing or incomplete — ./start.sh download')"
}

# --------------------------------- logs ------------------------------------
do_logs() {
    [ -f "$SERVE_LOG" ] || die "no log at $SERVE_LOG — nothing has been launched yet"
    log "following $SERVE_LOG"
    trap '' INT
    tail -n 100 -f "$SERVE_LOG" || true
    trap - INT
}

# -------------------------------- update -----------------------------------
do_update() {
    if [ -d "$SCRIPT_DIR/.git" ]; then
        local before after
        before="$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo none)"
        log "update: git pull"
        git -C "$SCRIPT_DIR" pull --ff-only
        after="$(git -C "$SCRIPT_DIR" rev-parse HEAD)"
        if [ "$before" = "$after" ]; then
            log "  already up to date"
        else
            log "  ${before:0:10} -> ${after:0:10}"
            # The pull may have moved the pin, which this shell read before the
            # pull. Re-exec the new start.sh so we act on the new value. exec
            # keeps our PID, so fd 9 keeps holding the lifecycle lock.
            if [ -z "${_QWEN38_REEXEC:-}" ]; then
                log "  re-executing the updated start.sh"
                export _QWEN38_REEXEC=1
                exec "$SCRIPT_DIR/start.sh" update
            fi
        fi
    else
        warn "update: not a git checkout — skipping the pull"
    fi
    if install_done; then
        log "update: engine unchanged at $IMAGE — no pull"
    else
        log "update: engine moved to $IMAGE — pulling it"
        do_install
    fi
    do_stop
    start_unlocked
}

# --------------------------------- start -----------------------------------
# DRY: report what a real start would do, then print the server environment
# and command. Nothing is installed, downloaded or launched.
dry_start() {
    preflight --with-port
    if command -v docker >/dev/null 2>&1 && install_done; then
        log "install: image present — a real start would skip it"
    else
        log "install: a real start would pull $IMAGE"
    fi
    if models_present; then
        log "download: checkpoint present — a real start would skip it"
    else
        log "download: a real start would fetch $TARGET_REPO @ $TARGET_REVISION into $MODELS_DIR"
    fi
    container_launch
}

# DRY stop/update: the lock is taken for real (that is what they check), and
# nothing is signalled, pulled or installed.
dry_stop() {
    local id; id="$(read_cid)"
    if [ -n "$id" ]; then
        log "stop: DRY=1, a real stop would docker stop container ${id:0:12} if it is ours"
    else
        log "stop: DRY=1, nothing started from this checkout is running"
    fi
}
dry_update() {
    log "update: DRY=1, a real update would git pull --ff-only, pull the image if the pin moved, then restart"
    dry_stop
    dry_start
}

start_unlocked() {
    if [ -n "${DRY:-}" ]; then dry_start; return 0; fi
    preflight --with-port
    do_install
    if models_present; then
        log "download: checkpoint present — skipping"
    else
        do_download
    fi
    container_launch
    [ -n "${DRY:-}" ] || wait_ready
}

# ---------------------------------- main -----------------------------------
main() {
    local cmd="${1:-start}"
    case "$cmd" in
        -h|--help|help) usage; exit 0 ;;
    esac
    banner "$cmd"
    case "$cmd" in
        start)    if [ -n "${DRY:-}" ]; then dry_start; else take_lock; start_unlocked; fi ;;
        install)  take_lock; preflight; do_install ;;
        download) take_lock; preflight; do_download ;;
        stop)     take_lock_for_stop; if [ -n "${DRY:-}" ]; then dry_stop; else do_stop; fi ;;
        restart)  take_lock; if [ -n "${DRY:-}" ]; then dry_stop; else do_stop; fi; start_unlocked ;;
        update)   take_lock; if [ -n "${DRY:-}" ]; then dry_update; else do_update; fi ;;
        status)   do_status ;;
        logs)     do_logs ;;
        smoke)    HOST="$CLIENT_HOST" exec "$SCRIPT_DIR/smoke.sh" ;;
        *) warn "unknown command: $cmd"; usage; exit 1 ;;
    esac
}

main "$@"
