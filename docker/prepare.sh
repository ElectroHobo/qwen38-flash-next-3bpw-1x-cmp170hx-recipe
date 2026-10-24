#!/usr/bin/env bash
# prepare.sh: storage and checkpoint for the server, run inside the engine
# image (this repository mounts it at /recipe/prepare.sh).
#
#   prepare.sh prepare           Docker Compose's prepare service. Started as
#                                root: makes the cache and model directories
#                                writable by the serving user, then, as that
#                                user, downloads the checkpoint parts and builds
#                                the served checkpoint unless it is complete.
#                                Hugging Face resumes a partial download; a
#                                complete checkpoint is skipped without
#                                contacting the Hub.
#   prepare.sh fetch REPO DIR [REVISION [hf filters...]]
#                                download one checkpoint, or the files the
#                                --include / --exclude filters select, into DIR
#                                as the current user (./start.sh download).
#   prepare.sh serve ARGS...     Docker Compose's server: runs serve.sh ARGS as
#                                the owner of /cache, the serving user chosen by
#                                prepare, never as root.
#   prepare.sh check DIR         exit 0 when DIR holds a complete checkpoint;
#                                runs on the host too (./start.sh).
#
# The serving user is the owner of the checkout (mounted at /checkout), or
# HOST_UID and HOST_GID when set, so `docker compose up` needs no settings for
# any user. ./start.sh runs the server as the user who runs it.
#
# Ownership: Docker creates a missing bind-mount directory as root. The cache
# is this recipe's own, so everything in it is given to the serving user. In
# the model directory only directories that are owned by root and hold nothing
# else (what Docker or this script just created) change owner; existing
# checkpoints and other files are never changed or removed.
set -euo pipefail

# The checkpoint parts and the DFlash 2 drafter at the revisions the release was
# validated with: klee100's 3 bpw build (the MTP head is inside it) without its BF16
# n-gram table shard, and halt95's FP8 table shards; docker/build_checkpoint.py makes
# the served checkpoint of them.
TARGET_REPO="${TARGET_REPO:-klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP}"
TARGET_REVISION="${TARGET_REVISION:-ce0e0b94083895bd836b916f29bf105c40a8162a}"
TABLE_REPO="${TABLE_REPO:-halt95/Qwen3.8-Flash-Next-W4A16-Merlin}"
TABLE_REVISION="${TABLE_REVISION:-cec53e9f90e5995b81691604ec0b32191debf797}"
SERVED_DIR_NAME=Qwen3.8-Flash-Next-3bpw-fp8ple
SOURCE_TABLE_SHARD=model-00001-of-00011.safetensors
DRAFTER_REPO="${DRAFTER_REPO:-incoai/Qwen3.8-Flash-Next-DFlash2}"
DRAFTER_REVISION="${DRAFTER_REVISION:-a22a1744e85c17364dcc54a6a420af7135d4c7f6}"
NEED_DISK_GB="${NEED_DISK_GB:-100}"
MODELS=/models
CACHE=/cache
CHECKOUT=/checkout
HF=/opt/venv/bin/hf

say() { printf 'prepare: %s\n' "$*"; }
fail() { printf 'prepare: %s\n' "$*" >&2; exit 1; }

# complete <dir>: the config, at least one weight file, every shard the
# index names, and no partial download left by an interrupted fetch.
complete() {
    local dir="$1" index shard
    [ -f "$dir/config.json" ] || return 1
    compgen -G "$dir/*.safetensors" >/dev/null || return 1
    if [ -d "$dir/.cache/huggingface/download" ] \
        && [ -n "$(find "$dir/.cache/huggingface/download" -name '*.incomplete' -print -quit)" ]; then
        return 1
    fi
    for index in "$dir"/*.safetensors.index.json; do
        [ -f "$index" ] || continue
        while IFS= read -r shard; do
            [ -f "$dir/$shard" ] || return 1
        done < <(grep -o '"[^"]*\.safetensors"' "$index" | tr -d '"' | sort -u)
    done
    return 0
}

fetch() {
    local repo="$1" dir="$2" revision="${3:-}"; shift 2; [ "$#" -gt 0 ] && shift
    [ -x "$HF" ] || fail "no Hugging Face client at $HF"
    say "downloading $repo${revision:+ @ $revision}${*:+ ($*)} (resumes a partial download)"
    "$HF" download "$repo" ${revision:+--revision "$revision"} --local-dir "$dir" "$@"
    if [ "$#" -gt 0 ]; then
        # A filtered download is checked when the served checkpoint is built from it.
        say "$repo: fetched"
        return 0
    fi
    complete "$dir" || fail "$repo: the download ended but $dir is still incomplete; run the same command again to resume"
    say "$repo: complete"
}

# claim <dir> [name...]: give <dir> to the serving user when root owns it and
# it holds nothing but the listed names.
claim() {
    local dir="$1" entry base; shift
    [ "$(stat -c %u "$dir")" = 0 ] || return 0
    for entry in "$dir"/* "$dir"/.[!.]* "$dir"/..?*; do
        [ -e "$entry" ] || [ -L "$entry" ] || continue
        base="${entry##*/}"
        case " $* " in *" $base "*) ;; *) return 0 ;; esac
    done
    chown "$SERVE_UID:$SERVE_GID" "$dir"
    say "$dir: now owned by $SERVE_UID:$SERVE_GID"
}

serving_ids() {
    if [ -n "${HOST_UID:-}" ]; then
        SERVE_UID="$HOST_UID"
        SERVE_GID="${HOST_GID:-$HOST_UID}"
    else
        [ -d "$CHECKOUT" ] || fail "mount the checkout at $CHECKOUT, or set HOST_UID and HOST_GID"
        SERVE_UID="$(stat -c %u "$CHECKOUT")"
        SERVE_GID="$(stat -c %g "$CHECKOUT")"
    fi
    [[ "$SERVE_UID" =~ ^[0-9]+$ && "$SERVE_GID" =~ ^[0-9]+$ ]] \
        || fail "HOST_UID and HOST_GID must be numeric ids"
}

# as_serving_user CMD...: from root, run CMD as the serving user in place of
# this process. Returns when the serving user is root itself.
as_serving_user() {
    [ "$SERVE_UID" != 0 ] || return 0
    exec setpriv --reuid="$SERVE_UID" --regid="$SERVE_GID" --clear-groups -- "$@"
}

prepare() {
    local target="$MODELS/$SERVED_DIR_NAME" source="$MODELS/${TARGET_REPO##*/}"
    local table="$MODELS/${TABLE_REPO##*/}" drafter="$MODELS/${DRAFTER_REPO##*/}"
    if [ "$(id -u)" = 0 ]; then
        serving_ids
        say "serving user: $SERVE_UID:$SERVE_GID"
        mkdir -p "$CACHE" "$target" "$source" "$table" "$drafter"
        find "$CACHE" -xdev \( ! -uid "$SERVE_UID" -o ! -gid "$SERVE_GID" \) \
            -exec chown -h "$SERVE_UID:$SERVE_GID" {} +
        claim "$target"
        claim "$source"
        claim "$table"
        claim "$drafter"
        claim "$MODELS" "${target##*/}" "${source##*/}" "${table##*/}" "${drafter##*/}"
        as_serving_user /bin/bash "$0" prepare
    fi
    local used avail
    if complete "$drafter"; then
        say "$drafter: complete, skipping"
    else
        [ -w "$drafter" ] || fail "$drafter is not writable by uid $(id -u)"
        fetch "$DRAFTER_REPO" "$drafter" "$DRAFTER_REVISION"
    fi
    if complete "$target"; then
        say "$target: complete, skipping"
        say "checkpoint ready"
        return 0
    fi
    for dir in "$target" "$source" "$table"; do
        [ -w "$dir" ] || fail "$dir is not writable by uid $(id -u); give it to that user, or set HOST_UID and HOST_GID to its owner"
    done
    used="$(du -s -BG "$source" "$table" | awk '{s += $1} END {print s + 0}')"
    avail="$(df -BG --output=avail "$MODELS" | tail -1 | tr -dc '0-9')"
    if [ -n "$avail" ] && [ "$((avail + used))" -lt "$NEED_DISK_GB" ]; then
        fail "only ${avail} GiB free under the model directory and ${used} GiB already downloaded; the checkpoint needs ~${NEED_DISK_GB} GiB"
    fi
    fetch "$TARGET_REPO" "$source" "$TARGET_REVISION" --exclude "$SOURCE_TABLE_SHARD"
    fetch "$TABLE_REPO" "$table" "$TABLE_REVISION" \
        --include "model-plefp8-*.safetensors" --include "model.safetensors.index.json"
    /opt/venv/bin/python "$CHECKOUT/docker/build_checkpoint.py" "$source" "$table" "$target"
    complete "$target" || fail "$target: built but incomplete"
    say "checkpoint ready"
}

serve() {
    if [ "$(id -u)" = 0 ]; then
        SERVE_UID="$(stat -c %u "$CACHE")"
        SERVE_GID="$(stat -c %g "$CACHE")"
        as_serving_user /bin/bash /recipe/serve.sh "$@"
    fi
    exec /bin/bash /recipe/serve.sh "$@"
}

case "${1:-}" in
    prepare) prepare ;;
    fetch) [ "$#" -ge 3 ] || fail "usage: prepare.sh fetch REPO DIR [REVISION [hf filters...]]"; shift; fetch "$@" ;;
    serve) shift; serve "$@" ;;
    check) [ "$#" = 2 ] || fail "usage: prepare.sh check DIR"; complete "$2" ;;
    *) fail "usage: prepare.sh prepare | fetch REPO DIR [REVISION [hf filters...]] | serve ARGS... | check DIR" ;;
esac
