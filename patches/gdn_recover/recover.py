# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Added 2026-10-23 by ElectroHobo for the qwen38-flash-next-3bpw-1x-cmp170hx-recipe:
# RecoverSSM for Qwen's GDN layers (patches/README.md).
# Written for Morrowmake/vllm-cmp170hx at 18305948acba3f4c4d912cdd56f2202d2ebd94e8.
"""RecoverSSM for Qwen's GDN layers on the fused sm_80 spec-decode path.

GLM-5 Next's KDA layers have this as vllm/sm80/kernels/kda/recover.py; this is
the same design for QwenGatedDeltaNetAttention (Qwen4Exp), opt-in with
VLLM_GDN_RECOVER=1.

Speculative verification normally keeps one recurrent state per draft
position per request (`num_speculative_blocks = num_spec` extra state pages per
GDN layer), so that the state at the accepted position can be picked up next
step. On one card that reservation, not the attention KV, decides how many
requests fit. With RecoverSSM:

  * the KV spec reserves no draft-position state pages
    (`num_speculative_blocks = 0`); the attention block size is unchanged;
  * the verify (gdn_recover_verify.cu, the fused CUDA spec-decode kernel with
    records) starts from the request's single state, writes no state and
    stores, per token, delta [HV, V], the normalised key k [H, K] and the decay
    exp(g) [HV] (fp32) into a per-row record buffer (one per GDN layer,
    allocated once, before CUDA-graph capture);
  * after sampling, the commit replays the accepted tokens over the state with
    the verify's operations, h = fma(delta, k, h * decay), and writes it once:
    the commit plan and conv-window compaction are the GLM recover's
    (vllm/sm80/kernels/kda/commit_plan.py, Kimi's _compact_conv_state_kernel);
    in align mode the block-boundary state is written as well.

GDN's decay is one scalar per value head, where KDA's is one per key channel,
and GDN shares each key head among HV / H value heads; the commit kernel below
is the KDA one with those two changes. The runner side (RecoverSSMState and
the align postprocess) is the generic one the KDA recover uses; the Qwen4Exp
model state turns it on.
"""

import dataclasses
import os
import re
from collections.abc import Sequence
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadata,
    PleShortConvAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

logger = init_logger(__name__)

WS_T = 8                     # token rows per record (the CUDA kernel's kMaxMtpTokens)
HEAD_DIM = 128
# The replay is bound by its record loads more than by state bandwidth; small
# tiles with one warp were fastest at 1-48 requests (BF16 and fp32 states).
COMMIT_BV = 8
COMMIT_WARPS = 1
_ARCHS = ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")


def _closed_reason(vllm_config) -> str | None:
    from vllm import envs
    from vllm.sm80.kernels._platform import _is_sm80

    num_spec = vllm_config.num_speculative_tokens or 0
    if num_spec < 1 or num_spec + 1 > WS_T:
        return f"needs 1..{WS_T - 1} speculative tokens (have {num_spec})"
    pc = vllm_config.parallel_config
    if pc.pipeline_parallel_size != 1:
        return "pipeline parallel is not supported"
    text = vllm_config.model_config.hf_text_config
    k_dim = int(getattr(text, "linear_key_head_dim", 0) or 0)
    v_dim = int(getattr(text, "linear_value_head_dim", 0) or 0)
    if k_dim != HEAD_DIM or v_dim != HEAD_DIM:
        return f"needs GDN heads of {HEAD_DIM} (have K={k_dim}, V={v_dim})"
    tp = pc.tensor_parallel_size
    hk = int(getattr(text, "linear_num_key_heads", 0) or 0)
    hv = int(getattr(text, "linear_num_value_heads", 0) or 0)
    if tp <= 0 or hk % tp or hv % tp or hv % hk or hv // hk not in (1, 2, 3, 4, 8):
        return f"needs HV/H in (1, 2, 3, 4, 8) per card (have {hv}/{hk} over {tp})"
    if vllm_config.cache_config.mamba_cache_mode not in ("align", "none"):
        return f"mamba cache mode {vllm_config.cache_config.mamba_cache_mode!r}"
    if getattr(vllm_config.cache_config, "use_replayssm", False):
        return "--use-replayssm is set"
    if envs.VLLM_GDN_DECODE_KERNEL.strip().lower() != "cuda":
        return "VLLM_GDN_DECODE_KERNEL is not cuda (the record verify is the CUDA one)"
    if not _is_sm80():
        return "needs sm_80"
    return None


def gdn_recover_enabled(vllm_config) -> bool:
    """RecoverSSM for this configuration: a Qwen4Exp model with
    VLLM_GDN_RECOVER=1 and the configuration the record verify is built for.

    Cheap; called at init by the GDN layers, the metadata builders and the
    model state. Logs the banner (or why it stays off) once."""
    if os.environ.get("VLLM_GDN_RECOVER", "0") != "1":
        return False
    archs = tuple(getattr(vllm_config.model_config.hf_config, "architectures", None) or ())
    if not any(a in _ARCHS for a in archs):
        logger.info_once("VLLM_GDN_RECOVER=1 set but off: not a Qwen4Exp model.")
        return False
    reason = _closed_reason(vllm_config)
    if reason is None:
        logger.info_once(
            "sm_80 GDN recover active: one GDN state per request, accepted "
            "tokens replayed after sampling."
        )
        return True
    logger.info_once(
        "sm_80 GDN recover off: %s; keeping one GDN state per draft position.", reason
    )
    return False


def records_rows(vllm_config) -> int:
    # A spec-decode row is a running request; captured graphs pad beyond
    # max_num_seqs only with rows that have no state slot (the verify skips
    # their records).
    return int(vllm_config.scheduler_config.max_num_seqs)


class RecordPool:
    """The fp32 records of every GDN layer of one card: (delta, key, decay)
    tensors of [layers, rows, WS_T, HV, 128], [layers, rows, WS_T, H, 128]
    and [layers, rows, WS_T, HV]."""

    _pools: dict = {}

    @classmethod
    def get(cls, device, num_layers: int, rows: int, hv: int, hk: int):
        key = (torch.device(device).type, torch.device(device).index)
        pool = cls._pools.get(key)
        if pool is None:
            f32 = dict(dtype=torch.float32, device=device)
            pool = (
                torch.zeros(num_layers, rows, WS_T, hv, HEAD_DIM, **f32),
                torch.zeros(num_layers, rows, WS_T, hk, HEAD_DIM, **f32),
                torch.zeros(num_layers, rows, WS_T, hv, **f32),
            )
            cls._pools[key] = pool
        c = pool[0]
        assert c.shape[0] == num_layers and c.shape[1] == rows and c.shape[3] == hv
        return pool


_EXT = None


def verify_ext():
    """The record verify (gdn_recover_verify.cu), built on first use into
    $VLLM_CACHE_ROOT/gdn_recover_ext and loaded from there afterwards."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        root = os.environ.get("VLLM_CACHE_ROOT") or os.path.expanduser("~/.cache/vllm")
        build = os.path.join(root, "gdn_recover_ext")
        os.makedirs(build, exist_ok=True)
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "gdn_recover_verify.cu")
        _EXT = load(
            name="gdn_recover_ext",
            sources=[src],
            # --use_fast_math as the engine builds fused_gdn_decode_kernel.cu
            # (CMakeLists.txt): the verify must round as the fused op does.
            extra_cuda_cflags=["-O3", "-std=c++20", "--use_fast_math",
                               "-gencode=arch=compute_80,code=sm_80"],
            build_directory=build,
        )
    return _EXT


def init_layer_recover(layer, vllm_config, config, prefix: str) -> None:
    """Give a QwenGatedDeltaNetAttention its record views and turn RecoverSSM
    on for it. Called from the layer's __init__ when gdn_recover_enabled()."""
    if layer.gdn_decode_kernel != "cuda":
        raise ValueError(
            "VLLM_GDN_RECOVER=1 needs the fused CUDA GDN decode path; this layer "
            f"fell back to {layer.gdn_decode_kernel!r}"
        )
    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    m = re.search(r"layers\.(\d+)\.", prefix + ".")
    if m is None or int(m.group(1)) not in gdn_layers:
        raise ValueError(f"GDN recover: no GDN layer index in {prefix!r}")
    hv = layer.num_v_heads // layer.tp_size
    hk = layer.num_k_heads // layer.tp_size
    device = torch.empty(0).device
    if device.type == "cpu" and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    pool = RecordPool.get(device, len(gdn_layers), records_rows(vllm_config), hv, hk)
    index = gdn_layers.index(int(m.group(1)))
    layer._gdn_recover = True
    layer._gdn_recover_pool = pool
    layer._gdn_recover_index = index
    layer._gdn_recover_records = (pool[0][index], pool[1][index], pool[2][index])
    verify_ext()


def record_verify(layer, mixed_qkv, a, b, state_indices, cu_seqlens, out,
                  output_gate=None) -> None:
    """The verify of `state_indices.shape[0]` spec-decode rows; with an
    output_gate, the gated RMSNorm is fused (the pure verify step), without
    it `out` gets the core output (mixed steps normalise every token after)."""
    c, k, e = layer._gdn_recover_records
    norm = output_gate is not None
    verify_ext().gdn_recover_verify(
        mixed_qkv, a, b, layer.A_log, layer.dt_bias, state_indices, cu_seqlens,
        layer.kv_cache[1], output_gate if norm else mixed_qkv,
        layer.norm.weight if norm else layer.A_log, out, c, k, e,
        layer.head_k_dim ** -0.5, layer.layer_norm_epsilon,
        layer.norm.activation, norm,
    )
    if _DEBUG and layer._gdn_recover_index in _DEBUG_LAYERS:
        _debug_compare(layer, mixed_qkv, a, b, state_indices, cu_seqlens, out,
                       output_gate)


# VLLM_GDN_RECOVER_DEBUG=1 (a diagnostic, never for serving): on eager steps
# (not CUDA-graph replays), run the path RecoverSSM replaces on copies of the
# same states and log how far the record verify's output is from it.
_DEBUG = os.environ.get("VLLM_GDN_RECOVER_DEBUG", "0") == "1"
_DEBUG_LAYERS = (0, 17, 35)
_debug_counts: dict = {}


def _debug_compare(layer, mixed_qkv, a, b, state_indices, cu_seqlens, out,
                   output_gate) -> None:
    if torch.cuda.is_current_stream_capturing():
        return
    kind = "pure" if output_gate is not None else "mixed"
    key = (kind, layer._gdn_recover_index)
    _debug_counts[key] = _debug_counts.get(key, 0) + 1
    state = layer.kv_cache[1]
    n = state_indices.shape[0]
    cu = cu_seqlens.tolist()
    width = max(cu[i + 1] - cu[i] for i in range(n))
    if width <= 0:
        return
    src = state_indices[:, 0].tolist()
    tmp = torch.zeros(1 + n * width, *state.shape[1:], dtype=state.dtype,
                      device=state.device)
    idx = torch.full((n, width), -1, dtype=torch.int32, device=state.device)
    for r in range(n):
        if src[r] > 0:
            tmp[1 + r * width] = state[src[r]]
            idx[r] = torch.arange(1 + r * width, 1 + (r + 1) * width,
                                  dtype=torch.int32, device=state.device)
    ones = torch.ones(n, dtype=torch.int32, device=state.device)
    tokens = cu[n]
    if output_gate is not None:
        from vllm import _custom_ops as ops

        ref = torch.empty_like(out)
        ops.fused_gdn_decode_post_conv_mtp(
            mixed_qkv=mixed_qkv, a=a, b=b, A_log=layer.A_log, dt_bias=layer.dt_bias,
            state_indices=idx, cu_seqlens=cu_seqlens, num_accepted_tokens=ones,
            state=tmp, output_gate=output_gate, norm_weight=layer.norm.weight,
            out=ref, scale=layer.head_k_dim ** -0.5, norm_eps=layer.layer_norm_epsilon,
            output_gate_activation=layer.norm.activation)
        ref = ref[:tokens]
    else:
        from vllm.third_party.flash_linear_attention.ops import (
            fused_sigmoid_gating_delta_rule_update,
        )

        q, kk, v = layer.rearrange_mixed_qkv(mixed_qkv)
        ref, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=layer.A_log, a=a, b=b, dt_bias=layer.dt_bias, q=q, k=kk, v=v,
            initial_state=tmp, inplace_final_state=True, cu_seqlens=cu_seqlens,
            ssm_state_indices=idx, num_accepted_tokens=ones,
            use_qk_l2norm_in_kernel=True)
        ref = ref.squeeze(0)[:tokens]
    got = out[:tokens].float()
    diff = (got - ref.float()).abs()
    _debug_stash[layer._gdn_recover_index] = (kind, tmp, idx)
    worst = diff.max().item()
    if (kind == "pure" and worst > 0) or worst > 1e-3 or _debug_counts[key] <= 3:
        logger.warning(
            "GDN recover debug %s layer %d: rows %d tokens %d src %s | max diff %.4g "
            "(ref max %.4g), mismatched %d of %d", kind, layer._gdn_recover_index, n,
            tokens, src[:8], worst, ref.float().abs().max().item(),
            int((diff > 0).sum().item()), diff.numel())


_debug_stash: dict = {}
# per commit kind: rows checked, wrong, checked in mixed steps
_debug_rows = {"GDN": [0, 0, 0], "PLE": [0, 0, 0]}


def _debug_check_rows(c, num_sampled, kind="GDN") -> None:
    """Each spec row's state slot against its batch row in the block table: a
    mismatch means the commit reads another request's sampled count. In "none"
    mode the slot is the row's column 0; in align mode it is one of the row's
    blocks (the PLE commit's rows are picked by recover.py, its slots by the
    engine's builder, so a wrong row shows here)."""
    si = c.state_indices.tolist()
    n = len(si)
    rows = c.request_indices.tolist() if c.request_indices is not None else list(range(n))
    align = c.block_table is not None
    bt = c.debug_block_table.tolist()
    ns = num_sampled.tolist()
    qsl = c.query_start_loc.tolist()
    counts = _debug_rows[kind]
    counts[0] += n
    counts[2] += n if c.request_indices is not None else 0
    bad = [(i, r, si[i], (([b for b in bt[r] if b > 0] if align else bt[r][0])
                          if r < len(bt) else None))
           for i, r in enumerate(rows)
           if r >= len(bt) or (si[i] not in bt[r] if align else bt[r][0] != si[i])]
    if bad:
        counts[1] += len(bad)
        logger.warning(
            "GDN recover debug rows (%s): %d of %d spec rows read another batch row "
            "(spec, batch, slot, batch slot or align-mode blocks) %s | pure=%s "
            "batch slots %s spec slots %s "
            "num_sampled %s qlens %s", kind, len(bad), n, bad[:6], c.request_indices is None,
            [b[0] for b in bt[:20]], si[:20], ns[:20],
            [qsl[i + 1] - qsl[i] for i in range(n)][:20])
    if counts[0] % 2000 < n:
        logger.warning("GDN recover debug rows (%s) checked %d, wrong %d (%d in mixed steps)",
                       kind, *counts)
_debug_commit_stats: dict = {"pure": [0, 0], "mixed": [0, 0]}


def _debug_check_commit(ctx, state_indices) -> None:
    """After a commit: each row's committed state against the replaced path's
    per-draft state at the accepted position (the verify's stash)."""
    lens = ctx.commit_lens[:state_indices.shape[0]].tolist()
    fins = ctx.final_state_indices[:state_indices.shape[0]].tolist()
    bnds = ctx.boundary_state_indices[:state_indices.shape[0]].tolist()
    blens = ctx.boundary_recovery_lens[:state_indices.shape[0]].tolist()
    for li, layer_index in enumerate(ctx.record_index.tolist()):
        stash = _debug_stash.pop(layer_index, None)
        if stash is None:
            continue
        kind, tmp, idx = stash
        has_state = (idx[:, 0] >= 0).tolist()
        for r, (c, f) in enumerate(zip(lens, fins)):
            # rows the verify saw no state for (slot 0): the startup dummy runs
            # stash these and commit nothing, so the first real commit after a
            # start can pop such a stash; idx -1 would read tmp's last row
            if c <= 0 or f <= 0 or r >= idx.shape[0] or not has_state[r]:
                continue
            checks = [("final", f, c)]
            if bnds[r] > 0 and 0 < blens[r] <= c:
                checks.append(("boundary", bnds[r], blens[r]))
            for what, slot, pos in checks:
                want = tmp[int(idx[r, pos - 1])]
                got = ctx.states[li][slot]
                d = (got.float() - want.float()).abs().max().item()
                st = _debug_commit_stats[kind]
                st[0] += 1
                if d > (0 if kind == "pure" else 1e-4):
                    st[1] += 1
                    logger.warning(
                        "GDN recover debug commit %s %s layer %d row %d len %d slot %d: "
                        "max diff %.4g (ref max %.4g)", kind, what, layer_index, r, pos,
                        slot, d, want.float().abs().max().item())
                elif what == "boundary":
                    _debug_commit_stats.setdefault("boundary_ok", [0])[0] += 1
    total = sum(v[0] for k, v in _debug_commit_stats.items() if k != "boundary_ok")
    if total and total % 500 < len(lens):
        logger.warning("GDN recover debug commit totals %s", _debug_commit_stats)


# ---------------------------------------------------------------------------
# commit: replay the accepted tokens over the state, all layers in one launch
# ---------------------------------------------------------------------------
# The verify is built with --use_fast_math, so its multiplies and fmas are the
# .ftz forms: denormal inputs and results flush to zero, keeping the sign. The
# replay uses the same instructions to stay bit-identical in that range too
# (emulating the flush with selects made the replay ~30 % slower).
@triton.jit
def _mul_ftz(a, b):
    return tl.inline_asm_elementwise("mul.rn.ftz.f32 $0, $1, $2;", "=r,r,r", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _fma_ftz(a, b, c):
    return tl.inline_asm_elementwise("fma.rn.ftz.f32 $0, $1, $2, $3;", "=r,r,r,r",
                                     [a, b, c], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _gdn_recover_commit_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    rec_c_ptr,                 # [pool layers, rows, WS_T, HV, D] fp32
    rec_k_ptr,                 # [pool layers, rows, WS_T, HK, D] fp32
    rec_e_ptr,                 # [pool layers, rows, WS_T, HV] fp32
    record_index_ptr,          # [L] int64: pool layer of each committed layer
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    stride_state_head,
    stride_state_v,
    stride_state_indices,
    stride_c_layer,
    stride_k_layer,
    stride_e_layer,
    HV: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
    WS_T: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_b = tl.program_id(1)
    i_lh = tl.program_id(2)
    i_l = i_lh // HV
    i_h = i_lh % HV
    i_kh = i_h // (HV // HK)
    src = tl.load(state_indices_ptr + i_b * stride_state_indices).to(tl.int64)
    if src <= null_block_id:
        return
    n = tl.load(commit_lens_ptr + i_b)
    if n == 0:
        return
    fin = tl.load(final_state_indices_ptr + i_b).to(tl.int64)
    if fin <= null_block_id:
        return
    bnd = tl.load(boundary_state_indices_ptr + i_b).to(tl.int64)
    blen = tl.load(boundary_recovery_lens_ptr + i_b)

    base = tl.load(state_base_addrs_ptr + i_l).to(
        tl.pointer_type(state_ref_ptr.dtype.element_ty))
    bstride = tl.load(state_block_strides_ptr + i_l)
    o_v = i_v * BV + tl.arange(0, BV)
    o_d = tl.arange(0, D)
    off = i_h * stride_state_head + o_v[:, None] * stride_state_v + o_d[None, :]
    h = tl.load(base + src * bstride + off).to(tl.float32)
    i_r = tl.load(record_index_ptr + i_l)
    rc = rec_c_ptr + i_r * stride_c_layer + (i_b * WS_T * HV + i_h) * D
    rk = rec_k_ptr + i_r * stride_k_layer + (i_b * WS_T * HK + i_kh) * D
    re_ = rec_e_ptr + i_r * stride_e_layer + i_b * WS_T * HV + i_h
    for t in tl.static_range(WS_T):
        if t < n:
            c = tl.load(rc + t * (HV * D) + o_v)
            k = tl.load(rk + t * (HK * D) + o_d)
            e = tl.load(re_ + t * HV)
            cb, kb = tl.broadcast(c[:, None], k[None, :])
            he = _mul_ftz(h, tl.broadcast_to(e, h.shape))
            h = _fma_ftz(cb, kb, he)
            if ALIGN_MODE:
                if (bnd > null_block_id) & (t + 1 == blen):
                    tl.store(base + bnd * bstride + off,
                             h.to(state_ref_ptr.dtype.element_ty))
    tl.store(base + fin * bstride + off, h.to(state_ref_ptr.dtype.element_ty))


@dataclasses.dataclass
class GDNRecoverCommitContext:
    conv_states: tuple
    conv_state_base_addrs: torch.Tensor
    conv_state_block_strides: torch.Tensor
    conv_state_dim_strides: torch.Tensor
    conv_state_token_strides: torch.Tensor
    conv_history_len: int
    states: tuple
    state_base_addrs: torch.Tensor
    state_block_strides: torch.Tensor
    rec_c: torch.Tensor
    rec_k: torch.Tensor
    rec_e: torch.Tensor
    record_index: torch.Tensor     # [L] int64 pool layer of each layer here
    commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor
    spec_query_len: int

    @classmethod
    def create(cls, layers: Sequence[Any], *, spec_query_len: int,
               max_num_reqs: int) -> "GDNRecoverCommitContext":
        if not layers:
            raise ValueError("GDN recover commit needs at least one layer")
        pool = layers[0]._gdn_recover_pool
        if any(layer._gdn_recover_pool is not pool for layer in layers):
            raise ValueError("GDN recover layers must share one record pool")
        index = [layer._gdn_recover_index for layer in layers]
        conv_states = _conv_views(layers)
        states = [layer.kv_cache[1] for layer in layers]
        ref = states[0]
        hv = pool[0].shape[3]
        if (ref.ndim != 4 or ref.dtype not in (torch.float32, torch.bfloat16)
                or ref.stride(3) != 1):
            raise ValueError("GDN recover needs fp32 or bf16 [blocks, HV, V, K] states")
        if tuple(ref.shape[1:]) != (hv, HEAD_DIM, HEAD_DIM):
            raise ValueError(f"GDN recover state shape {tuple(ref.shape)}")
        for s in states:
            if s.shape != ref.shape or s.dtype != ref.dtype or s.stride()[1:] != ref.stride()[1:]:
                raise ValueError("GDN recover layers need matching state layout")
        conv_ref = conv_states[0]
        conv_history_len = conv_ref.shape[2] - spec_query_len + 1
        if conv_history_len <= 0:
            raise ValueError("GDN recover conv state is shorter than its window")
        for s in conv_states:
            if s.shape != conv_ref.shape or s.dtype != conv_ref.dtype:
                raise ValueError("GDN recover layers need matching conv state")
        device = ref.device
        for t in pool:
            if not t.is_contiguous():
                raise ValueError("GDN recover record pool must be contiguous")

        def i64(vals):
            return torch.tensor(list(vals), dtype=torch.int64, device=device)

        def i32():
            return torch.empty(max_num_reqs, dtype=torch.int32, device=device)

        return cls(
            conv_states=tuple(conv_states),
            conv_state_base_addrs=i64(s.data_ptr() for s in conv_states),
            conv_state_block_strides=i64(s.stride(0) for s in conv_states),
            conv_state_dim_strides=i64(s.stride(1) for s in conv_states),
            conv_state_token_strides=i64(s.stride(2) for s in conv_states),
            conv_history_len=conv_history_len,
            states=tuple(states),
            state_base_addrs=i64(s.data_ptr() for s in states),
            state_block_strides=i64(s.stride(0) for s in states),
            rec_c=pool[0], rec_k=pool[1], rec_e=pool[2],
            record_index=i64(index),
            commit_lens=i32(), final_state_indices=i32(),
            boundary_state_indices=i32(), boundary_recovery_lens=i32(),
            spec_query_len=spec_query_len,
        )

    def commit(self, num_accepted_tokens: torch.Tensor, state_indices: torch.Tensor,
               query_start_loc: torch.Tensor, request_indices: torch.Tensor | None = None,
               block_table: torch.Tensor | None = None,
               num_computed_tokens: torch.Tensor | None = None,
               mamba_block_size: int | None = None) -> None:
        """Replay the accepted tokens of every spec-decode row in every layer."""
        batch = state_indices.shape[0]
        if batch == 0:
            return
        if batch > self.rec_c.shape[1]:
            raise ValueError("GDN recover commit batch exceeds its capacity")
        _plan_and_compact(self, num_accepted_tokens, state_indices, query_start_loc,
                          request_indices, block_table, num_computed_tokens,
                          mamba_block_size)
        num_layers = len(self.states)
        align = block_table is not None
        ref = self.states[0]
        hv = self.rec_c.shape[3]
        hk = self.rec_k.shape[3]
        _gdn_recover_commit_kernel[(HEAD_DIM // COMMIT_BV, batch, num_layers * hv)](
            ref, self.state_base_addrs, self.state_block_strides,
            self.rec_c, self.rec_k, self.rec_e, self.record_index,
            state_indices, self.commit_lens, self.final_state_indices,
            self.boundary_state_indices, self.boundary_recovery_lens,
            NULL_BLOCK_ID, ref.stride(1), ref.stride(2), state_indices.stride(0),
            self.rec_c.stride(0), self.rec_k.stride(0), self.rec_e.stride(0),
            HV=hv, HK=hk, D=HEAD_DIM, BV=COMMIT_BV, WS_T=WS_T, ALIGN_MODE=align,
            num_warps=COMMIT_WARPS,
        )
        if _DEBUG and not torch.cuda.is_current_stream_capturing():
            _debug_check_commit(self, state_indices)


def _conv_views(layers, attr_capacity=None):
    """The layers' conv states as [blocks, dim, window] views (both cache
    layouts), with their base addresses and strides."""
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    views = [layer.kv_cache[0] for layer in layers]
    if not is_conv_state_dim_first():
        views = [v.transpose(-1, -2) for v in views]
    if attr_capacity is not None:
        views = [v[..., -attr_capacity(layer):] for v, layer in zip(views, layers)]
    return views


def _plan_and_compact(ctx, num_accepted_tokens, state_indices, query_start_loc,
                      request_indices, block_table, num_computed_tokens,
                      mamba_block_size) -> None:
    """The commit plan (commit length, final and block-boundary state slots of
    every spec row) and the conv-window compaction of every layer in ctx."""
    from vllm.models.kimi_k3.nvidia.ops.recoverssm import _compact_conv_state_kernel
    from vllm.sm80.kernels.kda.commit_plan import _prepare_commit_plan_kernel

    batch = state_indices.shape[0]
    if batch > ctx.commit_lens.shape[0]:
        raise ValueError("RecoverSSM commit batch exceeds its capacity")
    if query_start_loc.shape[0] != batch + 1:
        raise ValueError("RecoverSSM commit metadata is incompatible")
    align = block_table is not None
    if align and (num_computed_tokens is None or mamba_block_size is None):
        raise ValueError("RecoverSSM align metadata is incomplete")
    bt_stride = block_table.stride() if align else (0, 0)
    _prepare_commit_plan_kernel[(batch,)](
        num_accepted_tokens, request_indices, state_indices, query_start_loc,
        block_table, num_computed_tokens, ctx.commit_lens,
        ctx.final_state_indices, ctx.boundary_state_indices,
        ctx.boundary_recovery_lens, NULL_BLOCK_ID, mamba_block_size or 1,
        block_table.shape[1] if align else 1,
        num_accepted_tokens.stride(0),
        request_indices.stride(0) if request_indices is not None else 0,
        state_indices.stride(0), query_start_loc.stride(0),
        bt_stride[0], bt_stride[1],
        num_computed_tokens.stride(0) if align else 0,
        SPEC_QUERY_LEN=ctx.spec_query_len,
        num_warps=1,
    )
    conv_ref = ctx.conv_states[0]
    conv_dim = conv_ref.shape[1]
    _compact_conv_state_kernel[(triton.cdiv(conv_dim, 256), batch, len(ctx.conv_states))](
        conv_ref, ctx.conv_state_base_addrs, ctx.conv_state_block_strides,
        ctx.conv_state_dim_strides, ctx.conv_state_token_strides,
        state_indices, ctx.commit_lens, ctx.final_state_indices,
        ctx.boundary_state_indices, ctx.boundary_recovery_lens,
        NULL_BLOCK_ID, conv_dim, ctx.conv_history_len, state_indices.stride(0),
        BLOCK_D=256, BLOCK_HISTORY=triton.next_power_of_2(ctx.conv_history_len),
        ALIGN_MODE=align, num_warps=4,
    )


@dataclasses.dataclass
class PLERecoverCommitContext:
    """The PLE short-conv layers' half of the commit in align mode: their conv
    windows are compacted to the accepted position (the plan and kernel of the
    GDN commit, with the PLE group's own block table), because the align
    postprocess resets every request's accepted count to 1 for every group."""
    conv_states: tuple
    conv_state_base_addrs: torch.Tensor
    conv_state_block_strides: torch.Tensor
    conv_state_dim_strides: torch.Tensor
    conv_state_token_strides: torch.Tensor
    conv_history_len: int
    commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor
    spec_query_len: int

    @classmethod
    def create(cls, layers: Sequence[Any], *, spec_query_len: int,
               max_num_reqs: int) -> "PLERecoverCommitContext":
        if not layers:
            raise ValueError("PLE recover commit needs at least one layer")
        conv_states = _conv_views(
            layers, lambda layer: layer.conv_state_len + layer.num_spec_tokens)
        ref = conv_states[0]
        for s in conv_states:
            if s.shape[1:] != ref.shape[1:] or s.dtype != ref.dtype:
                raise ValueError("PLE recover layers need matching conv state")
        conv_history_len = ref.shape[2] - spec_query_len + 1
        if conv_history_len != layers[0].conv_state_len:
            raise ValueError(
                f"PLE recover: conv window {ref.shape[2]} does not hold "
                f"{layers[0].conv_state_len} history + {spec_query_len - 1} drafts")
        device = ref.device

        def i64(vals):
            return torch.tensor(list(vals), dtype=torch.int64, device=device)

        def i32():
            return torch.empty(max_num_reqs, dtype=torch.int32, device=device)

        return cls(
            conv_states=tuple(conv_states),
            conv_state_base_addrs=i64(s.data_ptr() for s in conv_states),
            conv_state_block_strides=i64(s.stride(0) for s in conv_states),
            conv_state_dim_strides=i64(s.stride(1) for s in conv_states),
            conv_state_token_strides=i64(s.stride(2) for s in conv_states),
            conv_history_len=conv_history_len,
            commit_lens=i32(), final_state_indices=i32(),
            boundary_state_indices=i32(), boundary_recovery_lens=i32(),
            spec_query_len=spec_query_len,
        )

    def commit(self, num_accepted_tokens, state_indices, query_start_loc,
               request_indices=None, block_table=None, num_computed_tokens=None,
               mamba_block_size=None) -> None:
        if state_indices.shape[0] == 0:
            return
        _plan_and_compact(self, num_accepted_tokens, state_indices, query_start_loc,
                          request_indices, block_table, num_computed_tokens,
                          mamba_block_size)


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class GDNRecoverCommitInputs:
    state_indices: torch.Tensor           # [num_spec_decodes] state slots read by the verify
    query_start_loc: torch.Tensor         # [num_spec_decodes + 1]
    request_indices: torch.Tensor | None  # batch row of each spec decode (None: identity)
    block_table: torch.Tensor | None      # align mode only
    num_computed_tokens: torch.Tensor | None
    block_size: int | None
    # VLLM_GDN_RECOVER_DEBUG: the group's block table (batch order), to check
    # that each spec row reads its own batch row's sampled count
    debug_block_table: torch.Tensor | None = None


@dataclasses.dataclass
class GDNRecoverMetadata(GDNAttentionMetadata, RecoverSSMMetadata):
    recover_commit: GDNRecoverCommitInputs | None = None
    # the builder's BuilderRecover: the commit context is created on the first
    # commit (after the KV caches are bound), not when metadata is built
    recover_context: "BuilderRecover | None" = dataclasses.field(
        default=None, repr=False, compare=False)

    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        c = self.recover_commit
        if c is None or self.recover_context is None:
            return None
        context = self.recover_context.get_context()
        if _DEBUG and c.debug_block_table is not None:
            _debug_check_rows(c, num_accepted_tokens)
        context.commit(
            num_accepted_tokens, c.state_indices, c.query_start_loc,
            request_indices=c.request_indices, block_table=c.block_table,
            num_computed_tokens=c.num_computed_tokens, mamba_block_size=c.block_size)
        if c.block_table is None:
            return None
        return RecoverSSMPostprocessMetadata(
            num_spec_decodes=self.num_spec_decodes,
            request_indices=c.request_indices,
            block_table=c.block_table,
            num_computed_tokens=c.num_computed_tokens,
            block_size=c.block_size,
            commit_lens=context.commit_lens[:self.num_spec_decodes],
            source_state_indices=c.state_indices,
        )


class BuilderRecover:
    """Per GDN metadata builder state for RecoverSSM."""

    def __init__(self, builder) -> None:
        self.builder = builder
        # The verify reads num_accepted for every row the GDN builder can
        # stage, including a captured graph's padding rows.
        rows = max(builder.decode_cudagraph_max_bs,
                   builder.vllm_config.scheduler_config.max_num_seqs)
        self.ones = torch.ones(rows, dtype=torch.int32, device=builder.device)
        self.context: GDNRecoverCommitContext | None = None

    def get_context(self) -> GDNRecoverCommitContext:
        if self.context is None:
            b = self.builder
            fc = b.vllm_config.compilation_config.static_forward_context
            layers = [fc[name] for name in b.layer_names]
            self.context = GDNRecoverCommitContext.create(
                layers, spec_query_len=1 + b.num_spec,
                max_num_reqs=b.vllm_config.scheduler_config.max_num_seqs)
        return self.context

    def wrap(self, meta: GDNAttentionMetadata, m,
             spec_sequence_masks_cpu: torch.Tensor | None) -> GDNAttentionMetadata:
        """Add the commit inputs; the verify reads num_accepted = 1."""
        n = meta.num_spec_decodes
        if n == 0 or meta.spec_state_indices_tensor is None:
            return meta
        b = self.builder
        rows = meta.num_accepted_tokens.shape[0] if meta.num_accepted_tokens is not None else n
        if rows > self.ones.shape[0]:
            raise ValueError(f"GDN recover: {rows} rows exceed {self.ones.shape[0]}")
        request_indices = None
        pure = meta.num_prefills == 0 and meta.num_decodes == 0
        if not pure:
            assert spec_sequence_masks_cpu is not None
            idx = spec_sequence_masks_cpu.nonzero().flatten().to(torch.int32)
            request_indices = idx.pin_memory().to(b.device, non_blocking=True) \
                if torch.cuda.is_available() else idx.to(b.device)
        align = b.kv_cache_spec.mamba_cache_mode == "align"
        commit = GDNRecoverCommitInputs(
            state_indices=meta.spec_state_indices_tensor[:n, 0],
            query_start_loc=meta.spec_query_start_loc[: n + 1],
            request_indices=request_indices,
            block_table=m.block_table_tensor if align else None,
            num_computed_tokens=m.compute_num_computed_tokens() if align else None,
            block_size=b.kv_cache_spec.block_size if align else None,
            debug_block_table=m.block_table_tensor if _DEBUG else None,
        )
        fields = {f.name: getattr(meta, f.name) for f in dataclasses.fields(meta)}
        fields["num_accepted_tokens"] = self.ones[:rows]
        return GDNRecoverMetadata(**fields, recover_commit=commit, recover_context=self)


class QwenGDNRecoverMetadataBuilder(GDNAttentionMetadataBuilder):
    """The GDN builder plus the RecoverSSM commit inputs on spec-decode steps."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._gdn_recover = BuilderRecover(self) if self.use_spec_decode else None
        # Recovery commit indices and contexts belong to this group, unlike
        # the batch-level metadata shared by ordinary GDN.
        self.supports_update_block_table = False

    def build(self, common_prefix_len, common_attn_metadata,
              num_accepted_tokens=None, num_decode_draft_tokens_cpu=None,
              fast_build=False):
        metadata = super().build(common_prefix_len, common_attn_metadata,
                                 num_accepted_tokens, num_decode_draft_tokens_cpu,
                                 fast_build)
        if self._gdn_recover is None:
            return metadata
        return self._gdn_recover.wrap(metadata, common_attn_metadata,
                                      metadata.spec_sequence_masks_cpu)


class QwenGDNRecoverAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[QwenGDNRecoverMetadataBuilder]:
        return QwenGDNRecoverMetadataBuilder


@dataclasses.dataclass
class PLERecoverMetadata(PleShortConvAttentionMetadata, RecoverSSMMetadata):
    recover_commit: GDNRecoverCommitInputs | None = None
    recover_context: "PLEBuilderRecover | None" = dataclasses.field(
        default=None, repr=False, compare=False)

    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        c = self.recover_commit
        if c is None or self.recover_context is None:
            return None
        context = self.recover_context.get_context()
        if _DEBUG and c.debug_block_table is not None:
            _debug_check_rows(c, num_accepted_tokens, kind="PLE")
        context.commit(
            num_accepted_tokens, c.state_indices, c.query_start_loc,
            request_indices=c.request_indices, block_table=c.block_table,
            num_computed_tokens=c.num_computed_tokens, mamba_block_size=c.block_size)
        return RecoverSSMPostprocessMetadata(
            num_spec_decodes=self.num_spec_decodes,
            request_indices=c.request_indices,
            block_table=c.block_table,
            num_computed_tokens=c.num_computed_tokens,
            block_size=c.block_size,
            commit_lens=context.commit_lens[:self.num_spec_decodes],
            source_state_indices=c.state_indices,
        )


class PLEBuilderRecover:
    """Per PLE short-conv builder state for RecoverSSM in align mode."""

    def __init__(self, builder) -> None:
        self.builder = builder
        rows = max(builder.decode_cudagraph_max_bs,
                   builder.vllm_config.scheduler_config.max_num_seqs)
        self.ones = torch.ones(rows, dtype=torch.int32, device=builder.device)
        self.context: PLERecoverCommitContext | None = None

    def get_context(self) -> PLERecoverCommitContext:
        if self.context is None:
            b = self.builder
            fc = b.vllm_config.compilation_config.static_forward_context
            self.context = PLERecoverCommitContext.create(
                [fc[name] for name in b.layer_names], spec_query_len=1 + b.num_spec,
                max_num_reqs=b.vllm_config.scheduler_config.max_num_seqs)
        return self.context

    def wrap(self, meta: PleShortConvAttentionMetadata, m):
        n = meta.num_spec_decodes
        if n == 0 or meta.spec_state_indices_tensor is None:
            return meta
        b = self.builder
        rows = meta.num_accepted_tokens.shape[0] if meta.num_accepted_tokens is not None else n
        if rows > self.ones.shape[0]:
            raise ValueError(f"PLE recover: {rows} rows exceed {self.ones.shape[0]}")
        request_indices = None
        if meta.num_prefills or meta.num_decodes:
            mask = meta.num_decode_draft_tokens_cpu[: m.num_reqs] >= 0
            idx = mask.nonzero().flatten().to(torch.int32)
            request_indices = idx.pin_memory().to(b.device, non_blocking=True) \
                if torch.cuda.is_available() else idx.to(b.device)
        commit = GDNRecoverCommitInputs(
            state_indices=meta.spec_state_indices_tensor[:n],
            query_start_loc=meta.spec_query_start_loc[: n + 1],
            request_indices=request_indices,
            block_table=m.block_table_tensor,
            num_computed_tokens=m.compute_num_computed_tokens(),
            block_size=b.kv_cache_spec.block_size,
            debug_block_table=m.block_table_tensor if _DEBUG else None,
        )
        fields = {f.name: getattr(meta, f.name) for f in dataclasses.fields(meta)}
        fields["num_accepted_tokens"] = self.ones[:rows]
        return PLERecoverMetadata(**fields, recover_commit=commit, recover_context=self)


class QwenPLERecoverMetadataBuilder(PleShortConvAttentionMetadataBuilder):
    """The PLE short-conv builder plus the conv-only RecoverSSM commit; used
    in align mode only (in "none" mode the PLE conv keeps its own rollback)."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._ple_recover = PLEBuilderRecover(self) if self.use_spec_decode else None

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False, **kwargs):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build,
                                 **kwargs)
        if self._ple_recover is None:
            return metadata
        return self._ple_recover.wrap(metadata, common_attn_metadata)


class QwenPLERecoverAttentionBackend(PleShortConvAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[QwenPLERecoverMetadataBuilder]:
        return QwenPLERecoverMetadataBuilder
