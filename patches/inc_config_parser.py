# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified 2026-10-23 by ElectroHobo for the qwen38-flash-next-3bpw-1x-cmp170hx-recipe:
# AutoRound per-layer bits reach this engine's RoutedExperts, and the
# MTP layer's bits resolve (engine layer index, the checkpoint's pack plan) (patches/README.md).
# Otherwise identical to Morrowmake/vllm-cmp170hx at 18305948acba3f4c4d912cdd56f2202d2ebd94e8.

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import regex as re

from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

if TYPE_CHECKING:
    import torch

    from .inc import INCConfig


@dataclass(frozen=True)
class INCLayerConfig:
    bits: int
    group_size: int | tuple[int, int]
    sym: bool
    packing_format: str
    backend: str
    data_type: str
    quantized: bool

    @property
    def is_gptq(self) -> bool:
        return "gptq" in self.packing_format or "gptq" in self.backend

    @property
    def is_awq(self) -> bool:
        return "awq" in self.packing_format or "awq" in self.backend

    @property
    def is_wna16_int(self) -> bool:
        return self.data_type == "int" and self.quantized

    @property
    def is_mxfp4(self) -> bool:
        return "mx_fp" in self.data_type and self.bits == 4

    @property
    def is_mxfp8(self) -> bool:
        return "mx_fp" in self.data_type and self.bits == 8

    @property
    def is_fp8_block(self) -> bool:
        return (
            self.data_type == "fp"
            and self.bits == 8
            and self.sym
            and self.quantized
            and self.packing_format == "auto_round:fp8"
            and isinstance(self.group_size, tuple)
            and len(self.group_size) == 2
        )


class INCConfigParser:
    def __init__(self, config: "INCConfig") -> None:
        self._config = config

    def resolve(self, layer: "torch.nn.Module", layer_name: str) -> INCLayerConfig:
        bits, group_size, sym = self._resolve_raw(layer, layer_name)
        return INCLayerConfig(
            bits=bits,
            group_size=group_size,
            sym=sym,
            packing_format=self._config.packing_format,
            backend=self._config.backend,
            data_type=self._config.data_type,
            quantized=bits < 16,
        )

    def get_layer_config(
        self, layer: "torch.nn.Module", layer_name: str
    ) -> tuple[int, int | tuple[int, int], bool]:
        layer_config = self.resolve(layer, layer_name)
        return layer_config.bits, layer_config.group_size, layer_config.sym

    def _mtp_checkpoint_name(self, layer_name: str) -> str:
        """Recipe change: the engine numbers the MTP layer after the decoder layers
        (mtp.layers.48), AutoRound's extra_config by its own index (mtp.layers.0);
        when no entry names the engine's index and the config has one MTP layer,
        look that one up."""
        m = re.search(r"(?:^|\.)mtp\.layers\.(\d+)\.", layer_name)
        extra = self._config.extra_config
        if m is None or not extra:
            return layer_name
        if any(k.startswith(f"mtp.layers.{m.group(1)}.") for k in extra):
            return layer_name
        indices = {k.split(".")[2] for k in extra if k.startswith("mtp.layers.")}
        if len(indices) != 1:
            return layer_name
        return layer_name.replace(f"mtp.layers.{m.group(1)}.", f"mtp.layers.{indices.pop()}.", 1)

    def _mtp_pack_plan_experts(
        self, layer: "torch.nn.Module", layer_name: str
    ) -> tuple[int, int, bool] | None:
        """Recipe change: a checkpoint's mtp_pack_plan.json (klee100's 3 bpw MTP build) lists
        MTP experts packed at 4 bits that extra_config does not mention; their packed
        shapes (qweight 8 values per int32, one scale per 128 inputs) give groups of 128."""
        if "mtp." not in layer_name or layer.__class__.__name__.lower() not in (
            "routedexperts", "fusedmoe"
        ):
            return None
        plan = getattr(self, "_mtp_plan", False)
        if plan is False:
            plan = None
            try:
                from vllm.config import get_current_vllm_config

                path = os.path.join(
                    get_current_vllm_config().model_config.model, "mtp_pack_plan.json"
                )
                if os.path.exists(path):
                    with open(path) as f:
                        plan = json.load(f)
            except Exception:  # noqa: BLE001 - no config or no plan: the usual rules
                plan = None
            self._mtp_plan = plan
        if not plan:
            return None
        if any(m.startswith(layer_name + ".") for m in plan.get("expanded_expert_modules_w4", ())):
            return 4, 128, True
        return None

    def _resolve_raw(
        self, layer: "torch.nn.Module", layer_name: str
    ) -> tuple[int, int | tuple[int, int], bool]:
        REGEX_SPECIAL_CHARS = set(r"*+?^$()[]{}|\\")
        layer_name = self._mtp_checkpoint_name(layer_name)
        packed = self._mtp_pack_plan_experts(layer, layer_name)
        if packed is not None:
            return packed

        def is_explicitly_configured(name: str) -> bool:
            """Return True if *name* has an explicit entry in extra_config,
            either via exact key match or via a regex pattern key."""
            if not self._config.extra_config:
                return False
            if name in self._config.extra_config:
                return True
            for pattern in self._config.extra_config:
                if not isinstance(pattern, str) or not any(
                    c in REGEX_SPECIAL_CHARS for c in pattern
                ):
                    continue
                try:
                    if re.search(re.compile(pattern), name) is not None:
                        return True
                except re.error:
                    continue
            return False

        def get_config(
            name: str, quantized: bool = True
        ) -> tuple[int, int | tuple[int, int], bool]:
            if not self._config.extra_config:
                return (
                    self._config.weight_bits if quantized else 16,
                    self._config.group_size if quantized else -1,
                    self._config.sym if quantized else True,
                )

            if name in self._config.extra_config:
                cfg = self._config.extra_config[name]
                return (
                    cfg.get("bits", self._config.weight_bits if quantized else 16),
                    self._normalize_group_size(
                        cfg.get(
                            "group_size",
                            self._config.group_size if quantized else -1,
                        )
                    ),
                    cfg.get("sym", self._config.sym if quantized else True),
                )

            regex_special_chars = set(r"*+?^$()[]{}|\\")
            for pattern, cfg in self._config.extra_config.items():
                if not isinstance(pattern, str) or not any(
                    c in regex_special_chars for c in pattern
                ):
                    continue

                try:
                    if re.search(re.compile(pattern), name) is not None:
                        return (
                            cfg.get(
                                "bits",
                                self._config.weight_bits if quantized else 16,
                            ),
                            self._normalize_group_size(
                                cfg.get(
                                    "group_size",
                                    self._config.group_size if quantized else -1,
                                )
                            ),
                            cfg.get("sym", self._config.sym if quantized else True),
                        )
                except re.error:
                    continue

            return (
                self._config.weight_bits if quantized else 16,
                self._config.group_size if quantized else -1,
                self._config.sym if quantized else True,
            )

        if self._config.extra_config and layer_name in self._config.extra_config:
            return get_config(layer_name)

        # Suffix match: handle cases where extra_config keys use short names
        # (e.g. "lm_head") but the layer_name is fully qualified
        # (e.g. "model.language_model.lm_head") due to model nesting.
        if self._config.extra_config:
            for cfg_key in self._config.extra_config:
                if layer_name.endswith(f".{cfg_key}"):
                    return get_config(cfg_key)

        quantized = not isinstance(layer, ParallelLMHead)
        if self._config.block_name_to_quantize:
            quantized = any(
                layer_name.startswith(name)
                for name in self._config.block_name_to_quantize
            )

        # RoutedExperts is this engine's fused-MoE layer; without it the per-layer
        # expert overrides are skipped and every expert layer gets the default bits.
        moe_class = layer.__class__.__name__.lower()
        if self._config.extra_config and ("fusedmoe" in moe_class or moe_class == "routedexperts"):
            moe_configs = [
                get_config(name, quantized)
                for name in self._config.extra_config
                if name.startswith(layer_name)
            ]
            if moe_configs:
                if len(set(moe_configs)) == 1:
                    return moe_configs[0]
                raise ValueError(
                    f"Fused MoE layer '{layer_name}' requires "
                    f"consistent quant config for all sub-layers"
                )

        if self._config.extra_config:
            for fusion_key, sub_keys in self._config.packed_modules_mapping.items():
                if fusion_key in layer_name and layer_name.count(fusion_key) == 1:
                    sub_names = [
                        layer_name.replace(fusion_key, sub_key) for sub_key in sub_keys
                    ]
                    # Only trigger if at least one sub_name is explicitly
                    # configured in extra_config (via exact match or regex).
                    # This prevents false matches when a short fusion_key
                    # (e.g. "qkv") is merely a substring of a longer layer
                    # name (e.g. "in_proj_qkvz") and none of the generated
                    # sub_names are actually configured.
                    if not any(is_explicitly_configured(n) for n in sub_names):
                        continue
                    sub_configs = [get_config(name, quantized) for name in sub_names]
                    if len(set(sub_configs)) == 1:
                        return sub_configs[0]
                    raise ValueError(
                        f"Fused module '{layer_name}' requires "
                        f"consistent quant config for {sub_names}"
                    )

        return get_config(layer_name, quantized)

    @staticmethod
    def _normalize_group_size(
        group_size: int | list[int] | tuple[int, int],
    ) -> int | tuple[int, int]:
        """Normalize INC group_size into either an int or a 2-D int tuple."""
        if isinstance(group_size, (list, tuple)):
            if len(group_size) != 2 or not all(
                isinstance(value, int) for value in group_size
            ):
                raise ValueError(
                    "INC block-wise FP8 requires group_size to be a 2-D "
                    f"integer sequence, but found {group_size!r}."
                )
            return (group_size[0], group_size[1])

        if not isinstance(group_size, int):
            raise ValueError(
                "INC group_size must be an int or a 2-D integer sequence, "
                f"but found {type(group_size).__name__}."
            )

        return group_size
