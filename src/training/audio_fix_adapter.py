"""Generated-audio-scoped sidecar LoRA used by the production audio-fix trainer.

Why a sidecar instead of a third PEFT adapter inside the FSDP model:

* Stage-B and released Turbo remain ordinary frozen PEFT adapters at canonical scale;
* the audio fix is applied only to generated-audio sequence rows, never directly to
  video/text rows;
* its parameters stay outside FSDP/DTensor, so Prodigy-Plus sees normal fp32 tensors;
* multi-GPU training explicitly all-reduces the small sidecar gradients.

The exported weights still use normal PEFT LoRA A/B names, so the correction can be
stored as ``adapters/audio_fix`` in an otherwise standard VDN checkpoint directory.
"""
from __future__ import annotations

import contextvars
import copy
import json
import math
import os
import shutil
from contextlib import contextmanager

import torch
from torch import nn
from safetensors.torch import save_file

from src.models.model_spec import ModelSpec


_SCOPE = contextvars.ContextVar("vdn_audio_fix_scope", default=None)
# Portable generated-audio row targets. fc2 is deliberately excluded: ComfyUI's
# production INT8 H3 executes fc2 through a fused parent path rather than fc2.forward.
# q/k/v/out + fc1 remain exact ordinary per-row Linears in both layouts and give the
# correction substantial attention/MLP capacity without a train/deploy graph mismatch.
_SEQUENCE_SUFFIXES = (
    ".attn.orig.to_q", ".attn.orig.to_k", ".attn.orig.to_v", ".attn.orig.to_out.0",
    ".attn.to_q", ".attn.to_k", ".attn.to_v", ".attn.to_out.0",
    ".ff.net.0.proj",
)


def adapter_name(entry, index):
    cfg = entry.get("config", {})
    return cfg.get("name", "default" if index == 0 else None)


def adapter_entry(spec, name):
    for index, entry in enumerate(spec.get("adapters", [])):
        if adapter_name(entry, index) == name:
            return entry
    raise KeyError(f"checkpoint ModelSpec has no adapter {name!r}")


def require_unit_lora_scale(spec, name):
    """Fail closed unless every resolved LoRA target has alpha/rank == 1.

    The corrective model is deliberately learned on the canonical finished checkpoint,
    not on a deployment-time strength multiplier. PEFT's effective native LoRA scale is
    ``alpha / rank`` and may be overridden per target through rank/alpha patterns, so
    checking only the top-level pair would miss a mixed-scale adapter.
    """
    entry = adapter_entry(spec, name)
    if entry.get("type") != "lora":
        raise ValueError(f"frozen adapter {name!r} is not a LoRA")
    cfg = entry.get("config", {})
    targets = list(cfg.get("targets") or [])
    if not targets:
        raise ValueError(f"frozen adapter {name!r} has no resolved targets")
    try:
        default_rank = int(cfg["rank"])
        default_alpha = float(cfg["alpha"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"frozen adapter {name!r} has invalid rank/alpha") from exc
    rank_pattern = dict(cfg.get("rank_pattern") or {})
    alpha_pattern = dict(cfg.get("alpha_pattern") or {})
    bad = []
    for target in targets:
        rank = int(rank_pattern.get(target, default_rank))
        alpha = float(alpha_pattern.get(target, default_alpha))
        if rank <= 0 or not math.isfinite(alpha) or not math.isclose(
                alpha / rank, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            bad.append((target, alpha, rank))
    if bad:
        target, alpha, rank = bad[0]
        raise ValueError(
            f"audio-fix training requires frozen adapter {name!r} at native unit LoRA "
            f"scale on every target; {target!r} has alpha/rank={alpha}/{rank}")
    return entry


def sequence_linear_targets(spec, source_adapter="turbo"):
    """Exact Turbo targets that are portable generated-audio row projections.

    AdaLN, final-layer, token-refiner and fc2 targets are deliberately excluded. The
    first three do not use the generated-audio sequence-row axis; fc2 is fused into the
    parent MLP by the production ComfyUI INT8 H3 path. The retained q/k/v/out and fc1
    targets have identical per-row semantics in the Diffusers trainer and Comfy runtime.
    Target derivation also verifies that the frozen released Turbo itself is represented
    at canonical unit LoRA scale; deployment-time multipliers belong outside training.
    """
    entry = require_unit_lora_scale(spec, source_adapter)
    cfg = entry["config"]
    targets = list(cfg.get("targets") or [])
    if not cfg.get("exact_targets"):
        raise ValueError(
            f"source adapter {source_adapter!r} must carry exact_targets for audio-fix derivation")
    selected = sorted({
        target for target in targets
        if target.startswith("transformer_blocks.")
        and any(target.endswith(suffix) for suffix in _SEQUENCE_SUFFIXES)
    })
    if not selected:
        raise ValueError(
            f"source adapter {source_adapter!r} has no portable sequence-space DiT targets")
    return selected


def audio_fix_spec(source_spec, *, name, rank, alpha, source_adapter="turbo"):
    resolved = copy.deepcopy(source_spec)
    names = [adapter_name(entry, i) for i, entry in enumerate(resolved.get("adapters", []))]
    if name in names:
        raise ValueError(f"source checkpoint already contains adapter {name!r}")
    targets = sequence_linear_targets(resolved, source_adapter)
    config = {
        "name": name,
        "rank": int(rank),
        "alpha": int(alpha),
        "targets": targets,
        "rank_pattern": {},
        "alpha_pattern": {},
        "exact_targets": True,
        "scope": "generated_audio",
        "target_policy": "portable_sequence_linear",
        "source_adapter": source_adapter,
    }
    resolved.setdefault("adapters", []).append(
        {"type": "lora", "version": 1, "config": config})
    ModelSpec.from_dict(resolved)
    return resolved, config


class AudioFixPair(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32))
        self.scale = float(alpha) / float(rank)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return torch.nn.functional.linear(
            torch.nn.functional.linear(x.to(self.lora_A.dtype), self.lora_A),
            self.lora_B,
        ) * self.scale


class AudioFixAdapterBank(nn.Module):
    """One fp32 low-rank pair per exact portable generated-audio target."""

    def __init__(self, model, targets, rank, alpha, name="audio_fix"):
        super().__init__()
        self.name = name
        self.targets = tuple(targets)
        pairs = []
        for target in self.targets:
            module = model.get_submodule(target)
            base = getattr(module, "base_layer", module)
            in_features = getattr(base, "in_features", None)
            out_features = getattr(base, "out_features", None)
            if not isinstance(in_features, int) or not isinstance(out_features, int):
                raise TypeError(
                    f"audio-fix target {target} is not a Linear-like module: {type(module).__name__}")
            pairs.append(AudioFixPair(in_features, out_features, rank, alpha))
        self.pairs = nn.ModuleList(pairs)
        self._handles = []

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    @contextmanager
    def scope(self, audio_indices, *, enabled=True):
        token = _SCOPE.set((bool(enabled), audio_indices))
        try:
            yield
        finally:
            _SCOPE.reset(token)

    @contextmanager
    def disabled(self):
        token = _SCOPE.set((False, None))
        try:
            yield
        finally:
            _SCOPE.reset(token)

    def _hook(self, pair, target):
        def hook(_module, inputs, output):
            scope = _SCOPE.get()
            if scope is None:
                raise RuntimeError(
                    f"audio-fix target {target} executed without generated-audio scope")
            enabled, audio_indices = scope
            if not enabled:
                return output
            if not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(output, torch.Tensor):
                raise RuntimeError(f"audio-fix target {target} did not receive/return tensors")
            x = inputs[0]
            if x.ndim not in (2, 3) or output.ndim != x.ndim:
                raise RuntimeError(
                    f"audio-fix target {target} expected packed rank-2/3 rows, got "
                    f"input {tuple(x.shape)} output {tuple(output.shape)}")
            indices = audio_indices.to(device=x.device, dtype=torch.long)
            row_dim = 1 if x.ndim == 3 else 0
            xa = x.index_select(row_dim, indices)
            delta = pair(xa).to(dtype=output.dtype)
            output.index_add_(row_dim, indices, delta)
            return output
        return hook

    def install(self, model):
        if self._handles:
            raise RuntimeError("audio-fix hooks are already installed")
        for target, pair in zip(self.targets, self.pairs):
            module = model.get_submodule(target)
            self._handles.append(module.register_forward_hook(self._hook(pair, target)))
        return len(self._handles)

    def uninstall(self):
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def export_peft_state(self, dtype=torch.float32):
        state = {}
        for target, pair in zip(self.targets, self.pairs):
            state[f"{target}.lora_A.{self.name}.weight"] = pair.lora_A.detach().to(
                device="cpu", dtype=dtype).contiguous()
            state[f"{target}.lora_B.{self.name}.weight"] = pair.lora_B.detach().to(
                device="cpu", dtype=dtype).contiguous()
        return state


def average_gradients(module, world):
    if world <= 1:
        return
    import torch.distributed as dist
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world)


def broadcast_parameters(module, src=0):
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=src)


def _link_or_copy(src, dst):
    try:
        os.link(os.path.realpath(src), dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def export_composed_stage(source_dir, out_dir, model_spec, adapter_config, adapter_state,
                          metadata):
    """Write an immediately consumable VDN stage without duplicating the frozen 5 GB."""
    if not os.path.isdir(source_dir):
        raise ValueError("audio-fix deployment export requires an exploded source stage directory")
    required = ("linear_branch", "adapters", "model_spec.json", "metadata.json")
    missing = [name for name in required if not os.path.exists(os.path.join(source_dir, name))]
    if missing:
        raise FileNotFoundError(f"source stage {source_dir} is missing {missing}")
    if os.path.exists(out_dir):
        raise FileExistsError(f"{out_dir} already exists")
    tmp = out_dir.rstrip("/") + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    try:
        shutil.copytree(
            os.path.join(source_dir, "linear_branch"), os.path.join(tmp, "linear_branch"),
            copy_function=_link_or_copy)
        shutil.copytree(
            os.path.join(source_dir, "adapters"), os.path.join(tmp, "adapters"),
            copy_function=_link_or_copy)
        with open(os.path.join(tmp, "model_spec.json"), "w", encoding="utf-8") as f:
            json.dump(model_spec, f, indent=2, sort_keys=True)
            f.write("\n")
        with open(os.path.join(tmp, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({
                "kind": "weights",
                "checkpoint_format_version": 2,
                "weights_dtype": str(metadata.get("adapter_dtype", "float32")),
                "metadata": metadata,
            }, f, indent=2, sort_keys=True)
            f.write("\n")
        adapter_dir = os.path.join(tmp, "adapters", adapter_config["name"])
        if os.path.exists(adapter_dir):
            raise RuntimeError(f"source stage unexpectedly already contains {adapter_config['name']}")
        os.makedirs(adapter_dir)
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump({"type": "lora", "version": 1, "config": adapter_config},
                      f, indent=2, sort_keys=True)
            f.write("\n")
        save_file(adapter_state, os.path.join(adapter_dir, "adapter_model.safetensors"))
        os.replace(tmp, out_dir)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out_dir


__all__ = [
    "AudioFixAdapterBank",
    "adapter_entry",
    "audio_fix_spec",
    "average_gradients",
    "broadcast_parameters",
    "export_composed_stage",
    "require_unit_lora_scale",
    "sequence_linear_targets",
]
