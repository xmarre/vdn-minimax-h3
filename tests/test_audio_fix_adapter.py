from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from src.training.audio_fix_adapter import (
    AudioFixAdapterBank,
    audio_fix_spec,
    require_unit_lora_scale,
    sequence_linear_targets,
)


class _OrigAttention(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.to_q = nn.Linear(width, width, bias=False)
        self.to_k = nn.Linear(width, width, bias=False)
        self.to_v = nn.Linear(width, width, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(width, width, bias=False)])


class _Attention(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.orig = _OrigAttention(width)


class _Proj(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.proj = nn.Linear(width, width * 2, bias=False)


class _FF(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.net = nn.ModuleList([_Proj(width), nn.Identity(), nn.Linear(width, width, bias=False)])


class _Block(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.attn = _Attention(width)
        self.ff = _FF(width)


class _TinyModel(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block(width)])

    def forward(self, x):
        return self.transformer_blocks[0].attn.orig.to_q(x)


def _spec():
    return {
        "format_version": 2,
        "base": {
            "library": "diffusers",
            "class_name": "Fake",
            "source": "fake",
            "subfolder": "transformer",
            "revision": "unit",
            "resolved_config": {"hidden_size": 4},
        },
        "transforms": [{
            "type": "hybrid_attention",
            "version": 2,
            "config": {
                "enable_softmax_gate": True,
                "anchor_frames": "both",
                "softmax_attention": {"radius": 1, "chunk": 5},
                "linear_attention": {
                    "delta_rule": "vdn_solve",
                    "linear_head_dim": 2,
                    "bridge": "alpha",
                    "short_conv": {"targets": []},
                    "enable_text_state": True,
                    "a_fp32": True,
                },
            },
        }],
        "adapters": [
            {"type": "lora", "version": 1, "config": {
                "rank": 2, "alpha": 2,
                "targets": ["transformer_blocks.0.attn.orig.to_q"],
                "exact_targets": True,
            }},
            {"type": "lora", "version": 1, "config": {
                "name": "turbo", "rank": 2, "alpha": 2,
                "targets": [
                    "transformer_blocks.0.attn.orig.to_q",
                    "transformer_blocks.0.attn.orig.to_k",
                    "transformer_blocks.0.ff.net.0.proj",
                    "transformer_blocks.0.ff.net.2",
                    "transformer_blocks.0.adaln_proj.linear",
                    "token_refiner.refiner_blocks.0.attn.to_q",
                    "norm_out.linear",
                ],
                "rank_pattern": {}, "alpha_pattern": {}, "exact_targets": True,
            }},
        ],
    }


def test_target_derivation_keeps_only_portable_packed_sequence_linears():
    # fc2 is intentionally absent: the production Comfy INT8 path fuses it through
    # the parent MLP, so training a sidecar there would create a train/deploy mismatch.
    assert sequence_linear_targets(_spec()) == [
        "transformer_blocks.0.attn.orig.to_k",
        "transformer_blocks.0.attn.orig.to_q",
        "transformer_blocks.0.ff.net.0.proj",
    ]


def test_training_contract_requires_canonical_unit_scale_turbo_including_patterns():
    spec = _spec()
    assert require_unit_lora_scale(spec, "turbo")["config"]["alpha"] == 2

    global_scaled = copy.deepcopy(spec)
    global_scaled["adapters"][1]["config"]["alpha"] = 1
    with pytest.raises(ValueError, match="native unit LoRA scale"):
        sequence_linear_targets(global_scaled)

    patterned = copy.deepcopy(spec)
    target = "transformer_blocks.0.attn.orig.to_q"
    patterned["adapters"][1]["config"]["rank_pattern"] = {target: 4}
    patterned["adapters"][1]["config"]["alpha_pattern"] = {target: 2}
    with pytest.raises(ValueError, match="native unit LoRA scale"):
        sequence_linear_targets(patterned)

    unit_pattern = copy.deepcopy(spec)
    unit_pattern["adapters"][1]["config"]["rank_pattern"] = {target: 4}
    unit_pattern["adapters"][1]["config"]["alpha_pattern"] = {target: 4}
    assert require_unit_lora_scale(unit_pattern, "turbo")


def test_audio_fix_spec_is_explicitly_generated_audio_scoped():
    resolved, cfg = audio_fix_spec(_spec(), name="audio_fix", rank=8, alpha=8)
    assert cfg["scope"] == "generated_audio"
    assert cfg["target_policy"] == "portable_sequence_linear"
    assert cfg["exact_targets"] is True
    assert cfg["name"] == "audio_fix"
    assert len(resolved["adapters"]) == 3


def test_sidecar_zero_init_is_exact_noop_and_nonzero_only_touches_audio_rows():
    torch.manual_seed(3)
    model = _TinyModel()
    model.requires_grad_(False)
    target = "transformer_blocks.0.attn.orig.to_q"
    x = torch.randn(1, 5, 4)
    base = model(x).detach().clone()
    bank = AudioFixAdapterBank(model, [target], rank=2, alpha=2)
    bank.install(model)

    with bank.scope(torch.tensor([1, 3]), enabled=True):
        zero = model(x)
    assert torch.equal(zero, base)

    with torch.no_grad():
        bank.pairs[0].lora_B.fill_(0.25)
    with bank.scope(torch.tensor([1, 3]), enabled=True):
        changed = model(x)
    assert torch.equal(changed[:, [0, 2, 4]], base[:, [0, 2, 4]])
    assert not torch.equal(changed[:, [1, 3]], base[:, [1, 3]])

    with bank.disabled():
        disabled = model(x)
    assert torch.equal(disabled, base)
    bank.uninstall()


def test_sidecar_gradient_reaches_only_audio_fix_parameters():
    torch.manual_seed(4)
    model = _TinyModel()
    model.requires_grad_(False)
    target = "transformer_blocks.0.attn.orig.to_q"
    bank = AudioFixAdapterBank(model, [target], rank=2, alpha=2)
    with torch.no_grad():
        bank.pairs[0].lora_B.fill_(0.1)
    bank.install(model)
    x = torch.randn(1, 4, 4)
    with bank.scope(torch.tensor([1, 2]), enabled=True):
        model(x).square().mean().backward()
    assert bank.pairs[0].lora_A.grad is not None
    assert bank.pairs[0].lora_B.grad is not None
    assert all(parameter.grad is None for parameter in model.parameters())
    bank.uninstall()


def test_packed_target_fails_closed_without_scope():
    model = _TinyModel()
    bank = AudioFixAdapterBank(
        model, ["transformer_blocks.0.attn.orig.to_q"], rank=2, alpha=2)
    bank.install(model)
    with pytest.raises(RuntimeError, match="without generated-audio scope"):
        model(torch.randn(1, 3, 4))
    bank.uninstall()


def test_export_state_uses_peft_names_and_requested_dtype():
    model = _TinyModel()
    target = "transformer_blocks.0.attn.orig.to_q"
    bank = AudioFixAdapterBank(model, [target], rank=2, alpha=2, name="audio_fix")
    state = bank.export_peft_state(torch.bfloat16)
    assert set(state) == {
        target + ".lora_A.audio_fix.weight",
        target + ".lora_B.audio_fix.weight",
    }
    assert all(t.dtype == torch.bfloat16 for t in state.values())
