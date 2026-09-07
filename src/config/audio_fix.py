"""Configuration for the production audio-fix LoRA trainer.

The correction adapter is trained on top of the *finished* released VDN/Turbo stack.
The base H3, VDN linear branch, Stage-B adapter, and released Turbo adapter stay frozen
and run at their canonical/native unit LoRA scale. Deployment multipliers such as Turbo
0.75 are intentionally not training knobs; they are post-training validation settings.
Only a small generated-audio-scoped sidecar LoRA is optimized against a dense-H3 audio
teacher on the 10-step RES-multistep sampler contract and the current seven-second
per-H3-call production chunk geometry. References, Spectrum forecasting and progressive
mixed-grid handoff remain deployment-time validation dimensions rather than hidden
claims of this first corrective trainer.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from omegaconf import MISSING

from src.config.common import DataConfig, RuntimeConfig
from src.config.stage_b import BCheckpoint, BDistributed
from src.config.stage_dmd import GenerationConfig


@dataclass
class AudioFixInitialization:
    checkpoint: str = MISSING
    base_source: str | None = None
    source_step: int = 250
    vdn_adapter_name: str = "default"
    turbo_adapter_name: str = "turbo"


@dataclass
class AudioFixAdapterConfig:
    name: str = "audio_fix"
    rank: int = 32
    alpha: int = 32
    # Derive the exact portable row-wise target set from the same released Turbo
    # adapter selected by initialization.turbo_adapter_name. There is deliberately no
    # second source-adapter setting that could drift away from the verified stack.
    # q/k/v/out + fc1 are ordinary generated-audio row projections in both Diffusers
    # training and the production ComfyUI H3 path. Token-refiner/AdaLN/final and fused
    # INT8 fc2 are intentionally excluded so training and deployment execute the same
    # correction graph.
    target_policy: str = "portable_sequence_linear"


@dataclass
class AudioFixRolloutConfig:
    sampler: str = "res_multistep"
    # Production SIGMAS come from MiniMaxH3SASolverScheduler simple_control, which is
    # exact current-ComfyUI `simple` table selection. The trainer mirrors that discrete
    # table locally rather than substituting a continuous schedule.
    sigma_schedule: str = "simple_control"
    num_steps: int = 10
    video_shift: float = 12.0
    audio_shift: float = 3.0


@dataclass
class AudioFixGeneration(GenerationConfig):
    # Current Continuum production chunk request: 7 s * 24 fps = 168 display frames.
    # MiniMax H3's VAE alignment snaps that request to 175 pixel frames / 52 video
    # latent frames inside generation_geometry(). Keep the *request* here so the same
    # alignment code used by the real H3 pipeline remains authoritative.
    num_frames: int = 168


@dataclass
class AudioFixTraining:
    max_steps: int = 250
    seed: int = 0
    audio_teacher_weight: float = 1.0
    # Although the adapter is directly audio-row-only, altered audio K/V can affect
    # video queries in the same/later blocks. Preserve the frozen production video's
    # x0 explicitly so the correction cannot trade speech fidelity for visual drift.
    video_preserve_weight: float = 0.1
    ignore_stopped: bool = False
    auto_resume: bool = True


@dataclass
class ProdigyPlusConfig:
    name: str = "prodigy_plus_schedule_free"
    lr: float = 1.0
    beta1: float = 0.95
    beta2: float = 0.99
    weight_decay: float = 0.0
    d0: float = 1.0e-6
    d_coef: float = 1.0
    d_limiter: bool = True
    prodigy_steps: int = 0
    schedulefree_c: float = 0.0
    eps: float = 1.0e-8
    factored: bool = True
    factored_fp32: bool = True
    use_stableadamw: bool = True
    use_schedulefree: bool = True
    split_groups: bool = False


@dataclass
class AudioFixCheckpoint(BCheckpoint):
    output_dir: str = MISSING
    save_every: int = 10
    keep_states: int = 3
    early_saves: list[int] = field(default_factory=lambda: [0, 1, 2, 4, 8, 16, 32])
    adapter_dtype: str = "float32"


@dataclass
class AudioFixConfig:
    initialization: AudioFixInitialization = field(default_factory=AudioFixInitialization)
    adapter: AudioFixAdapterConfig = field(default_factory=AudioFixAdapterConfig)
    rollout: AudioFixRolloutConfig = field(default_factory=AudioFixRolloutConfig)
    generation: AudioFixGeneration = field(default_factory=AudioFixGeneration)
    data: DataConfig = field(default_factory=DataConfig)
    training: AudioFixTraining = field(default_factory=AudioFixTraining)
    optimizer: ProdigyPlusConfig = field(default_factory=ProdigyPlusConfig)
    distributed: BDistributed = field(default_factory=BDistributed)
    checkpoint: AudioFixCheckpoint = field(default_factory=AudioFixCheckpoint)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _validate_adapter_path_segment(name: str) -> None:
    """Require an adapter name that is safe as one ``adapters/<name>`` path segment."""
    if (not isinstance(name, str) or not name or name in {".", ".."}
            or os.path.isabs(name) or "/" in name or "\\" in name or "\x00" in name):
        raise ValueError(
            "audio-fix adapter name must be one non-empty safe path segment")


def validate_audio_fix(cfg) -> None:
    """Reject configuration drift away from the validated production training contract."""
    _validate_adapter_path_segment(cfg.adapter.name)
    if cfg.initialization.vdn_adapter_name == cfg.initialization.turbo_adapter_name:
        raise ValueError("VDN and Turbo adapter names must be distinct")
    if cfg.adapter.name in (cfg.initialization.vdn_adapter_name,
                            cfg.initialization.turbo_adapter_name):
        raise ValueError("audio-fix adapter name must be distinct from frozen adapters")
    if cfg.initialization.source_step != 250:
        raise ValueError("audio-fix training requires the released Stage-DMD step 250 source")
    if cfg.adapter.rank < 1 or cfg.adapter.alpha <= 0:
        raise ValueError("adapter rank/alpha must be positive")
    if cfg.adapter.target_policy != "portable_sequence_linear":
        raise ValueError(
            "only adapter.target_policy='portable_sequence_linear' is supported")
    if cfg.rollout.sampler != "res_multistep":
        raise ValueError("audio-fix training currently requires rollout.sampler='res_multistep'")
    if cfg.rollout.sigma_schedule != "simple_control":
        raise ValueError(
            "audio-fix production rollout requires rollout.sigma_schedule='simple_control'")
    if cfg.rollout.num_steps != 10:
        raise ValueError("audio-fix production rollout requires rollout.num_steps=10")
    if cfg.rollout.video_shift != 12.0 or cfg.rollout.audio_shift != 3.0:
        raise ValueError("audio-fix production rollout requires video/audio shifts 12/3")
    if cfg.generation.num_frames != 168:
        raise ValueError(
            "audio-fix production chunk contract requires generation.num_frames=168 "
            "(7 seconds at 24 fps; H3 aligns this to 175 frames internally)")
    if cfg.training.max_steps < 1:
        raise ValueError("training.max_steps must be >= 1")
    if cfg.training.audio_teacher_weight <= 0:
        raise ValueError("training.audio_teacher_weight must be > 0")
    if cfg.training.video_preserve_weight < 0:
        raise ValueError("training.video_preserve_weight must be >= 0")
    if cfg.optimizer.name != "prodigy_plus_schedule_free":
        raise ValueError("this trainer intentionally supports only Prodigy-Plus Schedule-Free")
    if cfg.optimizer.lr <= 0 or not (0 < cfg.optimizer.beta1 < 1) or not (0 < cfg.optimizer.beta2 < 1):
        raise ValueError("invalid Prodigy-Plus lr/betas")
    if cfg.optimizer.weight_decay != 0.0:
        raise ValueError("audio-fix baseline uses zero weight decay")
    if not cfg.optimizer.use_schedulefree:
        raise ValueError("audio-fix baseline requires Schedule-Free mode")
    if cfg.checkpoint.save_every < 1:
        raise ValueError("checkpoint.save_every must be >= 1")
    if cfg.checkpoint.keep_states < 1:
        raise ValueError("checkpoint.keep_states must be >= 1")
    if cfg.checkpoint.adapter_dtype not in ("float32", "bfloat16"):
        raise ValueError("checkpoint.adapter_dtype must be float32 or bfloat16")
    if cfg.generation.latent_height % 2 or cfg.generation.latent_width % 2:
        raise ValueError("generation latent height/width must be even")


__all__ = ["AudioFixConfig", "validate_audio_fix"]
