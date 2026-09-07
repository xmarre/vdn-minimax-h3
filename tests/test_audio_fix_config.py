from __future__ import annotations

import pytest

from src.config.audio_fix import AudioFixConfig, validate_audio_fix


def test_production_defaults_validate_and_have_single_turbo_selector():
    cfg = AudioFixConfig()
    validate_audio_fix(cfg)
    assert cfg.initialization.source_step == 250
    assert cfg.initialization.vdn_adapter_name == "default"
    assert cfg.initialization.turbo_adapter_name == "turbo"
    assert cfg.rollout.sampler == "res_multistep"
    assert cfg.rollout.sigma_schedule == "simple_control"
    assert cfg.rollout.num_steps == 10
    assert cfg.rollout.video_shift == 12.0
    assert cfg.rollout.audio_shift == 3.0
    assert cfg.generation.num_frames == 168
    # Frozen adapter strength is deliberately not exposed as a training knob. The
    # released adapters run at their canonical/native unit scale; deployment tuning
    # such as Turbo 0.75 is evaluated after training.
    assert not hasattr(cfg.initialization, "stage_b_strength")
    assert not hasattr(cfg.initialization, "turbo_strength")
    assert not hasattr(cfg.adapter, "source_adapter")


@pytest.mark.parametrize("name", ["", ".", "..", "/tmp/fix", "nested/fix", "nested\\fix", "bad\x00name"])
def test_adapter_name_must_be_one_safe_path_segment(name):
    cfg = AudioFixConfig()
    cfg.adapter.name = name
    with pytest.raises(ValueError, match="safe path segment"):
        validate_audio_fix(cfg)


def test_production_source_step_is_fixed():
    cfg = AudioFixConfig()
    cfg.initialization.source_step = 249
    with pytest.raises(ValueError, match="step 250"):
        validate_audio_fix(cfg)


def test_production_sigma_schedule_is_fixed_to_simple_control():
    cfg = AudioFixConfig()
    cfg.rollout.sigma_schedule = "normal"
    with pytest.raises(ValueError, match="simple_control"):
        validate_audio_fix(cfg)


@pytest.mark.parametrize("steps", [2, 8, 9, 11])
def test_production_rollout_requires_exactly_ten_steps(steps):
    cfg = AudioFixConfig()
    cfg.rollout.num_steps = steps
    with pytest.raises(ValueError, match="num_steps=10"):
        validate_audio_fix(cfg)


@pytest.mark.parametrize("video_shift,audio_shift", [(11.0, 3.0), (12.0, 2.0), (12.1, 3.0)])
def test_production_rollout_requires_12_3_shifts(video_shift, audio_shift):
    cfg = AudioFixConfig()
    cfg.rollout.video_shift = video_shift
    cfg.rollout.audio_shift = audio_shift
    with pytest.raises(ValueError, match="shifts 12/3"):
        validate_audio_fix(cfg)


@pytest.mark.parametrize("frames", [167, 169, 175, 345])
def test_production_chunk_request_is_fixed_to_seven_seconds(frames):
    cfg = AudioFixConfig()
    cfg.generation.num_frames = frames
    with pytest.raises(ValueError, match="generation.num_frames=168"):
        validate_audio_fix(cfg)


@pytest.mark.parametrize("field", ["save_every", "keep_states"])
def test_checkpoint_retention_values_must_be_positive(field):
    cfg = AudioFixConfig()
    setattr(cfg.checkpoint, field, 0)
    with pytest.raises(ValueError, match=field):
        validate_audio_fix(cfg)
