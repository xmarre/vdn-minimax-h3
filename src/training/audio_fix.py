"""On-policy audio-fix training primitives.

The frozen production student is VDN + Stage-B + released Turbo at their canonical
native unit LoRA scale. A generated-audio-scoped sidecar LoRA is the only trainable
component. Deployment multipliers such as Turbo 0.75 are not baked into the learned
correction. At one uniformly selected point on the production 10-step RES-multistep
sampler contract we compare:

* student audio x0 (canonical frozen stack + audio_fix)
* dense released H3 audio x0 on the same current state
* frozen canonical-stack video x0 (same stack without audio_fix)

The sigma source is the production ``MiniMax H3 SA-Solver Scheduler`` in
``simple_control`` mode. That mode delegates exactly to current ComfyUI ``simple``;
the discrete flow table selection is reproduced locally rather than approximated by a
continuous scheduler.

The other important sampler detail is MiniMax-H3's two-shift AV parameterisation.
Diffusers can carry video/audio directly on separate sigma grids, but ComfyUI's
``ModelSamplingAV`` puts both streams on the *video* sampler grid. Its carried audio
state is

    y_a = (sigma_v / sigma_a) * x_a

and therefore follows the ordinary video-sigma flow between audio noise and
``(video_shift / audio_shift) * x0_audio``. The model wrapper converts ``y_a`` back to
native ``x_a`` before H3 and converts H3's derivative back afterwards. A second-order
sampler is not invariant to silently solving the audio stream on ``sigma_a`` instead,
so this module explicitly reproduces the carried coordinate and applies RES history on
the video sigma grid for *both* streams.
"""
from __future__ import annotations

import torch

from src.models.hybrid_transform import set_teacher_mode
from src.training.dmd import FewStepSchedule
from src.training.res_multistep_math import (
    audio_carry_scale,
    carried_audio_x0,
    comfy_simple_flow_sigmas,
    native_audio_from_carry,
    res_multistep_update,
)
from src.training.turbo_adapter import set_active_adapters


class ResMultistepFewStepSchedule(FewStepSchedule):
    """Deterministic RES schedule matching the deployed ComfyUI H3 sampler math.

    The sigma table is exact ``simple_control``/Comfy ``simple``. ``video_rows`` are
    native video rows. ``audio_rows`` are the sampler-carried audio rows ``y_a``. Every
    model forward receives ``native_audio(index, audio_rows)``; every audio RES update
    uses the *video* sigma grid and the denoised carried target
    ``audio_scale * x0_audio``. At index 0 both sigmas are one, so initial audio noise
    already has the correct carried representation.
    """

    def __init__(self, num_steps, video_shift, audio_shift, sigma_schedule="simple_control"):
        if sigma_schedule != "simple_control":
            raise ValueError("audio-fix rollout requires sigma_schedule='simple_control'")
        self.num_steps = int(num_steps)
        self.video_shift, self.audio_shift = float(video_shift), float(audio_shift)
        self.sigma_schedule = sigma_schedule
        self.sigma_v = comfy_simple_flow_sigmas(self.num_steps, self.video_shift)
        self.sigma_a = comfy_simple_flow_sigmas(self.num_steps, self.audio_shift)
        self.t_v = 1.0 - self.sigma_v[:-1]
        self.t_a = 1.0 - self.sigma_a[:-1]
        self.audio_scale = audio_carry_scale(video_shift, audio_shift)
        self.reset()

    def reset(self):
        """Discard multistep history at the start of each independent rollout."""
        self._old_denoised_v = None
        self._old_denoised_a = None
        self._old_sigma_down = None
        self._old_sigma = None

    def native_audio(self, index, audio_carried):
        """Undo ComfyUI's ``sigma_v / sigma_a`` audio carry for one model forward."""
        return native_audio_from_carry(
            audio_carried, self.sigma_v[index], self.sigma_a[index])

    def x0(self, index, video_rows, audio_carried, velocity_v, velocity_a):
        """Return native video/audio x0 predictions from the carried sampler state."""
        native_audio = self.native_audio(index, audio_carried)
        return (
            video_rows + self.sigma_v[index].to(video_rows) * velocity_v,
            native_audio + self.sigma_a[index].to(native_audio) * velocity_a,
        )

    def step(self, index, video_rows, audio_carried, velocity_v, velocity_a):
        """Advance both streams on ComfyUI's common video-sigma RES grid."""
        if index == 0:
            self.reset()
        denoised_v, native_x0_a = self.x0(
            index, video_rows, audio_carried, velocity_v, velocity_a)
        denoised_a_carried = carried_audio_x0(
            native_x0_a, self.video_shift, self.audio_shift)
        sigma = self.sigma_v[index]
        sigma_next = self.sigma_v[index + 1]
        video_next = res_multistep_update(
            video_rows, denoised_v, sigma, sigma_next,
            previous_denoised=self._old_denoised_v,
            previous_sigma_down=self._old_sigma_down,
            previous_sigma=self._old_sigma,
        )
        audio_next = res_multistep_update(
            audio_carried, denoised_a_carried, sigma, sigma_next,
            previous_denoised=self._old_denoised_a,
            previous_sigma_down=self._old_sigma_down,
            previous_sigma=self._old_sigma,
        )
        self._old_denoised_v = denoised_v.detach()
        self._old_denoised_a = denoised_a_carried.detach()
        self._old_sigma_down = sigma_next
        self._old_sigma = sigma
        return video_next, audio_next


def set_production_role(model, vdn_adapter, turbo_adapter):
    """Enable canonical frozen VDN + Stage-B + Turbo at native PEFT scale."""
    set_teacher_mode(model, False)
    set_active_adapters(model, [vdn_adapter, turbo_adapter])


def set_dense_teacher_role(model):
    """Switch the shared 33B trunk to released dense H3 with all adapters disabled."""
    set_teacher_mode(model, True)
    set_active_adapters(model, [])


@torch.no_grad()
def rollout(model, bank, schedule, packed, video_rows, audio_rows, steps,
            vdn_adapter, turbo_adapter):
    """Roll the current audio-fix policy forward in ComfyUI's carried AV coordinates."""
    schedule.reset()
    set_production_role(model, vdn_adapter, turbo_adapter)
    with bank.scope(packed.audio_indices, enabled=True):
        for index in range(steps):
            t_v, t_a = schedule.times(index)
            native_audio = schedule.native_audio(index, audio_rows)
            velocity_v, velocity_a = model(
                **packed.inputs(video_rows, native_audio, t_v, t_a))
            video_rows, audio_rows = schedule.step(
                index, video_rows, audio_rows,
                velocity_v[0].float(), velocity_a[0].float())
    return video_rows, audio_rows


@torch.no_grad()
def dense_audio_target(model, bank, schedule, packed, video_rows, audio_rows, index):
    """Dense released-H3 audio x0 at the exact student's current carried state."""
    t_v, t_a = schedule.times(index)
    native_audio = schedule.native_audio(index, audio_rows)
    set_dense_teacher_role(model)
    with bank.disabled():
        velocity_v, velocity_a = model(
            **packed.inputs(video_rows, native_audio, t_v, t_a))
    _x0_v, x0_a = schedule.x0(
        index, video_rows, audio_rows,
        velocity_v[0].float(), velocity_a[0].float())
    return x0_a.detach()


@torch.no_grad()
def frozen_production_target(model, bank, schedule, packed, video_rows, audio_rows,
                             index, vdn_adapter, turbo_adapter):
    """Canonical VDN/Turbo x0 at the same carried state, used to preserve video."""
    t_v, t_a = schedule.times(index)
    native_audio = schedule.native_audio(index, audio_rows)
    set_production_role(model, vdn_adapter, turbo_adapter)
    with bank.disabled():
        velocity_v, velocity_a = model(
            **packed.inputs(video_rows, native_audio, t_v, t_a))
    x0_v, x0_a = schedule.x0(
        index, video_rows, audio_rows,
        velocity_v[0].float(), velocity_a[0].float())
    return x0_v.detach(), x0_a.detach()


__all__ = [
    "ResMultistepFewStepSchedule",
    "dense_audio_target",
    "frozen_production_target",
    "rollout",
    "set_dense_teacher_role",
    "set_production_role",
]
