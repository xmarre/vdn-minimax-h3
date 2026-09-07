"""Deterministic RES multistep, Comfy simple, and MiniMax-H3 AV carry math.

ComfyUI's ``res_multistep`` sampler advances one packed AV state on the video sigma
schedule. The production workflow feeds it ``MiniMaxH3SASolverScheduler`` in
``simple_control`` mode, which delegates *exactly* to ComfyUI's discrete ``simple``
scheduler. ``ModelSamplingAV`` then carries audio in a transformed coordinate so H3 can
keep its native audio shift internally. These helpers reproduce those pieces without a
ComfyUI runtime dependency, making the production rollout arithmetic directly testable
on CPU.
"""
from __future__ import annotations

import math

import torch


def _phi1(z: torch.Tensor) -> torch.Tensor:
    """Stable first exponential-integrator phi function for nonzero ``z``."""
    return torch.expm1(z) / z


def _phi2(z: torch.Tensor) -> torch.Tensor:
    """Second exponential-integrator phi function derived from :func:`_phi1`."""
    return (_phi1(z) - 1.0) / z


def shifted_sigma(base_sigma, shift):
    """Apply MiniMax/flow time-SNR shift ``s*u / (1 + (s-1)u)``."""
    base_sigma = torch.as_tensor(base_sigma)
    shift = float(shift)
    if shift <= 0.0:
        raise ValueError("flow shift must be positive")
    return shift * base_sigma / (1.0 + (shift - 1.0) * base_sigma)


def comfy_simple_flow_sigmas(steps: int, shift: float, *, table_steps: int = 1000) -> torch.Tensor:
    """Reproduce ComfyUI ``simple_scheduler`` for ``ModelSamplingDiscreteFlow``.

    Comfy's flow sampling table contains ``table_steps`` shifted base points generated
    from ``1/table_steps .. 1``. ``simple_scheduler`` walks that table backwards with
    ``int(x * len(table) / steps)`` and appends a terminal zero. Reproducing the table
    selection rather than assuming a continuous linspace makes the training contract
    exact even when ``steps`` does not evenly divide the model's discrete table.
    """
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("simple scheduler steps must be a positive integer")
    if not isinstance(table_steps, int) or table_steps < 1:
        raise ValueError("simple scheduler table_steps must be a positive integer")
    if steps > table_steps:
        raise ValueError("simple scheduler steps cannot exceed the discrete flow table")
    shift = float(shift)
    if not math.isfinite(shift) or shift <= 0.0:
        raise ValueError("flow shift must be finite and positive")

    base_table = torch.arange(1, table_steps + 1, dtype=torch.float32) / float(table_steps)
    shifted_table = shifted_sigma(base_table, shift)
    stride = len(shifted_table) / steps
    selected = [shifted_table[-(1 + int(x * stride))] for x in range(steps)]
    return torch.cat((torch.stack(selected), torch.zeros(1, dtype=torch.float32)))


def audio_carry_scale(video_shift: float, audio_shift: float) -> float:
    """Constant clean-audio target scale used by ComfyUI ``ModelSamplingAV``."""
    video_shift, audio_shift = float(video_shift), float(audio_shift)
    if video_shift <= 0.0 or audio_shift <= 0.0:
        raise ValueError("video/audio shifts must be positive")
    return video_shift / audio_shift


def native_audio_from_carry(audio_carried: torch.Tensor, sigma_v, sigma_a) -> torch.Tensor:
    """Undo the sampler carry ``y=(sigma_v/sigma_a)*x_audio`` before an H3 forward."""
    sigma_v = torch.as_tensor(
        sigma_v, dtype=audio_carried.dtype, device=audio_carried.device)
    sigma_a = torch.as_tensor(
        sigma_a, dtype=audio_carried.dtype, device=audio_carried.device)
    if float(sigma_v) <= 0.0 or float(sigma_a) <= 0.0:
        raise ValueError("audio carry inversion requires positive video/audio sigma")
    return audio_carried * (sigma_a / sigma_v)


def carried_audio_x0(native_x0: torch.Tensor, video_shift: float,
                     audio_shift: float) -> torch.Tensor:
    """Map native H3 audio x0 to the denoised target seen by the common AV sampler."""
    return native_x0 * audio_carry_scale(video_shift, audio_shift)


def res_multistep_update(
    sample: torch.Tensor,
    denoised: torch.Tensor,
    sigma,
    sigma_next,
    *,
    previous_denoised: torch.Tensor | None = None,
    previous_sigma_down=None,
    previous_sigma=None,
) -> torch.Tensor:
    """Advance one deterministic RES-multistep interval.

    ``previous_*`` are the immediately preceding interval's denoised prediction,
    sigma-down endpoint, and model-evaluation sigma. The first interval and the
    terminal clean interval intentionally fall back to Euler, matching the production
    sampler topology.
    """
    sigma = torch.as_tensor(sigma, dtype=sample.dtype, device=sample.device)
    sigma_next = torch.as_tensor(sigma_next, dtype=sample.dtype, device=sample.device)

    if previous_denoised is None or float(sigma_next) == 0.0:
        derivative = (sample - denoised) / sigma
        return sample + derivative * (sigma_next - sigma)

    if previous_sigma_down is None or previous_sigma is None:
        raise ValueError(
            "second-order RES multistep requires previous_sigma_down and previous_sigma"
        )
    previous_sigma_down = torch.as_tensor(
        previous_sigma_down, dtype=sample.dtype, device=sample.device)
    previous_sigma = torch.as_tensor(
        previous_sigma, dtype=sample.dtype, device=sample.device)
    if float(previous_sigma_down) <= 0.0 or float(previous_sigma) <= 0.0:
        raise ValueError("previous RES multistep sigma points must be positive")

    t = -torch.log(sigma)
    t_old = -torch.log(previous_sigma_down)
    t_next = -torch.log(sigma_next)
    t_prev = -torch.log(previous_sigma)
    h = t_next - t
    c2 = (t_prev - t_old) / h

    z = -h
    phi1 = _phi1(z)
    phi2 = _phi2(z)
    b2 = torch.nan_to_num(phi2 / c2, nan=0.0)
    b1 = torch.nan_to_num(phi1 - b2, nan=0.0)
    return torch.exp(-h) * sample + h * (b1 * denoised + b2 * previous_denoised)


__all__ = [
    "audio_carry_scale",
    "carried_audio_x0",
    "comfy_simple_flow_sigmas",
    "native_audio_from_carry",
    "res_multistep_update",
    "shifted_sigma",
]
