from __future__ import annotations

import math

import torch

from src.training.res_multistep_math import (
    audio_carry_scale,
    carried_audio_x0,
    comfy_simple_flow_sigmas,
    native_audio_from_carry,
    res_multistep_update,
    shifted_sigma,
)


def _euler(sample, denoised, sigma, sigma_next):
    return sample + (sample - denoised) / sigma * (sigma_next - sigma)


def _reference_second_order(sample, denoised, old_denoised,
                            sigma, sigma_next, old_sigma_down, previous_sigma):
    t = -math.log(float(sigma))
    t_old = -math.log(float(old_sigma_down))
    t_next = -math.log(float(sigma_next))
    t_prev = -math.log(float(previous_sigma))
    h = t_next - t
    c2 = (t_prev - t_old) / h
    z = -h
    phi1 = math.expm1(z) / z
    phi2 = (phi1 - 1.0) / z
    b2 = phi2 / c2
    b1 = phi1 - b2
    return math.exp(-h) * sample + h * (b1 * denoised + b2 * old_denoised)


def test_first_interval_is_euler():
    sample = torch.tensor([[1.0, -0.5]], dtype=torch.float64)
    denoised = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    got = res_multistep_update(sample, denoised, 0.9, 0.7)
    want = _euler(sample, denoised, 0.9, 0.7)
    assert torch.allclose(got, want, atol=1e-12, rtol=1e-12)


def test_terminal_interval_is_euler_even_with_history():
    sample = torch.tensor([[0.4, -0.2]], dtype=torch.float64)
    denoised = torch.tensor([[0.1, 0.3]], dtype=torch.float64)
    old = torch.tensor([[0.2, 0.1]], dtype=torch.float64)
    got = res_multistep_update(
        sample,
        denoised,
        0.2,
        0.0,
        previous_denoised=old,
        previous_sigma_down=0.2,
        previous_sigma=0.35,
    )
    want = _euler(sample, denoised, 0.2, 0.0)
    assert torch.allclose(got, want, atol=1e-12, rtol=1e-12)


def test_second_order_matches_independent_closed_form():
    sample = torch.tensor([[0.8, -0.7]], dtype=torch.float64)
    denoised = torch.tensor([[0.3, 0.1]], dtype=torch.float64)
    old = torch.tensor([[0.5, -0.2]], dtype=torch.float64)
    sigma = 0.6
    sigma_next = 0.42
    old_sigma_down = 0.6
    previous_sigma = 0.78
    got = res_multistep_update(
        sample,
        denoised,
        sigma,
        sigma_next,
        previous_denoised=old,
        previous_sigma_down=old_sigma_down,
        previous_sigma=previous_sigma,
    )
    want = _reference_second_order(
        sample,
        denoised,
        old,
        sigma,
        sigma_next,
        old_sigma_down,
        previous_sigma,
    )
    assert torch.allclose(got, want, atol=1e-12, rtol=1e-12)


def test_second_order_requires_complete_history():
    sample = torch.ones(1, 2)
    denoised = torch.zeros(1, 2)
    old = torch.zeros(1, 2)
    try:
        res_multistep_update(
            sample,
            denoised,
            0.6,
            0.4,
            previous_denoised=old,
            previous_sigma_down=None,
            previous_sigma=0.8,
        )
    except ValueError as exc:
        assert "previous_sigma_down" in str(exc)
    else:
        raise AssertionError("incomplete second-order history must fail")


def test_simple_control_matches_current_comfy_discrete_simple_at_ten_steps():
    """Mirror ComfyUI simple_scheduler's exact 1000-point flow-table selection."""
    table_steps = 1000
    steps = 10
    base_table = torch.arange(1, table_steps + 1, dtype=torch.float32) / table_steps
    stride = len(base_table) / steps
    simple_base = torch.tensor(
        [float(base_table[-(1 + int(i * stride))]) for i in range(steps)] + [0.0],
        dtype=torch.float32,
    )
    # At 10 steps the 1000-point table lands exactly at 1.0, .9, ... .1, 0.
    assert torch.equal(simple_base, torch.linspace(1.0, 0.0, 11, dtype=torch.float32))
    assert torch.equal(
        comfy_simple_flow_sigmas(steps, 12.0),
        shifted_sigma(simple_base, 12.0),
    )
    assert torch.equal(
        comfy_simple_flow_sigmas(steps, 3.0),
        shifted_sigma(simple_base, 3.0),
    )


def test_simple_control_preserves_discrete_table_rounding_for_nondivisor_steps():
    """Guard against accidentally replacing Comfy simple with a continuous linspace."""
    steps = 7
    got = comfy_simple_flow_sigmas(steps, 1.0)
    base_table = torch.arange(1, 1001, dtype=torch.float32) / 1000.0
    stride = 1000 / steps
    expected = torch.tensor(
        [float(base_table[-(1 + int(i * stride))]) for i in range(steps)] + [0.0],
        dtype=torch.float32,
    )
    assert torch.equal(got, expected)
    assert not torch.equal(got, torch.linspace(1.0, 0.0, steps + 1))


def test_minimax_audio_carry_is_one_common_video_sigma_flow():
    """Mirror ComfyUI ModelSamplingAV's algebra without importing ComfyUI.

    Native audio follows sigma_a while the sampler carries y=(sigma_v/sigma_a)*x_a.
    With shifted flow sigmas this must equal an ordinary flow on sigma_v whose clean
    endpoint is (video_shift/audio_shift)*x0_audio. That identity is why RES history
    must be built on sigma_v rather than separately on sigma_a.
    """
    dtype = torch.float64
    base = torch.tensor(0.37, dtype=dtype)
    sigma_v = shifted_sigma(base, 12.0)
    sigma_a = shifted_sigma(base, 3.0)
    noise = torch.tensor([[0.8, -0.5, 0.2]], dtype=dtype)
    x0 = torch.tensor([[-0.1, 0.7, 0.4]], dtype=dtype)

    native = sigma_a * noise + (1.0 - sigma_a) * x0
    carried = native * (sigma_v / sigma_a)
    scale = audio_carry_scale(12.0, 3.0)
    common_flow = sigma_v * noise + (1.0 - sigma_v) * (scale * x0)

    assert scale == 4.0
    assert torch.allclose(carried, common_flow, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        native_audio_from_carry(carried, sigma_v, sigma_a),
        native,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.equal(carried_audio_x0(x0, 12.0, 3.0), 4.0 * x0)


def test_audio_carry_helpers_fail_closed_at_invalid_geometry():
    x = torch.ones(1, 2)
    for video_shift, audio_shift in ((0.0, 3.0), (12.0, 0.0)):
        try:
            audio_carry_scale(video_shift, audio_shift)
        except ValueError:
            pass
        else:
            raise AssertionError("non-positive AV shifts must fail")
    try:
        native_audio_from_carry(x, 0.0, 0.5)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("terminal carry inversion must fail")
