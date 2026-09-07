from __future__ import annotations

import copy
import io
import math

import torch

from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree


def _make_optimizer(parameter):
    return ProdigyPlusScheduleFree(
        [parameter],
        lr=1.0,
        betas=(0.95, 0.99),
        weight_decay=0.0,
        use_schedulefree=True,
        use_stableadamw=True,
        split_groups=False,
        stochastic_rounding=False,
    )


def _assert_state_close(left, right):
    if torch.is_tensor(left):
        assert torch.is_tensor(right)
        # Prodigy-Plus may restore compact factored state into the parameter dtype.
        # That can change a BF16-vs-FP32 bookkeeping tensor by one rounding unit even
        # while the resumed parameter trajectory is numerically identical. Compare the
        # optimizer state numerically rather than requiring byte-identical dtypes.
        torch.testing.assert_close(
            left.float(), right.float(), rtol=1.0e-2, atol=1.0e-12,
            check_dtype=False, check_device=False)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_close(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_state_close(a, b)
    elif isinstance(left, float):
        assert isinstance(right, (float, int))
        assert math.isclose(left, float(right), rel_tol=1.0e-2, abs_tol=1.0e-12)
    else:
        assert left == right


def test_prodigy_plus_schedule_free_train_eval_and_state_roundtrip():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    initial = parameter.detach().clone()
    optimizer = _make_optimizer(parameter)
    optimizer.train()

    parameter.square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert not torch.equal(parameter.detach(), initial)

    train_value = parameter.detach().clone()
    optimizer.eval()
    eval_value = parameter.detach().clone()
    assert torch.isfinite(eval_value).all()
    optimizer.train()
    restored_train_value = parameter.detach().clone()
    assert torch.isfinite(restored_train_value).all()
    assert torch.allclose(restored_train_value, train_value, rtol=1e-6, atol=1e-7)

    state = copy.deepcopy(optimizer.state_dict())
    parameter2 = torch.nn.Parameter(restored_train_value.clone())
    optimizer2 = _make_optimizer(parameter2)
    optimizer2.load_state_dict(copy.deepcopy(state))
    optimizer2.train()
    assert optimizer2.param_groups[0]["use_schedulefree"] is True

    # A resumed optimizer must continue numerically from the same Schedule-Free state.
    gradient_probe = torch.tensor([0.25, -0.5], dtype=torch.float32)
    (parameter * gradient_probe).sum().backward()
    (parameter2 * gradient_probe).sum().backward()
    optimizer.step()
    optimizer2.step()

    assert torch.allclose(parameter, parameter2, rtol=1e-6, atol=1e-7)
    _assert_state_close(optimizer.state_dict(), optimizer2.state_dict())


def test_prodigy_state_is_compatible_with_restricted_torch_load():
    """The auto-resume payload must remain loadable without unrestricted pickle."""
    parameter = torch.nn.Parameter(torch.tensor([0.5, -0.25], dtype=torch.float32))
    optimizer = _make_optimizer(parameter)
    optimizer.train()
    parameter.square().sum().backward()
    optimizer.step()

    payload = {
        "format": 1,
        "stage": "audio_fix",
        "step": 1,
        "optimizer": optimizer.state_dict(),
        "bank": {"pair": torch.ones(2, 2)},
        "rng_per_rank": [{"torch": torch.get_rng_state()}],
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location="cpu", weights_only=True)

    assert restored["format"] == 1
    assert restored["stage"] == "audio_fix"
    assert restored["step"] == 1
    assert restored["optimizer"]["param_groups"][0]["use_schedulefree"] is True
