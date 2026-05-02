"""E1 cosine-VP smoke tests.

D3a — VP forward pass with `_smoke_test_vp = True` runs end-to-end without
NaN/Inf and emits the expected one-shot RuntimeWarning about `u_to_x_0`.

D3b — Calling `gmflow_posterior_mean` with explicit `alpha_t = 1 − sigma_t`
kwargs is bit-exact identical to the legacy no-kwargs path. Guards against
regressions on the linear codepath when adding the schedule-agnostic dispatch.

CPU only. No real checkpoints. Tests target the wrapper-level API; for D3a a
minimal `GMDiTPipeline` instance is built via `__new__` with stubbed
transformer/vae/scheduler so we exercise the actual `__call__` flow.

Run from repo root:
    pytest -x repos/piFlow/tests/test_e1_smoke.py
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# The full piFlow stack pulls mmcv via lakonlab/__init__.py -> apis/train.
# Skip cleanly if it is not installed in the test environment.
pytest.importorskip("mmcv")

from lakonlab.models.diffusions.gmflow import GMFlowMixin
from lakonlab.pipelines.pipeline_gmdit import GMDiTPipeline


# ---------------------------------------------------------------------------
# D3b — bit-exact equivalence: legacy linear path vs. general path with
#       alpha_t = 1 - sigma_t passed as kwargs.
# ---------------------------------------------------------------------------

class _BareMixin(GMFlowMixin):
    """Minimal GMFlowMixin instance that satisfies `time_scaling`.

    GMFlowMixin.time_scaling reads either `self.scheduler.config.num_train_timesteps`
    or `self.num_timesteps`. We use the latter to avoid pulling diffusers'
    scheduler machinery into a unit test.
    """
    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps


def _make_gm(B=1, K=2, C=3, H=4, W=4, seed=0):
    """Construct a deterministic GM dict with the shape contract the wrapper expects.

    Shape contract (from `gmflow_posterior_mean_jit`):
      - means        : (B, K, C, H, W)        gm_dim=-4
      - logweights   : (B, K, 1, H, W)
      - gm_vars      : size 1 at gm_dim=-4 AND channel_dim=-3 (asserted in JIT)
    """
    g = torch.Generator().manual_seed(seed)
    return dict(
        means=torch.randn(B, K, C, H, W, generator=g),
        logweights=torch.randn(B, K, 1, H, W, generator=g),
        gm_vars=torch.full((B, 1, 1, 1, 1), 0.1),
    )


def test_d3b_linear_equivalence():
    """D3b: passing alpha_t = 1 - sigma_t explicitly via kwargs must give
    the same output as the legacy no-kwargs path, bit-exact.

    Why bit-exact (not allclose): the wrapper docstring at gmflow.py:88-92
    promises this is bit-exact across 20 seeds * 5 (B,K,C,H,W) configs in
    fp32+fp64. A regression that drifts even one ULP would still indicate
    a divergence between the two code paths and is worth catching loudly.
    """
    mixin = _BareMixin(num_timesteps=1000)
    B, C, H, W = 1, 3, 4, 4
    torch.manual_seed(0)
    x_t = torch.randn(B, C, H, W)
    x_t_src = torch.randn(B, C, H, W)
    t = torch.tensor(500.0)
    t_src = torch.tensor(800.0)

    # Path 1: no alpha kwargs -> wrapper routes to legacy JIT,
    # which hardcodes alpha = 1 - sigma.
    gm_a = _make_gm(B=B, C=C, H=H, W=W)
    out_legacy = mixin.gmflow_posterior_mean(
        gm_a, x_t, x_t_src, t, t_src, prediction_type='x0')

    # Path 2: pass sigma + alpha kwargs explicitly with alpha = 1 - sigma.
    # Wrapper routes to gmflow_posterior_mean_jit_general.  The two should
    # agree bit-exact -- this is the regression contract on the linear path
    # when the schedule-agnostic dispatch is exercised.
    sigma_t_src = (t_src / mixin.time_scaling).reshape(1, 1, 1, 1)
    sigma_t = (t / mixin.time_scaling).reshape(1, 1, 1, 1)
    alpha_t_src = 1.0 - sigma_t_src
    alpha_t = 1.0 - sigma_t

    gm_b = _make_gm(B=B, C=C, H=H, W=W)  # fresh dict, same seed -> same data
    out_general = mixin.gmflow_posterior_mean(
        gm_b, x_t, x_t_src,
        sigma_t_src=sigma_t_src, sigma_t=sigma_t,
        alpha_t_src=alpha_t_src, alpha_t=alpha_t,
        prediction_type='x0')

    assert out_legacy.shape == out_general.shape, (
        f"shape mismatch: legacy {out_legacy.shape} vs general {out_general.shape}")
    assert torch.equal(out_legacy, out_general), (
        "Legacy linear path and general path with alpha=1-sigma "
        f"diverged. max|diff| = {(out_legacy - out_general).abs().max().item():.3e}")


# ---------------------------------------------------------------------------
# D3a — VP forward pass: full GMDiTPipeline.__call__ runs with
#       _smoke_test_vp = True, emits one RuntimeWarning, output is finite.
# ---------------------------------------------------------------------------

# Shapes used by the D3a fixture.  Kept tiny: 1 batch, 2 mixture components,
# 2 channels, 4x4 spatial.
_B = 1
_K = 2
_C = 2
_H = 4
_W = 4


class _FakeTransformerConfig:
    sample_size = _H
    in_channels = _C
    dtype = torch.float32


class _FakeTransformer:
    """Returns a fixed GM dict matching the shape contract u_to_x_0 expects."""
    config = _FakeTransformerConfig()
    dtype = torch.float32

    def __call__(self, x, timestep=None, class_labels=None):
        bs = x.size(0)
        return dict(
            means=torch.zeros(bs, _K, _C, _H, _W),
            logstds=torch.full((bs, 1, 1, 1, 1), -1.0),
            logweights=torch.zeros(bs, _K, 1, _H, _W),
        )


class _FakeVAE:
    """Stub VAE: returns a fixed-shape sample for `decode`. Spatial size
    upsampled by 8 to mimic SD-style VAE; values are zeros so output stays finite."""
    config = SimpleNamespace(scaling_factor=1.0)
    dtype = torch.float32

    @staticmethod
    def decode(x):
        return SimpleNamespace(sample=torch.zeros(x.size(0), 3, _H * 8, _W * 8))


class _FakeScheduler:
    """Minimal scheduler: descending timesteps in (eps, T) so sin/cos(pi*tbar/2)
    stay strictly inside (0, 1) -- both schedules avoid degenerate endpoints."""
    config = SimpleNamespace(num_train_timesteps=1000)
    timesteps = None  # populated by set_timesteps

    def set_timesteps(self, n, device=None):
        # Strictly inside (0, num_train_timesteps): avoids sin(0)=0 / cos(0)=1
        # boundary cases that would interact with the eps clamp.
        self.timesteps = torch.linspace(900.0, 100.0, n)

    def step(self, model_output, t, x, return_dict=False, prediction_type=None):
        # Identity step keeps the test deterministic and finite.
        return (x,)


def _build_pipeline(monkeypatch):
    """Construct a GMDiTPipeline that bypasses __init__/register_modules.
    All heavyweight components are stubbed; the load-bearing methods
    (`u_to_x_0`, `gmflow_posterior_mean`) come from GMFlowMixin and run for real.
    """
    pipe = GMDiTPipeline.__new__(GMDiTPipeline)
    pipe.transformer = _FakeTransformer()
    pipe.vae = _FakeVAE()
    pipe.scheduler = _FakeScheduler()
    pipe.spectrum_net = None  # Only consulted with output_mode='sample'.

    # DiffusionPipeline._execution_device is a property that walks model
    # components; without register_modules it would raise.  Override on the
    # class so attribute lookup finds a plain value.
    monkeypatch.setattr(GMDiTPipeline, '_execution_device',
                        torch.device('cpu'), raising=False)

    # Trivial overrides for DiffusionPipeline base methods we don't exercise.
    pipe.progress_bar = lambda x: x
    pipe.maybe_free_model_hooks = lambda: None
    return pipe


def test_d3a_vp_forward_no_nan(monkeypatch):
    """D3a: with `_smoke_test_vp = True`, the pipeline runs through the
    schedule-agnostic JIT dispatch end-to-end, emits exactly one
    RuntimeWarning about `u_to_x_0`, and produces finite output.

    The warning is captured (not silenced) per the E1-decisions D1b update:
    the gate is signal-only, allowing the smoke test to exercise the wiring
    while keeping the linear-`u_to_x_0` caveat visible to every future caller.
    """
    pipe = _build_pipeline(monkeypatch)
    pipe._smoke_test_vp = True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pipe(
            class_labels=[0],
            guidance_scale=0.0,
            num_inference_steps=2,
            num_inference_substeps=2,
            output_mode='mean',
            order=1,                # skip gm_2nd_order
            output_type='np',       # skip numpy_to_pil
        )

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) == 1, (
        f"expected exactly 1 RuntimeWarning, got {len(runtime_warnings)}: "
        f"{[str(w.message) for w in runtime_warnings]}")
    assert "u_to_x_0" in str(runtime_warnings[0].message), (
        "RuntimeWarning should mention u_to_x_0 (the linear-only conversion "
        f"that the smoke test bypasses): got {runtime_warnings[0].message!r}")

    images = result.images
    assert isinstance(images, np.ndarray), (
        f"expected np.ndarray output for output_type='np', got {type(images)}")
    assert np.isfinite(images).all(), (
        "VP smoke output contains NaN or Inf -- wrapper-dispatch path produced "
        "non-finite values; investigate posterior-mean computation under the "
        "schedule-agnostic JIT.")
