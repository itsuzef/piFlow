# Validation Tests — Schedule-Agnostic Dispatch & Fourier Policy

This document describes how to run the validation test suite that covers the
schedule-agnostic posterior-mean dispatch and the Fourier policy integration.

---

## Overview of What Is Being Validated

Two properties are verified by the test suite:

| Test | What it checks |
|---|---|
| `test_vp_forward_no_nan` | Full pipeline forward pass with a cosine VP schedule produces finite output and correctly signals the linear-`u_to_x_0` limitation via a `RuntimeWarning`. |
| `test_linear_equivalence` | Calling `gmflow_posterior_mean` with explicit `alpha_t = 1 − sigma_t` kwargs is bit-exact identical to the legacy no-kwargs (linear) path — guards regressions on the linear codepath when the schedule-agnostic dispatch is active. |

Both tests run on CPU with no real checkpoints. They target the
wrapper-level API by constructing minimal, stubbed pipeline instances.

---

## Prerequisites

### Python environment

The tests require the same conda/virtualenv used for training and inference.
Key dependencies (versions pinned in `requirements.txt`):

```
mmcv-full >= 1.7
mmgen == 0.7.*
torch >= 2.0
diffusers >= 0.20
pytest >= 7
```

Install the repo in editable mode from the repo root:

```bash
pip install -e .
```

### Note on macOS / CPU environments

The `conftest.py` in `tests/` patches two known incompatibilities that
arise on macOS arm64 or any CPU-only environment:

- **mmcv `SiLU` registry collision** — silenced transparently; no action required.
- **mmcv C-extension stub** — `mmcv.ops` symbols are replaced with stubs that
  raise `NotImplementedError` if actually called. The unit tests never invoke
  C-extension ops, so this is safe.

These patches are test-only and do not affect production code.

---

## Running the Tests

From the repository root:

```bash
# Run the full smoke test suite
pytest -x tests/test_e1_smoke.py -v
```

Expected output (both tests pass):

```
tests/test_e1_smoke.py::test_linear_equivalence   PASSED
tests/test_e1_smoke.py::test_vp_forward_no_nan    PASSED
```

The VP forward-pass test intentionally captures one `RuntimeWarning`; this is
expected and is part of the pass criteria (see notes below).

### Running a single test

```bash
pytest tests/test_e1_smoke.py::test_linear_equivalence -v
pytest tests/test_e1_smoke.py::test_vp_forward_no_nan -v
```

---

## GPU Validation

The unit tests above run on CPU. Before running on GPU, verify the environment:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Running the smoke tests on GPU

The tests use CPU tensors internally, but running pytest in a GPU environment
confirms that imports and CUDA-capable torch are healthy:

```bash
pytest tests/test_e1_smoke.py -v
```

### Full inference validation (GPU required)

To validate the full Fourier inference pipeline on GPU with the ImageNet
evaluation harness, use the provided config:

```bash
torchrun --nnodes=1 --nproc_per_node=8 tools/test.py \
    configs/piflow_imagenet/piflow_fourier_inference.py \
    --launcher pytorch --diff_seed
```

> **Note:** `pretrained=None` in the config — no Fourier-trained checkpoint
> exists yet. Set `pretrained` to a checkpoint path (or pass `--ckpt <path>`)
> once a trained model is available. Until then, the test run exercises the
> full data and inference pipeline with random weights, which is sufficient
> to validate wiring and shape correctness.

To pass a custom checkpoint:

```bash
torchrun --nnodes=1 --nproc_per_node=8 tools/test.py \
    configs/piflow_imagenet/piflow_fourier_inference.py \
    --ckpt <PATH_TO_CHECKPOINT> \
    --launcher pytorch --diff_seed
```

### Multi-node setup

For multi-node DDP, set `--nnodes` and `MASTER_ADDR` / `MASTER_PORT`
as appropriate for your cluster:

```bash
torchrun --nnodes=<N> --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    tools/test.py configs/piflow_imagenet/piflow_fourier_inference.py \
    --launcher pytorch --diff_seed
```

---

## Pass Criteria

### Unit tests

| Test | Pass condition |
|---|---|
| `test_linear_equivalence` | Output tensors are **bit-exact** (`torch.equal`) across the legacy and kwargs-dispatch paths. |
| `test_vp_forward_no_nan` | (1) Output `images` is a finite `np.ndarray`. (2) Exactly **one** `RuntimeWarning` is emitted, and its message mentions `u_to_x_0`. |

The bit-exact requirement on `test_linear_equivalence` is intentionally strict:
both code paths must follow identical floating-point operations on the linear
schedule; any ULP-level drift would indicate a path divergence.

### Inference validation (GPU)

With random weights, the run should complete without errors and produce
images (likely noise). FID / IS metrics are only meaningful with a trained
checkpoint. Once a checkpoint is available, expected FID targets are the same
as the GM-DiT baseline in `configs/piflow_imagenet/README.md`.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'mmcv'`**
Install `mmcv-full` with the correct CUDA version for your environment:
```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/<cu>/<torch>/index.html
```

**`KeyError: SiLU is already registered`** (outside tests)
This is an mmgen / mmcv version conflict; it is suppressed inside `conftest.py`
but will surface in production if mmgen is imported before the patch is applied.
Upgrade to `mmcv-full >= 1.7.2` or set `force=True` in the relevant
`Registry._register_module` call.

**`RuntimeWarning` about `u_to_x_0` appears during normal inference**
This warning is only emitted when `pipeline._smoke_test_vp = True`. It should
not appear during standard inference. If it does, confirm that the attribute
is not being set inadvertently.

**NCCL errors on multi-GPU runs**
Ensure `dist_params = dict(backend='nccl')` is set in the config (it is by
default) and that all GPUs are visible (`CUDA_VISIBLE_DEVICES`).
