"""Pytest test-collection fixtures: bypass parts of the lakonlab import
chain that aren't needed for the E1 smoke tests and that pull in heavy
training-only dependencies on macOS / CPU envs.

Two patches, both runtime-only and scoped to test collection:

1. Monkey-patch `mmcv.utils.registry.Registry._register_module` to ignore
   "already registered" collisions. mmgen 0.7.x unconditionally registers
   a `SiLU` activation that mmcv 1.7 already provides via torch >=1.7's
   native `nn.SiLU`, producing a `KeyError: SiLU is already registered`
   the moment any mmgen.models import runs. The collision is incidental
   to the registry's design (force=False is the safe default for
   production); silencing it during tests is harmless because we never
   exercise registry-backed model construction.

2. Stub `lakonlab.apis` and `lakonlab.datasets` in `sys.modules` so the
   `from .apis import *` and `from .datasets import *` lines in
   `lakonlab/__init__.py` are no-ops. Those branches pull in
   `apis/train.py` (mmcv DataParallel, full mmgen) and
   `datasets/image_prompt.py` (orjson + zstandard + s3 codecs), none of
   which the unit tests need.

Production code is unaffected: outside pytest, conftest.py is never
imported, so lakonlab/apis and lakonlab/datasets load normally.
"""

import sys
import types


# --- Patch 1: tolerate duplicate registrations under mmcv 1.7 + torch 2.0 ---
import mmcv  # noqa: E402 -- intentional ordering
from mmcv.utils.registry import Registry  # noqa: E402

_orig_register_module = Registry._register_module


def _tolerant_register_module(self, module, module_name=None, force=False):
    try:
        return _orig_register_module(self, module, module_name=module_name, force=force)
    except KeyError as exc:
        if 'already registered' in str(exc):
            return None  # the existing registration wins; mirror force=True semantics
        raise


Registry._register_module = _tolerant_register_module


# --- Patch 1b: stub the mmcv C extension so mmcv.ops modules can load ---
# The prebuilt `mmcv-full` wheels for macOS arm64 have ABI mismatches with
# torch's libc10 in this conda env. mmcv.ops is pulled in transitively via
# mmgen.models.architectures.stylegan, but the unit tests never actually
# call any C-extension op. Returning a stub from `load_ext` lets mmcv.ops
# import cleanly; any actual call into the stub raises NotImplementedError
# rather than corrupting silently.
import mmcv.utils.ext_loader as _ext_loader  # noqa: E402


class _StubExtModule:
    def __getattr__(self, name):
        def _unimplemented(*args, **kwargs):
            raise NotImplementedError(
                f'mmcv C-extension symbol {name!r} stubbed in tests; '
                'do not call mmcv.ops in unit tests')
        return _unimplemented


def _stub_load_ext(*args, **kwargs):
    return _StubExtModule()


_ext_loader.load_ext = _stub_load_ext


# --- Patch 2: stub lakonlab subpackages not needed by E1 smoke tests ---
# `lakonlab/__init__.py` does `from .X import *` for each of these. The
# stubs make those a no-op so the tests can reach
# `lakonlab.models.diffusions.gmflow` without dragging in training-only
# deps (mmcv DataParallel runner, bitsandbytes optimizer, S3 codecs, etc.).
for _name in ('lakonlab.apis', 'lakonlab.datasets'):
    if _name not in sys.modules:
        _stub = types.ModuleType(_name)
        _stub.__all__ = []
        sys.modules[_name] = _stub
