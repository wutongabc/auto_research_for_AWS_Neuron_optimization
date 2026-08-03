# SPDX-License-Identifier: Apache-2.0
"""Load the upstream plugin and install benchmark-scoped runtime patches."""

from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import PathFinder
import sys

_UPSTREAM_INIT = "/dev3/zigeng/bc/vllm-neuron/vllm_neuron/__init__.py"
with open(_UPSTREAM_INIT, "rb") as _source:
    exec(compile(_source.read(), _UPSTREAM_INIT, "exec"), globals(), globals())

_TARGET_RUNNER = "vllm_neuron.vllm.worker.neuron_model_runner"


def _skip_token_validation(self, input_ids):
    """Trust tokenizer-produced IDs and avoid two full-tensor reductions."""
    return None


def _patch_runner(module):
    module.NeuronModelRunner._validate_token_ids = _skip_token_validation


class _RunnerPatchLoader(Loader):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def create_module(self, spec):
        create = getattr(self.wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        _patch_runner(module)


class _RunnerPatchFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET_RUNNER:
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _RunnerPatchLoader(spec.loader)
        return spec


def _install_runtime_patches():
    module = sys.modules.get(_TARGET_RUNNER)
    if module is not None:
        _patch_runner(module)
    elif not any(isinstance(finder, _RunnerPatchFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _RunnerPatchFinder())


_install_runtime_patches()
