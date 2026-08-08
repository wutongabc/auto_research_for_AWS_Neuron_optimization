# SPDX-License-Identifier: Apache-2.0
"""Tensor capture for accuracy debugging.

Architecture:
    1. TensorCaptureModel (pre-compile wrapper): Registers hooks and returns
       captured tensors as extra outputs.
    2. Model runner extracts (strips) captures via writer.extract() and saves
       to disk via registry.write() with scheduler metadata.

Usage:
    from vllm_neuron.accuracy import TensorCaptureModel

    # Wrap before compile
    capture_model = TensorCaptureModel(model, ["model.layers.0-31"])
    compiled = torch.compile(capture_model, backend="vllm_neuron", fullgraph=True)

    # Model runner strips and saves after each forward pass:
    all_outputs = compiled(**kwargs)
    model_output, captures = writer.extract(all_outputs, capture_model.original_output_count)
    registry.write(captures, capture_names, req_ids, positions, is_prefill)
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from vllm_neuron.accuracy.tensor_io import (
    CapturedForwardPass,
    ForwardPassMetadata,
    write as tensor_io_write,
)

logger = logging.getLogger(__name__)


def expand_patterns(patterns: List[str]) -> List[str]:
    """Expand capture patterns supporting ranges and regex.

    Supports:
        - Exact: "model.layers.0"
        - Range: "model.layers.0-31" expands to layers 0 through 31
        - Range with suffix: "model.layers.0-31.self_attn"
        - Regex: Patterns containing regex metacharacters (*, +, ?, etc.)
          are passed through for regex matching in _matches()

    Args:
        patterns: List of module path patterns

    Returns:
        Expanded list of patterns

    Example:
        >>> expand_patterns(["model.layers.0-2", "lm_head"])
        ['model.layers.0', 'model.layers.1', 'model.layers.2', 'lm_head']
    """
    expanded = []
    # Match range anywhere: prefix.N-M or prefix.N-M.suffix
    range_pattern = re.compile(r"^(.+)\.(\d+)-(\d+)(\..*)?$")

    for pattern in patterns:
        match = range_pattern.match(pattern)
        if match:
            prefix, start, end, suffix = match.groups()
            suffix = suffix or ""
            for i in range(int(start), int(end) + 1):
                expanded.append(f"{prefix}.{i}{suffix}")
        else:
            expanded.append(pattern)

    return expanded


class TensorRegistry:
    """Registry for captured tensors during torch.compile tracing.

    Two usage modes:

    1. Hook-based captures (TensorCaptureModel): Each TensorCaptureModel creates
       its own TensorRegistry instance. This supports multiple models with
       overlapping module names (e.g., target + draft in speculative decoding).

    2. Manual captures (capture_tensor()): Uses the global singleton via
       get_instance(). For multi-model scenarios, call reset_instance() before
       creating each TensorCaptureModel to avoid stale captures.
    """

    _instance: Optional["TensorRegistry"] = None

    @classmethod
    def get_instance(cls) -> "TensorRegistry":
        """Get global singleton for manual capture_tensor() calls."""
        if cls._instance is None:
            cls._instance = TensorRegistry()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the global singleton.

        Call this before creating a new TensorCaptureModel when using manual
        capture_tensor() calls in multi-model scenarios.
        """
        cls._instance = None

    def __init__(self):
        # Use lists instead of OrderedDicts to avoid torch._dynamo errors
        # from dict mutation tracking ("mapping proxy affected by dictionary
        # mutation") introduced in torch 2.9.  Dynamo handles list appends.
        self._module_items: List[Tuple[str, torch.Tensor]] = []
        self._manual_items: List[Tuple[str, torch.Tensor]] = []
        self._enabled = False

    def configure(self, enabled: bool = True):
        self._enabled = enabled

    def clear(self) -> None:
        """Clear all registered tensors for new forward pass."""
        self._module_items.clear()
        self._manual_items.clear()

    def register_module_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if self._enabled:
            self._module_items.append((name, tensor.clone().detach()))

    def register_manual_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if self._enabled:
            key = f"manual.{name}"
            existing = {n for n, _ in self._manual_items}
            if key in existing:
                i = 1
                while f"{key}_{i}" in existing:
                    i += 1
                key = f"{key}_{i}"
            self._manual_items.append((key, tensor.clone().detach()))

    def get_all_tensors(self) -> List[torch.Tensor]:
        """Get all tensors in order (module first, then manual)."""
        return [t for _, t in self._module_items] + [t for _, t in self._manual_items]

    def get_all_names(self) -> List[str]:
        """Get all tensor names in order."""
        return [n for n, _ in self._module_items] + [n for n, _ in self._manual_items]

    @property
    def enabled(self) -> bool:
        return self._enabled


class TensorCaptureModel(nn.Module):
    """Pre-compile wrapper that captures module outputs via hooks.

    Registers forward hooks on specified modules and returns their outputs
    as additional model outputs. Must wrap model BEFORE torch.compile().

    Each TensorCaptureModel has its own TensorRegistry to support multiple
    models with overlapping module names (e.g., target + draft in spec decode).

    Args:
        model: The model to wrap
        modules_to_capture: Module paths to capture. Supports:
            - Exact: "model.layers.0"
            - Range: "model.layers.0-31"
            - Wildcard: "layers.*" matches module and all submodules
            - Suffix: "lm_head" matches "model.lm_head"
            - Regex: Standard regex patterns (e.g., "layers\\.\\d+\\.self_attn")
        model_prefix: Optional prefix for capture names to distinguish models
    """

    def __init__(
        self, model: nn.Module, modules_to_capture: List[str], model_prefix: str = ""
    ):
        super().__init__()
        self.model = model
        self._hooks: List[Any] = []
        self._module_names: List[str] = []
        self._capture_names: List[str] = []
        self._model_prefix = model_prefix

        # Per-model registry to avoid conflicts with other TensorCaptureModel instances
        self._registry = TensorRegistry()
        self._registry.configure(enabled=True)
        TensorRegistry._instance = self._registry

        patterns = expand_patterns(modules_to_capture)
        module_dict = dict(model.named_modules())
        matched = set()

        for name, module in module_dict.items():
            if not name:  # Skip root module
                continue
            for pattern in patterns:
                if self._matches(name, pattern) and name not in matched:
                    hook = module.register_forward_hook(self._make_hook(name))
                    self._hooks.append(hook)
                    self._module_names.append(name)
                    matched.add(name)
                    logger.info(f"Registered capture hook for: {name}")
                    break  # Don't match same module with multiple patterns

        unmatched = [
            p for p in patterns if not any(self._matches(n, p) for n in module_dict)
        ]
        if unmatched:
            logger.warning(f"No modules found for patterns: {unmatched}")

        logger.info(f"TensorCaptureModel: {len(self._module_names)} modules registered")

    @property
    def capture_names(self) -> List[str]:
        """Get capture names. Read AFTER compile to include manual registrations."""
        return self._capture_names if self._capture_names else list(self._module_names)

    def _matches(self, name: str, pattern: str) -> bool:
        """Match module name against pattern using regex.

        All patterns are treated as Python regex matched against the full module name.

        Examples:
            - "model\\.layers\\.0$" - exact match for model.layers.0
            - "model\\.layers\\..*" - model.layers and all descendants
            - ".*self_attn$" - all modules ending in self_attn
            - "model\\.layers\\.[0-9]+$" - layer modules only (not children)
        """
        try:
            return bool(re.fullmatch(pattern, name))
        except re.error:
            return name == pattern

    def _make_hook(self, name: str):
        # TODO: Support capturing input tensors from a module
        # TODO: Support complex output structures (nested tuples, dicts, NamedTuples)
        # Currently only captures first tensor from output tuple
        prefix = f"{self._model_prefix}." if self._model_prefix else ""

        def hook(module: nn.Module, input: Any, output: Any):
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor):
                self._registry.register_module_tensor(f"{prefix}{name}", tensor)

        return hook

    def forward(self, *args, **kwargs):
        """Forward pass returning (*original_outputs, *captures)."""
        original_output = self.model(*args, **kwargs)

        # Track original output count for writer.extract()
        if isinstance(original_output, tuple):
            self._original_output_count = len(original_output)
        else:
            self._original_output_count = 1

        captures = self._registry.get_all_tensors()
        self._capture_names = self._registry.get_all_names()

        if captures:
            if isinstance(original_output, tuple):
                return (*original_output, *captures)
            return (original_output, *captures)
        return original_output

    @property
    def original_output_count(self) -> int:
        """Original model output count (set after first forward)."""
        return getattr(self, "_original_output_count", 1)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._registry.configure(enabled=False)

    def register_clear_hook(self, compiled_model: nn.Module) -> None:
        """Register a pre-forward hook on the compiled model to clear the registry."""
        compiled_model.register_forward_pre_hook(
            lambda m, args, kwargs: self._registry.clear(),
            with_kwargs=True,
        )


class CaptureWriter:
    """Tensor capture I/O between model output and disk.

    extract() separates captured tensors from the model's output tuple,
    returning the original model output and the captured tensors separately.

    write() persists captured tensors to disk with this replica's local
    scheduler metadata (request IDs, positions, prefill/decode phase) in
    a 4-layer directory layout with per-bucket sequential counters. When
    no tensors are captured, only metadata JSON files are written
    (metadata-only mode for offline HF-to-Neuron conversion). For
    tensors captured after a cross-DP all-gather, the tensor contains
    tokens from all replicas but the metadata only describes this
    replica's portion. Cross-replica metadata can be assembled at read
    time by matching forward passes across dp{N}/ directories by
    dir_name.

    Disk layout::

        capture_dir/
          dp{dp_rank}/
            prefill_s{bucket_size}_{counter}/
              model.layers.0.input_layernorm/
                rank0.pt    # TP-local rank
              {dir_name}_meta.json
            decode_b{batch_size}_{counter}/
              ...
    """

    def __init__(
        self,
        capture_dir: str,
        dp_rank: int = 0,
        tp_rank: int = 0,
        capture_filter: Optional[set] = None,
    ):
        self.capture_dir = capture_dir
        self.dp_rank = dp_rank
        self.tp_rank = tp_rank
        # Optional write-time filter. When set, only tensors whose names
        # are in this set are written to disk; others are silently skipped.
        # None means write all captured tensors.
        self._capture_filter = capture_filter
        self._counters: Dict[tuple, int] = {}
        self._enabled = False
        os.makedirs(self.capture_dir, exist_ok=True)

    def enable(self) -> None:
        """Enable writing captures to disk."""
        self._enabled = True

    def disable(self) -> None:
        """Disable writing captures to disk."""
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def extract(
        self,
        model_output: Any,
        original_output_count: int,
    ) -> Tuple[Any, Tuple[torch.Tensor, ...]]:
        """Separate captured tensors from model output.

        Args:
            model_output: Raw output from the compiled model (includes captures
                appended by TensorCaptureModel).
            original_output_count: Number of original model outputs (from
                TensorCaptureModel.original_output_count).

        Returns:
            (original_outputs, captures) tuple.
        """
        if not isinstance(model_output, tuple):
            model_output = (model_output,)

        original = model_output[:original_output_count]
        captures = model_output[original_output_count:]

        if len(original) == 1:
            return original[0], captures
        return original, captures

    def _get_dir_name(self, phase: str, size_label: str) -> str:
        key = (phase, size_label)
        count = self._counters.get(key, 0)
        return f"{phase}_{size_label}_{count}"

    def write(
        self,
        captures: Tuple[torch.Tensor, ...],
        capture_names: List[str],
        req_ids: List[str],
        positions: torch.Tensor,
        is_prefill: bool,
    ) -> None:
        """Save captured tensors to disk for one forward pass.

        Args:
            captures: Tensor tuple from extract().
            capture_names: Module names corresponding to each tensor.
            req_ids: vLLM request IDs in this batch, from the Neuron scheduler.
            positions: Token positions tensor. Its shape determines the bucket
                label: s{len} for prefill, b{len} for decode.
            is_prefill: Whether this forward pass is prefill or decode.
        """
        phase = "prefill" if is_prefill else "decode"
        size_label = (
            f"s{positions.shape[0]}" if is_prefill else f"b{positions.shape[0]}"
        )

        dir_name = self._get_dir_name(phase, size_label)

        # Build tensors dict: {name: {rank: tensor}}, applying write-time filter
        tensors = {}
        for name, tensor in zip(capture_names, captures):
            if not isinstance(tensor, torch.Tensor):
                continue
            if self._capture_filter is not None and name not in self._capture_filter:
                continue
            tensors[name] = {self.tp_rank: tensor}

        metadata = ForwardPassMetadata(
            req_ids=req_ids,
            positions=positions.cpu().tolist(),
            dp_rank=self.dp_rank,
        )

        fwd = CapturedForwardPass(
            dir_name=dir_name,
            metadata=metadata,
            tensors=tensors,
        )

        capture_dir = os.path.join(self.capture_dir, f"dp{self.dp_rank}")
        tensor_io_write(capture_dir, [fwd])

        key = (phase, size_label)
        self._counters[key] = self._counters.get(key, 0) + 1

        logger.debug("Saved %d tensors to %s", len(captures), dir_name)


def capture_tensor(name: str, tensor: torch.Tensor) -> None:
    """Manually capture a tensor from within model code.

    Uses the global TensorRegistry singleton. No-op if TensorRegistry has not
    been initialized (i.e., TensorCaptureModel is not wrapping the model).

    Args:
        name: Identifier for the tensor (will be prefixed with "manual.")
        tensor: Tensor to capture
    """
    registry = TensorRegistry._instance
    if registry is not None:
        registry.register_manual_tensor(name, tensor)
