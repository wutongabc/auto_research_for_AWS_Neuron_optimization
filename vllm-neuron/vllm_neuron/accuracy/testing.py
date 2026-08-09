# SPDX-License-Identifier: Apache-2.0
"""Testing utilities for vLLM Neuron module accuracy validation.

Provides assertion functions that help engineers determine whether errors
are inherent to dtype quantization or specific to the Neuron target.

Two modes:

- **Two-way**: ``assert_close(actual, expected)`` — backward compatible with
  ``torch_neuronx.testing.assert_close`` semantics (rtol normalized by abs_max).

- **Three-way**: ``assert_close_three_way(baseline, expected, actual)`` — isolates
  target-specific errors from dtype-inherent errors using dynamic thresholds.
  Answers: "Should I relax the threshold or file a bug?"

Example::

    from vllm_neuron.accuracy.testing import assert_close, assert_close_three_way

    # Two-way (drop-in replacement for torch_neuronx.testing.assert_close)
    assert_close(neuron_output, hf_output)

    # Three-way (no magic rtol needed)
    assert_close_three_way(fp32_ref, bf16_cpu, bf16_neuron, name="attn.layer0")
"""

from dataclasses import dataclass
from typing import List, Union

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

from vllm_neuron.accuracy.tensor_compare import (
    _compute_bc,
    compare_tensors_three_way,
)

# Default tolerances matching neuron_allclose / torch.testing semantics
_DEFAULT_DTYPE_TOLERANCE = {
    torch.float16: (1e-3, 1e-5),
    torch.bfloat16: (1.6e-2, 1e-5),
    torch.float32: (1.3e-6, 1e-5),
}


@dataclass
class AssertCloseResult:
    """Result from assert_close with neuron_allclose-compatible fields."""

    allclose: bool
    num_mismatches: int
    max_rel_error: float
    max_abs_error: float
    max_rel_error_index: tuple
    max_abs_error_index: tuple
    linf_rel: float = 0.0
    l2_rel: float = 0.0


@dataclass
class ThreeWayAssertResult:
    """Result from assert_close_three_way with diagnostic info.

    Fields:
        passed: Whether the assertion passed (BC >= threshold or σ-ratio <= 1.0).
        name: Descriptive label for the comparison.
        base_linf: L-inf of |expected - baseline| (reference error).
        tgt_linf: L-inf of |actual - baseline| (target error).
        base_l2: L2 norm of |expected - baseline|.
        tgt_l2: L2 norm of |actual - baseline|.
        linf_ratio: tgt_linf / base_linf (worst-case across inputs if multi-input).
        l2_ratio: tgt_l2 / base_l2 (worst-case across inputs if multi-input).
        sigma_ratio: RMS(tgt_errors) / RMS(base_errors), aggregated across all inputs.
            < 1.0 means target is more accurate than reference.
        bc: Bhattacharyya Coefficient of error distributions, aggregated.
            1.0 = identical, >= 0.99 = excellent, < 0.95 = divergent.
        n_inputs: Number of input tensors aggregated.
    """

    passed: bool
    name: str
    base_linf: float
    tgt_linf: float
    base_l2: float
    tgt_l2: float
    linf_ratio: float  # worst-case across inputs
    l2_ratio: float  # worst-case across inputs
    sigma_ratio: float  # aggregated RMS(tgt_errors) / RMS(base_errors)
    bc: float  # aggregated BC
    n_inputs: int = 1

    def summary(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        lines = [f"{status} {self.name}"]
        if self.n_inputs > 1:
            lines.append(f"  inputs: {self.n_inputs}")
            lines.append(
                f"  worst L-inf ratio: {self.linf_ratio:.2f}x"
                f"  worst L2 ratio: {self.l2_ratio:.2f}x"
            )
        else:
            lines.append(
                f"  L-inf: base={self.base_linf:.6f}  tgt={self.tgt_linf:.6f}"
                f"  ratio={self.linf_ratio:.2f}x"
            )
            lines.append(
                f"  L2/σ:  base={self.base_l2:.6f}  tgt={self.tgt_l2:.6f}"
                f"  ratio={self.l2_ratio:.2f}x"
            )
        lines.append(f"  σ-ratio: {self.sigma_ratio:.3f}  BC: {self.bc:.4f}")
        return "\n".join(lines)


def _neuron_allclose(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = None,
    atol: float = None,
    equal_nan_inf: bool = False,
) -> AssertCloseResult:
    """Core comparison matching neuron_allclose semantics.

    Threshold formula::

        abs_diff = |actual - expected|
        rel_error = (abs_diff - atol) / max(|expected|)
        allclose = all(rel_error <= rtol)

    This differs from torch.allclose which uses per-element denominators.
    Here rtol is normalized by the global abs_max of expected.
    """
    if torch.equal(expected, actual):
        idx = torch.unravel_index(torch.tensor(0), expected.shape)
        return AssertCloseResult(
            allclose=True,
            num_mismatches=0,
            max_rel_error=0.0,
            max_abs_error=0.0,
            max_rel_error_index=idx,
            max_abs_error_index=idx,
        )

    if rtol is None or atol is None:
        default = _DEFAULT_DTYPE_TOLERANCE.get(expected.dtype, (1.6e-2, 1e-5))
        rtol = rtol if rtol is not None else default[0]
        atol = atol if atol is not None else default[1]

    actual_f, expected_f = actual.float(), expected.float()
    abs_diff = torch.abs(actual_f - expected_f)

    # Handle matching inf (same sign) as zero difference
    matching_inf = (
        torch.isinf(actual_f)
        & torch.isinf(expected_f)
        & (torch.sign(actual_f) == torch.sign(expected_f))
    )
    abs_diff = torch.where(matching_inf, torch.zeros_like(abs_diff), abs_diff)

    # Compute expected_abs_max excluding non-finite values
    finite_mask = torch.isfinite(expected_f)
    if finite_mask.any():
        expected_abs_max = torch.max(torch.abs(expected_f[finite_mask]))
    else:
        expected_abs_max = torch.tensor(0.0)

    if torch.is_nonzero(expected_abs_max):
        rel_error = (abs_diff - atol) / expected_abs_max
    else:
        rel_error = torch.full(expected_f.shape, float("inf"))

    close = rel_error <= rtol

    # If equal_nan_inf, treat matching NaN and matching inf as close
    if equal_nan_inf:
        matching_nan = torch.isnan(actual_f) & torch.isnan(expected_f)
        close = close | matching_nan | matching_inf
    allclose = close.all().item()
    num_mismatches = (~close).sum().item()
    max_rel_idx = torch.unravel_index(torch.argmax(rel_error), rel_error.shape)
    max_abs_idx = torch.unravel_index(torch.argmax(abs_diff), abs_diff.shape)

    linf_rel = (
        abs_diff.max().item() / expected_abs_max.item()
        if expected_abs_max > 0
        else float("inf")
    )
    l2_rel = (
        torch.norm(abs_diff).item() / torch.norm(expected_f).item()
        if torch.norm(expected_f) > 0
        else 0
    )

    return AssertCloseResult(
        allclose=allclose,
        num_mismatches=num_mismatches,
        max_rel_error=rel_error[max_rel_idx].item(),
        max_abs_error=abs_diff[max_abs_idx].item(),
        max_rel_error_index=max_rel_idx,
        max_abs_error_index=max_abs_idx,
        linf_rel=linf_rel,
        l2_rel=l2_rel,
    )


def assert_close(
    actual: Union[torch.Tensor, list],
    expected: Union[torch.Tensor, list],
    rtol: float = None,
    atol: float = None,
    equal_nan_inf: bool = False,
    name: str = "tensor",
    enable_histograms: bool = False,
) -> AssertCloseResult:
    """Two-way assertion compatible with torch_neuronx.testing.assert_close.

    Uses neuron_allclose semantics: rtol normalized by abs_max of expected.
    Supports nested lists/tuples of tensors (compared element-wise).

    Args:
        actual: Target tensor (or nested list/tuple of tensors).
        expected: Reference tensor (or nested list/tuple of tensors).
        rtol: Relative tolerance. Defaults to dtype-specific value.
        atol: Absolute tolerance. Defaults to dtype-specific value.
        equal_nan_inf: If True, matching NaN values and matching infinity
            values (same sign) are treated as equal.
        name: Label for error messages.
        enable_histograms: prints a diagnostic report
            (histograms, stats table, and 2D mismatch map) for each tensor pair
            on both pass and failure. If False (default), no diagnostic report is printed.

    Raises:
        AssertionError with diagnostic info (L-inf, L2, mismatch count).
    """
    pairs = _flatten_tensor_pairs(actual, expected, name)
    failures = []
    last_result = None

    for act, exp, pair_name in pairs:
        result = _neuron_allclose(
            act, exp, rtol=rtol, atol=atol, equal_nan_inf=equal_nan_inf
        )
        last_result = result

        # Resolve effective tolerances for the diagnostic report (matches
        # _neuron_allclose default-resolution logic).
        default = _DEFAULT_DTYPE_TOLERANCE.get(exp.dtype, (1.6e-2, 1e-5))
        eff_rtol = rtol if rtol is not None else default[0]
        eff_atol = atol if atol is not None else default[1]

        if enable_histograms:
            try:
                from vllm_neuron.accuracy.tensor_histogram import TensorHistogram

                TensorHistogram().print_full_comparison_report(
                    actual=act,
                    expected=exp,
                    name=pair_name,
                    atol=eff_atol,
                    rtol=eff_rtol,
                    passed=result.allclose,
                    enable_histograms=True,
                )
            except Exception as exc:  # pragma: no cover - diagnostic only
                logger.warning(
                    "Could not generate tensor histogram for %s: %s",
                    pair_name,
                    exc,
                )

        if not result.allclose:
            failures.append(
                f"  {pair_name}: {result.num_mismatches} mismatches, "
                f"max_rel={result.max_rel_error:.6f}, max_abs={result.max_abs_error:.6f}, "
                f"L-inf_rel={result.linf_rel:.6f}, L2_rel={result.l2_rel:.6f}"
            )

    if failures:
        raise AssertionError(
            f"assert_close failed for {len(failures)}/{len(pairs)} tensor(s):\n"
            + "\n".join(failures)
        )

    return last_result


def assert_close_three_way(
    baseline: Union[torch.Tensor, list],
    expected: Union[torch.Tensor, list],
    actual: Union[torch.Tensor, list],
    name: str = "tensor",
    bc_threshold: float = 0.99,
    max_linf_ratio: float = 5.0,
    max_l2_ratio: float = 3.0,
    plot_on_failure: bool = False,
    output_dir: str = None,
) -> ThreeWayAssertResult:
    """Three-way assertion that isolates target errors from dtype errors.

    Compares:
    - baseline (FP32) vs expected (same-dtype on CPU) → dtype error
    - baseline (FP32) vs actual (same-dtype on Neuron) → target error

    Passes if the Bhattacharyya Coefficient (BC) of the two error distributions
    is >= ``bc_threshold``, or if σ-ratio ≤ 1.0 (target more accurate than baseline).

    Additionally, hard guards ensure no extreme element-wise deviations:
    - worst L-inf ratio must be < ``max_linf_ratio`` (default 5.0)
    - worst L2 ratio must be < ``max_l2_ratio`` (default 3.0)

    When ``plot_on_failure=True`` (default), automatically generates error
    distribution and QQ-plot diagnostics on failure.

    Metrics reported:
        - **σ-ratio**: Ratio of target RMS error to baseline RMS error,
          aggregated across all inputs. σ-ratio ≤ 1.0 means the target is
          more accurate than the dtype baseline.
        - **BC** (Bhattacharyya Coefficient): Overlap between target and
          baseline error distributions (0–1). BC ≥ 0.99 means nearly
          identical distributions.
        - **worst L-inf ratio**: Maximum per-input ratio of target max
          absolute error to baseline max absolute error. Catches extreme
          element-wise outliers.
        - **worst L2 ratio**: Maximum per-input ratio of target L2 error
          to baseline L2 error. Catches systematic per-sample degradation.

    Raises:
        AssertionError with diagnostic summary.
    """
    base_list = _to_tensor_list(baseline)
    exp_list = _to_tensor_list(expected)
    act_list = _to_tensor_list(actual)
    assert len(base_list) == len(exp_list) == len(act_list)

    all_base_errors, all_tgt_errors = [], []
    worst_linf_ratio = worst_l2_ratio = 0.0

    for i, (b, e, a) in enumerate(zip(base_list, exp_list, act_list)):
        tag = f"{name}[{i}]" if len(base_list) > 1 else name
        r = compare_tensors_three_way(b, e, a, name=tag)
        worst_linf_ratio = max(worst_linf_ratio, r.linf_ratio)
        worst_l2_ratio = max(worst_l2_ratio, r.l2_ratio)
        if r.base_errors is not None:
            all_base_errors.append(r.base_errors)
            all_tgt_errors.append(r.tgt_errors)

    n_inputs = len(base_list)

    if all_base_errors:
        cat_base = np.concatenate(all_base_errors)
        cat_tgt = np.concatenate(all_tgt_errors)
        agg_bc = _compute_bc(cat_base, cat_tgt)
        base_rms = np.sqrt(np.mean(cat_base**2))
        tgt_rms = np.sqrt(np.mean(cat_tgt**2))
        sigma_ratio = tgt_rms / base_rms if base_rms > 0 else float("inf")
    else:
        agg_bc = 0.0
        sigma_ratio = float("inf")

    # Pass if BC >= threshold or target is more accurate than baseline
    # Hard guards: extreme element-wise deviations always fail regardless of BC/σ
    linf_guard_ok = worst_linf_ratio < max_linf_ratio
    l2_guard_ok = worst_l2_ratio < max_l2_ratio
    passed = (
        (agg_bc >= bc_threshold or sigma_ratio <= 1.0) and linf_guard_ok and l2_guard_ok
    )

    result = ThreeWayAssertResult(
        passed=passed,
        name=name,
        base_linf=r.base_linf,
        tgt_linf=r.tgt_linf,
        base_l2=r.base_l2,
        tgt_l2=r.tgt_l2,
        linf_ratio=worst_linf_ratio,
        l2_ratio=worst_l2_ratio,
        sigma_ratio=sigma_ratio,
        bc=agg_bc,
        n_inputs=n_inputs,
    )

    # Always print summary for visibility in test output
    logger.info(f"\n{result.summary()}")

    if not passed:
        if 0.95 <= agg_bc < bc_threshold:
            logger.info(
                f"  → distributions similar but below threshold (BC={agg_bc:.4f}), review plots"
            )
        if not linf_guard_ok:
            logger.info(
                f"  → hard guard: worst L-inf ratio {worst_linf_ratio:.2f}x exceeds limit {max_linf_ratio:.1f}x"
            )
        if not l2_guard_ok:
            logger.info(
                f"  → hard guard: worst L2 ratio {worst_l2_ratio:.2f}x exceeds limit {max_l2_ratio:.1f}x"
            )
        if plot_on_failure and all_base_errors:
            _generate_failure_plots(
                np.concatenate(all_base_errors),
                np.concatenate(all_tgt_errors),
                name,
                output_dir,
            )
        raise AssertionError(f"\n{result.summary()}")
    return result


def _generate_failure_plots(
    base_errors: np.ndarray,
    tgt_errors: np.ndarray,
    name: str,
    output_dir: str = None,
) -> None:
    """Generate a single combined diagnostic plot on three-way assertion failure."""
    try:
        from vllm_neuron.accuracy.plotting import (
            plot_error_distributions,
            plot_error_qqplot,
        )
        import os
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = output_dir or "accuracy_debug"
        safe_name = name.replace("/", "_").replace(" ", "_")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        plot_error_distributions(base_errors, tgt_errors, name=name, ax=ax1)
        plot_error_qqplot(base_errors, tgt_errors, name=name, ax=ax2)
        plt.suptitle(name, fontsize=10)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"{safe_name}_debug.png")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved debug plot to {output_path}")
    except Exception as e:
        logger.warning(f"Could not generate debug plots: {e}")


def _to_tensor_list(x) -> List[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            out.extend(_to_tensor_list(item))
        return out
    raise TypeError(f"Expected tensor or sequence, got {type(x)}")


def _flatten_tensor_pairs(actual, expected, prefix=""):
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        return [(actual, expected, prefix)]
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise ValueError(f"Length mismatch: {len(actual)} vs {len(expected)}")
        pairs = []
        for i, (a, e) in enumerate(zip(actual, expected)):
            pairs.extend(_flatten_tensor_pairs(a, e, f"{prefix}[{i}]"))
        return pairs
    raise TypeError(f"Type mismatch: {type(actual)} vs {type(expected)}")
