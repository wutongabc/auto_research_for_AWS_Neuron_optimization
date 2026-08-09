# SPDX-License-Identifier: Apache-2.0
"""Weight packing helpers for the Llama3 STATIC_MX (Trn3) path.

Forward direction of the layouts documented in
:class:`nkilib.core.utils.common_types.QKVWeightLayout`:

- ``MX_CONTIGUOUS``: ``[H, I] -> [H//4, I]`` x4-packed.
- ``MX_INTERLEAVED``: ``MX_CONTIGUOUS`` preceded by a row reorder that
  matches the interleaved layout produced by DMA transpose on the input.

The kernel consumes ``nl.float8_e4m3fn_x4`` (4 fp8 bytes packed
little-endian into one 32-bit element). torch lacks an x4 dtype, so we
store packed weights as ``torch.uint32``; the patched QKV CTE / TKG
kernels (CR-277644685 + the trn3 ``mlpcte-ws`` TKG patches) allocate
SBUF with the HBM dtype and view-cast back to ``nl.float8_e4m3fn_x4``
post-DMA.

The o-proj path does **not** pack — its kernel reshapes a contiguous
``[N*D, H]`` operand to ``[N*D//4, H, 4]`` via ``TensorView`` at access
time, so the host only applies a byte-shuffle (no dtype change).
"""

from __future__ import annotations

import torch


_FP8_DTYPE = torch.float8_e4m3fn


def x4_pack_fp8(w: torch.Tensor, *, contraction_axis: int) -> torch.Tensor:
    """Pack 4 consecutive fp8 bytes along ``contraction_axis`` into one uint32.

    Output shape has ``shape[contraction_axis] // 4``; output dtype is
    ``torch.uint32``. Bit-level reinterpret (no float math) — round-trips
    bit-exact with ``unpack_float8_e4m3fn_x4``.

    Args:
        w: ``torch.float8_e4m3fn`` tensor with
            ``shape[contraction_axis] % 4 == 0``.
        contraction_axis: axis along which 4 consecutive bytes form one x4.

    Returns:
        ``torch.uint32`` tensor of the same rank, with
        ``shape[contraction_axis] // 4``.
    """
    assert w.dtype == _FP8_DTYPE, f"x4_pack_fp8 expects {_FP8_DTYPE}, got {w.dtype}"
    contraction_axis = contraction_axis % w.ndim
    n = w.shape[contraction_axis]
    assert n % 4 == 0, (
        f"x4_pack_fp8 requires shape[{contraction_axis}]={n} divisible by 4"
    )

    # Move the contraction axis to the last position so we can fold it into
    # uint8 -> uint32. The little-endian byte order is what
    # ``nl.float8_e4m3fn_x4`` expects (byte 0 → low 8 bits, byte 3 → high
    # 8 bits); see ``nkilib/core/utils/mx_torch_common.py::
    # unpack_float8_e4m3fn_x4`` for the canonical reverse transform.
    moved = w.movedim(contraction_axis, -1).contiguous()
    bytes_view = moved.view(torch.uint8)
    new_last = n // 4
    packed = bytes_view.reshape(*moved.shape[:-1], new_last, 4)
    # Bit-pack 4 bytes (little-endian) into one uint32. Done in int64 because
    # older torch builds (e.g. 2.5.1) lack a CPU implementation of lshift on
    # uint32; the result fits in 32 bits, so the final cast is lossless.
    packed_i64 = (
        packed[..., 0].to(torch.int64)
        | (packed[..., 1].to(torch.int64) << 8)
        | (packed[..., 2].to(torch.int64) << 16)
        | (packed[..., 3].to(torch.int64) << 24)
    )
    return packed_i64.to(torch.uint32).movedim(-1, contraction_axis).contiguous()


def _qkv_weight_pack_mx_contiguous(w_HI: torch.Tensor) -> torch.Tensor:
    """Forward direction of ``MX_CONTIGUOUS`` (QKV-prefill helper).

    From the ``QKVWeightLayout`` docstring::

        w.reshape(H//4, 4, I).transpose(0, 2, 1).reshape(H//4, I*4)

    then x4-pack along the trailing axis (length ``I*4``) so the kernel
    sees ``[H//4, I]`` with x4 dtype.

    Module-private — the QKV CTE STATIC_MX path uses ``MX_INTERLEAVED``
    on top of this. Exposed for tests that exercise the contiguous
    layout directly.
    """
    assert w_HI.dim() == 2, f"expected 2D [H, I], got {tuple(w_HI.shape)}"
    H, I = w_HI.shape
    assert H % 4 == 0, f"H={H} must be divisible by 4"
    rearranged = w_HI.reshape(H // 4, 4, I).transpose(1, 2).reshape(H // 4, I * 4)
    return x4_pack_fp8(rearranged, contraction_axis=-1)


def _mx_interleaved_h_perm(H: int) -> torch.Tensor:
    """Row permutation for ``MX_INTERLEAVED``.

    Per ``QKVWeightLayout.MX_INTERLEAVED``::

        h_idx = arange(H).reshape(2, H//4, 2).transpose(1, 0, 2).reshape(H)
    """
    assert H % 4 == 0, f"H={H} must be divisible by 4"
    return torch.arange(H).reshape(2, H // 4, 2).transpose(0, 1).reshape(H).contiguous()


def qkv_weight_pack_mx_interleaved(w_HI: torch.Tensor) -> torch.Tensor:
    """Forward direction of ``MX_INTERLEAVED`` (DMA-transpose MX path).

    Reorder rows by ``h_idx``, then ``MX_CONTIGUOUS``-pack.

    Args:
        w_HI: ``torch.float8_e4m3fn``, shape ``[H, I]`` with ``H % 4 == 0``.

    Returns:
        ``torch.uint32``, shape ``[H//4, I]`` (x4-packed, interleaved row
        order).
    """
    H = w_HI.shape[0]
    h_idx = _mx_interleaved_h_perm(H)
    # Some torch CPU builds (2.5.1) lack ``index_cpu`` for float8; route
    # the gather through the byte view, which is bit-equivalent.
    permuted_bytes = w_HI.view(torch.uint8)[h_idx, :].contiguous()
    permuted = permuted_bytes.view(_FP8_DTYPE)
    return _qkv_weight_pack_mx_contiguous(permuted)


def mx_shuffle_o_proj(w_NDH: torch.Tensor) -> torch.Tensor:
    """Apply the STATIC_MX byte shuffle to a 2D ``[N*D, H]`` fp8 tensor.

    The o-proj CTE STATIC_MX kernel views the weight as
    ``[N*D//4, H, 4]`` (an x4-strided 3D operand) and slices on H. To
    produce that layout from a contiguous ``[N*D, H]`` source, the host
    applies::

        reshape(N*D//4, 4, H).transpose(0, 2, 1)  -> [N*D//4, H, 4]

    then reshapes back to ``[N*D, H]``. Shape and dtype are unchanged;
    only byte order shifts. The transpose is routed through a ``uint8``
    byte view because some torch CPU builds lack ``transpose`` for
    ``float8_e4m3fn``; at the byte level it is bit-equivalent.
    """
    assert w_NDH.dim() == 2, f"expected 2D [N*D, H], got {tuple(w_NDH.shape)}"
    nd, h = w_NDH.shape
    assert nd % 4 == 0, f"N*D={nd} must be divisible by 4 for STATIC_MX o-proj"
    bytes_view = w_NDH.contiguous().view(torch.uint8)
    shuffled = (
        bytes_view.reshape(nd // 4, 4, h).transpose(1, 2).reshape(nd, h).contiguous()
    )
    return shuffled.view(_FP8_DTYPE)
