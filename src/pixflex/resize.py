"""Pluggable image resizers, selected lazily per array library.

A resizer is ``fn(x, hw, axes) -> array`` where ``axes`` names the axes of ``x`` (e.g. ``"HWC"``);
it must return an array of the same dtype with H and W replaced by ``hw``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ._backend import backend_name
from .errors import MissingResizer

Resizer = Callable[[Any, "tuple[int, int]", str], Any]
_REGISTRY: dict[str, Resizer] = {}


def register_resizer(backend: str, fn: Resizer) -> None:
    """Use ``fn`` to resize arrays of ``backend`` (``"numpy"``, ``"torch"``, ...)."""
    _REGISTRY[backend] = fn


def resize(x: Any, hw: tuple[int, int], axes: str) -> Any:
    name = backend_name(x)
    fn = _REGISTRY.get(name) or _BUILTIN.get(name)
    if fn is None:
        raise MissingResizer(f"no resizer for {name!r} arrays; use pixflex.register_resizer")
    return fn(x, hw, axes)


def _numpy_resize(x: Any, hw: tuple[int, int], axes: str) -> Any:
    import numpy as np

    h, w = axes.index("H"), axes.index("W")
    moved = np.moveaxis(x, (h, w), (0, 1))  # (H, W, rest...)
    rest = moved.shape[2:]
    flat = moved.reshape(moved.shape[0], moved.shape[1], -1)
    out = _numpy_hwk(flat, hw)
    out = out.reshape(hw[0], hw[1], *rest)
    return np.moveaxis(out, (0, 1), (h, w))


def _numpy_hwk(a: Any, hw: tuple[int, int]) -> Any:
    """Resize an (H, W, K) numpy array with the best installed library."""
    import numpy as np

    oh, ow = hw
    try:
        import cv2  # pyright: ignore[reportMissingImports]

        if a.dtype in (np.uint8, np.uint16, np.int16, np.float32, np.float64):
            interp = cv2.INTER_AREA if oh * ow < a.shape[0] * a.shape[1] else cv2.INTER_LINEAR
            out = np.empty((oh, ow, a.shape[2]), a.dtype)
            for lo in range(0, a.shape[2], 4):  # cv2 handles <=4 channels per call robustly
                chunk = cv2.resize(
                    np.ascontiguousarray(a[..., lo : lo + 4]), (ow, oh), interpolation=interp
                )
                out[..., lo : lo + 4] = chunk.reshape(oh, ow, -1)
            return out
    except ImportError:
        pass
    try:
        from PIL import Image

        resample = Image.Resampling.BILINEAR
        if a.dtype == np.uint8 and a.shape[2] in (1, 3, 4):  # one call for the whole image
            mode = {1: "L", 3: "RGB", 4: "RGBA"}[a.shape[2]]
            src = np.ascontiguousarray(a[..., 0] if a.shape[2] == 1 else a)
            img = Image.fromarray(src, mode).resize((ow, oh), resample)
            return np.asarray(img).reshape(oh, ow, a.shape[2])
        modes = {np.dtype(np.uint8): "L", np.dtype(np.float32): "F"}
        if a.dtype in modes:  # any other channel count: plane by plane
            out = np.empty((oh, ow, a.shape[2]), a.dtype)
            for k in range(a.shape[2]):
                img = Image.fromarray(np.ascontiguousarray(a[..., k]), modes[a.dtype])
                out[..., k] = np.asarray(img.resize((ow, oh), resample))
            return out
    except ImportError:
        pass
    try:
        from scipy.ndimage import zoom  # pyright: ignore[reportMissingImports]

        return zoom(a, (oh / a.shape[0], ow / a.shape[1], 1), order=1).astype(a.dtype, copy=False)
    except ImportError:
        pass
    raise MissingResizer(
        f"cannot resize numpy arrays of dtype {a.dtype}: install one of "
        "opencv-python-headless, pillow (uint8/float32 only) or scipy, or register your own"
    )


def _torch_resize(x: Any, hw: tuple[int, int], axes: str) -> Any:
    import torch
    import torch.nn.functional as F  # noqa: N812

    h, w = axes.index("H"), axes.index("W")
    moved = x.movedim((h, w), (-2, -1))
    lead = moved.shape[:-2]
    flat = moved.reshape(-1, 1, moved.shape[-2], moved.shape[-1])
    integral = not (flat.is_floating_point())
    work = flat.to(torch.float32) if integral or flat.dtype == torch.float16 else flat
    out = F.interpolate(work, size=hw, mode="bilinear", align_corners=False, antialias=True)
    if integral:
        info = torch.iinfo(x.dtype)
        out = out.round().clamp(info.min, info.max)
    out = out.to(x.dtype).reshape(*lead, *hw)
    return out.movedim((-2, -1), (h, w))


_BUILTIN: dict[str, Resizer] = {"numpy": _numpy_resize, "torch": _torch_resize}
