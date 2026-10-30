"""Thin helpers over array-api-compat so the rest of pixflex is array-library agnostic."""

from __future__ import annotations

from typing import Any

from array_api_compat import array_namespace, device, is_numpy_array, is_torch_array


def namespace(x: Any) -> Any:
    return array_namespace(x)


def backend_name(x: Any) -> str:
    """Top-level module that owns the array type: ``"numpy"``, ``"torch"``, ..."""
    return type(x).__module__.split(".")[0]


def device_of(x: Any) -> Any:
    return device(x)


def copy(x: Any) -> Any:
    return array_namespace(x).asarray(x, copy=True)


def to_numpy(x: Any) -> Any:
    import numpy as np

    if is_torch_array(x):
        return x.detach().cpu().numpy().copy()
    if is_numpy_array(x):
        return x.copy()
    return np.array(x)


def default_namespace() -> Any:
    import array_api_compat.numpy as xp

    return xp
