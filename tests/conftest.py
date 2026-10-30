import numpy as np
import pytest

BACKENDS = ["numpy", "torch"]


@pytest.fixture(params=BACKENDS)
def be(request):
    """A tiny adapter so every test runs unchanged on each array library."""
    name = request.param
    if name == "torch":
        torch = pytest.importorskip("torch")

        class Torch:
            name = "torch"

            @staticmethod
            def arr(a):
                return torch.as_tensor(np.ascontiguousarray(a))

            @staticmethod
            def np(a):
                return a.detach().cpu().numpy()

            @staticmethod
            def to_u8(a):
                return (a * 255).to(torch.uint8)

            @staticmethod
            def shares_memory(a, b):
                return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()

        return Torch
    else:

        class Numpy:
            name = "numpy"

            @staticmethod
            def arr(a):
                return np.asarray(a)

            @staticmethod
            def np(a):
                return np.asarray(a)

            @staticmethod
            def to_u8(a):
                return (a * 255).astype(np.uint8)

            @staticmethod
            def shares_memory(a, b):
                return np.shares_memory(a, b)

        return Numpy


def solid(h, w, color, dtype: type = np.uint8):
    """An (h, w, len(color)) image filled with one colour."""
    return np.broadcast_to(np.asarray(color, dtype), (h, w, len(color))).copy()
