"""Every gallery example must keep running and return a plausible RGB image."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
pytest.importorskip("torch")
pytest.importorskip("PIL")
from gallery.examples import EXAMPLES  # noqa: E402


@pytest.mark.parametrize("slug,fn", [(e.slug, e.fn) for e in EXAMPLES], ids=lambda v: str(v)[:20])
def test_example_renders(slug, fn):
    img = fn()
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
    assert img.shape[0] > 100 and img.shape[1] > 100
    assert len(np.unique(img.reshape(-1, 3), axis=0)) > 3  # not blank
