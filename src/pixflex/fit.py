"""Pure-integer placement math, following CSS ``object-fit`` / ``object-position``.

All functions work on plain ints so they are independent of any array library.
"""

from __future__ import annotations

from typing import NamedTuple

from .errors import ShapeMismatch

FITS = ("contain", "cover", "fill", "none", "scale-down", "exact")
Align = str | tuple[float, float]


class Placement(NamedTuple):
    """How to put content into a slot rectangle.

    ``src`` is the (y0, y1, x0, x1) window of the content to use (a crop), ``out_hw`` the size
    that window is resized to (equal to the window size when no resize is needed) and ``dst`` the
    (y, x) offset of the result inside the slot.
    """

    src: tuple[int, int, int, int]
    out_hw: tuple[int, int]
    dst: tuple[int, int]

    @property
    def needs_resize(self) -> bool:
        y0, y1, x0, x1 = self.src
        return self.out_hw != (y1 - y0, x1 - x0)


def parse_align(align: Align) -> tuple[float, float]:
    """Return ``(x, y)`` fractions in [0, 1]. Words follow CSS: ``"top left"``, ``"center"``..."""
    if isinstance(align, tuple):
        x, y = align
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"align fractions must be within [0, 1], got {align!r}")
        return float(x), float(y)
    x = y = 0.5
    for word in align.lower().replace("-", " ").split():
        if word == "left":
            x = 0.0
        elif word == "right":
            x = 1.0
        elif word == "top":
            y = 0.0
        elif word == "bottom":
            y = 1.0
        elif word != "center":
            raise ValueError(f"unknown align word {word!r} in {align!r}")
    return x, y


def _at(extent: int, size: int, frac: float) -> int:
    return round((extent - size) * frac)


def place(
    content_hw: tuple[int, int],
    rect_hw: tuple[int, int],
    fit: str = "contain",
    align: Align = "center",
    label: str = "slot",
) -> Placement:
    h0, w0 = content_hw
    h, w = rect_hw
    if h0 <= 0 or w0 <= 0:
        raise ValueError(f"{label}: content has an empty shape {content_hw}")
    ax, ay = parse_align(align)

    if fit == "scale-down":
        fit = "none" if h0 <= h and w0 <= w else "contain"
    if fit == "exact":
        if (h0, w0) != (h, w):
            raise ShapeMismatch(f'{label} needs (h={h}, w={w}), got (h={h0}, w={w0}) (fit="exact")')
        return Placement((0, h0, 0, w0), (h0, w0), (0, 0))
    if fit == "fill":
        return Placement((0, h0, 0, w0), (h, w), (0, 0))
    if fit == "none":
        sh, sw = min(h0, h), min(w0, w)
        sy, sx = _at(h0, sh, ay), _at(w0, sw, ax)
        return Placement((sy, sy + sh, sx, sx + sw), (sh, sw), (_at(h, sh, ay), _at(w, sw, ax)))
    if fit == "contain":
        s = min(h / h0, w / w0)
        oh, ow = min(h, max(1, round(h0 * s))), min(w, max(1, round(w0 * s)))
        return Placement((0, h0, 0, w0), (oh, ow), (_at(h, oh, ay), _at(w, ow, ax)))
    if fit == "cover":
        # Crop the source to the slot's aspect first so only kept pixels are resized.
        s = max(h / h0, w / w0)
        ch, cw = min(h0, max(1, round(h / s))), min(w0, max(1, round(w / s)))
        sy, sx = _at(h0, ch, ay), _at(w0, cw, ax)
        return Placement((sy, sy + ch, sx, sx + cw), (h, w), (0, 0))
    raise ValueError(f"unknown fit {fit!r}; expected one of {', '.join(FITS)}")
