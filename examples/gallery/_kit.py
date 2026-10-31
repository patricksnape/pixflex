"""Helpers shared by the gallery examples: palette, text, diagrams, procedural 'photos', sheets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pixflex import Box, Column, Plan, Row, Slot

BG = (243, 244, 246)
CARD = (255, 255, 255)
INK = (31, 41, 55)
MUTED = (107, 114, 128)
LINE = (209, 213, 219)
COLORS = [
    (96, 165, 250), (52, 211, 153), (251, 191, 36), (248, 113, 113),
    (167, 139, 250), (244, 114, 182), (45, 212, 191), (251, 146, 60),
]  # fmt: skip


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def text_img(
    hw: tuple[int, int], text: str, *, size: int = 14, fg=INK, bg=BG, align: str = "center"
) -> np.ndarray:
    """Text rendered at *exactly* the size of its slot, so it is written with a single copy."""
    img = Image.new("RGB", (hw[1], hw[0]), bg)
    x = {"left": 4, "center": hw[1] // 2, "right": hw[1] - 4}[align]
    anchor = {"left": "lm", "center": "mm", "right": "rm"}[align]
    ImageDraw.Draw(img).text((x, hw[0] // 2), text, font=font(size), fill=fg, anchor=anchor)
    return np.array(img)


def draw_plan(plan: Plan[Any], scale: int = 2, size: int = 18) -> np.ndarray:
    """Draw a solved plan as a diagram: coloured slots (numbered by key), outlined containers."""
    h, w = plan.size
    img = Image.new("RGB", (w * scale, h * scale), CARD)
    d = ImageDraw.Draw(img)
    for n, (item, r) in enumerate(plan.rects.items()):
        box = (r.x * scale, r.y * scale, r.x1 * scale - 1, r.y1 * scale - 1)
        if isinstance(item, Box):
            d.rounded_rectangle(
                box, 8 if item is plan.root else 5, fill=(248, 250, 252), outline=LINE, width=2
            )
            continue
        i = item.key if isinstance(item.key, int) else n
        d.rounded_rectangle(box, 5, fill=COLORS[(i - 1) % len(COLORS)])
        if isinstance(item.key, (int, str)):
            c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            d.text(c, str(item.key), font=font(size), fill=(255, 255, 255), anchor="mm")
    return np.array(img)


def photo(h: int, w: int, seed: int, caption: bool = True) -> np.ndarray:
    """A procedural 'photograph' (soft colour blobs) with a frame and its own size printed on it,
    so crops, letterboxing and resizes are easy to see."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy, xx = yy / h, xx / w
    img = np.stack(
        [
            rng.uniform(0.25, 0.8) + rng.uniform(-0.4, 0.4) * xx + rng.uniform(-0.4, 0.4) * yy
            for _ in range(3)
        ],
        -1,
    )
    for _ in range(7):
        cy, cx, r = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0.08, 0.3)
        blob = np.exp(-(((yy - cy) ** 2) * (h / w) ** -2 + (xx - cx) ** 2) / (2 * r**2))
        img += blob[..., None] * rng.uniform(-0.5, 0.6, 3)
    out = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    b = max(2, min(h, w) // 60)
    k = max(1, b // 2)  # thin dark outer edge so the white frame stays visible on light backgrounds
    d.rectangle((0, 0, w - 1, h - 1), outline=(20, 20, 24), width=k)
    if min(h, w) > 2 * k + 2:
        d.rectangle((k, k, w - 1 - k, h - 1 - k), outline=(255, 255, 255), width=max(1, b - k))
    if caption:
        d.text((w / 2, h / 2), f"{w}x{h}", font=font(max(10, min(h, w) // 7)), fill=(255, 255, 255),
               anchor="mm", stroke_width=2, stroke_fill=(0, 0, 0))  # fmt: skip
    return np.array(out)


def sheet(
    tiles: Sequence[tuple[str, np.ndarray]], cols: int, title: str | None = None, pad: int = 24,
    gap: int = 18, caption: int = 26, width: int | None = None,
) -> np.ndarray:  # fmt: skip
    """A captioned contact sheet, laid out and composed with pixflex itself."""
    content: dict[Any, np.ndarray] = {}
    head: Slot | None = None
    with Column(padding=pad, gap=gap, background=BG, width=width, align_items="flex-start") as root:
        if title:
            head = Slot(height=36, align_self="stretch", key="title", fit="exact")
        for r in range(0, len(tiles), cols):
            with Row(gap=gap, align_items="flex-start"):
                for i, (cap, img) in enumerate(tiles[r : r + cols], start=r):
                    with Column(gap=6):
                        Slot(key=("img", i), width=img.shape[1], height=img.shape[0], fit="exact")
                        Slot(key=("cap", i), height=caption, width=img.shape[1], fit="exact")
                    content[("img", i)] = img
                    content[("cap", i)] = text_img((caption, img.shape[1]), cap, size=15, fg=MUTED)
    plan = root.solve()
    if head is not None and title:
        content[head] = text_img(plan[head].hw, title, size=24, align="left")
    return plan.compose(content, like=tiles[0][1])
