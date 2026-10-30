"""The yogalayout.dev homepage layout, filled with images (numpy + Pillow).

Run: python examples/figure_panel.py  ->  out/figure_panel.png
"""

from pathlib import Path

import numpy as np
from PIL import Image

from pixflex import Column, Row, Slot


def noise(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (h // 20 + 1, w // 20 + 1, 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(base).resize((w, h), Image.Resampling.BICUBIC))


with Column(width=250, height=475, padding=10, background=(30, 30, 30)) as root:  # noqa: SIM117
    with Column(flex=1, row_gap=10):
        header = Slot(height=60)
        a = Slot(flex=1, margin_inline=10, fit="cover")
        b = Slot(flex=2, margin_inline=10, fit="contain", background=(0, 0, 0))
        with Row(
            position="absolute",
            width="100%",
            bottom=0,
            height=64,
            align_items="center",
            justify_content="space-around",
            background=(60, 60, 60),
        ):
            icons = [Slot(width=40, height=40, fit="exact") for _ in range(4)]

plan = root.solve()
print(plan)

img = plan.compose(
    {
        header: np.full((*plan[header].hw, 3), (200, 80, 80), np.uint8),  # exact size: one copy
        a: noise(3000, 4000, 1),  # cover: centre-cropped, then resized
        b: noise(200, 300, 2),  # contain: letterboxed
        **{
            icon: np.full((40, 40, 3), (40 + 50 * i, 180, 90), np.uint8)
            for i, icon in enumerate(icons)
        },
    },
    like=np.zeros((1, 1, 3), np.uint8),
)
out = Path("out")
out.mkdir(exist_ok=True)
Image.fromarray(img).save(out / "figure_panel.png")
print("wrote", out / "figure_panel.png", img.shape)
