"""Place a whole torch clip at once: static parts are written once and broadcast over time.

Run: python examples/batch_torch.py
"""

import torch

from pixflex import Column, Row, Slot

T = 8
with Column(width=64, height=48, background=(0, 0, 0)) as root:
    banner = Slot(height=8, fit="exact")
    with Row(flex=1, gap=2):
        left = Slot(flex=1, fit="contain")
        right = Slot(flex=1, fit="cover")

plan = root.solve()
like = torch.zeros((T, 1, 1, 3), dtype=torch.uint8)  # T H W C
cv = plan.canvas(like, axes="THWC", static={banner: torch.full((8, 64, 3), 255, dtype=torch.uint8)})
clip = torch.randint(0, 256, (T, 100, 100, 3), dtype=torch.uint8)
cv.update({left: clip, right: clip})  # one batched resize + write each

out = cv.array
print(type(out).__name__, tuple(out.shape), out.dtype, out.device)
assert (out[:, :8] == 255).all()  # banner present in every frame
