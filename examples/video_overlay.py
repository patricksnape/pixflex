"""Video with a constant header/footer: only the changing slots are rewritten each frame.

Writes out/video_overlay.mp4 and compares per-frame cost against recomposing everything.
Run: python examples/video_overlay.py
"""

import time
from pathlib import Path
from typing import NamedTuple

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

from pixflex import Box, Column, Row, ShapeMismatch, Slot

W, H, FRAMES = 1280, 720, 90


class Dashboard(NamedTuple):
    root: Box
    title: Slot
    video: Slot
    plot: Slot
    thumb: Slot
    footer: Slot


def dashboard() -> Dashboard:
    with Column(width=W, height=H, background=(16, 16, 20)) as root:
        title = Slot(height=80)
        with Row(flex=1, gap=8, padding_horizontal=8):
            video = Slot(flex=3, fit="contain", background=(0, 0, 0))
            with Column(flex=1, gap=8):
                plot = Slot(flex=1, fit="exact")
                thumb = Slot(aspect_ratio=16 / 9, fit="cover")
        footer = Slot(height=40)
    return Dashboard(root, title, video, plot, thumb, footer)


def text_bar(hw: tuple[int, int], text: str) -> np.ndarray:
    img = Image.new("RGB", (hw[1], hw[0]), (40, 60, 120))
    ImageDraw.Draw(img).text((16, hw[0] // 2 - 6), text, fill=(255, 255, 255))
    return np.asarray(img)


def source_frame(t: int, w: int = 1920, h: int = 1080) -> np.ndarray:
    """A moving gradient with a bouncing square; arbitrary size, not the slot's."""
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    img = np.empty((h, w, 3), np.uint8)
    img[..., 0] = (x + 4 * t) % 256
    img[..., 1] = y
    img[..., 2] = 128
    cx = int((w - 200) * abs(((t * 17) % 200) / 100 - 1))
    img[h // 2 - 100 : h // 2 + 100, cx : cx + 200] = 255
    return img


def make_plot(t: int, hw: tuple[int, int]) -> np.ndarray:
    img = np.full((*hw, 3), 255, np.uint8)
    xs = np.arange(hw[1])
    ys = (hw[0] / 2 + (hw[0] / 3) * np.sin(xs / 25 + t / 5)).astype(int)
    img[np.clip(ys, 0, hw[0] - 1), xs] = (200, 30, 30)
    return img


def main() -> None:
    ui = dashboard()
    plan = ui.root.solve()
    plot_hw = plan[ui.plot].hw
    print(plan)
    title_img = text_bar(plan[ui.title].hw, "pixflex video_overlay")
    footer_img = text_bar(plan[ui.footer].hw, "constant footer")
    thumb_img = source_frame(0, 640, 480)

    cv = plan.canvas(title_img, static={ui.title: title_img, ui.footer: footer_img})
    out = Path("out")
    out.mkdir(exist_ok=True)

    t_incremental = 0.0
    plot_img = make_plot(0, plot_hw)
    with iio.imopen(out / "video_overlay.mp4", "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=30)
        for t in range(FRAMES):
            frame = source_frame(t, *((1920, 1080) if t < 45 else (640, 480)))  # size changes
            if t % 10 == 0:
                plot_img = make_plot(t, plot_hw)
            t0 = time.perf_counter()  # time placement only, not content generation
            cv[ui.video] = frame  # contain: letterboxed, stale bars cleared only if needed
            if t % 10 == 0:
                cv[ui.plot] = plot_img
            if t % 30 == 0:
                cv[ui.thumb] = thumb_img
            t_incremental += time.perf_counter() - t0
            writer.write_frame(cv.array)  # zero-copy view of the live buffer
    try:
        cv[ui.plot] = make_plot(0, (plot_hw[0] - 1, plot_hw[1]))
    except ShapeMismatch as e:
        print("ShapeMismatch as expected:", e)

    t_full = 0.0
    for t in range(0, FRAMES, 3):
        frame = source_frame(t, *((1920, 1080) if t < 45 else (640, 480)))
        t0 = time.perf_counter()
        plan.compose(
            {
                ui.title: title_img,
                ui.footer: footer_img,
                ui.video: frame,
                ui.plot: make_plot(t, plot_hw),
                ui.thumb: thumb_img,
            },
            like=title_img,
        )
        t_full += (time.perf_counter() - t0) * 3
    print(f"incremental updates: {1000 * t_incremental / FRAMES:.2f} ms/frame")
    print(f"full recompose:      {1000 * t_full / FRAMES:.2f} ms/frame")
    Image.fromarray(cv.to_numpy()).save(out / "video_overlay_last_frame.png")
    print(f"wrote {out / 'video_overlay.mp4'}")


if __name__ == "__main__":
    main()
