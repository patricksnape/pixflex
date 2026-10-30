"""Gallery examples. Each function returns an image; run `python examples/gallery.py` to render.

Every function is deliberately short: its source is embedded verbatim in docs/GALLERY.md.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from pixflex import Box, Column, Row, Slot

from ._kit import BG, CARD, COLORS, INK, LINE, MUTED, draw_plan, photo, sheet, text_img


class Example(NamedTuple):
    slug: str
    title: str
    blurb: str
    fn: Callable[[], np.ndarray]
    uses: tuple[Callable[..., Any], ...]  # helper functions whose source is shown with the example


EXAMPLES: list[Example] = []
_HELPERS: dict[str, Callable[..., Any]] = {}


def example(slug: str, title: str, blurb: str, uses: tuple[str, ...] = ()):
    """Register an example; ``uses`` names module-level helpers to show alongside it."""

    def register(fn: Callable[[], np.ndarray]):
        EXAMPLES.append(Example(slug, title, blurb, fn, tuple(_HELPERS[n] for n in uses)))
        return fn

    return register


def helper(fn: Callable[..., Any]):
    _HELPERS[fn.__name__] = fn
    return fn


# ---------------------------------------------------------------- the Yoga homepage layout
@helper
def phone(u: int) -> Box:
    """The yogalayout.dev homepage layout, scaled by ``u``."""
    with Column(width=250 * u, height=475 * u, padding=10 * u, background=(30, 30, 30)) as root:
        with Column(flex=1, row_gap=10 * u):
            Slot(key="header", height=60 * u)
            Slot(key="a", flex=1, margin_inline=10 * u, fit="cover")
            Slot(key="b", flex=2, margin_inline=10 * u, fit="contain", background=(0, 0, 0))
            with Row(position="absolute", width="100%", bottom=0, height=64 * u, align_items="center",
                     justify_content="space-around", background=(70, 70, 76)):  # fmt: skip
                for i in range(1, 5):
                    Slot(key=i, width=40 * u, height=40 * u, fit="exact")
    return root


@example(
    "homepage",
    "The yogalayout.dev homepage layout",
    "The layout from the Yoga homepage, solved once and drawn as a diagram (left), then filled "
    "with images (right). `a` is `cover` (cropped), `b` is `contain` (letterboxed), the icon bar "
    "is absolutely positioned.",
    uses=("phone",),
)
def homepage() -> np.ndarray:
    plan = phone(2).solve()
    icons = {}
    for i in range(1, 5):
        im = Image.new("RGB", (80, 80), (70, 70, 76))
        ImageDraw.Draw(im).ellipse((10, 10, 70, 70), fill=COLORS[i - 1])
        icons[i] = np.asarray(im)
    shot = plan.compose(
        {
            "header": text_img(
                plan["header"].hw, "Header", size=28, fg=(255, 255, 255), bg=(59, 82, 139)
            ),
            "a": photo(900, 1200, 3),  # 4:3 into a wide slot -> cropped
            "b": photo(400, 600, 8),  # 3:2 into a tall slot -> letterboxed
            **icons,
        },
        like=icons[1],
    )
    return sheet([("solved by Yoga", draw_plan(plan, scale=1)), ("composed by pixflex", shot)], 2)


# ---------------------------------------------------------------- real-world layouts
@example(
    "collage",
    "Photo collage",
    "A hero image plus a 2x2 grid, all `cover`: whatever shape the source has, each slot is "
    "filled and only the pixels that are kept get resized.",
)
def collage() -> np.ndarray:
    with Column(width=1000, height=560, padding=8, gap=8, background=CARD) as root:
        with Row(flex=1, gap=8):
            hero = Slot(flex=2, fit="cover")
            with Column(flex=1, gap=8):
                with Row(flex=1, gap=8):
                    p = Slot(flex=1, fit="cover")
                    q = Slot(flex=1, fit="cover", align="top")
                with Row(flex=1, gap=8):
                    r = Slot(flex=2, fit="cover")
                    s = Slot(flex=1, fit="cover")
    return root.solve().compose(
        {
            hero: photo(900, 1200, 11),
            p: photo(600, 600, 12),
            q: photo(1200, 500, 13),
            r: photo(500, 900, 14),
            s: photo(300, 300, 15),
        },  # fmt: skip
        like=photo(2, 2, 0),
    )


@example(
    "justified",
    "Justified rows (Flickr-style)",
    "Each row's height is `(width - gaps) / sum(aspect ratios)` and each image gets `flex_grow` "
    "equal to its aspect ratio with `flex_basis=0`, so every row fills the width exactly, with no "
    "cropping or bars.",
)
def justified() -> np.ndarray:
    width, pad, gap = 900, 8, 8
    rows = [[(400, 600), (400, 400), (400, 700)], [(500, 500), (300, 800)],
            [(400, 500), (400, 400), (400, 650), (400, 500)]]  # fmt: skip
    imgs: dict[Any, np.ndarray] = {}
    with Column(width=width, padding=pad, gap=gap, background=CARD) as root:
        for ri, row in enumerate(rows):
            aspects = [w / h for h, w in row]
            row_h = round((width - 2 * pad - gap * (len(row) - 1)) / sum(aspects))
            with Row(gap=gap, height=row_h):
                for ci, ((h, w), aspect) in enumerate(zip(row, aspects, strict=True)):
                    cell = Slot(flex_grow=aspect, flex_basis=0, fit="contain", background=BG)
                    imgs[cell] = photo(h, w, 20 + ri * 5 + ci)
    return root.solve().compose(imgs, like=photo(2, 2, 0))


@example(
    "contact_sheet",
    "Contact sheet with captions",
    "Mixed landscape/portrait/square sources in identical cells (`contain` on a dark cell "
    "background) with a caption slot under each. Cell and caption sizes come from Yoga.",
)
def contact_sheet() -> np.ndarray:
    shapes = [(300, 400), (400, 300), (300, 300), (600, 300), (300, 600), (240, 400),
              (400, 400), (300, 450), (450, 300), (350, 350), (200, 500), (500, 250)]  # fmt: skip
    content: dict[Any, np.ndarray] = {}
    with Column(padding=16, gap=14, background=BG) as root:
        title = Slot(height=34, align_self="stretch")
        with Column(gap=12):
            for r in range(3):
                with Row(gap=12):
                    for c in range(4):
                        i = r * 4 + c
                        with Column(gap=4):
                            cell = Slot(
                                width=200, height=150, fit="contain", background=(34, 34, 40)
                            )
                            cap = Slot(height=22, width=200, fit="exact")
                        h, w = shapes[i]
                        content[cell] = photo(h, w, 40 + i)
                        content[cap] = text_img(
                            (22, 200), f"IMG_{i + 1:04d}  {w}x{h}", size=13, fg=MUTED
                        )
    plan = root.solve()
    content[title] = text_img(plan[title].hw, "Contact sheet", size=22, align="left")
    return plan.compose(content, like=photo(2, 2, 0))


@helper
def colormap(x: np.ndarray) -> np.ndarray:
    """A small viridis-like colormap: x in [0, 1] -> uint8 RGB."""
    stops = np.array(
        [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], np.float32
    )
    pos = np.linspace(0, 1, len(stops))
    return np.stack([np.interp(x, pos, stops[:, c]) for c in range(3)], -1).astype(np.uint8)


@example(
    "before_after",
    "Before / after / difference",
    "Three panels and a colour bar, all `exact`. The panel size is read from the plan, so the "
    "processed images are produced at precisely that size (no resizing anywhere).",
    uses=("colormap",),
)
def before_after() -> np.ndarray:
    with Column(padding=20, gap=14, background=BG) as root:
        title = Slot(height=34, align_self="stretch")
        with Row(gap=14):
            panels = [Slot(width=300, height=225, fit="exact") for _ in range(3)]
            bar = Slot(width=22, height=225, fit="exact")
        with Row(gap=14):
            caps = [Slot(width=300, height=24, fit="exact") for _ in range(3)]
    plan = root.solve()
    h, w = plan[panels[0]].hw
    original = photo(h, w, 61, caption=False)
    smooth = np.asarray(
        Image.fromarray(original)
        .convert("L")
        .resize((w // 8, h // 8))
        .resize((w, h), Image.Resampling.BICUBIC)
    )
    processed = np.stack([smooth] * 3, -1)
    diff = np.abs(original.astype(np.float32).mean(-1) - smooth) / 255
    return plan.compose(
        {
            title: text_img(plan[title].hw, "Low-pass filter", size=22, align="left"),
            panels[0]: original, panels[1]: processed,
            panels[2]: colormap(diff / diff.max()),
            bar: colormap(np.linspace(1, 0, h)[:, None].repeat(22, 1)),
            caps[0]: text_img((24, w), "original", fg=MUTED),
            caps[1]: text_img((24, w), "grayscale, 8x downsampled", fg=MUTED),
            caps[2]: text_img((24, w), "|difference|", fg=MUTED),
        },
        like=original,
    )  # fmt: skip


@helper
def page(width: int) -> Box:
    wide = width >= 720
    with Column(width=width, padding=8, gap=8, background=CARD) as root:
        Slot(key="header", height=52)
        with Row(flex_wrap="wrap", gap=8):
            Slot(
                key="nav", width=170 if wide else "100%", height=(260 if wide else 44), flex_grow=0
            )
            Slot(key="main", flex_grow=1, flex_basis=300, min_height=260)
            Slot(
                key="aside",
                width=170 if width >= 1000 else "100%",
                height=260 if width >= 1000 else 60,
            )
        Slot(key="footer", height=40)
    return root


@example(
    "responsive",
    "One layout, three widths",
    "The same tree at 360, 720 and 1100 px. `flex_wrap` plus `flex_basis` reflow the sidebar "
    "and aside as space allows, exactly like CSS. Re-solving is cheap and touches no pixels.",
    uses=("page",),
)
def responsive() -> np.ndarray:
    tiles = [(f"{w} px", draw_plan(page(w).solve(), scale=1, size=16)) for w in (360, 720, 1100)]
    return sheet(tiles, 3, title="Responsive page (flex-wrap)", width=None)


@example(
    "feature_maps",
    "Torch feature maps",
    "A real tensor workflow: 16 conv activations (a `(16, H, W, 3)` torch tensor) laid out as a "
    "grid next to their input, composed on the tensor's own device with no numpy round trip.",
)
def feature_maps() -> np.ndarray:
    g = torch.Generator().manual_seed(0)
    x = torch.from_numpy(photo(128, 128, 5, caption=False)).permute(2, 0, 1)[None].float() / 255
    maps = torch.relu(
        torch.nn.functional.conv2d(x, torch.randn(16, 3, 5, 5, generator=g), stride=2)
    )[0]
    maps = (maps - maps.amin((1, 2), keepdim=True)) / (
        maps.amax((1, 2), keepdim=True) - maps.amin((1, 2), keepdim=True) + 1e-6
    )
    lut = torch.from_numpy(colormap(np.linspace(0, 1, 256)))
    tiles = lut[(maps * 255).long()]  # (16, 62, 62, 3) uint8 on the tensor device

    with Column(padding=16, gap=12, background=BG) as root:
        title = Slot(height=32, align_self="stretch")
        with Row(gap=16):
            inp = Slot(width=402, height=402, fit="fill")
            with Column(gap=6):
                cells = []
                for _ in range(4):
                    with Row(gap=6):
                        cells += [Slot(width=96, height=96, fit="fill") for _ in range(4)]
    plan = root.solve()
    like = torch.zeros((1, 1, 3), dtype=torch.uint8)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
    cv = plan.canvas(like, fill=0)
    cv.update({title: t(text_img(plan[title].hw, "Conv activations (16 filters, 5x5, stride 2)", size=20, align="left")),
               inp: t(photo(128, 128, 5, caption=False))})  # fmt: skip
    cv.update(dict(zip(cells, tiles, strict=True)))
    return cv.to_numpy()


# ---------------------------------------------------------------- Yoga docs property playgrounds
@helper
def demo(children: list[dict[str, Any]] | None = None, **container: Any) -> np.ndarray:
    """The Yoga docs playground: a 200x200 container, padding 10, 50x50 children with margin 5."""
    children = children or [dict(width=50, height=50, margin=5)] * 2
    container = {"padding": 10, **container}
    with Column(width=200, height=200, **container) as root:
        for i, style in enumerate(children, start=1):
            Slot(key=i, **style)
    return draw_plan(root.solve())


@example(
    "justify_content",
    "justify-content",
    "Main-axis distribution in the Yoga docs' playground (200x200, padding 10, two 50x50 "
    "children with margin 5, default column direction).",
)
def justify_content() -> np.ndarray:
    values = ["flex-start", "center", "flex-end", "space-between", "space-around", "space-evenly"]
    return sheet([(v, demo(justify_content=v)) for v in values], 3, title="justify-content")


@example(
    "align_items",
    "align-items",
    "Cross-axis alignment in a row container, children of different heights; `stretch` needs "
    "children without a fixed height.",
)
def align_items() -> np.ndarray:
    fixed = [dict(width=40, height=h, margin=5) for h in (30, 70, 50)]
    free = [dict(width=40, margin=5) for _ in range(3)]
    tiles = [
        (v, demo(fixed, flex_direction="row", align_items=v))
        for v in ("flex-start", "center", "flex-end")
    ]
    tiles.append(("stretch (no height)", demo(free, flex_direction="row", align_items="stretch")))
    return sheet(tiles, 4, title="align-items (row)")


@example(
    "flex_direction",
    "flex-direction",
    "The four directions with three children; reversed directions pack from the opposite edge.",
)
def flex_direction() -> np.ndarray:
    three = [dict(width=50, height=50, margin=5)] * 3
    values = ("column", "column-reverse", "row", "row-reverse")
    return sheet([(v, demo(three, flex_direction=v)) for v in values], 4, title="flex-direction")


@example(
    "flex_wrap",
    "flex-wrap",
    "Eight 40x40 items in a row container: no wrapping overflows (clipped by the tile), `wrap` "
    "starts a new line, `wrap-reverse` stacks lines from the bottom.",
)
def flex_wrap() -> np.ndarray:
    eight = [dict(width=40, height=40, margin=5)] * 8
    tiles = [
        (v, demo(eight, flex_direction="row", flex_wrap=v))
        for v in ("nowrap", "wrap", "wrap-reverse")
    ]
    return sheet(tiles, 3, title="flex-wrap")


@example(
    "gap",
    "gap, row-gap, column-gap",
    "Gaps between wrapped items, in points, per axis, and as a percentage of the container.",
)
def gap() -> np.ndarray:
    six = [dict(width=40, height=40)] * 6
    opts: list[tuple[str, dict[str, Any]]] = [
        ("gap = 0", {}), ("gap = 10", dict(gap=10)),
        ("row_gap=20, column_gap=4", dict(row_gap=20, column_gap=4)), ("gap = 8%", dict(gap="8%")),
    ]  # fmt: skip
    tiles = [(cap, demo(six, flex_direction="row", flex_wrap="wrap", **kw)) for cap, kw in opts]
    return sheet(tiles, 4, title="gap")


@example(
    "absolute",
    "position: absolute",
    "Absolute children leave the flow and are placed by insets (measured from the container's "
    "edge, ignoring padding), or pinned with a percentage width like the homepage's icon bar. "
    "Unlike CSS, Yoga does not centre an over-constrained child with `margin: auto`; centre it "
    "with `justify_content` / `align_items` on the parent instead.",
)
def absolute() -> np.ndarray:
    corner = dict(position="absolute", width=40, height=40)
    corners = [dict(corner, **kw) for kw in (dict(top=10, left=10), dict(top=10, right=10),
                                             dict(bottom=10, left=10), dict(bottom=10, right=10))]  # fmt: skip
    centred = [dict(position="absolute", width=60, height=60)]
    bar = [dict(flex=1, margin=5), dict(position="absolute", width="100%", height=50, bottom=0)]
    return sheet(
        [
            ("insets (from the edge)", demo(corners)),
            (
                "centred by the parent",
                demo(centred, justify_content="center", align_items="center"),
            ),
            ("pinned bar, width 100%", demo(bar, padding=0)),
        ],
        3,
        title="position: absolute",
    )


@example(
    "sizing",
    "Percentages, aspect-ratio, min/max",
    "Percent widths, `aspect_ratio` (width 100% => height follows) and `max_width`/`min_height`.",
)
def sizing() -> np.ndarray:
    tiles = [
        ("width 25% / 50% / 25%", demo([dict(width=f"{p}%", height=50, margin_vertical=5) for p in (25, 50, 25)])),
        ("aspect_ratio 16/9, width 100%", demo([dict(width="100%", aspect_ratio=16 / 9)])),
        ("aspect_ratio 1, width 50%", demo([dict(width="50%", aspect_ratio=1)])),
        ("flex 1 + max_width 60", demo([dict(flex=1, max_width=60, margin=5), dict(flex=1, margin=5)], flex_direction="row")),
    ]  # fmt: skip
    return sheet(tiles, 4, title="sizing")


# silence unused imports used by individual examples
_ = (INK, LINE)
