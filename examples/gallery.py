"""Render every gallery example to docs/gallery/*.png and write docs/GALLERY.md.

Run: python examples/gallery.py [slug ...]
"""

import inspect
import sys
import textwrap
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from gallery.examples import (
    EXAMPLES,  # noqa: E402
    demo,  # noqa: E402
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "gallery"

SECTIONS = {
    "homepage": ("Real layouts", None),
    "justify_content": (
        "Yoga docs property playgrounds",
        "These recreate the small playgrounds on the Yoga docs pages: a 200x200 container with "
        "padding 10 and 50x50 children with margin 5. Each tile is solved by Yoga, drawn with "
        "`draw_plan`, and the tiles are laid out and composed with pixflex too. They all share "
        "this helper:",
    ),
}


def source(fn) -> str:
    """A function's source without its decorators."""
    src = textwrap.dedent(inspect.getsource(fn))
    return src[src.index("def ") :].rstrip()


def main(only: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    md = [
        "# Gallery",
        "",
        "Every image below is produced by pixflex itself: layouts are solved by Yoga, then arrays "
        "are composed into one buffer. The code under each picture is the example function "
        "verbatim (regenerate with `python examples/gallery.py`).",
        "",
    ]
    for ex in EXAMPLES:
        if ex.slug in SECTIONS:
            heading, intro = SECTIONS[ex.slug]
            md += [f"## {heading}", ""]
            if intro:
                md += [intro, "", "```python", source(demo), "```", ""]
        if only and ex.slug not in only:
            continue
        img = ex.fn()
        Image.fromarray(img).save(OUT / f"{ex.slug}.png", optimize=True)
        print(f"{ex.slug}: {img.shape[1]}x{img.shape[0]}")
        code = "\n\n\n".join([*(source(h) for h in ex.uses), source(ex.fn)])
        md += [
            f"### {ex.title}", "", ex.blurb, "", f"![{ex.title}](gallery/{ex.slug}.png)", "",
            "<details><summary>Code</summary>", "", "```python", code, "```", "", "</details>", "",
        ]  # fmt: skip
    if not only:
        (ROOT / "docs" / "GALLERY.md").write_text("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1:])
