"""Solving a layout into whole-pixel rectangles (no arrays involved)."""

from __future__ import annotations

import difflib
from collections.abc import Hashable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar

import yogalayout as yl

from .dsl import Box, Node, Slot
from .errors import DuplicateKeyError, ForeignSlotError

if TYPE_CHECKING:
    from .canvas import Canvas

K = TypeVar("K", bound=Hashable)
A = TypeVar("A")


class Rect(NamedTuple):
    """An absolute pixel rectangle, ready for slicing: ``img[r.y:r.y1, r.x:r.x1]``."""

    y: int
    x: int
    h: int
    w: int

    @property
    def hw(self) -> tuple[int, int]:
        return (self.h, self.w)

    @property
    def y1(self) -> int:
        return self.y + self.h

    @property
    def x1(self) -> int:
        return self.x + self.w


def _measure_for(content: Any, axes: str):
    shape = content.shape
    h0, w0 = shape[axes.index("H")], shape[axes.index("W")]

    def measure(width: float, wm: Any, height: float, hm: Any) -> tuple[float, float]:
        exact_w, exact_h = wm == yl.MeasureMode.EXACTLY, hm == yl.MeasureMode.EXACTLY
        if exact_w and exact_h:
            return width, height
        if exact_w:
            h = width * h0 / w0
            return width, min(h, height) if hm == yl.MeasureMode.AT_MOST else h
        if exact_h:
            w = height * w0 / h0
            return min(w, width) if wm == yl.MeasureMode.AT_MOST else w, height
        s = 1.0
        if wm == yl.MeasureMode.AT_MOST and w0 > width:
            s = min(s, width / w0)
        if hm == yl.MeasureMode.AT_MOST and h0 > height:
            s = min(s, height / h0)
        return w0 * s, h0 * s

    return measure


def _build(item: Node, cfg: yl.Config, nodes: dict[Node, yl.Node]) -> yl.Node:
    node = yl.Node(cfg, **item.style)
    nodes[item] = node
    if isinstance(item, Box):
        for child in item.children:
            node.insert_child(_build(child, cfg, nodes))
    elif isinstance(item, Slot) and item.content is not None:
        node.set_measure(_measure_for(item.content, item.content_axes))
    return node


def _walk(item: Node) -> Iterator[Node]:
    yield item
    if isinstance(item, Box):
        for child in item.children:
            yield from _walk(child)


def solve(root: Box, width: float | None, height: float | None, key_type: Any = None) -> Plan[Any]:
    if root.parent is not None:
        raise ValueError("solve() must be called on the root of the layout")
    cfg = yl.Config(use_web_defaults=False, point_scale_factor=1)  # whole pixels
    nodes: dict[Node, yl.Node] = {}
    yroot = _build(root, cfg, nodes)
    yroot.calculate_layout(width, height)

    items = list(_walk(root))
    keys: dict[Hashable, Node] = {}
    for it in items:
        if it.key is not None:
            if it.key in keys:
                raise DuplicateKeyError(
                    f"key {it.key!r} used by both {keys[it.key].label} and {it.label}"
                )
            keys[it.key] = it

    rects: dict[Node, Rect] = {}

    def place(item: Node, oy: int, ox: int) -> None:
        lay = nodes[item].layout
        r = Rect(oy + round(lay.top), ox + round(lay.left), round(lay.height), round(lay.width))
        rects[item] = r
        if isinstance(item, Box):
            for child in item.children:
                place(child, r.y, r.x)

    place(root, 0, 0)
    return Plan(root, items, rects, keys)


class Plan(Generic[K]):
    """A solved layout: ``plan[k]`` is the :class:`Rect` of a node, addressed by handle or key."""

    def __init__(
        self, root: Box, items: list[Node], rects: dict[Node, Rect], keys: dict[Hashable, Node]
    ) -> None:
        self.root = root
        self._items = items
        self._rects = rects
        self._keys = keys

    @property
    def size(self) -> tuple[int, int]:
        r = self._rects[self.root]
        return (r.h, r.w)

    @property
    def rects(self) -> Mapping[Node, Rect]:
        return self._rects

    def keys(self) -> list[Hashable]:
        return list(self._keys)

    @property
    def untyped(self) -> Plan[Hashable]:
        return self  # type: ignore[return-value]

    def find(self, key: Hashable) -> Node:
        return self._resolve(key)

    def _resolve(self, k: Any) -> Node:
        if isinstance(k, Node):
            if k not in self._rects:
                raise ForeignSlotError(f"{k.label} does not belong to this layout")
            return k
        try:
            return self._keys[k]
        except KeyError:
            msg = f"no node with key {k!r}"
            if isinstance(k, str):
                close = difflib.get_close_matches(k, [x for x in self._keys if isinstance(x, str)])
                if close:
                    msg += f" (did you mean {close[0]!r}?)"
            raise KeyError(msg) from None

    def __getitem__(self, k: Node | K) -> Rect:
        return self._rects[self._resolve(k)]

    def __contains__(self, k: object) -> bool:
        try:
            self._resolve(k)
        except KeyError:
            return False
        return True

    def _subplan(self, box: Box) -> Plan[K]:
        origin = self._rects[box]
        items = list(_walk(box))
        rects = {
            i: Rect(
                self._rects[i].y - origin.y,
                self._rects[i].x - origin.x,
                self._rects[i].h,
                self._rects[i].w,
            )
            for i in items
        }
        keys = {k: n for k, n in self._keys.items() if n in rects}
        sub: Plan[K] = Plan(box, items, rects, keys)
        return sub

    def __repr__(self) -> str:
        lines = [f"Plan(size={self.size})"]

        def show(item: Node, depth: int) -> None:
            r = self._rects[item]
            lines.append(f"{'  ' * (depth + 1)}{item.label}  y={r.y} x={r.x} h={r.h} w={r.w}")
            if isinstance(item, Box):
                for c in item.children:
                    show(c, depth + 1)

        show(self.root, 0)
        return "\n".join(lines)

    # ---- placing content (delegates to Canvas)
    def canvas(self, like: A | None = None, **kw: Any) -> Canvas[A, K]:
        from .canvas import Canvas

        return Canvas(self, like, **kw)

    def allocate(self, like: Any = None, **kw: Any) -> Any:
        """One backend-native array of the plan's size, backgrounds painted. No content written."""
        return self.canvas(like, **kw).array

    def view(self, array: Any, k: Node | K, axes: str = "HWC") -> Any:
        """A no-copy view of ``array`` (a canvas array) covering node ``k``."""
        r = self[k]
        idx = tuple(
            slice(r.y, r.y1) if a == "H" else slice(r.x, r.x1) if a == "W" else slice(None)
            for a in axes
        )
        return array[idx]

    def compose(self, content: Mapping[Any, Any], *, like: Any = None, **kw: Any) -> Any:
        """Allocate a canvas, write every ``{key: array}`` and return the finished array."""
        if like is None and content:
            like = next(iter(content.values()))
        cv = self.canvas(like, **kw)
        cv.update(content)
        return cv.array

    def debug_image(self) -> Any:
        """A numpy uint8 image outlining every rectangle (needs numpy)."""
        import numpy as np

        h, w = self.size
        img = np.full((h, w, 3), 255, np.uint8)
        for n, r in enumerate(self._rects.values()):
            color = [(230, 80, 80), (80, 160, 90), (70, 110, 220), (200, 150, 40)][n % 4]
            if r.h and r.w:
                img[r.y, r.x : r.x1] = color
                img[r.y1 - 1, r.x : r.x1] = color
                img[r.y : r.y1, r.x] = color
                img[r.y : r.y1, r.x1 - 1] = color
        return img
