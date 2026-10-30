"""Declaring a layout: ``Box``/``Row``/``Column`` containers and ``Slot`` leaves.

Nodes are plain objects hashed by identity, so a node *is* a valid key for plans and canvases.
Optionally a node can also carry a user ``key`` (str, Enum member, tuple, ...).
"""

from __future__ import annotations

import contextvars
import os
import sys
from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Any, TypeVar, overload

from .fit import FITS, Align

if TYPE_CHECKING:
    from .plan import Plan

K = TypeVar("K", bound=Hashable)

_stack: contextvars.ContextVar[tuple[Box, ...]] = contextvars.ContextVar(
    "pixflex_stack", default=()
)
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def _callsite() -> str:
    frame = sys._getframe(1)
    while frame and os.path.abspath(frame.f_code.co_filename).startswith(_PKG_DIR):
        frame = frame.f_back
    if frame is None:
        return "<unknown>"
    return f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"


class Node:
    """Common base of :class:`Box` and :class:`Slot` (identity-hashed; never compare by value)."""

    parent: Box | None

    def __init__(self, *, key: Hashable | None, background: Any, style: dict[str, Any]) -> None:
        if key is not None:
            hash(key)  # reject unhashable keys right here, not later
        self.key = key
        self.background = background
        self.style = style
        self.parent = None
        self.site = _callsite()
        stack = _stack.get()
        if stack:
            stack[-1]._add(self)

    @property
    def label(self) -> str:
        return repr(self.key) if self.key is not None else f"{type(self).__name__}@{self.site}"

    def __repr__(self) -> str:
        return f"<{self.label}>"


class Box(Node):
    """A container. Use as ``with Box(...) as b:`` to nest children, or pass ``children=[...]``."""

    def __init__(
        self,
        *,
        children: Iterable[Node] = (),
        key: Hashable | None = None,
        background: Any = None,
        **style: Any,
    ) -> None:
        self.children: list[Node] = []
        super().__init__(key=key, background=background, style=style)
        for child in children:
            self._add(child)

    def _add(self, child: Node) -> None:
        if child is self or self._is_descendant_of(child):
            raise ValueError("a node cannot contain itself")
        if child.parent is not None:
            child.parent.children.remove(child)  # re-parent (constructor-style inside a `with`)
        child.parent = self
        self.children.append(child)

    def _is_descendant_of(self, other: Node) -> bool:
        node: Node | None = self
        while node is not None:
            if node is other:
                return True
            node = node.parent
        return False

    def __enter__(self) -> Box:
        self._token = _stack.set((*_stack.get(), self))
        return self

    def __exit__(self, *exc: object) -> None:
        _stack.reset(self._token)

    @overload
    def solve(
        self, width: float | None = ..., height: float | None = ..., *, key_type: None = ...
    ) -> Plan[Hashable]: ...
    @overload
    def solve(
        self, width: float | None = ..., height: float | None = ..., *, key_type: type[K]
    ) -> Plan[K]: ...
    def solve(
        self, width: float | None = None, height: float | None = None, *, key_type: Any = None
    ) -> Plan[Any]:
        """Lay the tree out with Yoga in whole pixels; no arrays are touched."""
        from .plan import solve

        return solve(self, width, height, key_type)


class Row(Box):
    def __init__(self, **kw: Any) -> None:
        kw.setdefault("flex_direction", "row")
        super().__init__(**kw)


class Column(Box):
    def __init__(self, **kw: Any) -> None:
        kw.setdefault("flex_direction", "column")
        super().__init__(**kw)


class Slot(Node):
    """A leaf that receives array content.

    ``content`` (optional) only provides a natural size for layout (e.g. ``Slot(content=img)``);
    nothing is copied. ``fit``/``align`` say what happens when real content differs from the
    slot's rectangle (see :mod:`pixflex.fit`).
    """

    def __init__(
        self,
        content: Any = None,
        *,
        fit: str = "contain",
        align: Align = "center",
        axes: str = "HWC",
        key: Hashable | None = None,
        background: Any = None,
        **style: Any,
    ) -> None:
        if fit not in FITS:
            raise ValueError(f"unknown fit {fit!r}; expected one of {', '.join(FITS)}")
        self.content = content
        self.fit = fit
        self.align = align
        self.content_axes = axes
        super().__init__(key=key, background=background, style=style)


def grid(items: Iterable[Node], cols: int, **style: Any) -> Column:
    """Rows of ``cols`` items each, built from nodes you already created. Yoga has no CSS grid."""
    items = list(items)
    gap = style.pop("gap", 0)
    rows = [Row(children=items[i : i + cols], gap=gap) for i in range(0, len(items), cols)]
    return Column(children=rows, gap=gap, **style)
