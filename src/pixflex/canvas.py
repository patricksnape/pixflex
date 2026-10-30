"""A reusable output buffer you write content into, slot by slot."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import Any, Generic, TypeVar

from . import _backend as be
from .dsl import Box, Node, Slot
from .errors import (
    BackendMismatch,
    ChannelMismatch,
    DeviceMismatch,
    DTypeMismatch,
    ForeignSlotError,
)
from .fit import Align, place
from .plan import Plan, Rect
from .resize import resize

K = TypeVar("K", bound=Hashable)
A = TypeVar("A")

_ALLOWED_AXES = set("HWCTN")


class Canvas(Generic[A, K]):
    """One allocation holding the finished image; slots are written in place.

    Constant parts (headers, footers...) are drawn once and stay; ``cv[k] = x`` rewrites only
    the rectangle of slot ``k``. ``cv.array`` is the live buffer (no copy); use
    :meth:`snapshot` when you need to keep a frame.
    """

    def __init__(
        self,
        plan: Plan[K],
        like: A | None = None,
        *,
        axes: str = "HWC",
        channels: int | None = None,
        dtype: Any = None,
        fill: float = 0,
        lead: Mapping[str, int] | None = None,
        static: Mapping[Any, Any] | None = None,
        out: A | None = None,
        xp: Any = None,
        _array: Any = None,
    ) -> None:
        if not {"H", "W"} <= set(axes) or set(axes) - _ALLOWED_AXES or len(set(axes)) != len(axes):
            raise ValueError(
                f"axes must use distinct letters from HWCTN and include H, W: {axes!r}"
            )
        self.plan = plan
        self.axes = axes
        self._fill = fill
        self._placed: dict[Node, tuple[int, int, int, int] | None] = {}
        if _array is not None:  # sub-canvas / clone: adopt an existing buffer
            self._array = _array
            self._xp = be.namespace(_array)
            self._device = be.device_of(_array)
            return

        if out is not None:
            array = out
        else:
            array = self._allocate(like, channels, dtype, fill, lead or {}, xp)
        self._array: Any = array
        self._xp = be.namespace(array)
        self._device = be.device_of(array)
        if out is not None:
            self._check_shape(out)
        self._paint()
        if static:
            self.update(static)

    # ------------------------------------------------------------------ allocation
    def _allocate(
        self,
        like: Any,
        channels: int | None,
        dtype: Any,
        fill: float,
        lead: Mapping[str, int],
        xp: Any,
    ) -> Any:
        axes = self.axes
        if like is not None:
            xp = be.namespace(like)
            dtype = dtype if dtype is not None else like.dtype
            device = be.device_of(like)
        else:
            xp = xp or be.default_namespace()
            dtype = dtype if dtype is not None else xp.uint8
            device = None
        sizes: dict[str, int] = dict(lead)
        if like is not None:
            if like.ndim == len(axes):
                for letter in "CTN":
                    if letter in axes and letter not in sizes:
                        sizes[letter] = like.shape[axes.index(letter)]
            elif like.ndim == len(axes) - 1 and "C" in axes:
                sizes.setdefault("C", 1)
        if channels is not None:
            sizes["C"] = channels
        if "C" in axes:
            sizes.setdefault("C", 3)
        for letter in "TN":
            if letter in axes and letter not in sizes:
                raise ValueError(
                    f"axes {axes!r} has {letter!r}: pass lead={{'{letter}': size}} or like="
                )
        sizes["H"], sizes["W"] = self.plan.size
        shape = tuple(sizes[a] for a in axes)
        kwargs = {"device": device} if device is not None else {}
        return xp.full(shape, fill, dtype=dtype, **kwargs)

    def _check_shape(self, arr: Any) -> None:
        h, w = self.plan.size
        got = (arr.shape[self.axes.index("H")], arr.shape[self.axes.index("W")])
        if got != (h, w) or arr.ndim != len(self.axes):
            raise ValueError(
                f"out has shape {tuple(arr.shape)}, plan needs H={h}, W={w} ({self.axes})"
            )

    # ------------------------------------------------------------------ geometry helpers
    @property
    def array(self) -> A:
        """The live buffer (no copy)."""
        return self._array

    def _idx(self, y0: int, y1: int, x0: int, x1: int) -> tuple[slice, ...]:
        return tuple(
            slice(y0, y1) if a == "H" else slice(x0, x1) if a == "W" else slice(None)
            for a in self.axes
        )

    def _effective_background(self, item: Node) -> Any:
        node: Node | None = item
        while node is not None:
            if node.background is not None:
                return node.background
            node = node.parent
        return self._fill

    def _bg_value(self, bg: Any) -> Any:
        if not isinstance(bg, (tuple, list)):
            return bg
        c = self.axes.index("C") if "C" in self.axes else None
        if c is None or len(bg) != self._array.shape[c]:
            raise ChannelMismatch(f"background {tuple(bg)} does not match the canvas channels")
        shape = [1] * len(self.axes)
        shape[c] = len(bg)
        return self._xp.reshape(
            self._xp.asarray(list(bg), dtype=self._array.dtype, device=self._device), tuple(shape)
        )

    def _fill_rect(self, rect: tuple[int, int, int, int], bg: Any) -> None:
        y0, y1, x0, x1 = rect
        if y1 > y0 and x1 > x0:
            self._array[self._idx(y0, y1, x0, x1)] = self._bg_value(bg)

    def _paint(self) -> None:
        for item in self.plan._items:  # pre-order: containers first, nested ones on top
            if item.background is not None:
                r = self.plan._rects[item]
                self._fill_rect((r.y, r.y1, r.x, r.x1), item.background)

    # ------------------------------------------------------------------ writing content
    def __setitem__(self, k: Node | K, content: Any) -> None:
        self.set(k, content)

    def update(self, content: Mapping[Any, Any]) -> None:
        """Write several slots at once: ``cv.update({a: x, b: y})``."""
        for k, v in content.items():
            self.set(k, v)

    def set(
        self,
        k: Node | K,
        content: Any,
        *,
        fit: str | None = None,
        align: Align | None = None,
        alpha: str | None = None,
        convert: Callable[[Any], Any] | None = None,
        to_device: bool = False,
    ) -> None:
        """Write ``content`` into slot ``k``.

        ``fit``/``align`` override the slot's own; ``alpha`` is ``"drop"`` or ``"over"`` for RGBA
        content into an RGB canvas; ``convert`` runs first (e.g. float 0-1 -> uint8);
        ``to_device`` allows a device transfer.
        """
        item = self.plan._resolve(k)
        if not isinstance(item, Slot):
            raise TypeError(f"{item.label} is a container; write to its slots (or use .sub())")
        rect = self.plan._rects[item]
        if convert is not None:
            content = convert(content)
        src = self._normalize(content, item, to_device)
        src = self._channels(src, rect, alpha)

        hi, wi = self.axes.index("H"), self.axes.index("W")
        # Content may lack leading axes (static content broadcast across T); H/W sit from the end.
        offset = src.ndim - len(self.axes)
        h0, w0 = src.shape[hi + offset], src.shape[wi + offset]
        pl = place(
            (h0, w0),
            rect.hw,
            fit or item.fit,
            align if align is not None else item.align,
            label=f"slot {item.label}",
        )
        y0, y1, x0, x1 = pl.src
        view = src[self._src_idx(src.ndim, y0, y1, x0, x1)]
        if pl.needs_resize:
            view = resize(view, pl.out_hw, self._src_axes(src.ndim))
        view = self._cast(view)

        oh, ow = pl.out_hw
        dy, dx = rect.y + pl.dst[0], rect.x + pl.dst[1]
        placed = (dy, dy + oh, dx, dx + ow)
        slot_rect = (rect.y, rect.y1, rect.x, rect.x1)
        if placed != slot_rect and self._placed.get(item) != placed:
            self._clear_strips(slot_rect, placed, self._effective_background(item))
        self._placed[item] = placed
        if oh and ow:
            self._array[self._idx(*placed)] = view

    def _src_axes(self, ndim: int) -> str:
        return self.axes[len(self.axes) - ndim :]

    def _src_idx(self, ndim: int, y0: int, y1: int, x0: int, x1: int) -> tuple[slice, ...]:
        return tuple(
            slice(y0, y1) if a == "H" else slice(x0, x1) if a == "W" else slice(None)
            for a in self._src_axes(ndim)
        )

    def _normalize(self, content: Any, item: Slot, to_device: bool) -> Any:
        if be.backend_name(content) != be.backend_name(self._array):
            raise BackendMismatch(
                f"{item.label}: got a {be.backend_name(content)} array for a "
                f"{be.backend_name(self._array)} canvas"
            )
        if be.device_of(content) != self._device:
            if not to_device:
                raise DeviceMismatch(
                    f"{item.label}: content is on {be.device_of(content)}, canvas on "
                    f"{self._device} (pass to_device=True to allow the transfer)"
                )
            content = self._xp.asarray(content, device=self._device)
        n = len(self.axes)
        if content.ndim == n - 1 and "C" in self.axes and not set("TN") & set(self.axes):
            content = content[..., None] if self.axes.index("C") == n - 1 else content[None]
        elif content.ndim not in (n, n - 1):
            raise ValueError(
                f"{item.label}: content has {content.ndim} dims, canvas axes {self.axes!r} "
                f"expect {n}"
            )
        return content

    def _channels(self, src: Any, rect: Rect, alpha: str | None) -> Any:
        if "C" not in self.axes:
            return src
        c_axis = self.axes.index("C") - (len(self.axes) - src.ndim)
        have, want = src.shape[c_axis], self._array.shape[self.axes.index("C")]
        if have == want or have == 1:
            return src
        if have == want + 1 and alpha in ("drop", "over"):

            def ch(lo: int | None, hi: int | None) -> Any:
                idx = [slice(None)] * src.ndim
                idx[c_axis] = slice(lo, hi)
                return src[tuple(idx)]

            rgb = ch(0, want)
            if alpha == "drop":
                return rgb
            return self._over(rgb, ch(want, None), rect)
        raise ChannelMismatch(
            f"content has {have} channels, canvas has {want}"
            + (" (pass alpha='drop' or 'over' for RGBA)" if have == want + 1 else "")
        )

    def _over(self, rgb: Any, a: Any, rect: Rect) -> Any:
        """Composite RGBA onto what is currently on the canvas (same size as the slot only)."""
        xp = self._xp
        dtype = self._array.dtype
        top = float(xp.iinfo(dtype).max) if xp.isdtype(dtype, "integral") else 1.0
        af = xp.astype(a, xp.float32) / top
        dst = xp.astype(self._array[self._idx(rect.y, rect.y1, rect.x, rect.x1)], xp.float32)
        if tuple(dst.shape) != tuple(rgb.shape):
            raise ValueError("alpha='over' needs content with the slot's exact size")
        out = xp.astype(rgb, xp.float32) * af + dst * (1 - af)
        if xp.isdtype(dtype, "integral"):
            out = xp.round(out)
        return xp.astype(out, dtype)

    def _cast(self, arr: Any) -> Any:
        xp, want = self._xp, self._array.dtype
        if arr.dtype == want:
            return arr
        if not xp.can_cast(arr.dtype, want):
            raise DTypeMismatch(
                f"cannot safely convert {arr.dtype} to the canvas dtype {want}; "
                "pass convert=... (e.g. float 0-1 -> uint8) or choose a canvas dtype"
            )
        return xp.astype(arr, want)

    def _clear_strips(
        self, slot: tuple[int, int, int, int], placed: tuple[int, int, int, int], bg: Any
    ) -> None:
        sy0, sy1, sx0, sx1 = slot
        py0, py1, px0, px1 = placed
        self._fill_rect((sy0, py0, sx0, sx1), bg)  # above
        self._fill_rect((py1, sy1, sx0, sx1), bg)  # below
        self._fill_rect((py0, py1, sx0, px0), bg)  # left
        self._fill_rect((py0, py1, px1, sx1), bg)  # right

    # ------------------------------------------------------------------ whole-canvas operations
    def clear(self, k: Node | K) -> None:
        """Reset a slot, or a whole box subtree, to its background."""
        item = self.plan._resolve(k)
        r = self.plan._rects[item]
        self._fill_rect((r.y, r.y1, r.x, r.x1), self._effective_background(item))
        for n in _subtree(item):
            self._placed.pop(n, None)
            if n.background is not None and n is not item:
                rr = self.plan._rects[n]
                self._fill_rect((rr.y, rr.y1, rr.x, rr.x1), n.background)

    def sub(self, box: Node | K) -> Canvas[A, K]:
        """A canvas scoped to ``box``: its array is a view into this one; rects are box-relative."""
        item = self.plan._resolve(box)
        if not isinstance(item, Box):
            raise TypeError(f"{item.label} is not a container")
        r = self.plan._rects[item]
        view = self._array[self._idx(r.y, r.y1, r.x, r.x1)]
        sub_plan: Plan[K] = self.plan._subplan(item)
        return Canvas(sub_plan, axes=self.axes, fill=self._fill, _array=view)

    def view(self, k: Node | K) -> Any:
        """A no-copy view of the canvas inside node ``k``."""
        r = self.plan[k]
        return self._array[self._idx(r.y, r.y1, r.x, r.x1)]

    def snapshot(self) -> A:
        """A copy of the current image (safe to keep while the canvas is updated)."""
        return be.copy(self._array)

    def to_numpy(self) -> Any:
        """A numpy copy (the single device->host transfer when saving from a GPU)."""
        return be.to_numpy(self._array)

    def ring(self, n: int) -> Iterator[Canvas[A, K]]:
        """Cycle through ``n`` copies of this canvas (static content already drawn in each).

        Hand each to a consumer (encoder thread, async copy) and write the next frame into the
        next buffer; a buffer is only reused after ``n`` steps.
        """
        if n < 1:
            raise ValueError("ring needs at least one buffer")
        buffers = [self._clone() for _ in range(n)]
        return itertools.cycle(buffers)

    def _clone(self) -> Canvas[A, K]:
        twin: Canvas[A, K] = Canvas(
            self.plan, axes=self.axes, fill=self._fill, _array=be.copy(self._array)
        )
        twin._placed = dict(self._placed)
        return twin

    @property
    def untyped(self) -> Canvas[A, Hashable]:
        return self  # type: ignore[return-value]


def _subtree(item: Node) -> Iterator[Node]:
    yield item
    if isinstance(item, Box):
        for c in item.children:
            yield from _subtree(c)


__all__ = ["Canvas", "ForeignSlotError"]
