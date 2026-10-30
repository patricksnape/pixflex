import numpy as np
import pytest

import pixflex
from pixflex import (
    BackendMismatch,
    ChannelMismatch,
    Column,
    DTypeMismatch,
    MissingResizer,
    Row,
    ShapeMismatch,
    Slot,
)
from tests.conftest import solid

RED, GREEN, BLUE, GREY = (255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)


def two_slots(**slot_kw):
    """Header (20x100) over a body slot (40x100) on a blue background."""
    with Column(width=100, height=60, background=BLUE) as root:
        head = Slot(height=20, fit="exact")
        body = Slot(flex=1, **slot_kw)
    return root.solve(), head, body


def px(be, cv):
    return be.np(cv.array)


def test_compose_exact_and_single_write_per_pixel(be):
    plan, head, body = two_slots(fit="exact")
    img = plan.compose({head: be.arr(solid(20, 100, RED)), body: be.arr(solid(40, 100, GREEN))})
    out = be.np(img)
    assert out.shape == (60, 100, 3) and out.dtype == np.uint8
    assert (out[:20] == RED).all() and (out[20:] == GREEN).all()


def test_exact_size_never_touches_the_resizer(be, monkeypatch):
    def boom(*a):
        raise AssertionError("resizer called")

    monkeypatch.setitem(pixflex.resize._REGISTRY, be.name, boom)
    plan, head, body = two_slots(fit="contain")
    cv = plan.canvas(be.arr(solid(1, 1, RED)))
    cv[body] = be.arr(solid(40, 100, GREEN))  # already the slot's size
    assert (px(be, cv)[20:] == GREEN).all()


def test_views_share_memory_with_canvas(be):
    plan, head, body = two_slots()
    cv = plan.canvas(be.arr(solid(1, 1, RED)))
    v = cv.view(body)
    assert tuple(v.shape) == (40, 100, 3) and be.shares_memory(v, cv.array)
    v[...] = be.arr(solid(40, 100, GREEN))  # writing the view writes the canvas
    assert (px(be, cv)[20:] == GREEN).all()
    v2 = plan.view(cv.array, body)
    assert be.shares_memory(v2, cv.array)


def test_contain_letterboxes_with_slot_background(be):
    plan, head, body = two_slots(fit="contain")
    cv = plan.canvas(be.arr(solid(1, 1, RED)), static={head: be.arr(solid(20, 100, RED))})
    cv[body] = be.arr(solid(10, 20, GREEN))  # 1:2 into 100x40 -> 80x40, centred
    out = px(be, cv)
    assert (out[20:, 10:90] == GREEN).all()
    assert (out[20:, :10] == BLUE).all() and (out[20:, 90:] == BLUE).all()  # bars
    assert (out[:20] == RED).all()


def test_cover_crops_then_resizes(be):
    plan, head, body = two_slots(fit="cover", align="left")
    src = np.zeros((40, 200, 3), np.uint8)
    src[:, :100] = GREEN
    src[:, 100:] = RED
    cv = plan.canvas(be.arr(src))
    cv[body] = be.arr(src)
    out = px(be, cv)
    assert (out[20:] == GREEN).all()  # left-aligned crop keeps only the green half


def test_none_crops_and_pads_by_alignment(be):
    plan, head, body = two_slots(fit="none", align="top left")
    cv = plan.canvas(be.arr(solid(1, 1, RED)))
    cv[body] = be.arr(solid(10, 10, GREEN))
    out = px(be, cv)
    assert (out[20:30, :10] == GREEN).all() and (out[30:, :] == BLUE).all()
    big = solid(100, 300, RED)
    cv[body] = be.arr(big)  # larger than the slot: cropped, never spills
    assert (px(be, cv)[20:] == RED).all() and (px(be, cv)[:20] == BLUE).all()


def test_scale_down_only_shrinks(be):
    plan, head, body = two_slots(fit="scale-down", align="top left")
    cv = plan.canvas(be.arr(solid(1, 1, RED)))
    cv[body] = be.arr(solid(10, 10, GREEN))  # fits: untouched size
    assert (px(be, cv)[20:30, :10] == GREEN).all() and (px(be, cv)[20:30, 10:] == BLUE).all()
    cv[body] = be.arr(solid(80, 80, RED))  # too big: contained to 40x40
    assert (px(be, cv)[20:, :40] == RED).all()


def test_exact_mismatch_message(be):
    plan, head, body = two_slots(fit="exact")
    cv = plan.canvas(be.arr(solid(1, 1, RED)))
    with pytest.raises(ShapeMismatch, match=r"needs \(h=40, w=100\), got \(h=39, w=100\)"):
        cv[body] = be.arr(solid(39, 100, RED))


def test_update_touches_only_its_rect(be):
    plan, head, body = two_slots(fit="contain")
    cv = plan.canvas(be.arr(solid(1, 1, RED)), static={head: be.arr(solid(20, 100, RED))})
    cv[body] = be.arr(solid(40, 100, GREEN))
    before = px(be, cv).copy()
    cv[body] = be.arr(solid(40, 100, BLUE))
    after = px(be, cv)
    assert (after[:20] == before[:20]).all() and (after[20:] == BLUE).all()


def test_stale_letterbox_is_cleared_when_placement_changes(be):
    plan, head, body = two_slots(fit="contain")
    cv = plan.canvas(be.arr(solid(1, 1, RED)))
    cv[body] = be.arr(solid(40, 50, GREEN))  # 1.25 aspect -> 50x40 centred: side bars
    assert (px(be, cv)[20:, :25] == BLUE).all() and (px(be, cv)[20:, 25:75] == GREEN).all()
    cv[body] = be.arr(solid(10, 100, RED))  # wide -> 100x10 centred: bars above/below instead
    out = px(be, cv)[20:]
    assert (out[:15] == BLUE).all() and (out[15:25] == RED).all() and (out[25:] == BLUE).all()
    cv[body] = be.arr(solid(40, 100, GREEN))  # full again
    assert (px(be, cv)[20:] == GREEN).all()


def test_clear_slot_and_box(be):
    plan, head, body = two_slots()
    cv = plan.canvas(be.arr(solid(1, 1, RED)))
    cv.update({head: be.arr(solid(20, 100, RED)), body: be.arr(solid(40, 100, GREEN))})
    cv.clear(body)
    assert (px(be, cv)[20:] == BLUE).all() and (px(be, cv)[:20] == RED).all()
    cv.clear(plan.root)
    assert (px(be, cv) == BLUE).all()


def test_snapshot_is_independent_and_array_is_live(be):
    plan, head, body = two_slots(fit="exact")
    cv = plan.canvas(be.arr(solid(1, 1, RED)))
    live = cv.array
    cv[body] = be.arr(solid(40, 100, GREEN))
    snap = cv.snapshot()
    cv[body] = be.arr(solid(40, 100, RED))
    assert (be.np(snap)[20:] == GREEN).all() and (be.np(live)[20:] == RED).all()
    assert isinstance(cv.to_numpy(), np.ndarray)


def test_sub_canvas_is_a_view_with_box_relative_rects(be):
    with Column(width=100, height=60, background=BLUE) as root:
        Slot(height=20, key="top")
        with Row(flex=1, padding=5, key="panel") as panel:
            inner = Slot(flex=1, fit="exact", key="inner")
    plan = root.solve()
    cv = plan.canvas(be.arr(solid(1, 1, RED)))
    sub = cv.sub(panel)
    assert sub.plan[inner].y == 5 and sub.plan[inner].x == 5
    assert be.shares_memory(sub.array, cv.array)
    sub["inner"] = be.arr(solid(30, 90, GREEN))
    out = px(be, cv)
    assert (out[25:55, 5:95] == GREEN).all() and (out[:25] == BLUE).all()


def test_ring_buffers_keep_static_content_and_rotate(be):
    plan, head, body = two_slots(fit="exact")
    cv = plan.canvas(be.arr(solid(1, 1, RED)), static={head: be.arr(solid(20, 100, RED))})
    bufs = cv.ring(2)
    a, b, a2 = next(bufs), next(bufs), next(bufs)
    assert a is a2 and a is not b
    a[body] = be.arr(solid(40, 100, GREEN))
    assert (be.np(a.array)[:20] == RED).all() and (be.np(b.array)[:20] == RED).all()
    assert (be.np(b.array)[20:] == BLUE).all() and (be.np(cv.array)[20:] == BLUE).all()


def test_grayscale_broadcasts_and_hw_content_gets_a_channel(be):
    plan, head, body = two_slots(fit="exact")
    cv = plan.canvas(be.arr(solid(1, 1, RED)))
    cv[body] = be.arr(np.full((40, 100, 1), 77, np.uint8))
    assert (px(be, cv)[20:] == 77).all()
    cv[body] = be.arr(np.full((40, 100), 99, np.uint8))  # no channel axis at all
    assert (px(be, cv)[20:] == 99).all()


def test_channel_dtype_backend_errors(be):
    plan, head, body = two_slots(fit="exact")
    cv = plan.canvas(be.arr(solid(1, 1, RED)))
    with pytest.raises(ChannelMismatch, match="2 channels"):
        cv[body] = be.arr(np.zeros((40, 100, 2), np.uint8))
    with pytest.raises(DTypeMismatch):
        cv[body] = be.arr(np.zeros((40, 100, 3), np.float32))
    cv.set(body, be.arr(np.full((40, 100, 3), 0.5, np.float32)), convert=be.to_u8)
    assert (px(be, cv)[20:] == 127).all()
    if be.name == "torch":
        with pytest.raises(BackendMismatch):
            cv[body] = np.zeros((40, 100, 3), np.uint8)


def test_safe_widening_dtype_is_cast(be):
    plan, head, body = two_slots(fit="exact")
    cv = plan.canvas(be.arr(solid(1, 1, RED, np.uint16)))  # uint16 canvas
    cv[body] = be.arr(solid(40, 100, GREEN, np.uint8))  # uint8 -> uint16 is lossless
    assert px(be, cv).dtype == np.uint16 and (px(be, cv)[20:] == GREEN).all()


def test_rgba_into_rgb(be):
    plan, head, body = two_slots(fit="exact")
    cv = plan.canvas(be.arr(solid(1, 1, RED)))
    rgba = solid(40, 100, (0, 255, 0, 128))
    with pytest.raises(ChannelMismatch, match="alpha"):
        cv[body] = be.arr(rgba)
    cv.set(body, be.arr(rgba), alpha="drop")
    assert (px(be, cv)[20:] == GREEN).all()
    cv.clear(body)
    cv.set(body, be.arr(rgba), alpha="over")  # half-transparent green over blue
    out = px(be, cv)[20]
    assert abs(int(out[0, 1]) - 128) <= 1 and abs(int(out[0, 2]) - 127) <= 1


def test_float_canvas_and_fill(be):
    with Column(width=4, height=4) as root:
        s = Slot(flex=1, fit="exact")
    p = root.solve()
    cv = p.canvas(be.arr(np.zeros((1, 1, 3), np.float32)), fill=0.25)
    assert px(be, cv).dtype == np.float32 and (px(be, cv) == 0.25).all()
    cv[s] = be.arr(np.ones((4, 4, 3), np.float32))
    assert (px(be, cv) == 1).all()


def test_channels_first_and_hw_axes(be):
    plan, head, body = two_slots(fit="exact")
    cv = (
        plan.canvas(axes="CHW", channels=3, xp=None)
        if be.name == "numpy"
        else plan.canvas(be.arr(np.zeros((3, 1, 1), np.uint8)), axes="CHW")
    )
    cv[body] = be.arr(np.full((3, 40, 100), 9, np.uint8))
    assert px(be, cv).shape == (3, 60, 100) and (px(be, cv)[:, 20:] == 9).all()
    with Column(width=100, height=60) as root:  # no RGB background to clash with a gray canvas
        flat = Slot(flex=1, fit="exact")
    gray = root.solve().canvas(be.arr(np.zeros((1, 1), np.uint8)), axes="HW")
    gray[flat] = be.arr(np.full((60, 100), 5, np.uint8))
    assert px(be, gray).shape == (60, 100) and (px(be, gray) == 5).all()
    with pytest.raises(ChannelMismatch):  # tuple background cannot paint a channel-less canvas
        plan.canvas(be.arr(np.zeros((1, 1), np.uint8)), axes="HW")


def test_clip_with_time_axis_broadcasts_static_parts(be):
    plan, head, body = two_slots(fit="exact")
    like = be.arr(np.zeros((4, 1, 1, 3), np.uint8))
    cv = plan.canvas(like, axes="THWC", static={head: be.arr(solid(20, 100, RED))})
    clip = np.zeros((4, 40, 100, 3), np.uint8)
    clip[2] = GREEN
    cv[body] = be.arr(clip)
    out = px(be, cv)
    assert out.shape == (4, 60, 100, 3) and (out[:, :20] == RED).all()
    assert (out[2, 20:] == GREEN).all() and (out[1, 20:] == 0).all()


def test_clip_resize_is_batched(be):
    plan, head, body = two_slots(fit="fill")
    cv = plan.canvas(be.arr(np.zeros((3, 1, 1, 3), np.uint8)), axes="THWC")
    cv[body] = be.arr(np.full((3, 8, 8, 3), 200, np.uint8))
    assert (px(be, cv)[:, 20:] == 200).all()


def test_missing_resizer_is_actionable(be, monkeypatch):
    monkeypatch.delitem(pixflex.resize._BUILTIN, be.name)
    plan, head, body = two_slots(fit="fill")
    cv = plan.canvas(be.arr(solid(1, 1, RED)))
    with pytest.raises(MissingResizer, match="register_resizer"):
        cv[body] = be.arr(solid(5, 5, RED))


def test_container_cannot_be_written(be):
    plan, head, body = two_slots()
    cv = plan.canvas(be.arr(solid(1, 1, RED)))
    with pytest.raises(TypeError, match="container"):
        cv[plan.root] = be.arr(solid(60, 100, RED))


def test_out_buffer_is_used_in_place(be):
    plan, head, body = two_slots(fit="exact")
    out = be.arr(np.zeros((60, 100, 3), np.uint8))
    img = plan.compose({body: be.arr(solid(40, 100, GREEN))}, out=out)
    assert img is out and (be.np(out)[20:] == GREEN).all()


def test_slot_background_and_nested_painting(be):
    with Column(width=10, height=10, background=BLUE) as root:
        s = Slot(height=4, background=GREEN)
    cv = root.solve().canvas(be.arr(solid(1, 1, RED)))
    out = px(be, cv)
    assert (out[:4] == GREEN).all() and (out[4:] == BLUE).all()
    assert s.background == GREEN
