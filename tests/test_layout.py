import pytest

from pixflex import Box, Column, DuplicateKeyError, ForeignSlotError, Rect, Row, Slot, grid


def phone():
    with Column(width=250, height=475, padding=10, background=(30, 30, 30)) as root:  # noqa: SIM117
        with Column(flex=1, row_gap=10):
            header = Slot(height=60)
            a = Slot(flex=1, margin_inline=10, fit="cover")
            b = Slot(flex=2, margin_inline=10)
            with Row(
                position="absolute",
                width="100%",
                bottom=0,
                height=64,
                align_items="center",
                justify_content="space-around",
                background=(0, 0, 0),
            ):
                icons = [Slot(width=40, height=40) for _ in range(4)]
    return root, header, a, b, icons


def test_homepage_rects_are_absolute_whole_pixels():
    root, header, a, b, icons = phone()
    plan = root.solve()
    assert plan.size == (475, 250)
    assert plan[header] == Rect(10, 10, 60, 230)
    assert plan[a] == Rect(80, 20, 125, 210)
    assert plan[b] == Rect(215, 20, 250, 210)
    assert [plan[i].x for i in icons] == [19, 76, 134, 191]
    assert plan[icons[0]].y == 413


def test_rect_helpers():
    r = Rect(2, 3, 4, 5)
    assert (r.hw, r.y1, r.x1) == ((4, 5), 6, 8)


def test_content_gives_natural_size_and_keeps_aspect():
    import numpy as np

    img = np.zeros((20, 40, 3), np.uint8)
    with Column() as root:
        s = Slot(content=img)
    assert root.solve()[s].hw == (20, 40)
    with Column(width=100) as root:
        s = Slot(content=img, align_self="flex-start")
    assert root.solve()[s].hw == (20, 40)
    with Column(width=20) as root:  # at-most width shrinks, aspect preserved
        s = Slot(content=img, align_self="flex-start")
    assert root.solve()[s].hw == (10, 20)


def test_constructor_style_children_inside_with():
    with Column(width=100, height=100) as root:
        row = Row(children=[Slot(flex=1), Slot(flex=1)])
    assert root.children == [row] and len(row.children) == 2
    plan = root.solve()
    assert [plan[c].w for c in row.children] == [50, 50]


def test_grid_helper():
    cells = [Slot(width=10, height=10) for _ in range(6)]
    root = grid(cells, cols=3, gap=2)
    plan = root.solve()
    assert plan[cells[0]] == Rect(0, 0, 10, 10) and plan[cells[4]] == Rect(12, 12, 10, 10)


def test_handles_are_identity_hashed():
    a, b = Slot(), Slot()
    assert a != b and len({a, b}) == 2 and hash(a) == object.__hash__(a)


def test_solve_must_be_called_on_root():
    with Column() as root:
        inner = Column()
    with pytest.raises(ValueError):
        inner.solve()
    assert isinstance(root, Box)


@pytest.mark.parametrize(
    "keys",
    [("a", "b"), (1, 2), (("cam", 0), ("cam", 1)), (frozenset({1}), frozenset({2}))],
)
def test_keys_of_any_hashable_type(keys):
    with Row(width=20, height=10) as root:
        s0 = Slot(key=keys[0], flex=1)
        s1 = Slot(key=keys[1], flex=1)
    plan = root.solve()
    assert plan[keys[0]] == plan[s0] and plan[keys[1]] == plan[s1]
    assert set(plan.keys()) == set(keys)
    assert keys[0] in plan and "nope" not in plan


def test_enum_key():
    import enum

    class P(enum.Enum):
        A = 1

    with Row(width=10, height=10) as root:
        s = Slot(key=P.A, flex=1)
    assert root.solve(key_type=P)[P.A] == root.solve()[s]


def test_key_errors():
    with pytest.raises(TypeError):
        Slot(key=["unhashable"])  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    with Row(width=10, height=10) as root:
        Slot(key="dup")
        Slot(key="dup")
    with pytest.raises(DuplicateKeyError, match="dup"):
        root.solve()
    with Row(width=10, height=10) as root:
        Slot(key="cam0")
    plan = root.solve()
    with pytest.raises(KeyError, match="did you mean 'cam0'"):
        plan["cam9"]
    with pytest.raises(ForeignSlotError):
        plan[Slot()]


def test_unknown_fit_rejected():
    with pytest.raises(ValueError):
        Slot(fit="stretchy")
