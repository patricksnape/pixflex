# pixflex (+ yogalayout): handover

State as of 2026-10-31. Binding-specific details: `yogalayout/docs/HANDOVER.md`.

## What it is
Lay out images like a web page. Describe a flexbox tree, Yoga solves every rectangle in whole
pixels, then arrays (numpy, torch, anything array-API) are placed into slots with as few copies as
possible. Pure Python, MIT, depends on `yogalayout` and `array-api-compat`.

## Layout
| Path | What |
|---|---|
| `src/pixflex/dsl.py` | `Box`/`Row`/`Column`/`Slot`, `with`-block nesting (contextvars), `grid()`, identity-hashed handles, `key=` (any hashable) |
| `src/pixflex/plan.py` | `solve()` builds the Yoga tree (measure function from `content` shape); `Plan[K]`: `plan[k] -> Rect(y,x,h,w)`, `.size`, `.canvas()`, `.compose()`, `.view()`, `.debug_image()`; errors `ForeignSlotError`, `DuplicateKeyError`, KeyError with "did you mean" |
| `src/pixflex/canvas.py` | `Canvas[A, K]`: one allocation from `like=`, `set/__setitem__/update/clear/sub/view/snapshot/to_numpy/ring`; fit/align/dtype/channel/device rules; **painter's-algorithm stacking** so later overlapping items (e.g. an absolute logo) stay on top across incremental writes |
| `src/pixflex/fit.py` | pure-int CSS `object-fit`/`object-position` math (contain/cover/fill/none/scale-down/exact) |
| `src/pixflex/resize.py` | resizer registry (`register_resizer`); torch `F.interpolate(antialias=True)`; numpy cv2 > Pillow > scipy |
| `examples/` | `video_overlay.py` (real MP4, incremental vs full recompose timing), `figure_panel.py`, `batch_torch.py`, `gallery/` + `gallery.py` (14 rendered PNGs -> `docs/gallery/`, `docs/GALLERY.md` is **generated**, rerun `python examples/gallery.py`) |
| `tests/` | 102 tests: layout, canvas (numpy+torch via `be` fixture), key typing (runs pyright on snippets), gallery smoke (~30 s, torch) |

## Design decisions
- **Layout and pixels are separate.** `root.solve()` involves no arrays; render content at
  `plan[slot].hw` and no resize is needed. **A write never changes the layout and never spills
  outside its slot's rect.**
- **Any hashable is a key; handles are recommended.** `plan[k]`/`cv[k]` take a Slot/Box handle or
  a user `key=` (str, Enum, tuple...). Strings are not mixed with kwargs, so static analysis works;
  `solve(key_type=Panel)` makes pyright reject `cv["typo"]`.
- **Copies.** Views are zero-copy; each pixel is written once unless resized. `cv.array` is the
  live buffer (zero-copy to an encoder), `snapshot()` copies, `ring(n)` for async consumers.
  Incremental update (static header/footer drawn once): ~25 ms/frame vs ~45 ms full recompose.
- **Wrong-size data** follows the slot's `fit` (default `contain`, letterbox cleared only when the
  placement changes); `fit="exact"` raises `ShapeMismatch` naming the slot, needed and got shapes.
  dtype casts only within a kind; float->uint8 and cross-device raise unless `convert=`/`to_device=`.
- **Stacking** (added after the gallery exposed it): later siblings draw over earlier ones;
  overlays are redrawn clipped to the overlap on every write; unwritten overlay slots stay transparent.
- Sample photos in the gallery have **no frame** (a white frame made slots look misaligned).

## Dev setup
```
uv venv --python 3.14 && uv pip install --python .venv -e ../yogalayout -e . pytest pyright ruff numpy pillow imageio av
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu   # for torch tests/gallery
.venv/bin/python -m pytest -q    # 102 tests
```
**Do not run `uv sync`** in a venv with manually installed CPU torch: it removes it. `uv.lock` is
gitignored. `pyproject.toml` points `yogalayout` at `../yogalayout` (editable path source), so clone
both repos side by side until `yogalayout` is on PyPI.

## Known limits
- JAX and cupy are **untested/unsupported** (README says so); the write shim for immutable arrays
  was not built. Torch CPU is tested; CUDA is not.
- Layout is frozen per canvas: `plan.resolve/diff/relayout` (re-layout reusing the Yoga tree) are
  not built. Yoga 3.2 has no CSS grid; use `grid()`.
- No text rendering: pass arrays (the gallery uses Pillow to draw captions).
- The gallery smoke test is slow (~30 s); consider marking it `slow`.
- `ruff format --check examples` flags one file (not looked at).

## Next steps (priority order)
1. Release plumbing (see yogalayout handover): wheels, PyPI names, then switch the `yogalayout`
   source from the path to the PyPI dependency and add CI for this repo.
2. `T`-axis clips on tensor backends are implemented; benchmark the batched torch path on GPU.
3. `plan.resolve()/diff()/relayout()` for changing layouts mid-video.
4. JAX (`.at[].set`) and cupy backends with tests.
5. Docs page "Addressing slots" (handles -> typed NamedTuple -> Enum keys -> strings) from the
   README/gallery material; mark the gallery test `slow`.
