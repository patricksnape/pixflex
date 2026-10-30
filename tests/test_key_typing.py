"""Static-typing guarantees, checked by running pyright on small snippets."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PYRIGHT = shutil.which("pyright") or str(Path(sys.executable).parent / "pyright")
pytestmark = pytest.mark.skipif(not Path(PYRIGHT).exists(), reason="pyright not installed")

PRELUDE = """
import enum
import numpy as np
from pixflex import Column, Slot

class Panel(enum.Enum):
    VIDEO = 1

with Column(width=10, height=10) as root:
    s = Slot(key=Panel.VIDEO, flex=1)
frame = np.zeros((1, 1, 3), np.uint8)
"""


def check(tmp_path, body):
    f = tmp_path / "snippet.py"
    f.write_text(PRELUDE + body)
    out = subprocess.run(
        [PYRIGHT, "--outputjson", str(f)], capture_output=True, text=True, cwd=Path(__file__).parent
    )
    diags = json.loads(out.stdout)["generalDiagnostics"]
    return [(d["severity"], d["message"].splitlines()[0]) for d in diags]


def errors(tmp_path, body):
    return [m for sev, m in check(tmp_path, body) if sev == "error"]


def test_enum_keys_accept_members_and_handles(tmp_path):
    body = """
plan = root.solve(key_type=Panel)
cv = plan.canvas(frame)
cv[Panel.VIDEO] = frame
cv[s] = frame
plan[Panel.VIDEO]
plan[s]
"""
    assert errors(tmp_path, body) == []


def test_enum_keys_reject_strings_and_typos(tmp_path):
    body = """
plan = root.solve(key_type=Panel)
cv = plan.canvas(frame)
cv["video"] = frame
cv[Panel.VIDOE] = frame
plan["video"]
"""
    errs = errors(tmp_path, body)
    assert len(errs) == 3, errs
    assert any("VIDOE" in e for e in errs)


def test_untyped_plan_accepts_any_hashable(tmp_path):
    body = """
plan = root.solve()
cv = plan.canvas(frame)
cv["anything"] = frame
cv[(1, 2)] = frame
cv.untyped[Panel.VIDEO] = frame
"""
    assert errors(tmp_path, body) == []


def test_canvas_array_type_follows_the_input_array(tmp_path):
    body = """
reveal_type(root.solve().canvas(frame).array)
"""
    infos = [m for sev, m in check(tmp_path, body) if sev == "information"]
    assert infos and "ndarray" in infos[0], infos
