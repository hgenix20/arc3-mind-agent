"""Tests for the rule-synthesis harness. No network; engine tests skip without the wheels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from arcagent.core.types import RESET_ID, Action, Phase
from arcagent.synth import run as synth_run
from arcagent.synth.prompt import PROMPT_CHAR_CAP, SYSTEM_PROMPT, build_messages, extract_code
from arcagent.synth.render import (
    crop,
    parse_grid,
    render_diff,
    render_grid,
    render_transition,
    rle_row,
)
from arcagent.synth.trace import Trace, Transition, click_targets, collect_trace
from arcagent.synth.verify import IDENTITY_CODE, identity_exact_rate, verify

# ---------------------------------------------------------------- helpers


def _grid(fill: int = 0) -> np.ndarray:
    return np.full((64, 64), fill, dtype=np.int8)


def _synthetic_trace(n: int = 10) -> Trace:
    """Even steps change one cell, odd steps change nothing."""
    ts: list[Transition] = []
    g = _grid(0)
    for i in range(n):
        after = g.copy()
        if i % 2 == 0:
            after[i, i] = 3
        ts.append(Transition(i, g, Action(1), after, "NOT_FINISHED", 0, 0))
        g = after
    return Trace(game="syn", seed=0, win_levels=1, transitions=ts)


_TRACES: dict[tuple[str, int], Trace] = {}


def _trace(env_dir: Path, game: str, steps: int) -> Trace:
    key = (game, steps)
    if key not in _TRACES:
        _TRACES[key] = collect_trace(env_dir, game, steps, seed=0)
    return _TRACES[key]


# ---------------------------------------------------------------- trace


@pytest.mark.parametrize("game", ["ls20", "vc33"])
def test_collect_trace_shapes_and_split(env_dir: Path, game: str) -> None:
    tr = _trace(env_dir, game, 40)
    assert len(tr.transitions) == 40
    assert (len(tr.train), len(tr.heldout)) == (28, 12)
    assert tr.train + tr.heldout == tr.transitions
    assert [t.step for t in tr.transitions] == list(range(40))
    for prev, nxt in zip(tr.transitions, tr.transitions[1:], strict=False):
        assert np.array_equal(prev.grid_after, nxt.grid_before)
        if prev.phase_after in (Phase.GAME_OVER, Phase.WIN):
            assert nxt.action.id == RESET_ID
    for t in tr.transitions:
        assert t.grid_before.shape == (64, 64)
        assert t.grid_before.dtype == np.int8


def test_collect_trace_cycles_every_action(env_dir: Path) -> None:
    tr = _trace(env_dir, "ls20", 40)
    assert {t.action.id for t in tr.transitions} >= {1, 2, 3, 4}


def test_trace_npz_round_trip(env_dir: Path, tmp_path: Path) -> None:
    tr = _trace(env_dir, "vc33", 40)
    path = tr.save(tmp_path)
    assert path.with_suffix(".json").exists()
    back = Trace.load(path)
    assert (back.game, back.seed, back.win_levels) == (tr.game, tr.seed, tr.win_levels)
    assert len(back.transitions) == len(tr.transitions)
    for a, b in zip(tr.transitions, back.transitions, strict=True):
        assert a.action == b.action
        assert (a.step, a.phase_after, a.levels_before, a.levels_after) == (
            b.step,
            b.phase_after,
            b.levels_before,
            b.levels_after,
        )
        assert np.array_equal(a.grid_before, b.grid_before)
        assert np.array_equal(a.grid_after, b.grid_after)


def test_split_is_integer_floor() -> None:
    assert _synthetic_trace(10).n_train == 7
    assert _synthetic_trace(9).n_train == 6


def test_click_targets() -> None:
    assert len(click_targets(_grid(0))) == 16  # no objects: 4x4 lattice
    g = _grid(0)
    g[10:13, 20:23] = 4  # 3x3 block, center (21, 11)
    g[40, 5] = 7
    assert click_targets(g) == [(21, 11), (5, 40)]
    assert click_targets(g, limit=1) == [(21, 11)]


# ---------------------------------------------------------------- render


def test_crop_and_rle() -> None:
    g = np.full((64, 64), -1, dtype=np.int8)
    g[2:4, 5:9] = [[1, 1, 2, 15], [0, 0, 0, 0]]
    sub, x0, y0 = crop(g)
    assert (x0, y0, sub.shape) == (5, 2, (2, 4))
    assert rle_row(sub[0]) == "1*2 2 f"
    assert rle_row(sub[1]) == "0*4"
    assert rle_row(np.array([-1, -1, 3])) == ".*2 3"
    assert render_grid(g) == "crop x=5 y=2 w=4 h=2\n2: 1*2 2 f\n3: 0*4"
    empty, _, _ = crop(np.full((64, 64), -1, dtype=np.int8))
    assert empty.shape == (0, 0)


def test_render_grid_round_trip_and_size() -> None:
    rng = np.random.default_rng(0)
    g = rng.integers(0, 16, size=(64, 64)).astype(np.int8)
    g[:, 60:] = -1
    assert np.array_equal(parse_grid(render_grid(g)), g)
    flat = render_grid(_grid(7))
    assert len(flat) < 300
    assert np.array_equal(parse_grid(flat), _grid(7))


def test_render_diff_cap_and_transition_modes() -> None:
    a, b = _grid(0), _grid(0)
    b[0, :] = 1
    b[1, :10] = 2
    d = render_diff(a, b)
    assert d.startswith("(0,0) 0->1; (1,0) 0->1")
    assert d.endswith("... and 14 more")
    assert d.count("->") == 60
    assert render_diff(a, a) == "no cells changed"
    t = Transition(5, a, Action(6, 3, 4), b, "NOT_FINISHED", 0, 1)
    out = render_transition(t)
    assert out.splitlines()[0] == "#5 ACTION6(x=3, y=4) changed=74 level 0->1 phase=NOT_FINISHED"
    assert "next grid:" in render_transition(t, mode="full")
    with pytest.raises(ValueError):
        render_transition(t, mode="bogus")


# ---------------------------------------------------------------- prompt


@pytest.mark.parametrize("game", ["ls20", "vc33"])
def test_prompt_size_bound(env_dir: Path, game: str) -> None:
    tr = _trace(env_dir, game, 60)  # 42 train transitions, so 40 are shown
    msgs = build_messages(tr, max_transitions=40)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert sum(len(m["content"]) for m in msgs) < PROMPT_CHAR_CAP
    assert msgs[1]["content"].count("\n#") == 40


def test_prompt_contract_and_repair_turn() -> None:
    tr = _synthetic_trace(10)
    assert "def step(grid: list[list[int]], action: int" in SYSTEM_PROMPT
    msgs = build_messages(tr, 40)
    user = msgs[1]["content"]
    assert "Starting grid" in user
    assert "#7 " not in user  # held-out transitions are never shown
    assert "```python" in user
    res = verify("def step(grid, action, x=-1, y=-1):\n    return grid\n", tr)
    rep = build_messages(
        tr, 40, prior_code="def step(g, a, x=-1, y=-1):\n    return g", mismatches=res.mismatches
    )
    assert [m["role"] for m in rep] == ["system", "user", "assistant", "user"]
    assert "return g" in rep[2]["content"]
    assert "train #0 ACTION1: 1 cells wrong" in rep[3]["content"]
    assert "(0,0) expected 3 got 0" in rep[3]["content"]
    assert "Do not special-case" in rep[3]["content"]


def test_extract_code() -> None:
    reply = "text\n```\nnot this\n```\n```python\ndef step(g, a, x=-1, y=-1):\n    return g\n```\n"
    assert extract_code(reply) == "def step(g, a, x=-1, y=-1):\n    return g\n"
    assert extract_code("```\nx = 1\n```") == "x = 1\n"
    assert extract_code("no code here") is None


# ---------------------------------------------------------------- verify


def test_verify_identity_matches_numpy_baseline() -> None:
    tr = _synthetic_trace(10)
    res = verify(IDENTITY_CODE, tr)
    assert res.crashed == 0
    assert not res.timed_out
    assert res.exact_rate_train == pytest.approx(identity_exact_rate(tr.train))
    assert res.exact_rate_heldout == pytest.approx(identity_exact_rate(tr.heldout))
    assert res.cell_accuracy == pytest.approx(1 - 1 / (3 * 4096))
    assert [m.index for m in res.mismatches] == [0, 2, 4, 6]
    assert res.mismatches[0].wrong == ((0, 0, 3, 0),)


def test_verify_candidate_that_raises() -> None:
    tr = _synthetic_trace(6)
    res = verify("raise RuntimeError('boom at import')\n", tr)
    assert res.crashed == 6
    assert res.load_error is not None
    assert "boom at import" in res.load_error
    res2 = verify("def step(grid, action, x=-1, y=-1):\n    raise ValueError('bad rule')\n", tr)
    assert res2.crashed == 6
    assert res2.exact_rate_train == 0.0
    assert res2.cell_accuracy == 0.0
    assert res2.mismatches[0].error is not None
    assert "bad rule" in res2.mismatches[0].error
    res3 = verify("def step(grid, action, x=-1, y=-1):\n    return [[0]]\n", tr)
    assert res3.crashed == 6
    assert res3.mismatches[0].error is not None
    assert "64x64" in res3.mismatches[0].error


def test_verify_timeout_is_honored() -> None:
    tr = _synthetic_trace(6)
    code = "def step(grid, action, x=-1, y=-1):\n    while True:\n        pass\n"
    import time  # noqa: PLC0415

    t0 = time.perf_counter()
    res = verify(code, tr, timeout_s=2.0)
    assert time.perf_counter() - t0 < 10
    assert res.timed_out
    assert res.crashed == 6
    assert all(r.error == "timeout" for r in res.results)


def test_verify_network_is_blocked() -> None:
    tr = _synthetic_trace(4)
    code = (
        "import socket\n"
        "def step(grid, action, x=-1, y=-1):\n"
        "    socket.create_connection(('127.0.0.1', 9))\n"
        "    return grid\n"
    )
    res = verify(code, tr)
    assert res.crashed == 4
    assert res.mismatches[0].error is not None
    assert "network is disabled" in res.mismatches[0].error


# ---------------------------------------------------------------- run


ROUND_KEYS = {
    "round",
    "exact_rate_train",
    "exact_rate_heldout",
    "cell_accuracy",
    "crashed",
    "prompt_tokens",
    "completion_tokens",
    "seconds",
    "code_path",
    "error",
}


def test_run_fake_end_to_end(env_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    argv = ["--env-dir", str(env_dir), "--games", "ls20", "vc33", "--steps", "40"]
    argv += ["--rounds", "2", "--fake", "--out", str(out)]
    synth_run.main(argv)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["config"]["model"] == "fake-identity"
    for game in ("ls20", "vc33"):
        g = doc["games"][game]
        assert g["trace"]["train"] == 28
        assert g["trace"]["heldout"] == 12
        assert 0.0 <= g["baseline"]["identity_exact_rate_heldout"] <= 1.0
        assert 1 <= len(g["rounds"]) <= 2
        for r in g["rounds"]:
            assert set(r) >= ROUND_KEYS
            assert Path(r["code_path"]).exists()
        # the identity candidate run through the subprocess must agree with numpy
        final = g["final"]
        assert final["exact_rate_heldout"] == pytest.approx(
            g["baseline"]["identity_exact_rate_heldout"]
        )
    assert (tmp_path / "code" / "ls20-r1.py").exists()
    assert (tmp_path / "prompts" / "ls20-r1.json").exists()
    assert list((tmp_path / "traces").glob("*.npz"))


def test_run_dry_run_writes_prompts(env_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    argv = ["--env-dir", str(env_dir), "--games", "vc33", "--steps", "40", "--dry-run"]
    synth_run.main([*argv, "--out", str(out)])
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["games"]["vc33"]["dry_run"]["prompt_chars_r1"] > 0
    assert (tmp_path / "prompts" / "vc33-r1.txt").exists()
    assert (tmp_path / "prompts" / "vc33-repair-sample.json").exists()
    assert not (tmp_path / "code").exists()


def test_run_requires_model_for_live_mode() -> None:
    with pytest.raises(SystemExit):
        synth_run.parse_args(["--games", "ls20"])
