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


def test_request_body_disables_thinking() -> None:
    body = synth_run.request_body("m", [{"role": "user", "content": "x"}], seed=42)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] == synth_run.MAX_TOKENS
    assert body["model"] == "m"
    assert body["seed"] == 42
    seeds = {synth_run.request_seed(s, r, a) for s in (0, 1) for r in (1, 2, 3) for a in (0, 1)}
    assert len(seeds) == 12  # distinct per (trace seed, round, attempt)


def test_parse_completion_reads_finish_reason_and_reasoning() -> None:
    doc = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": "", "reasoning_content": "thinking " * 10},
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6000},
    }
    comp = synth_run.parse_completion(doc, 1.5)
    assert comp.text == ""
    assert comp.finish_reason == "length"
    assert comp.reasoning_chars == 90
    assert (comp.prompt_tokens, comp.completion_tokens, comp.seconds) == (5, 6000, 1.5)
    # the v4 Kaggle failure shape, minus the usage field, still parses
    comp = synth_run.parse_completion({"choices": [{"message": {"content": None}}]}, 0.0)
    assert comp.text == "" and comp.finish_reason == "" and comp.reasoning_chars == 0
    with pytest.raises(KeyError):
        synth_run.parse_completion({"choices": [{}]}, 0.0)


def test_run_game_stops_after_a_reply_without_code(tmp_path: Path) -> None:
    calls = 0

    def no_code_chat(_messages: list[synth_run.Message], _seed: int) -> synth_run.Completion:
        nonlocal calls
        calls += 1
        return synth_run.Completion("I thought about it.", 10, 6000, 2.0, "length", 1234)

    rec = synth_run.run_game(
        _synthetic_trace(10), no_code_chat, rounds=3, max_transitions=5, root=tmp_path
    )
    assert calls == 2  # one re-sample with the next seed, then the game stops
    assert len(rec["rounds"]) == 1
    r = rec["rounds"][0]
    assert r["no_code"] is True
    assert r["attempts"] == 2
    assert r["prompt_tokens"] == 20  # accounting sums over attempts
    assert (tmp_path / "replies" / "syn-r1-retry0.txt").exists()
    assert rec["final"] is None
    assert r["finish_reason"] == "length"
    assert r["reasoning_chars"] == 1234
    assert r["reply_chars"] == len("I thought about it.")
    assert r["load_error"]
    assert (tmp_path / "replies" / "syn-r1.txt").read_text(
        encoding="utf-8"
    ) == "I thought about it."
    assert (tmp_path / "code" / "syn-r1.py").read_text(encoding="utf-8") == synth_run.NO_CODE
    assert "none" in synth_run.format_table({"games": {"syn": rec}})


def test_same_program_ignores_comments_and_formatting() -> None:
    a = "def step(g, a, x=-1, y=-1):\n    return g\n"
    b = "# a comment\ndef step(g, a, x=-1, y=-1):\n\n    return g  # same\n"
    c = "def step(g, a, x=-1, y=-1):\n    return [r[:] for r in g]\n"
    assert synth_run.same_program(a, b)
    assert not synth_run.same_program(a, c)
    assert synth_run.same_program("def (", "def (")
    assert not synth_run.same_program("def (", a)


def test_verify_records_the_candidates_own_change() -> None:
    tr = _synthetic_trace(10)
    # paints (5,5) every step: wrong everywhere, but a visible change of exactly one cell
    code = "def step(g, a, x=-1, y=-1):\n    n = [r[:] for r in g]\n    n[5][5] = 7\n    return n\n"
    res = verify(code, tr)
    r0 = res.results[0]
    assert r0.predicted_changes == 1
    assert r0.predicted == ((5, 5, 0, 7),)
    identity = verify(IDENTITY_CODE, tr).results[0]
    assert identity.predicted_changes == 0 and identity.predicted == ()


def test_repair_turn_shows_score_and_own_change() -> None:
    tr = _synthetic_trace(10)
    res = verify(IDENTITY_CODE, tr)
    k = sum(r.exact for r in res.results if r.split == "train")
    rep = build_messages(tr, 40, IDENTITY_CODE, res.mismatches, exact_train=(k, len(tr.train)))
    turn = rep[3]["content"]
    assert f"It reproduced {k} of {len(tr.train)} training transitions exactly." in turn
    assert "your change: (nothing)" in turn
    assert "at most five plain lines" in turn
    assert "wastes this round" in turn
    # without a score the line is absent, and a candidate that changes cells lists them
    code = "def step(g, a, x=-1, y=-1):\n    n = [r[:] for r in g]\n    n[5][5] = 7\n    return n\n"
    rep = build_messages(tr, 40, code, verify(code, tr).mismatches)
    assert "reproduced" not in rep[3]["content"]
    assert "your change: (5,5) 0->7" in rep[3]["content"]


def test_run_game_stops_when_the_repair_is_the_same_program(tmp_path: Path) -> None:
    calls = 0
    first = f"```python\n{IDENTITY_CODE}```"
    again = "I changed nothing of substance.\n```python\n# new comment\n" + IDENTITY_CODE + "```"

    def chat(_messages: list[synth_run.Message], _seed: int) -> synth_run.Completion:
        nonlocal calls
        calls += 1
        return synth_run.Completion(first if calls == 1 else again, 1, 1, 0.1, "stop", 0)

    rec = synth_run.run_game(_synthetic_trace(10), chat, rounds=3, max_transitions=5, root=tmp_path)
    assert calls == 2
    assert [r["unchanged"] for r in rec["rounds"]] == [False, True]
    # the round-2 prompt carried the round-1 score line
    r2 = json.loads((tmp_path / "prompts" / "syn-r2.json").read_text(encoding="utf-8"))
    assert "training transitions exactly" in r2[3]["content"]
    assert "same" in synth_run.format_table({"games": {"syn": rec}})


def test_run_game_resamples_once_when_the_reply_has_no_code(tmp_path: Path) -> None:
    seeds: list[int] = []
    paint = (
        "def step(g, a, x=-1, y=-1):\n    n = [r[:] for r in g]\n    n[5][5] = 7\n    return n\n"
    )

    def chat(_messages: list[synth_run.Message], seed: int) -> synth_run.Completion:
        seeds.append(seed)
        if len(seeds) == 1:
            return synth_run.Completion(
                "Let's try... no. Let's try... no.", 5, 16000, 9.0, "length", 0
            )
        return synth_run.Completion(f"```python\n{paint}```", 5, 100, 1.0, "stop", 0)

    tr = _synthetic_trace(10)
    rec = synth_run.run_game(tr, chat, rounds=1, max_transitions=5, root=tmp_path)
    assert seeds == [synth_run.request_seed(tr.seed, 1, 0), synth_run.request_seed(tr.seed, 1, 1)]
    r = rec["rounds"][0]
    assert r["no_code"] is False and r["attempts"] == 2
    assert r["completion_tokens"] == 16100 and r["seconds"] == 10.0
    assert "n[5][5] = 7" in (tmp_path / "code" / "syn-r1.py").read_text(encoding="utf-8")
    assert "Let's try" in (tmp_path / "replies" / "syn-r1-retry0.txt").read_text(encoding="utf-8")
    assert rec["final"] is r


def test_final_is_the_round_that_fits_train_best(tmp_path: Path) -> None:
    paint = (
        "def step(g, a, x=-1, y=-1):\n    n = [r[:] for r in g]\n    n[5][5] = 7\n    return n\n"
    )
    replies = [f"```python\n{IDENTITY_CODE}```", f"```python\n{paint}```", "no fence", "no fence"]

    def chat(_messages: list[synth_run.Message], _seed: int) -> synth_run.Completion:
        return synth_run.Completion(replies.pop(0), 1, 1, 0.1, "stop", 0)

    rec = synth_run.run_game(_synthetic_trace(10), chat, rounds=3, max_transitions=5, root=tmp_path)
    assert [r["no_code"] for r in rec["rounds"]] == [False, False, True]
    # identity is exact on the odd (no-change) steps, paint on none: round 1 fits train best
    assert rec["rounds"][0]["exact_rate_train"] > rec["rounds"][1]["exact_rate_train"]
    assert rec["final"]["round"] == 1
    assert rec["final"]["code_path"].endswith("syn-r1.py")
    assert 0.0 < rec["rounds"][1]["cell_accuracy_train"] < 1.0


def test_select_final_prefers_train_fit_then_train_cells_then_earliest() -> None:
    def rnd(
        n: int, train: float, cells: float, held: float, no_code: bool = False
    ) -> dict[str, object]:
        return {
            "round": n,
            "exact_rate_train": train,
            "cell_accuracy_train": cells,
            "exact_rate_heldout": held,
            "no_code": no_code,
        }

    def pick(*rounds: dict[str, object]) -> int:
        final = synth_run.select_final(list(rounds))
        assert final is not None
        return int(str(final["round"]))

    # a later round that fits train better wins even with a lower held-out score
    assert pick(rnd(1, 0.5, 0.99, 0.9), rnd(2, 0.6, 0.98, 0.1)) == 2
    # a regressing repair does not replace the better earlier round (vc33 in run v8)
    assert pick(rnd(1, 0.56, 0.99, 0.53), rnd(2, 0.45, 0.99, 0.47)) == 1
    # equal exact rate: train cell accuracy decides, then the earliest round
    assert pick(rnd(1, 0.0, 0.980, 0.0), rnd(2, 0.0, 0.984, 0.0)) == 2
    assert pick(rnd(1, 0.0, 0.98, 0.0), rnd(2, 0.0, 0.98, 0.0)) == 1
    # rounds without code never win; none with code -> None
    assert pick(rnd(1, 0.2, 0.9, 0.2), rnd(2, 0.0, 0.0, 0.0, True)) == 1
    assert synth_run.select_final([rnd(1, 0.0, 0.0, 0.0, True)]) is None
    assert synth_run.select_final([{"round": 1, "error": "boom"}]) is None


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
