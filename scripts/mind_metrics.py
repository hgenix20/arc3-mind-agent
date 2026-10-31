"""Metrics for the mind entry: the one place the paper's numbers are defined.

Usage: ``python scripts/mind_metrics.py LOG_DIR [LOG_DIR ...] [--json out.json]``.
Importable: ``load(paths)`` reads JSONL run logs (files or directories), and
``compute(records)`` turns the records of any mix of games into
``{"games": {game_id: metrics}, "overall": metrics}``.

Input contract (schema in ``arcagent/mind/runlog.py``). Games are keyed by the
``game`` field of action records (the full id, e.g. ``ft09-0d8bbf25``). Event
records carry no ``game``, so ``load`` stamps each with its file's game. Records
passed straight to ``compute`` must already carry ``game``; a record without it
is filed under ``"?"``. One game's records are assumed to be one run, in file
order.

Definitions
-----------
Action ``i`` is the i-th ``action`` record of a game. The agent logs action ``i``
inside ``choose(obs_i)``, after it has processed the step that produced
``obs_i`` (the result of action ``i-1``). So the ``programs`` list of record
``i+1`` scores the prediction made for action ``i``, and a ``game_over`` or
``level_up`` event whose ``step`` is ``s`` belongs to the action record with
``step == s`` (agent.py writes both events with ``step=step.index``, and no
phase field is logged, so the event is the only game-over signal). RESET
actions (id 0) produce no step and no predictions and are left out of the
outcome and repeat statistics.

Outcome of action ``i`` (first match wins), judged against the next action
record ``i+1`` only when it is on the same level:
  * ``game_over``: a ``game_over`` event with ``step == i.step`` exists.
  * ``noop``: record ``i+1`` has the same ``frame_hash`` as record ``i``.
  * ``disconfirmed``: record ``i`` names an ``active`` program and record
    ``i+1`` lists that program in ``programs`` with ``matched == false``.
    ``active`` is the program in force when action ``i`` was chosen.
Anything else has no bad outcome. The last action of a level has none.

Per game:
  * ``levels_completed``: the number of ``level_up`` events.
  * ``actions_per_completed_level``: for each ``level_up``, the count of action
    records on the level of the action that triggered it, in event order.
  * ``score_estimate``: competition formula. Level ``k`` (1-based) has weight
    ``k``; a completed level scores ``min(h/a, 1)^2`` with ``h`` the human
    action count and ``a`` the agent's; an unfinished level scores 0. The
    result is ``100 * sum(weight * score) / sum(weights over all levels of the
    game)``. ``None`` for games absent from the human file. A completed level
    beyond the human list, or with zero actions, is skipped.
  * ``prediction_accuracy``: ``matched`` over every ``programs`` entry whose
    ``matched`` is a bool; ``None`` with no entries.
  * ``abandonment_latency``: mean of ``step - first_miss_step`` over
    ``program_dropped`` events with reason ``low_fit`` or ``misses_in_row`` and
    a non-null ``first_miss_step``; ``abandonment_latency_n`` is the count used.
    ``drop_reasons`` counts every drop event by reason, all reasons included.
  * ``disconfirmed_repeat_rate``: over non-RESET actions, the share taken in a
    ``(frame_hash, action)`` pair that an earlier action on the same level
    already took with a bad outcome. Computed from raw records, independent of
    whether inhibition was enabled.
  * ``entropy_change_per_action``: mean of ``entropy[t] - entropy[t+1]`` over
    consecutive action records on the same level, skipping non-numeric values.
  * ``transfer_actions``: mean actions per completed level, ``level_1`` for the
    first level and ``levels_2_plus`` for the rest.
  * ``outcomes``: counts of the three bad outcomes.

Fluid-agent metrics (slice 3). Each is ``None`` when the log lacks the events
it needs (old logs, other agents), so an absent signal never reads as zero:
  * ``hit_rate``: ``matched`` over every ``programs`` entry whose ``matched``
    is a bool (the same count as ``prediction_accuracy``).
  * ``hit_rate_halves``: ``{"first", "second"}``. Within each level the action
    records are split at ``n // 2`` (levels with fewer than 2 records are
    skipped); each half's rate pools the entries of its records across levels.
    ``None`` with no entries in either half.
  * ``rule_reuse``: among matched entries that carry a numeric ``level`` (the
    level the fact was created on), the share whose ``level`` is lower than the
    ``level`` of the action record holding the entry. ``None`` with no such
    entry.
  * ``tests_per_level``: ``test`` events divided by the distinct levels that
    have action records. ``None`` with no ``test`` event.
  * ``impasses``: ``wake`` events with reason ``"no plan and no test"``.
    ``None`` with no ``wake`` event.
  * ``search_depth_mean``: mean ``depth`` over ``search`` events with a numeric
    depth. ``None`` with none.
  * ``goals``: ``{"final", "transitions"}``, counts by status. ``final`` counts
    the last status logged for each goal id; ``transitions`` counts every
    ``goal`` event. ``None`` with no ``goal`` event.
  * ``actions_vs_human``: for each completed level, agent actions divided by
    the human count for that level, in ``level_up`` order. A level beyond the
    human list or with zero human actions is skipped. ``None`` for games absent
    from the human file or with no usable level.

``overall`` pools the raw counts of all games (not a mean of per-game means),
except ``score_estimate``, the mean over games that have one, and
``levels_completed``, the sum. Pooled ``actions_vs_human`` concatenates the
per-game lists; pooled ``goals`` sums the per-game counts.

Planner agent (v2) metrics
--------------------------
Reported under ``v2`` per game and pooled, from the fields and events the
planner agent writes (``arcagent/mind/runlog.py``, "Planner agent (v2)
fields"). Every key is present whatever the log holds, so a report never
KeyErrors: with no v2 records the counts are 0, the dicts empty and the rates
``None``.
  * ``calibration``: ten bins ``{"bin": "0.0-0.1", "n", "matched", "rate"}``.
    A pair is an action record's ``p_step`` with the next action record's
    ``prev_matched`` when both records carry the same ``level`` and no
    ``level_up`` or ``game_over`` event lies between them; a pair is skipped
    when ``p_step`` is not a number in [0, 1] or ``prev_matched`` is not a
    bool. Bin ``i`` holds ``i/10 <= p < (i+1)/10`` and ``p == 1`` joins the
    last bin. ``rate`` is ``matched / n``, ``None`` when ``n`` is 0.
  * ``moves_vs_typical``: one ``{"level", "moves", "typical", "ratio"}`` per
    ``consolidate`` event, in event order. ``moves`` counts the action records
    of the level instance the event closes: the records since the previous
    ``level_up``, ``game_over`` or ``consolidate`` event, or, when a
    ``level_up`` or ``game_over`` precedes the ``consolidate`` with no action
    record between them, the records of the instance that event closed (the
    agent logs ``level_up`` and then ``consolidate``). ``typical`` is the
    event's value when numeric, else ``None``; ``ratio`` is ``moves /
    typical`` when ``typical`` is positive, else ``None``.
  * ``gates``: ``trips`` and ``by_gate`` count distinct ``(step, gate name)``
    pairs over ``gate_trip`` events and action records whose ``gate.trip`` is
    not null, so a trip logged both ways counts once. ``seconds_total`` and
    ``tokens_total`` sum the numeric ``gate.seconds`` and ``gate.tokens`` of
    the action records.
  * ``handoffs``: ``handoff`` events counted by ``rule``.
  * ``planner``: ``calls`` counts ``planner_call`` events and ``tokens`` sums
    their numeric ``tokens``; ``skills`` counts their ``skill`` (a name, or the
    ``name`` of a skill dict) and ``questions`` their ``question`` when
    present. ``segments`` counts ``segment`` events of every source and
    ``mean_segment_len`` is the mean of their numeric ``length`` (``None``
    with none).
  * ``forks``: ``opened``, ``failed``, ``returns`` and ``root_replans`` count
    the ``fork_open``, ``branch_failed``, ``return_route`` and ``replan_root``
    events.
  * ``harm``: ``events`` counts ``harm`` events and ``by_kind`` groups them by
    ``kind``.
A ``gate_trip``, ``handoff`` or ``harm`` event without its ``gate``, ``rule`` or
``kind`` field is counted under the key ``"None"``, the same convention as
``drop_reasons``.

Pooled ``v2`` sums the counts, recomputes ``rate`` and ``mean_segment_len``
from the pooled totals, and concatenates ``moves_vs_typical`` in the order the
games first appear in the records. ``v2_metrics(records)`` computes the same
block for one game's records on its own.

Failure modes: ``load`` skips lines that are not JSON objects and returns their
count; ``compute`` ignores records that are not dicts or lack a string ``type``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeGuard

ROOT = Path(__file__).resolve().parent.parent
HUMAN_FILE = ROOT / "data" / "public_human_actions.json"
RESET_ID = 0
LATENCY_REASONS = frozenset({"low_fit", "misses_in_row"})
UNKNOWN_GAME = "?"
BAD_OUTCOMES = ("noop", "game_over", "disconfirmed")
IMPASSE_REASON = "no plan and no test"
CAL_BINS = 10
LEVEL_BOUNDARIES = frozenset({"level_up", "game_over"})
FORK_EVENTS = {
    "fork_open": "opened",
    "branch_failed": "failed",
    "return_route": "returns",
    "replan_root": "root_replans",
}

Rec = dict[str, Any]


def load_human(path: Path = HUMAN_FILE) -> dict[str, list[int]]:
    """Per-level human action counts keyed by the 4-character game id."""
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _files(paths: Iterable[Path | str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        out.extend(sorted(p.rglob("*.jsonl")) if p.is_dir() else [p])
    return out


def load(paths: Iterable[Path | str]) -> tuple[list[Rec], int]:
    """Read JSONL logs from files and directories. Returns ``(records, malformed_lines)``.

    Events without a ``game`` get the game of the file's first action record,
    else the file stem.
    """
    records: list[Rec] = []
    malformed = 0
    for path in _files(paths):
        file_recs: list[Rec] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(rec, dict):
                    malformed += 1
                    continue
                file_recs.append(rec)
        game = next(
            (r["game"] for r in file_recs if r.get("type") == "action" and "game" in r), path.stem
        )
        for rec in file_recs:
            rec.setdefault("game", game)
        records.extend(file_recs)
    return records, malformed


def _is_num(x: object) -> TypeGuard[int | float]:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _is_reset(rec: Rec) -> bool:
    action = rec.get("action")
    return isinstance(action, list) and bool(action) and action[0] == RESET_ID


def _entries(rec: Rec) -> list[Rec]:
    return [p for p in rec.get("programs") or () if isinstance(p, dict)]


def _matched_flags(rec: Rec) -> list[bool]:
    return [p["matched"] for p in _entries(rec) if isinstance(p.get("matched"), bool)]


def _rate(hit: int, total: int) -> float | None:
    return hit / total if total else None


def _vs_human(raw: dict[str, Any], human: list[int] | None) -> list[float]:
    if not human:
        return []
    return [
        actions / human[level]
        for level, actions in zip(raw["level_indices"], raw["per_level_actions"], strict=True)
        if 0 <= level < len(human) and human[level] > 0
    ]


def _disconfirmed(rec: Rec, nxt: Rec) -> bool:
    active = rec.get("active")
    if active is None:
        return False
    return any(
        isinstance(p, dict) and p.get("id") == active and p.get("matched") is False
        for p in nxt.get("programs") or ()
    )


def _bin_index(p: float) -> int:
    """Calibration bin of a probability in [0, 1]; ``p == 1`` joins the last bin."""
    return min(int(p * CAL_BINS), CAL_BINS - 1)


def _bin_label(i: int) -> str:
    return f"{i / CAL_BINS:.1f}-{(i + 1) / CAL_BINS:.1f}"


def _skill_name(value: object) -> str | None:
    """The skill a ``planner_call`` names: a string, or the ``name`` of a skill dict."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        name = value.get("name")
        return name if isinstance(name, str) else None
    return None


class _V2Tally:
    """Planner-agent tallies over one game's records, taken in file order.

    ``add`` takes every record of the game; ``raw`` returns additive tallies
    (``v2_`` keys) that ``compute`` pools across games and ``_v2_summary``
    reduces. Definitions: the module docstring, "Planner agent (v2) metrics".
    """

    def __init__(self) -> None:
        self.cal_n: Counter[int] = Counter()
        self.cal_hit: Counter[int] = Counter()
        self.typical: list[dict[str, Any]] = []
        self.trips: set[tuple[object, str]] = set()
        self.gate_seconds = 0.0
        self.gate_tokens = 0
        self.handoffs: Counter[str] = Counter()
        self.calls = 0
        self.call_tokens = 0
        self.segments = 0
        self.segment_lens: list[float] = []
        self.skills: Counter[str] = Counter()
        self.questions: Counter[str] = Counter()
        self.forks: Counter[str] = Counter()
        self.harm: Counter[str] = Counter()
        self.harm_events = 0
        self._prev: Mapping[str, Any] | None = None  # the last action record seen
        self._boundary = False  # a level_up or game_over event since ``_prev``
        self._open_moves = 0  # action records of the level instance still open
        self._closed_moves = 0  # action records of the instance a boundary last closed

    def add(self, rec: Mapping[str, Any]) -> None:
        kind = rec.get("type")
        if not isinstance(kind, str):
            return
        if kind == "action":
            self._action(rec)
        elif kind in LEVEL_BOUNDARIES:
            self._boundary = True
            self._close()
        elif kind == "consolidate":
            self._consolidate(rec)
        elif kind == "gate_trip":
            self.trips.add((rec.get("step"), str(rec.get("gate"))))
        elif kind == "handoff":
            self.handoffs[str(rec.get("rule"))] += 1
        elif kind == "planner_call":
            self._planner_call(rec)
        elif kind == "segment":
            self.segments += 1
            if _is_num(rec.get("length")):
                self.segment_lens.append(float(rec["length"]))
        elif kind in FORK_EVENTS:
            self.forks[FORK_EVENTS[kind]] += 1
        elif kind == "harm":
            self.harm_events += 1
            self.harm[str(rec.get("kind"))] += 1

    def _action(self, rec: Mapping[str, Any]) -> None:
        prev = self._prev
        if prev is not None and not self._boundary and prev.get("level") == rec.get("level"):
            self._pair(prev.get("p_step"), rec.get("prev_matched"))
        self._prev = rec
        self._boundary = False
        self._open_moves += 1
        gate = rec.get("gate")
        if isinstance(gate, dict):
            self._gate(rec.get("step"), gate)

    def _pair(self, p: Any, matched: object) -> None:
        """One calibration pair, skipped unless ``p`` is in [0, 1] and ``matched`` a bool."""
        if not isinstance(matched, bool) or not _is_num(p) or not 0.0 <= p <= 1.0:
            return
        b = _bin_index(float(p))
        self.cal_n[b] += 1
        self.cal_hit[b] += int(matched)

    def _gate(self, step: object, gate: Mapping[str, Any]) -> None:
        if _is_num(gate.get("seconds")):
            self.gate_seconds += float(gate["seconds"])
        if _is_num(gate.get("tokens")):
            self.gate_tokens += int(gate["tokens"])
        trip = gate.get("trip")
        if trip is not None:
            self.trips.add((step, str(trip)))

    def _close(self) -> None:
        """Close the open level instance, when it has any action records."""
        if self._open_moves:
            self._closed_moves = self._open_moves
            self._open_moves = 0

    def _consolidate(self, rec: Mapping[str, Any]) -> None:
        # A level_up logged just before the consolidate has already closed the
        # instance, so an empty open instance falls back to the one it closed.
        moves = self._open_moves or self._closed_moves
        self._close()
        self._closed_moves = 0
        typical = rec.get("typical")
        if not _is_num(typical):
            typical = None
        self.typical.append(
            {
                "level": rec.get("level"),
                "moves": moves,
                "typical": typical,
                "ratio": moves / typical if typical is not None and typical > 0 else None,
            }
        )

    def _planner_call(self, rec: Mapping[str, Any]) -> None:
        self.calls += 1
        if _is_num(rec.get("tokens")):
            self.call_tokens += int(rec["tokens"])
        skill = _skill_name(rec.get("skill"))
        if skill is not None:
            self.skills[skill] += 1
        question = rec.get("question")
        if isinstance(question, str):
            self.questions[question] += 1

    def raw(self) -> dict[str, Any]:
        """Additive tallies under ``v2_`` keys, so they pool beside the other raw keys."""
        return {
            "v2_cal_n": self.cal_n,
            "v2_cal_hit": self.cal_hit,
            "v2_typical": self.typical,
            "v2_trips": Counter(name for _, name in self.trips),
            "v2_gate_seconds": self.gate_seconds,
            "v2_gate_tokens": self.gate_tokens,
            "v2_handoffs": self.handoffs,
            "v2_calls": self.calls,
            "v2_call_tokens": self.call_tokens,
            "v2_segments": self.segments,
            "v2_segment_lens": self.segment_lens,
            "v2_skills": self.skills,
            "v2_questions": self.questions,
            "v2_forks": self.forks,
            "v2_harm": self.harm,
            "v2_harm_events": self.harm_events,
        }


class _Game:
    """Raw per-game tallies; ``raw`` reduces them to numbers ``_summary`` reports."""

    def __init__(self) -> None:
        self.v2 = _V2Tally()
        self.actions: list[Rec] = []
        self.by_step: dict[int, int] = {}
        self.game_over_idx: set[int] = set()
        self.level_up_idx: list[int] = []
        self.drops: list[Rec] = []
        self.tests: list[Rec] = []
        self.wakes: list[Rec] = []
        self.searches: list[Rec] = []
        self.goal_events: list[Rec] = []
        self.modes: Counter[str] = Counter()  # DriveAgent's moves per mode (``drive_end``)

    def add(self, rec: Rec) -> None:
        self.v2.add(rec)  # the v2 tally keeps its own order-dependent state over every record
        kind = rec["type"]
        if kind == "action":
            self.actions.append(rec)
            step = rec.get("step")
            if isinstance(step, int):
                self.by_step[step] = len(self.actions) - 1
        elif kind == "program_dropped":
            self.drops.append(rec)
        elif kind == "test":
            self.tests.append(rec)
        elif kind == "wake":
            self.wakes.append(rec)
        elif kind == "search":
            self.searches.append(rec)
        elif kind == "goal":
            self.goal_events.append(rec)
        elif kind == "drive_end" and isinstance(rec.get("modes"), dict):
            self.modes.update({str(k): int(v) for k, v in rec["modes"].items() if _is_num(v)})
        elif kind in ("game_over", "level_up"):
            idx = self.by_step.get(rec.get("step", -1))
            if idx is None:
                return  # no action record to attach the event to
            if kind == "game_over":
                self.game_over_idx.add(idx)
            else:
                self.level_up_idx.append(idx)

    def outcomes(self) -> list[str | None]:
        """Bad outcome (or None) of each action record, per the module docstring."""
        out: list[str | None] = []
        for i, rec in enumerate(self.actions):
            nxt = self.actions[i + 1] if i + 1 < len(self.actions) else None
            same = nxt is not None and nxt.get("level") == rec.get("level")
            if _is_reset(rec):
                out.append(None)
            elif i in self.game_over_idx:
                out.append("game_over")
            elif same and nxt is not None and nxt.get("frame_hash") == rec.get("frame_hash"):
                out.append("noop")
            elif same and nxt is not None and _disconfirmed(rec, nxt):
                out.append("disconfirmed")
            else:
                out.append(None)
        return out

    def fluid(self) -> dict[str, Any]:
        """Raw counts behind the slice-3 metrics (see the module docstring)."""
        by_level: dict[Any, list[Rec]] = {}
        for r in self.actions:
            by_level.setdefault(r.get("level"), []).append(r)
        halves = [0, 0, 0, 0]  # first matched/total, second matched/total
        for recs in by_level.values():
            if len(recs) < 2:
                continue
            for half, chunk in enumerate((recs[: len(recs) // 2], recs[len(recs) // 2 :])):
                for r in chunk:
                    for m in _matched_flags(r):
                        halves[2 * half] += m
                        halves[2 * half + 1] += 1
        reused = reuse_total = 0
        for r in self.actions:
            lvl = r.get("level")
            if not _is_num(lvl):
                continue
            for p in _entries(r):
                if p.get("matched") is True and _is_num(p.get("level")):
                    reuse_total += 1
                    reused += p["level"] < lvl
        final: dict[str, str] = {}
        for g in self.goal_events:
            final[str(g.get("id"))] = str(g.get("status"))
        depths = [s["depth"] for s in self.searches if _is_num(s.get("depth"))]
        return {
            "halves": halves,
            "reused": reused,
            "reuse_total": reuse_total,
            "tests": len(self.tests),
            "level_count": len(by_level),
            "wakes": len(self.wakes),
            "impasses": sum(w.get("reason") == IMPASSE_REASON for w in self.wakes),
            "depths": depths,
            "goal_events": len(self.goal_events),
            "goal_final": Counter(final.values()),
            "goal_transitions": Counter(str(g.get("status")) for g in self.goal_events),
            "modes": Counter(self.modes),
        }

    def raw(self) -> dict[str, Any]:
        outcomes = self.outcomes()
        seen_bad: set[tuple[Any, ...]] = set()
        repeats = total = 0
        for rec, outcome in zip(self.actions, outcomes, strict=True):
            if _is_reset(rec):
                continue
            total += 1
            key = (rec.get("level"), rec.get("frame_hash"), tuple(rec.get("action", ())))
            repeats += key in seen_bad
            if outcome is not None:
                seen_bad.add(key)
        matched = [
            p["matched"]
            for r in self.actions
            for p in r.get("programs") or ()
            if isinstance(p, dict) and isinstance(p.get("matched"), bool)
        ]
        latencies = [
            d["step"] - d["first_miss_step"]
            for d in self.drops
            if d.get("reason") in LATENCY_REASONS
            and _is_num(d.get("first_miss_step"))
            and _is_num(d.get("step"))
        ]
        deltas = [
            a["entropy"] - b["entropy"]
            for a, b in zip(self.actions, self.actions[1:], strict=False)
            if a.get("level") == b.get("level")
            and _is_num(a.get("entropy"))
            and _is_num(b.get("entropy"))
        ]
        per_level = Counter(r.get("level") for r in self.actions)
        counted = Counter(o for o in outcomes if o)
        return {
            "levels_completed": len(self.level_up_idx),
            "per_level_actions": [
                per_level[self.actions[i].get("level")] for i in self.level_up_idx
            ],
            "level_indices": [self.actions[i].get("level", 0) for i in self.level_up_idx],
            "matched": matched,
            "latencies": latencies,
            "drop_reasons": Counter(str(d.get("reason")) for d in self.drops),
            "repeats": repeats,
            "repeat_total": total,
            "entropy_deltas": deltas,
            "outcomes": Counter({k: counted[k] for k in BAD_OUTCOMES}),
            **self.fluid(),
            **self.v2.raw(),
        }


def _score(raw: dict[str, Any], human: list[int] | None) -> float | None:
    if not human:
        return None
    total = 0.0
    for level, actions in zip(raw["level_indices"], raw["per_level_actions"], strict=True):
        if actions <= 0 or not 0 <= level < len(human):
            continue
        total += (level + 1) * min(human[level] / actions, 1.0) ** 2
    return 100.0 * total / (len(human) * (len(human) + 1) / 2)


def _summary(raw: dict[str, Any], score: float | None) -> dict[str, Any]:
    pairs = list(zip(raw["level_indices"], raw["per_level_actions"], strict=True))
    matched = raw["matched"]
    total = raw["repeat_total"]
    return {
        "levels_completed": raw["levels_completed"],
        "actions_per_completed_level": raw["per_level_actions"],
        "score_estimate": score,
        "prediction_accuracy": sum(matched) / len(matched) if matched else None,
        "abandonment_latency": _mean(raw["latencies"]),
        "abandonment_latency_n": len(raw["latencies"]),
        "drop_reasons": dict(raw["drop_reasons"]),
        "disconfirmed_repeat_rate": raw["repeats"] / total if total else None,
        "entropy_change_per_action": _mean(raw["entropy_deltas"]),
        "transfer_actions": {
            "level_1": _mean([a for lv, a in pairs if lv == 0]),
            "levels_2_plus": _mean([a for lv, a in pairs if lv > 0]),
        },
        "outcomes": dict(raw["outcomes"]),
        "actions": total,
        **_fluid_summary(raw),
        "v2": _v2_summary(raw),
    }


def _fluid_summary(raw: dict[str, Any]) -> dict[str, Any]:
    h = raw["halves"]
    first, second = _rate(h[0], h[1]), _rate(h[2], h[3])
    return {
        "hit_rate": _rate(sum(raw["matched"]), len(raw["matched"])),
        "hit_rate_halves": None if h[1] + h[3] == 0 else {"first": first, "second": second},
        "rule_reuse": _rate(raw["reused"], raw["reuse_total"]),
        "tests_per_level": _rate(raw["tests"], raw["level_count"]) if raw["tests"] else None,
        "impasses": raw["impasses"] if raw["wakes"] else None,
        "search_depth_mean": _mean(raw["depths"]),
        "goals": {"final": dict(raw["goal_final"]), "transitions": dict(raw["goal_transitions"])}
        if raw["goal_events"]
        else None,
        "actions_vs_human": raw["vs_human"] or None,
        "modes": dict(sorted(raw["modes"].items())) or None,
    }


def _sorted(counts: Counter[str]) -> dict[str, int]:
    return dict(sorted(counts.items()))


def _v2_summary(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The ``v2`` block of a summary, from per-game or pooled raw tallies."""
    n, hit, forks = raw["v2_cal_n"], raw["v2_cal_hit"], raw["v2_forks"]
    return {
        "calibration": [
            {"bin": _bin_label(i), "n": n[i], "matched": hit[i], "rate": _rate(hit[i], n[i])}
            for i in range(CAL_BINS)
        ],
        "moves_vs_typical": list(raw["v2_typical"]),
        "gates": {
            "trips": sum(raw["v2_trips"].values()),
            "by_gate": _sorted(raw["v2_trips"]),
            "seconds_total": raw["v2_gate_seconds"],
            "tokens_total": raw["v2_gate_tokens"],
        },
        "handoffs": _sorted(raw["v2_handoffs"]),
        "planner": {
            "calls": raw["v2_calls"],
            "tokens": raw["v2_call_tokens"],
            "segments": raw["v2_segments"],
            "mean_segment_len": _mean(raw["v2_segment_lens"]),
            "skills": _sorted(raw["v2_skills"]),
            "questions": _sorted(raw["v2_questions"]),
        },
        "forks": {name: forks[name] for name in FORK_EVENTS.values()},
        "harm": {"events": raw["v2_harm_events"], "by_kind": _sorted(raw["v2_harm"])},
    }


def v2_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Planner-agent metrics of one game's records, in file order.

    Every key of the ``v2`` block is present whatever the records hold (module
    docstring, "Planner agent (v2) metrics"). ``compute`` reports the same block
    per game under ``"v2"`` and pools it under ``overall``.
    """
    tally = _V2Tally()
    for rec in records:
        tally.add(rec)
    return _v2_summary(tally.raw())


def compute(records: Iterable[Rec], human: dict[str, list[int]] | None = None) -> dict[str, Any]:
    """Per-game and pooled metrics for ``records`` (all games mixed).

    ``human`` defaults to ``data/public_human_actions.json``; pass ``{}`` to
    disable score estimates.
    """
    if human is None:
        human = load_human()
    games: dict[str, _Game] = {}
    for rec in records:
        if isinstance(rec, dict) and isinstance(rec.get("type"), str):  # pyright: ignore[reportUnnecessaryIsInstance]  # input is untrusted JSON
            games.setdefault(str(rec.get("game", UNKNOWN_GAME)), _Game()).add(rec)
    raws = {g: state.raw() for g, state in games.items()}
    for g, raw in raws.items():
        raw["vs_human"] = _vs_human(raw, human.get(g[:4]))
    scores = {g: _score(raw, human.get(g[:4])) for g, raw in raws.items()}
    pooled: dict[str, Any] = {
        "levels_completed": 0,
        "per_level_actions": [],
        "level_indices": [],
        "matched": [],
        "latencies": [],
        "drop_reasons": Counter(),
        "repeats": 0,
        "repeat_total": 0,
        "entropy_deltas": [],
        "outcomes": Counter(),
        "halves": [0, 0, 0, 0],
        "reused": 0,
        "reuse_total": 0,
        "tests": 0,
        "level_count": 0,
        "wakes": 0,
        "impasses": 0,
        "depths": [],
        "goal_events": 0,
        "goal_final": Counter(),
        "goal_transitions": Counter(),
        "modes": Counter(),
        "vs_human": [],
        **_V2Tally().raw(),  # a fresh tally is the zero state of every v2 key
    }
    for raw in raws.values():
        for key, value in raw.items():
            if key == "halves":
                pooled[key] = [a + b for a, b in zip(pooled[key], value, strict=True)]
            elif isinstance(pooled[key], list):
                pooled[key].extend(value)
            else:
                pooled[key] += value
    known = [s for s in scores.values() if s is not None]
    return {
        "games": {g: _summary(raws[g], scores[g]) for g in sorted(raws)},
        "overall": _summary(pooled, _mean(known)),
    }


def _fmt(x: object) -> str:
    return "-" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)


def format_table(result: dict[str, Any]) -> str:
    """A compact fixed-width table, one row per game plus the overall row."""
    cols = [
        ("levels", "levels_completed"),
        ("score", "score_estimate"),
        ("pred_acc", "prediction_accuracy"),
        ("latency", "abandonment_latency"),
        ("repeat", "disconfirmed_repeat_rate"),
        ("dH/act", "entropy_change_per_action"),
        ("hit", "hit_rate"),
        ("reuse", "rule_reuse"),
        ("tests/lv", "tests_per_level"),
        ("impasse", "impasses"),
        ("depth", "search_depth_mean"),
    ]
    rows = [["game", *[c[0] for c in cols]]]
    for name, m in [*result["games"].items(), ("OVERALL", result["overall"])]:
        rows.append([name, *[_fmt(m[c[1]]) for c in cols]])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in rows)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Metrics over mind run logs.")
    parser.add_argument("paths", nargs="+", type=Path, help="log files or directories")
    parser.add_argument("--json", type=Path, help="write the full result here")
    args = parser.parse_args(argv)
    records, malformed = load(args.paths)
    result = compute(records)
    result["malformed_lines"] = malformed
    print(format_table(result))
    if malformed:
        print(f"skipped {malformed} malformed line(s)", file=sys.stderr)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
