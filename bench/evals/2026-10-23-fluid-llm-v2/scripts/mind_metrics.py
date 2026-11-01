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

``overall`` pools the raw counts of all games (not a mean of per-game means),
except ``score_estimate``, the mean over games that have one, and
``levels_completed``, the sum.

Failure modes: ``load`` skips lines that are not JSON objects and returns their
count; ``compute`` ignores records that are not dicts or lack a string ``type``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
HUMAN_FILE = ROOT / "data" / "public_human_actions.json"
RESET_ID = 0
LATENCY_REASONS = frozenset({"low_fit", "misses_in_row"})
UNKNOWN_GAME = "?"
BAD_OUTCOMES = ("noop", "game_over", "disconfirmed")

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


def _is_num(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _is_reset(rec: Rec) -> bool:
    action = rec.get("action")
    return isinstance(action, list) and bool(action) and action[0] == RESET_ID


def _disconfirmed(rec: Rec, nxt: Rec) -> bool:
    active = rec.get("active")
    if active is None:
        return False
    return any(
        isinstance(p, dict) and p.get("id") == active and p.get("matched") is False
        for p in nxt.get("programs") or ()
    )


class _Game:
    """Raw per-game tallies; ``raw`` reduces them to numbers ``_summary`` reports."""

    def __init__(self) -> None:
        self.actions: list[Rec] = []
        self.by_step: dict[int, int] = {}
        self.game_over_idx: set[int] = set()
        self.level_up_idx: list[int] = []
        self.drops: list[Rec] = []

    def add(self, rec: Rec) -> None:
        kind = rec["type"]
        if kind == "action":
            self.actions.append(rec)
            step = rec.get("step")
            if isinstance(step, int):
                self.by_step[step] = len(self.actions) - 1
        elif kind == "program_dropped":
            self.drops.append(rec)
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
    }


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
    }
    for raw in raws.values():
        for key, value in raw.items():
            if isinstance(pooled[key], list):
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
