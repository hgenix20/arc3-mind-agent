"""Noise band of the Duck 25-game evaluation from two runs of the same configuration.

Reads two ``summary.txt`` files written by the harness (lines of the form
``  ar25-0c556536: score=58.33, ...``) and prints the paired statistics used
in ``docs/experiment-log.md`` ("Noise band"): per-game differences, the
per-game standard deviation of a single run (rms difference / sqrt 2), the
standard error of the 25-game mean, and the 95% band for a difference
between two single runs. Any further summary files are printed alongside
for comparison but do not enter the noise estimate.

Usage: python scripts/duck_noise.py RUN_A/summary.txt RUN_B/summary.txt [OTHER/summary.txt ...]
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

LINE = re.compile(r"\s+(\w{4})-\w+: score=([\d.]+)")


def load(path: Path) -> dict[str, float]:
    """Per-game mean score keyed by the 4-character game id; raises if the file has none."""
    scores: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LINE.match(line)
        if m:
            scores[m.group(1)] = float(m.group(2))
    if not scores:
        raise ValueError(f"no per-game score lines in {path}")
    return scores


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("summaries", nargs="+", type=Path, help="two or more summary.txt files")
    a = ap.parse_args(argv)
    if len(a.summaries) < 2:
        ap.error("need at least two summary files")
    runs = [load(p) for p in a.summaries]
    games = sorted(set(runs[0]) & set(runs[1]))
    n = len(games)
    first, second = runs[0], runs[1]
    diffs = [second[g] - first[g] for g in games]
    rms = math.sqrt(sum(d * d for d in diffs) / n)
    per_game_sd = rms / math.sqrt(2)
    mean_se = per_game_sd / math.sqrt(n)
    means = ", ".join(f"{sum(r[g] for g in games) / n:.2f}" for r in runs)
    print(f"games {n}; means {means}")
    mean_abs = sum(map(abs, diffs)) / n
    band = 1.96 * mean_se * math.sqrt(2)
    print(f"rms per-game difference {rms:.1f}; mean absolute difference {mean_abs:.1f}")
    print(f"per-game sd of one run ~{per_game_sd:.1f}; se of the {n}-game mean ~{mean_se:.1f}")
    print(f"95% band for a difference between two single runs ~ +/-{band:.1f}")
    moved = [
        f"{g} {first[g]:.0f}->{second[g]:.0f}" for g in games if abs(second[g] - first[g]) > 30
    ]
    print("moved by more than 30: " + ", ".join(moved))
    print("won in both: " + ", ".join(g for g in games if first[g] >= 90 and second[g] >= 90))
    for g in games:
        print(g + " " + " ".join(f"{r.get(g, float('nan')):6.1f}" for r in runs))


if __name__ == "__main__":
    main()
