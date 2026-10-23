"""Guard: nothing from the Franzen period ships in the entry."""

import json
import re
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = [
    ROOT / "arcagent",
    ROOT / "scripts" / "kaggle_mind_notebook.py",
    ROOT / "scripts" / "kaggle_synth_notebook.py",
    ROOT / "data",
]
SUFFIXES = {".py", ".json", ".md"}


def iter_lines(paths: list[Path]) -> Iterator[tuple[Path, int, str]]:
    """Yield (file, line number, line) for every shipped text file that exists."""
    for path in paths:
        if not path.exists():
            continue
        files = [path] if path.is_file() else sorted(path.rglob("*"))
        for f in files:
            if f.is_file() and f.suffix in SUFFIXES and "__pycache__" not in f.parts:
                for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                    yield f, n, line


def test_shipped_tree_has_no_franzen_period_names() -> None:
    hits = [
        f"{p}:{n}"
        for p, n, line in iter_lines(SHIPPED)
        if re.search(r"duck|franzen", line, re.IGNORECASE)
    ]
    assert hits == []


def test_human_actions_cover_the_public_games() -> None:
    data = json.loads((ROOT / "data" / "public_human_actions.json").read_text())
    assert len(data) == 25
    assert all(len(v) >= 1 and min(v) > 0 for v in data.values())
