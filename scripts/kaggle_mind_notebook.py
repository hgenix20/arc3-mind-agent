"""Build the Kaggle notebook that plays ARC-AGI-3 with the mind agent.

Intent: one self-contained .ipynb plus kernel-metadata.json under build/kaggle-mind/
(or ``--out``), ready for ``kaggle kernels push``. The notebook:

1. records ``NOTEBOOK_T0`` in its first code cell;
2. installs ``arc-agi`` from the competition's wheel folder (``INSTALL_CELL`` from
   ``kaggle_agent_notebook.py``);
3. installs the vLLM wheelhouse and serves Qwen3.6-27B-FP8 on 127.0.0.1:1234,
   waiting for ``/v1/models`` (``SERVE_CELL`` from ``kaggle_synth_notebook.py``);
4. writes every ``.py`` under ``arcagent/``, ``data/public_human_actions.json`` and
   ``scripts/mind_metrics.py`` into /kaggle/working at their repo-relative paths
   (the sandbox loads ``arcagent/mind/_worker.py`` and ``arcagent/plan/planner.py``
   by file path, so the layout has to match the repo);
5. runs ``python -m arcagent.agent.submit --mode auto --policy mind``, which plays
   the public games offline in an interactive commit (only ``--games`` if given)
   and every private game through the gateway in the competition rerun, writing
   ``/kaggle/working/submission.parquet``, logs under ``/kaggle/working/logs`` and
   model replies under ``/kaggle/working/replies``;
6. runs ``scripts/mind_metrics.py`` on the logs into ``/kaggle/working/metrics.json``.

Wall clock: Kaggle stops a notebook at 9 h. Setup (wheel installs, model load) took
several minutes in the probe kernel and the serve cell allows up to an hour, so the
agent's budget is computed in the notebook at run time:
``--total-seconds = max(1800, 8.5 * 3600 - (time.time() - NOTEBOOK_T0))``. submit
measures that from its own start and keeps its scorecard reserve inside it; only
the scheduler's hang grace (120 s) and the metrics cell run past it, which stays
under 9 h.

Failure modes: the run cell raises only when ``submission.parquet`` is missing;
summary printing and the metrics cell report errors without failing the notebook,
so a reporting bug can't cost a competition rerun its submission.

Constraints: internet off, so every dependency comes from an attached dataset or
the competition data. The image is pinned to Python 3.12 to match the cp312 wheels.

Usage: python scripts/kaggle_mind_notebook.py [--slug arc3-mind-agent]
       [--games all|<ids>...] [--config FULL] [--workers 8] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_agent_notebook import DOCKER_IMAGE, INSTALL_CELL  # noqa: E402  (needs the path above)
from kaggle_synth_notebook import (  # noqa: E402  (needs the path above)
    MODEL_SNAPSHOT,
    SERVE_CELL,
    SERVED_NAME,
    WHEELHOUSE,
)

from arcagent.mind.config import CONFIGS  # noqa: E402  (needs the path above)

OUT = ROOT / "build" / "kaggle-mind"
USERNAME = "hgenix20"
COMPETITION = "arc-prize-2026-arc-agi-3"
COMP_DIR = f"/kaggle/input/competitions/{COMPETITION}"
ENDPOINT = "http://127.0.0.1:1234/v1"


def source_files() -> list[Path]:
    """Every file the notebook writes, in a stable order."""
    py = sorted(p for p in (ROOT / "arcagent").rglob("*.py") if "__pycache__" not in p.parts)
    return [*py, ROOT / "data" / "public_human_actions.json", ROOT / "scripts" / "mind_metrics.py"]


T0_CELL = """
import time
NOTEBOOK_T0 = time.time()  # the agent's budget is what is left of 8.5 h after setup
"""

RUN_CELL = r"""
import json, os, subprocess, sys, time
from pathlib import Path
total = max(1800, 8.5 * 3600 - (time.time() - NOTEBOOK_T0))
env = dict(os.environ, PYTHONPATH="/kaggle/working")
cmd = [sys.executable, "-m", "arcagent.agent.submit", "--mode", "auto", "--policy", "mind",
       "--env-dir", %(env_dir)r, "--config", %(config)r, "--workers", %(workers)r,
       "--endpoint", "http://127.0.0.1:1234/v1", "--model", %(served)r,
       "--total-seconds", str(int(total)), "--out", "/kaggle/working",
       "--log-dir", "/kaggle/working/logs", "--replies-dir", "/kaggle/working/replies"]
cmd += %(games_args)r  # empty means every game; a game list applies offline only
print(f"setup took {time.time() - NOTEBOOK_T0:.0f}s; agent budget {total:.0f}s")
print(" ".join(cmd))
r = subprocess.run(cmd, env=env, cwd="/kaggle/working")
print("exit", r.returncode)
if not Path("/kaggle/working/submission.parquet").exists():
    raise RuntimeError("submission.parquet was not written")
try:
    doc = json.load(open("/kaggle/working/results.json"))
    print("mode", doc["config"]["mode"], "games", len(doc["games"]), "errors", len(doc["errors"]),
          "scorecard score", (doc["scorecard"] or {}).get("score"))
    for g in doc["games"]:
        print(g["game_id"], g["levels_completed"], "/", g["win_levels"], g["actions"], "actions",
              g["stop"])
except Exception as e:  # reporting only; the submission is already written
    print("results summary failed:", repr(e))
"""

METRICS_CELL = r"""
import json, subprocess, sys
r = subprocess.run([sys.executable, "scripts/mind_metrics.py", "/kaggle/working/logs",
                    "--json", "/kaggle/working/metrics.json"], cwd="/kaggle/working")
print("metrics exit", r.returncode)
"""


def code_cell(src: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src,
    }


def md_cell(src: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slug", default="arc3-mind-agent")
    ap.add_argument(
        "--games", nargs="+", default=["all"], help="'all' or 4-char ids (interactive commit only)"
    )
    ap.add_argument("--config", choices=sorted(CONFIGS), default="FULL")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    if a.workers < 1:
        ap.error("need --workers >= 1")
    if "all" in a.games and len(a.games) > 1:
        ap.error("--games takes 'all' alone or a list of ids")
    return a


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    games: list[str] = [] if a.games == ["all"] else list(a.games)
    files = source_files()
    # POSIX keys built here, so no host path separator leaks into the notebook.
    sources = {f.relative_to(ROOT).as_posix(): f.read_text(encoding="utf-8") for f in files}
    cells: list[dict[str, object]] = [
        md_cell(
            "# ARC-AGI-3: the mind agent\n\nPlays every game through the public action interface "
            "with the `arcagent` mind policy, backed by Qwen3.6-27B-FP8 served locally with vLLM "
            "(thinking off). Offline on the public games when committed, through the gateway in "
            "the competition rerun. Source: the files written below."
        ),
        code_cell(T0_CELL),
        code_cell(INSTALL_CELL % {"wheels": f"{COMP_DIR}/arc_agi_3_wheels"}),
        code_cell(
            SERVE_CELL % {"wheelhouse": WHEELHOUSE, "model": MODEL_SNAPSHOT, "served": SERVED_NAME}
        ),
        code_cell(
            "from pathlib import Path\n"
            f"FILES = {json.dumps(sources)}\n"
            "for rel, body in FILES.items():\n"
            "    p = Path('/kaggle/working') / rel\n"
            "    p.parent.mkdir(parents=True, exist_ok=True)\n"
            "    p.write_text(body, encoding='utf-8')\n"
            "print('wrote', len(FILES), 'source files')\n"
        ),
        code_cell(
            RUN_CELL
            % {
                "env_dir": f"{COMP_DIR}/environment_files",
                "config": a.config,
                "workers": str(a.workers),
                "served": SERVED_NAME,
                "games_args": ["--games", *games] if games else [],
            }
        ),
        code_cell(METRICS_CELL),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out: Path = a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {
        "id": f"{USERNAME}/{a.slug}",
        # Kaggle derives the kernel slug from the title; this title slugifies back to --slug.
        "title": a.slug.replace("-", " "),
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [WHEELHOUSE, MODEL_SNAPSHOT],
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaRtxPro6000",
        "docker_image_pinning_type": "original",
        "docker_image": DOCKER_IMAGE,
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out / 'notebook.ipynb'} ({len(cells)} cells, {len(sources)} source files)")


if __name__ == "__main__":
    main()
