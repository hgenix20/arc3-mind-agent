"""Build the Kaggle notebook that plays ARC-AGI-3 with the mind agent.

Intent: one self-contained .ipynb plus kernel-metadata.json under build/kaggle-mind/
(or ``--out``), ready for ``kaggle kernels push``. The notebook:

1. records ``NOTEBOOK_T0`` in its first code cell;
2. installs ``arc-agi`` from the competition's wheel folder (``INSTALL_CELL`` from
   ``kaggle_agent_notebook.py``);
3. installs the vLLM wheelhouse and serves Qwen3.6-27B-FP8 on 127.0.0.1:1234,
   waiting for ``/v1/models`` (``SERVE_CELL`` from ``kaggle_synth_notebook.py``, run
   inside a ``try`` that sets ``MODEL_OK``);
4. writes every ``.py`` under ``arcagent/`` except ``core/oracle.py``,
   ``data/public_human_actions.json`` and
   ``scripts/mind_metrics.py`` into /kaggle/working at their repo-relative paths
   (the sandbox loads ``arcagent/mind/_worker.py`` and ``arcagent/plan/planner.py``
   by file path, so the layout has to match the repo);
5. runs ``python -m arcagent.agent.submit --mode auto --policy mind``, which plays
   the public games offline in an interactive commit (only ``--games`` if given)
   and every private game through the gateway in the competition rerun, writing
   ``/kaggle/working/submission.parquet``, logs under ``/kaggle/working/logs`` and
   model replies under ``/kaggle/working/replies``; its combined stdout and stderr
   stream into the cell unbuffered and are kept in ``/kaggle/working/submit.log``;
6. runs ``scripts/mind_metrics.py`` on the logs into ``/kaggle/working/metrics.json``
   (interactive commit only; skipped when ``KAGGLE_IS_COMPETITION_RERUN`` is set).

Wall clock: Kaggle stops a notebook at 9 h. Setup (wheel installs, model load) took
several minutes in the probe kernel and the serve cell allows up to an hour, so the
agent's budget is computed in the notebook at run time:
``--total-seconds = max(1800, 8.5 * 3600 - (time.time() - NOTEBOOK_T0))`` in the
competition rerun. The interactive commit caps that at ``RERUN_SECONDS * n_games /
PRIVATE_GAMES`` (or ``--interactive-seconds``), floor 1800 s. submit
measures that from its own start and keeps its scorecard reserve inside it; only
the scheduler's hang grace (120 s) runs past it in the rerun (plus the metrics
cell in the interactive commit), which stays under 9 h.

Failure modes: a model server that exits or never answers sets ``MODEL_OK = False``
(the serve step runs inside a ``try``) and the run cell passes ``--config EXPLORER``,
so the games are still played without the model. submit writes the parquet in a
``finally``; if its process leaves none anyway, the run cell writes a zero-score
fallback (one row per game from the gateway's list in the rerun or the public
metadata offline, else one placeholder row). The run cell raises only when even
that fails. Summary printing and the metrics cell report errors without failing the
notebook, so a reporting bug can't cost a competition rerun its submission.

Constraints: internet off, so every dependency comes from an attached dataset or
the competition data. The image is pinned to Python 3.12 to match the cp312 wheels.

Usage: python scripts/kaggle_mind_notebook.py [--slug arc3-mind-agent]
       [--games all|<ids>...] [--config FULL] [--workers 8] [--out DIR]
       [--interactive-seconds S]
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
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

from arcagent.agent.submit import (  # noqa: E402  (needs the path above)
    GATEWAY_KEY,
    GATEWAY_URL,
    PARQUET_COLUMNS,
)
from arcagent.mind.config import CONFIGS  # noqa: E402  (needs the path above)

OUT = ROOT / "build" / "kaggle-mind"
USERNAME = "hgenix20"
COMPETITION = "arc-prize-2026-arc-agi-3"
COMP_DIR = f"/kaggle/input/competitions/{COMPETITION}"
ENDPOINT = "http://127.0.0.1:1234/v1"

# The interactive commit plays the 25 public games offline. Giving them the whole 8.5 h
# burns the weekly GPU quota, so its budget is scaled to the same per-game time as the
# private run: RERUN_SECONDS for PRIVATE_GAMES games.
RERUN_SECONDS = 8 * 3600
PRIVATE_GAMES = 110
PUBLIC_GAMES = 25


# The oracle wraps the real engine and reads game source. Only the probe experiment's
# trace collection imports it (lazily); the agent never does, so it is left out.
NOT_SHIPPED = frozenset({ROOT / "arcagent" / "core" / "oracle.py"})


def source_files() -> list[Path]:
    """Every file the notebook writes, in a stable order."""
    py = sorted(
        p
        for p in (ROOT / "arcagent").rglob("*.py")
        if "__pycache__" not in p.parts and p not in NOT_SHIPPED
    )
    return [*py, ROOT / "data" / "public_human_actions.json", ROOT / "scripts" / "mind_metrics.py"]


T0_CELL = """
import time
NOTEBOOK_T0 = time.time()  # the agent's budget is what is left of 8.5 h after setup
"""

# The shared serve cell raises when vLLM exits or never answers. Here that must not stop
# the notebook: the run cell then plays the model-free EXPLORER config and still writes a
# submission. The serve cell itself is unchanged, so the probe kernel keeps raising.
SERVE_WRAPPER = """
MODEL_OK = False
try:
%(serve)s
    MODEL_OK = True
except Exception as exc:  # the serve cell printed vLLM's root cause before raising
    print("model server failed, the agent plays EXPLORER:", repr(exc))
"""


def serve_cell() -> str:
    """The shared serve cell inside a ``try`` that sets ``MODEL_OK`` instead of raising."""
    serve = SERVE_CELL % {"wheelhouse": WHEELHOUSE, "model": MODEL_SNAPSHOT, "served": SERVED_NAME}
    return SERVE_WRAPPER % {"serve": textwrap.indent(serve.strip("\n"), "    ")}


# Runs after the agent's process exits when it left no submission.parquet (it could not
# start, or died before its own finally). Zero scores, but the rerun has a file to score.
FALLBACK_DEFS = r'''
import json, os, urllib.request
from pathlib import Path

PARQUET_COLUMNS = %(columns)r

def fallback_game_ids(env_dir, rerun, wanted):
    """Game ids for a fallback submission, or [] when no list is readable.

    The rerun asks the gateway's /api/games; offline reads the public metadata under
    env_dir, keeping only the ``wanted`` 4-char prefixes when any are given.
    """
    ids = []
    try:
        if rerun:
            base = os.environ.get("ARC_BASE_URL", %(gateway)r).rstrip("/")
            key = os.environ.get("ARC_API_KEY", %(key)r)
            req = urllib.request.Request(base + "/api/games",
                                         headers={"X-Api-Key": key, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                ids = [str(g["game_id"]) for g in json.loads(r.read()) if g.get("game_id")]
        else:
            for meta in sorted(Path(env_dir).glob("*/*/metadata.json")):
                ids.append(str(json.loads(meta.read_text(encoding="utf-8"))["game_id"]))
    except Exception as exc:
        print("fallback: no game list:", repr(exc))
        ids = []
    if wanted:
        ids = [g for g in ids if g.split("-", 1)[0] in set(wanted)]
    return sorted(set(ids))

def write_fallback_parquet(path, game_ids):
    """One zero-score row per game, or one placeholder row; submit's schema and columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    ids = list(game_ids) or ["placeholder"]
    rows = [{"row_id": f"{i}_0", "game_id": g, "end_of_game": True, "score": 0.0}
            for i, g in enumerate(ids)]
    schema = pa.schema([("row_id", pa.string()), ("game_id", pa.string()),
                        ("end_of_game", pa.bool_()), ("score", pa.float64())])
    if tuple(schema.names) != PARQUET_COLUMNS:
        raise ValueError(f"fallback columns {schema.names} differ from {PARQUET_COLUMNS}")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), str(path))
'''


def fallback_defs() -> str:
    """The fallback functions with submit's columns and gateway defaults filled in."""
    return FALLBACK_DEFS % {"columns": PARQUET_COLUMNS, "gateway": GATEWAY_URL, "key": GATEWAY_KEY}


RUN_CELL = r"""
import json, os, subprocess, sys, time
from pathlib import Path
SUBMISSION = Path("/kaggle/working/submission.parquet")
ENV_DIR = %(env_dir)r
GAMES = %(games)r
left = 8.5 * 3600 - (time.time() - NOTEBOOK_T0)
# Same test as submit's --mode auto.
rerun = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
RERUN_SECONDS = %(rerun_seconds)d
PRIVATE_GAMES = %(private_games)d
N_GAMES = %(n_games)d
if rerun:
    total = max(1800, left)
else:
    %(scaled)s
    total = max(1800, min(left, scaled))
env = dict(os.environ, PYTHONPATH="/kaggle/working", PYTHONUNBUFFERED="1")
config = %(config)r if MODEL_OK else "EXPLORER"
if config != %(config)r:
    print("model server not ready: playing", config)
cmd = [sys.executable, "-m", "arcagent.agent.submit", "--mode", "auto", "--policy", "mind",
       "--env-dir", ENV_DIR, "--config", config, "--workers", %(workers)r,
       "--endpoint", "http://127.0.0.1:1234/v1", "--model", %(served)r,
       "--total-seconds", str(int(total)), "--out", "/kaggle/working",
       "--log-dir", "/kaggle/working/logs", "--replies-dir", "/kaggle/working/replies"]
cmd += %(games_args)r  # empty means every game; a game list applies offline only
print(f"setup took {time.time() - NOTEBOOK_T0:.0f}s; rerun {rerun}; agent budget {total:.0f}s")
print(" ".join(cmd))
# Stream the child's output into the cell and keep a copy, so a long run shows progress
# and its log survives in the kernel output.
returncode = None
try:
    with open("/kaggle/working/submit.log", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, env=env, cwd="/kaggle/working", stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = proc.wait()
except Exception as exc:  # a launch failure still gets the fallback below
    print("submit run failed:", repr(exc))
print("exit", returncode)
if not SUBMISSION.exists():
    print("submission.parquet missing: writing the zero-score fallback")
    try:
        ids = fallback_game_ids(ENV_DIR, rerun, [] if rerun else GAMES)
        write_fallback_parquet(SUBMISSION, ids)
    except Exception as exc:
        print("fallback parquet failed:", repr(exc))
if not SUBMISSION.exists():
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

# Metrics are for the interactive commit; the competition rerun's time is the agent's.
METRICS_CELL = r"""
import os, subprocess, sys
if os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}:
    print("competition rerun: metrics skipped")
else:
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
    ap.add_argument(
        "--interactive-seconds",
        type=int,
        default=None,
        help="interactive-commit budget in seconds, instead of the scaled per-game value",
    )
    a = ap.parse_args(argv)
    if a.workers < 1:
        ap.error("need --workers >= 1")
    if a.interactive_seconds is not None and a.interactive_seconds < 1:
        ap.error("need --interactive-seconds >= 1")
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
        code_cell(serve_cell()),
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
            fallback_defs()
            + RUN_CELL
            % {
                "env_dir": f"{COMP_DIR}/environment_files",
                "config": a.config,
                "workers": str(a.workers),
                "served": SERVED_NAME,
                "rerun_seconds": RERUN_SECONDS,
                "private_games": PRIVATE_GAMES,
                "n_games": len(games) if games else PUBLIC_GAMES,
                "scaled": (
                    "scaled = RERUN_SECONDS * N_GAMES / PRIVATE_GAMES"
                    if a.interactive_seconds is None
                    else f"scaled = {a.interactive_seconds}"
                ),
                "games_args": ["--games", *games] if games else [],
                "games": games,
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
