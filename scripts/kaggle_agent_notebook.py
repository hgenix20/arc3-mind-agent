"""Build the Kaggle submission notebook for the playing agent.

Intent: one self-contained .ipynb plus kernel-metadata.json under build/kaggle-agent/,
ready for ``kaggle kernels push``. The notebook installs ``arc-agi`` from the
competition's wheel folder, writes the ``arcagent`` sources into /kaggle/working and
runs ``arcagent.agent.submit --mode auto``: offline on the 25 public games during an
interactive commit, through the Kaggle gateway during the competition rerun
(``KAGGLE_IS_COMPETITION_RERUN``). Both write ``/kaggle/working/submission.parquet``.

Constraints: no internet, CPU only (the probe baseline needs no GPU, so it does not
touch the RTX pool), image pinned to Python 3.12 to match the cp312 wheels.

Usage: python scripts/kaggle_agent_notebook.py [--slug SLUG] [--policy probe|explore] [--max-actions 300]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "kaggle-agent"
USERNAME = "hgenix20"
COMPETITION = "arc-prize-2026-arc-agi-3"
COMP_DIR = f"/kaggle/input/competitions/{COMPETITION}"
DOCKER_IMAGE = (
    "gcr.io/kaggle-private-byod/python@sha256:"
    "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c"
)

# The agent needs the core contracts, the probe policy (synth/trace.py) and the agent
# package; the synthesis harness itself stays out of the submission.
SOURCE_FILES = [
    ROOT / "arcagent" / "__init__.py",
    ROOT / "arcagent" / "core" / "__init__.py",
    ROOT / "arcagent" / "core" / "types.py",
    ROOT / "arcagent" / "core" / "frames.py",
    ROOT / "arcagent" / "synth" / "__init__.py",
    ROOT / "arcagent" / "synth" / "trace.py",
    *sorted((ROOT / "arcagent" / "agent").glob("*.py")),
]

INSTALL_CELL = r"""
import os, subprocess, sys, time
t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links",
    %(wheels)r, "arc-agi", "--no-warn-conflicts", "--disable-pip-version-check"], check=True)
import arc_agi, arcengine, pyarrow
print(f"arc_agi {arc_agi.__file__} installed in {time.time() - t0:.0f}s; pyarrow {pyarrow.__version__}")
print("competition rerun:", os.environ.get("KAGGLE_IS_COMPETITION_RERUN", ""))
"""

RUN_CELL = r"""
import json, os, subprocess, sys
from pathlib import Path
env = dict(os.environ, PYTHONPATH="/kaggle/working")
cmd = [sys.executable, "-m", "arcagent.agent.submit", "--mode", "auto", "--policy", %(policy)r,
       "--env-dir", %(env_dir)r, "--max-actions", %(max_actions)r,
       "--game-seconds", %(game_seconds)r, "--total-seconds", %(total_seconds)r,
       "--out", "/kaggle/working"]
print(" ".join(cmd))
r = subprocess.run(cmd, env=env, cwd="/kaggle/working")
print("exit", r.returncode)
sub = Path("/kaggle/working/submission.parquet")
if not sub.exists():
    raise RuntimeError("submission.parquet was not written")
doc = json.load(open("/kaggle/working/results.json"))
print("mode", doc["config"]["mode"], "games", len(doc["games"]), "errors", len(doc["errors"]),
      "scorecard score", (doc["scorecard"] or {}).get("score"))
for g in doc["games"]:
    print(g["game_id"], g["levels_completed"], "/", g["win_levels"], g["actions"], "actions", g["stop"])
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="arc3-agent-probe-baseline")
    ap.add_argument("--policy", choices=["probe", "explore"], default="probe")
    ap.add_argument("--max-actions", default="300")
    ap.add_argument("--game-seconds", default="240")
    ap.add_argument("--total-seconds", default=str(8 * 3600))
    a = ap.parse_args()

    sources = {f.relative_to(ROOT).as_posix(): f.read_text(encoding="utf-8") for f in SOURCE_FILES}
    cells: list[dict[str, object]] = [
        md_cell(
            "# ARC-AGI-3 agent: probe baseline\n\nPlays every game through the public action "
            "interface with the same probe policy the rule-synthesis harness uses to collect "
            "traces. Offline on the public games when committed, through the gateway in the "
            "competition rerun. Source: the `arcagent` package written below."
        ),
        code_cell(INSTALL_CELL % {"wheels": f"{COMP_DIR}/arc_agi_3_wheels"}),
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
                "policy": a.policy,
                "max_actions": a.max_actions,
                "game_seconds": a.game_seconds,
                "total_seconds": a.total_seconds,
            }
        ),
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
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {
        "id": f"{USERNAME}/{a.slug}",
        # Kaggle derives the kernel slug from the title; keep them in step with --slug.
        "title": "ARC3 agent probe baseline",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
        "model_sources": [],
        "docker_image_pinning_type": "original",
        "docker_image": DOCKER_IMAGE,
    }
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'notebook.ipynb'} ({len(cells)} cells, {len(sources)} source files)")


if __name__ == "__main__":
    main()
