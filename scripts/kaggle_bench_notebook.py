"""Build the Kaggle CPU notebook that runs the explorer bench (no GPU, no model).

Why: several policies times several seeds don't fit in the laptop's memory. A Kaggle
CPU notebook (4 cores, about 30 GB) plays the competition's offline game files the
same way the agent notebook does. The notebook installs ``arc-agi`` from the
competition wheels, writes the ``arcagent`` sources and ``scripts/bench_explore.py``
into /kaggle/working, and runs the bench there. Output: ``bench/bench.json``, every
``results.json`` and ``run.log``, and ``bench.log`` (the bench's own stdout).

    python scripts/kaggle_bench_notebook.py --policies explore guided --seeds 0 1 2 3 \\
        --max-actions 10000
    kaggle kernels push -p build/kaggle-bench

``--gpu`` builds a GPU notebook instead: it serves Qwen3.6-27B-FP8 with vLLM first (the
agent notebook's serve cell), so the ``advised`` policy can ask it; its replies and
questions land under ``bench/<policy>/seed<n>/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_agent_notebook import DOCKER_IMAGE, INSTALL_CELL  # noqa: E402  (needs the path above)
from kaggle_mind_notebook import (  # noqa: E402  (needs the path above)
    COMP_DIR,
    COMPETITION,
    USERNAME,
    code_cell,
    md_cell,
    serve_cell,
    source_files,
)
from kaggle_synth_notebook import MODEL_SNAPSHOT, WHEELHOUSE  # noqa: E402  (needs the path above)

OUT = ROOT / "build" / "kaggle-bench"

RUN_CELL = """
import subprocess, sys
cmd = [sys.executable, "scripts/bench_explore.py", "--env-dir", %(env_dir)r,
       "--out", "/kaggle/working/bench", *%(args)r]
with open("/kaggle/working/bench.log", "w") as log:
    proc = subprocess.Popen(cmd, cwd="/kaggle/working", stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
        log.write(line)
print("exit", proc.wait())
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--slug", default="arc3-bench")
    p.add_argument("--policies", nargs="+", default=["explore", "guided"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--max-actions", type=int, default=10000)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--gpu", action="store_true", help="serve the model first (advised policy)")
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    files = [*source_files(), ROOT / "scripts" / "bench_explore.py"]
    sources = {f.relative_to(ROOT).as_posix(): f.read_text(encoding="utf-8") for f in files}
    args = [
        "--policies", *a.policies, "--seeds", *map(str, a.seeds),
        "--max-actions", str(a.max_actions), "--jobs", str(a.jobs),
    ]  # fmt: skip
    cells = [
        md_cell("# ARC-AGI-3 explorer bench\n\nExplorer policies over seeds, no model."),
        code_cell(INSTALL_CELL % {"wheels": f"{COMP_DIR}/arc_agi_3_wheels"}),
        *([code_cell(serve_cell())] if a.gpu else []),
        code_cell(
            "from pathlib import Path\n"
            f"FILES = {json.dumps(sources)}\n"
            "for rel, body in FILES.items():\n"
            "    p = Path('/kaggle/working') / rel\n"
            "    p.parent.mkdir(parents=True, exist_ok=True)\n"
            "    p.write_text(body, encoding='utf-8')\n"
            "print('wrote', len(FILES), 'source files')\n"
        ),
        code_cell(RUN_CELL % {"env_dir": f"{COMP_DIR}/environment_files", "args": args}),
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
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {
        "id": f"{USERNAME}/{a.slug}",
        "title": a.slug.replace("-", " "),
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": a.gpu,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [WHEELHOUSE, MODEL_SNAPSHOT] if a.gpu else [],
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
        "model_sources": [],
        **({"machine_shape": "NvidiaRtxPro6000"} if a.gpu else {}),
        "docker_image_pinning_type": "original",
        "docker_image": DOCKER_IMAGE,
    }
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {a.out / 'notebook.ipynb'} ({len(sources)} source files)")


if __name__ == "__main__":
    main()
