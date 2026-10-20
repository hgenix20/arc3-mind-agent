"""Build the Kaggle notebook that runs the rule-synthesis experiment offline.

Intent: one self-contained .ipynb plus kernel-metadata.json under build/kaggle-synth/,
ready for ``kaggle kernels push``. The notebook serves Qwen3.6-27B-FP8 with vLLM
from two public Kaggle datasets (the wheelhouse and the model snapshot Tufa Labs'
Milestone 1 notebook used on the same RTX PRO 6000 machine), writes our
``arcagent`` sources into /kaggle/working, and runs ``arcagent.synth.run``
against the public games shipped with the competition data.

Constraints: no internet in the notebook, so every dependency comes from an
attached dataset or the competition's wheel folder. The notebook never reads
game source files; the harness only drives games through the action interface.

Usage: python scripts/kaggle_synth_notebook.py [--games ls20 ft09 ...] [--steps 120]
       [--rounds 3] [--max-transitions 40] [--slug arc3-rule-synthesis-probe]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "kaggle-synth"
USERNAME = "hgenix20"
WHEELHOUSE = "driessmit1/arc3-vllm-h100-wheelhouse-v3"
MODEL_SNAPSHOT = "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"
SERVED_NAME = "vrfai/Qwen3.6-27B-FP8"
COMPETITION = "arc-prize-2026-arc-agi-3"
COMP_DIR = f"/kaggle/input/competitions/{COMPETITION}"

SOURCE_FILES = [
    *sorted((ROOT / "arcagent" / "core").glob("*.py")),
    *sorted((ROOT / "arcagent" / "synth").glob("*.py")),
    ROOT / "arcagent" / "__init__.py",
]

SERVE_CELL = r"""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

WORK = Path("/kaggle/working")
SITE = Path("/tmp/vllm-site")  # outside /kaggle/working so the 5 GB wheel tree is not saved as output
LOG = WORK / "vllm.log"
# The CUDA driver library is off the default linker and loader paths on Kaggle GPU images.
NVLIB = "/usr/local/nvidia/lib64"
for var in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
    os.environ[var] = os.pathsep.join(p for p in [NVLIB, os.environ.get(var, "")] if p)

def root_cause(text: str) -> str:
    lines = text.splitlines()
    hits = [i for i, ln in enumerate(lines) if "EngineCore" in ln and ("Error" in ln or "error" in ln or "Exception" in ln)]
    hits += [i for i, ln in enumerate(lines) if ln.strip().startswith(("ERROR", "RuntimeError", "ValueError", "ImportError", "ModuleNotFoundError", "OSError", "torch.OutOfMemoryError"))]
    out = []
    for i in sorted(set(hits))[:12]:
        out.extend(lines[max(0, i - 3): i + 2])
    return "\n".join(out) if out else text[-4000:]

def find_input(slug: str) -> Path:
    owner, name = slug.split("/")
    for p in (Path("/kaggle/input") / name, Path("/kaggle/input/datasets") / owner / name):
        if p.exists():
            return p
    raise FileNotFoundError(slug)

WHEELHOUSE = find_input(%(wheelhouse)r)
MODEL = find_input(%(model)r)
print("gpu:", subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
      capture_output=True, text=True).stdout.strip())

t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(WHEELHOUSE),
    "--requirement", str(WHEELHOUSE / "requirements.lock"), "--target", str(SITE), "--upgrade",
    "--ignore-installed", "--only-binary", ":all:", "--no-compile", "--disable-pip-version-check",
    "--no-warn-conflicts", "-q"], check=True)
print(f"vllm wheelhouse installed in {time.time() - t0:.0f}s")

env = dict(os.environ, PYTHONPATH=str(SITE), VLLM_NO_USAGE_STATS="1", USE_TF="0", TRANSFORMERS_NO_TF="1",
           TRANSFORMERS_NO_TORCHVISION="1")
cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(MODEL),
       "--served-model-name", %(served)r, "--host", "127.0.0.1", "--port", "1234",
       "--max-model-len", "65536", "--enable-prefix-caching", "--reasoning-parser", "qwen3",
       "--generation-config", "vllm", "--gpu-memory-utilization", "0.92"]
server = subprocess.Popen(cmd, env=env, stdout=LOG.open("w"), stderr=subprocess.STDOUT)
# Weight loading from the Kaggle input disk varied 2.4x between runs (v4 ready in 535 s,
# v5 killed at 1500 s with warmup finished and the server seconds from ready).
deadline = time.time() + 3600
while time.time() < deadline:
    if server.poll() is not None:
        print(root_cause(LOG.read_text())); raise RuntimeError("vllm exited")
    try:
        with urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=5) as r:
            print("server ready after", f"{time.time() - t0:.0f}s:", json.loads(r.read())["data"][0]["id"]); break
    except Exception:
        time.sleep(5)
else:
    print(root_cause(LOG.read_text())); raise TimeoutError("vllm did not start")
"""

ARC_CELL = r"""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links",
    %(wheels)r, "arc_agi", "arcengine", "--no-deps"], check=True)
import arc_agi, arcengine
print("arc_agi ok")
"""

RUN_CELL = r"""
import os, subprocess, sys, json
env = dict(os.environ, PYTHONPATH="/kaggle/working")
cmd = [sys.executable, "-m", "arcagent.synth.run", "--env-dir", %(env_dir)r, "--games", *%(games)r,
       "--steps", %(steps)r, "--rounds", %(rounds)r, "--max-transitions", %(max_transitions)r,
       "--endpoint", "http://127.0.0.1:1234/v1", "--model", %(served)r,
       "--out", "/kaggle/working/results.json"]
print(" ".join(cmd))
r = subprocess.run(cmd, env=env, cwd="/kaggle/working")
print("exit", r.returncode)
if os.path.exists("/kaggle/working/results.json"):
    doc = json.load(open("/kaggle/working/results.json"))
    for game, g in doc["games"].items():
        f = g.get("final") or {}
        print(game, "final:", {k: f.get(k) for k in ("round", "exact_rate_heldout", "cell_accuracy",
              "no_code", "finish_reason", "reasoning_chars", "reply_chars", "completion_tokens")})
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
    ap.add_argument("--games", nargs="+", default=["ls20", "ft09", "vc33"])
    ap.add_argument("--steps", default="120")
    ap.add_argument("--rounds", default="3")
    ap.add_argument("--max-transitions", default="40")
    ap.add_argument("--slug", default="arc3-rule-synthesis-probe")
    a = ap.parse_args()

    cells: list[dict[str, object]] = [
        md_cell(
            "# ARC-AGI-3 rule-synthesis probe\n\nCan an open-weight 27B model write an executable rule model "
            "of a public game from observed transitions alone? Private experiment, no submission."
        ),
        code_cell(
            SERVE_CELL % {"wheelhouse": WHEELHOUSE, "model": MODEL_SNAPSHOT, "served": SERVED_NAME}
        ),
        code_cell(ARC_CELL % {"wheels": f"{COMP_DIR}/arc_agi_3_wheels"}),
    ]
    # One cell writes every source file; paths are POSIX strings built here, so no
    # host path separator can leak into the notebook.
    sources = {f.relative_to(ROOT).as_posix(): f.read_text(encoding="utf-8") for f in SOURCE_FILES}
    cells.append(
        code_cell(
            "from pathlib import Path\n"
            f"FILES = {json.dumps(sources)}\n"
            "for rel, body in FILES.items():\n"
            "    p = Path('/kaggle/working') / rel\n"
            "    p.parent.mkdir(parents=True, exist_ok=True)\n"
            "    p.write_text(body, encoding='utf-8')\n"
            "print('wrote', len(FILES), 'source files')\n"
        )
    )
    cells.append(
        code_cell(
            RUN_CELL
            % {
                "env_dir": f"{COMP_DIR}/environment_files",
                "games": a.games,
                "steps": a.steps,
                "rounds": a.rounds,
                "max_transitions": a.max_transitions,
                "served": SERVED_NAME,
            }
        )
    )

    fixed = cells
    nb = {
        "cells": fixed,
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
        "title": "ARC3 rule synthesis probe",
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
        # The wheelhouse holds cp312 wheels; Kaggle's current default image runs Python 3.13.
        # Pin the image the Tufa and Franzen notebooks ran on (Python 3.12.13).
        "docker_image_pinning_type": "original",
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c",
    }
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"wrote {OUT / 'notebook.ipynb'} ({len(fixed)} cells, {len(SOURCE_FILES)} source files) and kernel-metadata.json"
    )


if __name__ == "__main__":
    main()
