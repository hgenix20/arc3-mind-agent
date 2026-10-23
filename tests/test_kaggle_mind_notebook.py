"""Kaggle notebook builder for the mind agent: structure, metadata and shipped text."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "kaggle_mind_notebook.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("kaggle_mind_notebook", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(tmp_path: Path, *extra: str) -> tuple[dict[str, Any], dict[str, Any]]:
    load_script().main(["--out", str(tmp_path), *extra])
    nb = json.loads((tmp_path / "notebook.ipynb").read_text(encoding="utf-8"))
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text(encoding="utf-8"))
    return nb, meta


def cell_sources(nb: dict[str, Any]) -> list[str]:
    return [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]


def written_files(nb: dict[str, Any]) -> dict[str, str]:
    """The FILES dict of the cell that writes the sources into /kaggle/working."""
    for src in cell_sources(nb):
        for line in src.splitlines():
            if line.startswith("FILES = "):
                return json.loads(line.removeprefix("FILES = "))
    raise AssertionError("no FILES cell")


def run_cell(nb: dict[str, Any]) -> str:
    hits = [s for s in cell_sources(nb) if "arcagent.agent.submit" in s and "FILES = " not in s]
    assert len(hits) == 1
    return hits[0]


def test_notebook_builds_valid_json_with_all_mind_sources(tmp_path: Path) -> None:
    nb, _ = build(tmp_path, "--games", "all")
    assert nb["nbformat"] == 4
    files = written_files(nb)
    expected = {
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "arcagent").rglob("*.py")
        if "__pycache__" not in p.parts
    }
    expected |= {"data/public_human_actions.json", "scripts/mind_metrics.py"}
    assert set(files) == expected
    # The sandbox loads these two by file path, so they must keep their layout.
    assert "arcagent/plan/planner.py" in files and "arcagent/mind/_worker.py" in files
    assert files["arcagent/mind/sandbox.py"] == (ROOT / "arcagent/mind/sandbox.py").read_text(
        encoding="utf-8"
    )
    for src in cell_sources(nb):
        compile(src, "<cell>", "exec")  # every cell is valid Python
    run = run_cell(nb)
    for token in ('"--policy", "mind"', '"--mode", "auto"', "http://127.0.0.1:1234/v1"):
        assert token in run
    assert "'vrfai/Qwen3.6-27B-FP8'" in run
    assert "NOTEBOOK_T0" in run and "8.5 * 3600" in run
    assert "'--games'" not in run and "cmd += []" in run  # all games: submit picks them
    assert "/kaggle/working/submission.parquet" in run
    sources = "\n".join(cell_sources(nb))
    assert "NOTEBOOK_T0 = time.time()" in cell_sources(nb)[0]
    assert "scripts/mind_metrics.py" in sources and "/kaggle/working/metrics.json" in sources
    assert "--max-model-len" in sources  # the shared serve cell is included


def test_notebook_passes_named_games_for_the_interactive_commit(tmp_path: Path) -> None:
    nb, _ = build(tmp_path, "--games", "ls20", "ft09", "--config", "EXPLORER", "--workers", "4")
    run = run_cell(nb)
    assert "cmd += ['--games', 'ls20', 'ft09']" in run
    assert "'EXPLORER'" in run and "'4'" in run


def test_metadata_has_gpu_no_internet_and_datasets(tmp_path: Path) -> None:
    _, meta = build(tmp_path, "--slug", "arc3-mind-test")
    assert meta["id"] == "hgenix20/arc3-mind-test"
    assert meta["is_private"] is True
    assert meta["enable_gpu"] is True and meta["enable_internet"] is False
    assert meta["machine_shape"] == "NvidiaRtxPro6000"
    assert meta["competition_sources"] == ["arc-prize-2026-arc-agi-3"]
    assert meta["dataset_sources"] == [
        "driessmit1/arc3-vllm-h100-wheelhouse-v3",
        "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
    ]
    assert meta["code_file"] == "notebook.ipynb"
    assert meta["docker_image_pinning_type"] == "original"
    assert meta["docker_image"].startswith("gcr.io/kaggle-private-byod/python@sha256:")


def test_notebook_text_has_no_franzen_period_names(tmp_path: Path) -> None:
    build(tmp_path)
    for name in ("notebook.ipynb", "kernel-metadata.json"):
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert re.search(r"duck|franzen", text, re.IGNORECASE) is None, name
