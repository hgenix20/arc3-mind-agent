"""Kaggle notebook builder for the mind agent: structure, metadata and shipped text."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

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


def test_run_cell_streams_submit_output_to_a_log(tmp_path: Path) -> None:
    nb, _ = build(tmp_path)
    run = run_cell(nb)
    assert 'PYTHONUNBUFFERED="1"' in run
    assert "/kaggle/working/submit.log" in run and "stderr=subprocess.STDOUT" in run
    assert 'raise RuntimeError("submission.parquet was not written")' in run


def test_run_cell_scales_the_interactive_budget_and_keeps_the_rerun_formula(
    tmp_path: Path,
) -> None:
    nb, _ = build(tmp_path)
    run = run_cell(nb)
    assert '"KAGGLE_IS_COMPETITION_RERUN"' in run and '{"1", "true"}' in run
    assert "if rerun:" in run and "total = max(1800, left)" in run
    assert "RERUN_SECONDS = 28800" in run and "PRIVATE_GAMES = 110" in run
    assert "N_GAMES = 25" in run
    assert "scaled = RERUN_SECONDS * N_GAMES / PRIVATE_GAMES" in run
    assert "total = max(1800, min(left, scaled))" in run


def test_run_cell_counts_named_games_for_the_scaled_budget(tmp_path: Path) -> None:
    nb, _ = build(tmp_path, "--games", "ls20", "ft09")
    assert "N_GAMES = 2" in run_cell(nb)


def test_interactive_seconds_overrides_the_scaled_value(tmp_path: Path) -> None:
    nb, _ = build(tmp_path, "--interactive-seconds", "5400")
    run = run_cell(nb)
    assert "scaled = 5400" in run
    assert "scaled = RERUN_SECONDS * N_GAMES / PRIVATE_GAMES" not in run
    compile(run, "<cell>", "exec")


def serve_cell(nb: dict[str, Any]) -> str:
    hits = [s for s in cell_sources(nb) if "--max-model-len" in s]
    assert len(hits) == 1
    return hits[0]


def test_serve_failure_sets_model_ok_false_instead_of_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    nb, _ = build(tmp_path)
    src = serve_cell(nb)
    # The cell edits these; setting them first lets monkeypatch restore them.
    monkeypatch.setenv("LIBRARY_PATH", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    ns: dict[str, Any] = {}
    exec(compile(src, "<serve>", "exec"), ns)  # no /kaggle/input here: find_input raises
    assert ns["MODEL_OK"] is False
    out = capsys.readouterr().out
    assert "model server failed" in out and "FileNotFoundError" in out


def test_run_cell_plays_explorer_when_the_model_server_failed(tmp_path: Path) -> None:
    nb, _ = build(tmp_path, "--config", "FULL")
    run = run_cell(nb)
    assert "config = 'FULL' if MODEL_OK else \"EXPLORER\"" in run
    assert '"--config", config' in run


def fallback_ns() -> dict[str, Any]:
    ns: dict[str, Any] = {}
    exec(compile(load_script().fallback_defs(), "<fallback>", "exec"), ns)
    return ns


def test_fallback_parquet_has_one_zero_row_per_public_game(tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    env = tmp_path / "env"
    for gid in ("ls20-aaaa", "ft09-bbbb", "vc33-cccc"):
        d = env / gid[:4] / gid[5:]
        d.mkdir(parents=True)
        (d / "metadata.json").write_text(json.dumps({"game_id": gid}), encoding="utf-8")
    ns = fallback_ns()
    assert ns["fallback_game_ids"](str(env), False, []) == ["ft09-bbbb", "ls20-aaaa", "vc33-cccc"]
    ids = ns["fallback_game_ids"](str(env), False, ["ls20"])
    assert ids == ["ls20-aaaa"]
    out = tmp_path / "submission.parquet"
    ns["write_fallback_parquet"](out, ids)
    table = pq.read_table(str(out))
    assert table.to_pylist() == [
        {"row_id": "0_0", "game_id": "ls20-aaaa", "end_of_game": True, "score": 0.0}
    ]
    # Same schema as the one submit writes.
    from arcagent.agent import submit  # noqa: PLC0415

    ref = tmp_path / "ref.parquet"
    submit.write_parquet([], ref)
    assert pq.read_schema(str(out)).equals(pq.read_schema(str(ref)))
    assert tuple(table.column_names) == submit.PARQUET_COLUMNS


def test_fallback_parquet_writes_a_placeholder_when_no_game_list_is_readable(
    tmp_path: Path,
) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    ns = fallback_ns()
    assert ns["fallback_game_ids"](str(tmp_path / "missing"), False, []) == []
    out = tmp_path / "submission.parquet"
    ns["write_fallback_parquet"](out, [])
    rows = pq.read_table(str(out)).to_pylist()
    assert len(rows) == 1 and rows[0]["score"] == 0.0 and rows[0]["end_of_game"] is True


def test_run_cell_writes_the_fallback_only_when_submit_left_no_parquet(tmp_path: Path) -> None:
    nb, _ = build(tmp_path)
    run = run_cell(nb)
    assert load_script().fallback_defs() in run
    i_exit = run.index('print("exit", returncode)')
    i_fallback = run.index("write_fallback_parquet(SUBMISSION")
    i_raise = run.index('raise RuntimeError("submission.parquet was not written")')
    assert i_exit < i_fallback < i_raise
    assert "if not SUBMISSION.exists():" in run[i_exit:i_fallback]
