"""Build our Kaggle kernel from Franzen's Milestone 2 notebook by named-cell patches.

Intent: the LLM layer of the agent is Daniel Franzen's Duck-derived harness
(see ``third_party/franzen-m2/NOTICE.md``). This script turns the unchanged
reference notebook into ``build/kaggle-duck/`` for ``kaggle kernels push``.
Every change we make is a ``Patch`` below, applied by exact string replacement
on one cell, so the diff against the reference is this file.

Patches in the baseline build:
- cell 0 (markdown): attribution header for our copy.
- cell 16 (code): the interactive "Save & Run" commit plays only ``--smoke-games``
  once (the reference plays all 25 games four times, nine GPU hours per commit).
- cell 4 (code): a game's first analyzer request waits up to 60 minutes (the
  reference: 15) for the server, because rerun machines load the model slowly.
- a new code cell after cell 16: ``scripts/duck_keepalive.py`` verbatim, which
  makes the warm-up game repeat its RESET every 5 minutes until the first model
  action. The gateway ends a run with no move for 15 minutes; duck v1 scored
  0.00 that way (see ``docs/experiment-log.md``).
The competition rerun branch (``TRUE_SUBMISSION``) is otherwise untouched.

Constraints: RTX PRO 6000, no internet, the same dataset and model sources as the
reference, image pinned to the same Python 3.12 digest.

Usage: python scripts/kaggle_duck_notebook.py [--slug S] [--smoke-games ...|all] [--variant V]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "third_party" / "franzen-m2" / "arc-agi-3-milestone-2-solution.ipynb"
KEEPALIVE_SRC = ROOT / "scripts" / "duck_keepalive.py"
KEEPALIVE_AFTER_CELL = 16  # the benchmark-config cell; the run cell comes later
OUT = ROOT / "build" / "kaggle-duck"
USERNAME = "hgenix20"
COMPETITION = "arc-prize-2026-arc-agi-3"
DOCKER_IMAGE = (
    "gcr.io/kaggle-private-byod/python@sha256:"
    "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c"
)
DATASET_SOURCES = ["dfranzen/pennyroyal-v253", "dfranzen/taaf-kaggle-source-bundle-copy"]
MODEL_SOURCES = [
    "dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1",
    "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1",
]
PUBLIC_GAMES = [
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85",
    "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48", "sp80", "su15",
    "tn36", "tr87", "tu93", "vc33", "wa30",
]  # fmt: skip

HEADER = """### ARC3 duck agent (hgenix20)

This kernel is Kameron Green's copy of Daniel Franzen's public Milestone 2 notebook
(`dfranzen/arc-agi-3-milestone-2-solution`, Apache-2.0 at github.com/da-fr/arc-agi-3-solution),
itself built on Tufa Labs' Duck harness. Credit for the harness, the patch, the serving
stack and the model choice goes to them. Our changes are applied as named-cell patches by
`scripts/kaggle_duck_notebook.py` in the `arc-agent` repository; this version's changes are
listed in the cell below. The original header follows.

---

"""


@dataclass(frozen=True)
class Patch:
    """Replace ``old`` (which must occur exactly once) with ``new`` in cell ``cell``."""

    cell: int
    old: str
    new: str
    why: str


def smoke_patches(smoke_games: list[str]) -> list[Patch]:
    """``smoke_games``: 4-char public game ids, or ``["all"]`` for a 25-game evaluation run."""
    if smoke_games == ["all"]:
        smoke_games = list(PUBLIC_GAMES)
    unknown = sorted(set(smoke_games) - set(PUBLIC_GAMES))
    if unknown:
        raise ValueError(f"not public games: {unknown}")
    excluded = [g for g in PUBLIC_GAMES if g not in smoke_games]
    return [
        Patch(
            cell=16,
            old="demo_excluded_games = [] if TRUE_SUBMISSION else []\n",
            new=(
                "# hgenix20: an interactive commit is a smoke test, not a benchmark.\n"
                f"SMOKE_EXCLUDED = {excluded!r}\n"
                "demo_excluded_games = [] if TRUE_SUBMISSION else SMOKE_EXCLUDED\n"
            ),
            why=f"interactive commits play only {smoke_games}",
        ),
        Patch(
            cell=16,
            old="bm.n_passes = 4\n",
            new="bm.n_passes = 1  # hgenix20: one pass in the interactive commit\n",
            why="one pass in the interactive commit",
        ),
    ]


RERUN_PATCHES: list[Patch] = [
    Patch(
        cell=4,
        old="os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '900'\n",
        new="os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '3600'  # hgenix20: slow hosts\n",
        why="a game's first analyzer request waits up to 60 min (reference: 15) for the server",
    ),
]


def keepalive_cell() -> dict[str, Any]:
    """The keep-alive module as a code cell, installed on the harness session class."""
    src = KEEPALIVE_SRC.read_text(encoding="utf-8")
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": (
            "# hgenix20: keep the competition scorecard alive while the server loads.\n"
            "# Verbatim copy of scripts/duck_keepalive.py from the arc-agent repository.\n"
            + src
            + "\nfrom inference.framework.solver import _HarnessGameSession\n\n"
            "install(_HarnessGameSession)\n"
        ),
    }


VARIANTS: dict[str, list[Patch]] = {
    "baseline": [],
    "streams16": [
        Patch(
            cell=4,
            old="'ARC3_MAX_ACTIVE_STREAMS': 10,\n",
            new="'ARC3_MAX_ACTIVE_STREAMS': 16,  # hgenix20: streams16 variant\n",
            why="16 active game streams instead of 10 (throughput is the published bottleneck)",
        ),
        Patch(
            cell=12, old="    MAXREQ=10,\n", new="    MAXREQ=16,\n", why="server runs 16 requests"
        ),
        Patch(
            cell=12,
            old="    CUDAGRAPH_MAXBS=10,\n",
            new="    CUDAGRAPH_MAXBS=16,\n",
            why="CUDA graphs cover batch 16",
        ),
    ],
    "spec4": [
        Patch(
            cell=12,
            old="    SPEC_STEPS=3,\n",
            new="    SPEC_STEPS=4,  # hgenix20: spec4 variant\n",
            why=(
                "draft 4 tokens per step instead of 3; baseline mean accept length is 2.67"
                " of 4, so a longer draft may raise decoded tokens per step"
            ),
        ),
    ],
}


def apply(nb: dict[str, Any], patches: list[Patch]) -> None:
    for p in patches:
        cell = nb["cells"][p.cell]
        src = "".join(cell["source"])
        if src.count(p.old) != 1:
            raise ValueError(f"cell {p.cell}: expected exactly one match for {p.old!r}")
        cell["source"] = src.replace(p.old, p.new)


def build(slug: str, smoke_games: list[str], out: Path = OUT, variant: str = "baseline") -> Path:
    nb = json.loads(REFERENCE.read_text(encoding="utf-8"))
    patches = smoke_patches(smoke_games) + RERUN_PATCHES + VARIANTS[variant]
    apply(nb, patches)
    cells = nb["cells"]
    cells.insert(KEEPALIVE_AFTER_CELL + 1, keepalive_cell())
    changes = "\n".join(f"- cell {p.cell}: {p.why}" for p in patches)
    changes += (
        f"\n- new cell after {KEEPALIVE_AFTER_CELL}: the warm-up game repeats its RESET every "
        "5 min until the first model action (the gateway ends a run idle for 15 min)"
    )
    cells[0]["source"] = HEADER + "".join(cells[0]["source"])
    cells.insert(
        1,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": f"**Changes in this version (hgenix20):**\n\n{changes}\n",
        },
    )
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
    nb["metadata"].pop("papermill", None)
    out.mkdir(parents=True, exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {
        "id": f"{USERNAME}/{slug}",
        "title": slug.replace("-", " ").replace("arc3", "ARC3", 1),
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": DATASET_SOURCES,
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
        "model_sources": MODEL_SOURCES,
        "machine_shape": "NvidiaRtxPro6000",
        "docker_image_pinning_type": "original",
        "docker_image": DOCKER_IMAGE,
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--slug", default="arc3-duck-agent")
    ap.add_argument("--smoke-games", nargs="+", default=["ls20", "ft09", "vc33"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="baseline")
    a = ap.parse_args(argv)
    out = build(a.slug, a.smoke_games, Path(a.out), a.variant)
    print(f"wrote {out} (smoke games {a.smoke_games}, variant {a.variant})")


if __name__ == "__main__":
    main()
