"""Collect observed transitions of a public game through the action interface.

Contract:
- ``collect_trace`` drives the oracle only through ``observe``/``step``, the
  same interface the gateway exposes. It never reads game source or hidden
  state.
- Every executed action becomes one ``Transition``, RESET included.
- ``Trace.train`` is the first 70% of transitions by time and ``Trace.heldout``
  the rest: what an agent would have at synthesis time versus what comes after.

Probe policy: a fixed cycle through every available action in order
(ACTION6 expands into a small set of click targets computed from the current
grid), interleaved with uniformly random choices, with RESET whenever the game
reaches GAME_OVER or WIN.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arcagent.core.types import COORD_ACTION_ID, GRID_SIZE, RESET_ID, Action, Grid, Phase

TRAIN_FRACTION_NUM = 7
TRAIN_FRACTION_DEN = 10
MAX_CLICK_TARGETS = 8
RANDOM_PROB = 0.5


@dataclass(frozen=True, slots=True)
class Transition:
    """One observed step. ``step`` is the 0-based index in the trace."""

    step: int
    grid_before: Grid
    action: Action
    grid_after: Grid
    phase_after: str
    levels_before: int
    levels_after: int


@dataclass(slots=True)
class Trace:
    """An ordered list of transitions from one game and seed."""

    game: str
    seed: int
    win_levels: int
    transitions: list[Transition]

    @property
    def n_train(self) -> int:
        """Train size: floor(0.7 * n), computed in integers to avoid float drift."""
        return len(self.transitions) * TRAIN_FRACTION_NUM // TRAIN_FRACTION_DEN

    @property
    def train(self) -> list[Transition]:
        return self.transitions[: self.n_train]

    @property
    def heldout(self) -> list[Transition]:
        return self.transitions[self.n_train :]

    def save(self, directory: str | Path) -> Path:
        """Write ``<game>-s<seed>-n<len>.npz`` plus a ``.json`` sidecar; return the npz path."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        stem = f"{self.game}-s{self.seed}-n{len(self.transitions)}"
        npz_path = d / f"{stem}.npz"
        ts = self.transitions
        empty = np.zeros((0, GRID_SIZE, GRID_SIZE), dtype=np.int8)
        np.savez_compressed(
            npz_path,
            before=np.stack([t.grid_before for t in ts]) if ts else empty,
            after=np.stack([t.grid_after for t in ts]) if ts else empty,
            actions=np.array([t.action.key() for t in ts], dtype=np.int16).reshape(-1, 3),
            levels=np.array(
                [(t.levels_before, t.levels_after) for t in ts], dtype=np.int16
            ).reshape(-1, 2),
            steps=np.array([t.step for t in ts], dtype=np.int32),
        )
        meta = {
            "game": self.game,
            "seed": self.seed,
            "win_levels": self.win_levels,
            "n": len(ts),
            "n_train": self.n_train,
            "phases_after": [t.phase_after for t in ts],
        }
        npz_path.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return npz_path

    @classmethod
    def load(cls, npz_path: str | Path) -> Trace:
        """Inverse of ``save``. Raises ``FileNotFoundError`` if either file is missing."""
        p = Path(npz_path)
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(p) as z:
            before = z["before"].astype(np.int8)
            after = z["after"].astype(np.int8)
            actions = z["actions"]
            levels = z["levels"]
            steps = z["steps"]
        phases: list[str] = meta["phases_after"]
        ts = [
            Transition(
                step=int(steps[i]),
                grid_before=before[i],
                action=Action(int(actions[i, 0]), int(actions[i, 1]), int(actions[i, 2])),
                grid_after=after[i],
                phase_after=phases[i],
                levels_before=int(levels[i, 0]),
                levels_after=int(levels[i, 1]),
            )
            for i in range(len(phases))
        ]
        return cls(
            game=meta["game"],
            seed=int(meta["seed"]),
            win_levels=int(meta["win_levels"]),
            transitions=ts,
        )


def background_color(grid: Grid) -> int:
    """Most frequent visible color (ties broken by the smaller color id); -1 if none."""
    vals = grid[grid >= 0]
    if vals.size == 0:
        return -1
    counts = Counter(int(v) for v in vals.ravel())
    return min(counts, key=lambda c: (-counts[c], c))


def click_targets(
    grid: Grid, limit: int = MAX_CLICK_TARGETS, rng: random.Random | None = None
) -> list[tuple[int, int]]:
    """Click coordinates ``(x, y)`` for ACTION6.

    One target per 4-connected single-color component of non-background
    color, at most ``limit``. Without ``rng`` the largest components are
    taken; with ``rng`` a uniform sample of components (still listed largest
    first), so repeated probing covers small objects such as buttons too.
    The target is the component cell nearest its centroid, so it lies on the
    object even for non-convex shapes. With no components, a 4x4 lattice.
    """
    bg = background_color(grid)
    h, w = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for y0 in range(h):
        for x0 in range(w):
            c = int(grid[y0, x0])
            if seen[y0, x0] or c < 0 or c == bg:
                continue
            stack = [(y0, x0)]
            seen[y0, x0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] == c:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(cells)
    if not comps:
        step = GRID_SIZE // 4
        return [
            (x, y)
            for y in range(step // 2, GRID_SIZE, step)
            for x in range(step // 2, GRID_SIZE, step)
        ]
    comps.sort(key=len, reverse=True)
    if rng is not None and len(comps) > limit:
        keep = sorted(rng.sample(range(len(comps)), limit))
        comps = [comps[i] for i in keep]
    out: list[tuple[int, int]] = []
    for cells in comps[:limit]:
        cy = sum(c[0] for c in cells) / len(cells)
        cx = sum(c[1] for c in cells) / len(cells)
        y, x = min(cells, key=lambda c: ((c[0] - cy) ** 2 + (c[1] - cx) ** 2, c))
        out.append((x, y))
    return out


def _expand(action_id: int, grid: Grid, rng: random.Random) -> list[Action]:
    if action_id == COORD_ACTION_ID:
        return [Action(COORD_ACTION_ID, x, y) for x, y in click_targets(grid, rng=rng)]
    return [Action(action_id)]


def collect_trace(env_dir: str | Path, game: str, steps: int, seed: int = 0) -> Trace:
    """Run the probe policy for ``steps`` actions and return the trace.

    Deterministic for a given (game, steps, seed). Needs the engine wheels;
    the import is local so the rest of the package works without them.
    Raises ``ValueError`` for ``steps < 1`` and ``FileNotFoundError`` for an
    unknown game.
    """
    from arcagent.core.oracle import OracleModel  # noqa: PLC0415  # optional dependency

    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    rng = random.Random(seed)
    model = OracleModel.load(env_dir, game, seed=seed)
    obs = model.observe()
    queue: list[Action] = []
    transitions: list[Transition] = []
    for i in range(steps):
        avail = [a for a in obs.available_actions if a != RESET_ID]
        if obs.phase in (Phase.GAME_OVER, Phase.WIN) or not avail:
            action = Action(RESET_ID)
            queue = []
        elif rng.random() < RANDOM_PROB:
            action = rng.choice(_expand(rng.choice(avail), obs.grid, rng))
        else:
            if not queue:
                queue = [act for a in avail for act in _expand(a, obs.grid, rng)]
            action = queue.pop(0)
        after = model.step(action)
        transitions.append(
            Transition(
                step=i,
                grid_before=obs.grid.copy(),
                action=action,
                grid_after=after.grid.copy(),
                phase_after=str(after.phase),
                levels_before=obs.levels_completed,
                levels_after=after.levels_completed,
            )
        )
        obs = after
    return Trace(game=game, seed=seed, win_levels=obs.win_levels, transitions=transitions)
