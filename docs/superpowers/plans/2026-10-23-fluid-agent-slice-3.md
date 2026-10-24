# Fluid agent, slice 3 (sleep, transfer, modules, metrics) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the fluid agent as the full design: sleep at level-up and game end (consolidate, check the winning facts, carry facts and goal forms forward), the paper's four modules as swappable protocols with null versions, the HCQM metrics, and an end to undirected B0 moves. Kameron asked for the 2026-10-24 submission to be the finished product.

**Architecture:**
- **New `arcagent/mind/fluid_sleep.py`:** sleep and the winning-facts check.
- **New `arcagent/mind/fluid_modules.py`:** four protocols plus nulls for the fluid agent.
- **Changes:** `fluid.py`, `config.py` and `scripts/mind_metrics.py`.
- **Exploration:** a novelty search replaces B0 as the fallback when no goal plan and no test exist. It plans through known moves to a mover position not yet visited on this level.

**Tech Stack:** as slices 1 and 2.

**Spec:** `docs/superpowers/specs/2026-10-23-fluid-agent-design.md`, components 9 (sleep), 10 (drive and time), 12 (the four modules), "Three failures" item 3, "Metrics we log". Slice ledgers hold the rulings so far: `.superpowers/sdd/2026-10-23-fluid-agent-slice-1/progress.md` and `.superpowers/sdd/2026-10-23-fluid-agent-slice-2/progress.md`.

## Global Constraints

- Everything from slices 1 and 2 holds.
- Sleep is bounded: `SLEEP_MAX_S = 0.5` per call; whatever isn't done by then is skipped and logged.
- Modules advise; the agent applies and logs, as in `arcagent/mind/modules/base.py`.
- `FLUID` and `FLUID-LLM` run with all four modules on. Ablations are `FLUID-LLM-<module>` (one module null), built the same way `_all_configs` builds `FULL-<module>`.
- Log fields `scripts/mind_metrics.py` already reads stay unchanged; new metrics read new events.

## Decisions this plan makes

- **Winning-facts check (failure 3):** at level-up, collect the fact keys used by the moves of the level that just ended (the `used` keys of each executed prediction). Any of those facts with a miss on this level drops to thin: `hits = min(hits, THIN_BELOW - 1)`. A `demoted` event is logged for each. Facts with no misses are promoted as usual.
- **Consolidation:** merge facts that share a label and context, have identical effects, and whose subjects share a shape (sibling sigs). The merged fact keeps the higher counts, and the others are deleted and logged as `merged`. Thin facts with zero hits that were never used are dropped.
- **Transfer:** facts and the table of contents persist across levels (already the case). The confirmed goal form goes first (already). Each fact gains `level: int` (the level it was created on), so rule reuse can be measured. NullTransfer clears facts, blockers, walkable and safe sets at level-up and keeps only the tag vocabulary.
- **The four modules:**
  - **Voi:** `choose(scene, actions, mem, frontier, tried, toward) -> Test | None`. Real = `tester.choose_test`. Null = the first untried action in `Action.key()` order whose prediction has an unknown pair.
  - **Inhibition:** `veto(scene, action, mem, tried) -> str | None`. Real = `tester.veto`. Null = never vetoes.
  - **Metacognition:** `broken_after() -> int`, the consecutive-miss threshold. Real = `BROKEN_MISSES` (2). Null = a very large number, so facts never break and tests are never forced by misses.
  - **Transfer:** `carry(mem, goals) -> None` at level-up. Real keeps everything. Null clears as above.
- **Novelty fallback:** when there's no goal plan and no test, run `search` with a single pseudo-goal "some mover reaches a cell it hasn't occupied on this level". Use the same caps, and keep the visited mover cells per level. The plan's first step is taken with branch `plan`, why `explore novel`. B0 remains only when that also fails, and is logged as now.
- **Metrics** added to `scripts/mind_metrics.py` per game and overall, from events:
  - `hit_rate` (matched over all prediction entries) and `hit_rate_halves` (first and second half of each level)
  - `rule_reuse` (share of matched entries whose fact level is lower than the current level; needs `level` in the `programs` entries)
  - `tests_per_level`
  - `impasses` (wakes with reason "no plan and no test")
  - `search_depth_mean` (from `search` events)
  - `goals` counts by status
  - `actions_vs_human` per completed level

## Review Focus

1. Sleep on a level-up that arrives on the very first move of a level (no moves used): nothing demoted, no crash. Pinned in Task 1.
2. Null modules must never crash the agent or stop it acting: a full fake game played with every null module at once. Pinned in Task 2.
3. Novelty search on a click game with no movers returns no plan and falls to B0 cleanly. Pinned in Task 3.
4. Metrics on logs from before slice 3 (no `level` in programs entries, no search events): every new metric is None, and nothing raises. Pinned in Task 4.
5. Sleep time cap on a fact memory of 5,000 facts. Pinned in Task 1.

---

### Task 1: Sleep and the winning-facts check

**Files:** Create `arcagent/mind/fluid_sleep.py`; modify `arcagent/mind/facts.py` (`Fact.level`), `arcagent/mind/fluid.py` (call sleep at level-up and in `close`; track used keys per level; log `sleep`, `demoted`, `merged`). Test `tests/mind/test_fluid_sleep.py`.

**Interfaces:**
```python
SLEEP_MAX_S = 0.5
@dataclass(frozen=True, slots=True)
class FluidSleepReport:
    demoted: tuple[FactKey, ...]; merged: tuple[FactKey, ...]; dropped: tuple[FactKey, ...]
    seconds: float; truncated: bool
def fluid_sleep(mem: FactMemory, used: Iterable[FactKey], level: int, *,
                time_s: float = SLEEP_MAX_S, clock: Callable[[], float] = time.monotonic) -> FluidSleepReport
```
- `Fact` gets `level: int = 0`, set by `record` from a new `FactMemory.level` attribute that the agent updates on level change. The `programs` log entries get `"level": fact.level`.

- [ ] Tests:
  - A fact used on a winning level with one miss on that level is demoted to thin.
  - A fact with no misses keeps solid.
  - Two sibling facts with identical effects merge into one.
  - A zero-hit unused thin fact is dropped.
  - An empty `used` set changes nothing.
  - 5,000 facts and `time_s=0.01` with a fake clock give `truncated=True`.
- [ ] Fail, implement, pass with gates, commit "Fluid slice 3: sleep and the winning-facts check".

### Task 2: The four modules for the fluid agent, and ablation configs

**Files:** Create `arcagent/mind/fluid_modules.py`; modify `arcagent/mind/fluid.py` (take modules from config), `arcagent/mind/config.py` (`FLUID`/`FLUID-LLM` set all four flags True; add `FLUID-LLM-<module>` configs), the `facts.py` broken threshold (read from a parameter that Metacognition provides). Test `tests/mind/test_fluid_modules.py`.

**Interfaces:**
```python
class FluidVoi(Protocol):
    def choose(self, scene: Scene, actions: list[Action], mem: FactMemory,
               frontier: Sequence[tuple[str, str]], tried: Tried, toward: Action | None) -> Test | None: ...
class FluidInhibition(Protocol):
    def veto(self, scene: Scene, action: Action, mem: FactMemory, tried: Tried) -> str | None: ...
class FluidMetacognition(Protocol):
    def broken_after(self) -> int: ...
class FluidTransfer(Protocol):
    def carry(self, mem: FactMemory, goals: GoalBook) -> None: ...
@dataclass(frozen=True, slots=True)
class FluidModules:
    voi: FluidVoi; inhibition: FluidInhibition; metacognition: FluidMetacognition; transfer: FluidTransfer
def build_fluid_modules(cfg: MindConfig) -> FluidModules
```
- [ ] Tests:
  - Each null behaves as in "Decisions".
  - A fake game plays to the end with all nulls.
  - `CONFIGS["FLUID-LLM-voi"].voi is False` and the rest are True.
  - `build_fluid_modules` maps each flag to the real or null class.
- [ ] Fail, implement, pass with gates, commit "Fluid slice 3: the four modules as protocols".

### Task 3: Novelty fallback

**Files:** Modify `arcagent/mind/fluid.py` (visited mover cells per level; novelty search before B0), and `arcagent/mind/search.py` if a predicate goal hook is needed (`search(..., goal_test: Callable[[Scene], bool] | None = None)`). Test in `tests/mind/test_fluid_agent.py` and `tests/mind/test_search.py`.
- [ ] Tests:
  - On a fake maze with no reachable goal, the agent's fallback moves reach new cells, and none are B0 while a novel cell is reachable.
  - On a click-only fake game with no movers, novelty returns no plan and B0 acts.
  - `search` with `goal_test` finds the nearest novel cell.
- [ ] Fail, implement, pass with gates, commit "Fluid slice 3: novelty fallback before B0".

### Task 4: Metrics

**Files:** Modify `scripts/mind_metrics.py` (new fields, docstring definitions), `tests/test_mind_metrics.py`.
- [ ] Tests: the new metrics on a hand-built log; every new metric is None on a pre-slice-3 log (Review Focus 4); the overall aggregates.
- [ ] Fail, implement, pass with gates, commit "Fluid slice 3: HCQM metrics".

### Task 5: Final eval and submission (controller)

- [ ] Full suite and gates on the branch. Final whole-branch review on the slice 3 diff. Merge `fluid-slice-1` to main (Kameron asked for the finished product to be the submission).
- [ ] Push the kernel with `FLUID-LLM`, pull the output, score it, and log it in `docs/experiment-log.md`.
- [ ] Submit the finished agent on 2026-10-24, with the eval numbers in the message (Kameron's instruction 2026-10-23: "tomorrow's submission is going be a full finished product").
