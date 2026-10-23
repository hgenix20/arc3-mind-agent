# Fluid agent, slice 1 (no LLM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new mind config `FLUID` that plays ARC-AGI-3 games on CPU with no LLM by perceiving objects, learning short facts about how actions change them, predicting every move before taking it, searching inside those predictions toward guessed goals, and testing only when prediction can't decide.

**Architecture:** Seven small pure modules under `arcagent/mind/` (objects, noise, facts, predictor, goals, search, tester) and one policy, `FluidAgent`, that wires them per move and logs to the existing `RunLog`. `MindConfig` gains a `fluid` flag; `MindRun.make_agent` builds a `FluidAgent` for it. The v1 `MindAgent` path is untouched, so v1 and `FLUID` can be scored side by side.

**Tech Stack:** Python 3.12, numpy, pytest, ruff, pyright strict. Interpreter `C:/arc-agent-venv/Scripts/python.exe`.

**Spec:** `docs/superpowers/specs/2026-10-23-fluid-agent-design.md` (approved 2026-10-23). This plan covers delivery slice 1 only.

## Global Constraints

- No Franzen code or structure, and no other team's harness (`tests/test_no_franzen.py` must stay green).
- No genre priors: nothing in code, names or comments names a known game. Root mechanic words only: move, block, push, collect, toggle, die, win, spawn, rotate, select, no_effect.
- No memorized solutions: facts describe effects of actions on objects, never action sequences.
- A fact is `thin` under 5 observations (`THIN_BELOW = 5`), `solid` at 5 or more, `broken` after 2 misses it can't absorb (`BROKEN_MISSES = 2`).
- A test happens only when the move needed is unknown, a fact went broken, or the search found no plan. Never random, never a repeat of a settled test.
- Every choose call is bounded: search time cap `SEARCH_TIME_S = 0.25`, node cap `SEARCH_MAX_NODES = 20000`.
- Grids are `(64, 64)` int8, values 0..15, padding `-1` (`arcagent/core/types.py`). Padding and the grid edge always block movement.
- Run log: every record keeps the envelope in `arcagent/mind/runlog.py`; action records keep `level, step, frame_hash, action, branch, why, programs, vetoes, voi, entropy, wake, active` so `scripts/mind_metrics.py` scores them unchanged. `branch` uses the existing `Branch` literal.
- Repo gates per task: `ruff format`, `ruff check`, `pyright`, `pytest` all clean. Commit messages end with exactly `Claude-Session: https://claude.ai/code/session_01ED1JLjSNaMyvkVda5AScAg` and no Co-Authored-By line. No em-dashes in prose or comments.

## Decisions this plan makes (the spec leaves them to the builder)

- Shape signature: `shape` = first 10 hex of SHA-1 over the sorted cell offsets from the bbox top-left; `sig = f"c{color}:{shape}"`. Siblings share a shape (any color) or a color (any shape).
- Object ids: a `Tracker` matches by same `sig` nearest top-left (within `MATCH_RADIUS = 16` cells, Manhattan), then same `shape` at the same top-left (recolor), then same color with overlapping cells (reshape); the rest disappear or appear. Ids reset on level change and on RESET.
- Noise is per object track, never per cell: a track is noise when its change descriptor was identical under 2 different action ids, or it had a non-move change on each of the last 4 transitions spanning 2 or more action ids. The mask is the union of the noise tracks' bboxes, and masked cells read as background before object extraction.
- Action labels: `"A1".."A5"`, `"A7"`; a click is `f"A6:{clicked_sig}"`. Subjects: an object's `sig`, `"self"` for the clicked object of a click, or `"any"`. Contexts: `"any"`, `"blocked"`, `"pushed"`, `f"touch:{color}"`. Event facts use `("event", name, "any")`.
- Search: breadth-first over predicted scenes, deduped by `Scene.key()`. A branch ends at an unknown prediction and its `(label, subject)` pair is returned as a test candidate.
- Slice 1 goal guesses come from heuristics only: `reach(A, B)` for every sig A that has a move fact against every other non-noise sig B, and `clear(color)` for every color whose objects disappeared or were recolored away by an action. `match` and `count` are evaluated but not proposed until slice 2.
- The paper's four modules stay as they are for v1. In slice 1 the tester, vetoes and broken threshold are plain functions; wrapping them as the VOI, Inhibition and Metacognition protocols is slice 3 work.
- Facts persist across levels within a game with no consolidation (sleep and the winning-facts check are slice 3).

## Review Focus

1. A level change replaces the whole screen: the tracker must reset rather than report every object as moved, and no fact may learn from that transition. Pinned in Task 1 and Task 8.
2. A game with only clicks (no simple actions): goals and tests must still work with `A6:` labels and no movable object. Pinned in Task 7 and Task 8.
3. The GAME_OVER frame and the frame after RESET carry no rule: the agent returns RESET on GAME_OVER and learns nothing from the RESET transition, except the `touch:` game-over fact from the move that caused it. Pinned in Task 3 and Task 8.
4. Screens with hundreds of tiny objects: extraction, tracking and one search must stay under the time cap. Pinned in Task 1 and Task 6.
5. A fact that predicts move but the object is pressed against a wall: this must become a `blocked` fact, never a broken move fact. Pinned in Task 3 and Task 4.

---

### Task 1: Objects, scenes, tracking and diffs

**Files:**
- Create: `arcagent/mind/objects.py`
- Test: `tests/mind/test_objects.py`

**Interfaces:**
- Consumes: `arcagent.synth.trace.background_color`, `components`.
- Produces:
  ```python
  @dataclass(frozen=True, slots=True)
  class Thing:
      id: int; color: int; shape: str; sig: str; x: int; y: int   # bbox top-left
      cells: frozenset[tuple[int, int]]                           # absolute (x, y)
  @dataclass(frozen=True, slots=True)
  class Scene:
      things: tuple[Thing, ...]          # sorted by id
      background: int
      blocked: frozenset[tuple[int, int]]  # padding cells; the 64x64 edge is implicit
      def key(self) -> tuple[tuple[str, int, int], ...]   # sorted (sig, x, y), ids ignored
      def by_id(self) -> dict[int, Thing]
      def occupancy(self) -> dict[tuple[int, int], int]   # cell -> thing id
  ChangeKind = Literal["moved", "recolored", "reshaped", "disappeared", "appeared", "same"]
  @dataclass(frozen=True, slots=True)
  class Change:
      id: int; kind: ChangeKind; dx: int = 0; dy: int = 0; color: int = -1; sig: str = ""
  @dataclass(frozen=True, slots=True)
  class SceneDiff:
      changes: tuple[Change, ...]   # one per id in before or after, sorted by id
      stray_cells: int              # changed cells outside any thing
      @property
      def empty(self) -> bool       # every change is "same" and stray_cells == 0
  def shape_of(cells: Iterable[tuple[int, int]]) -> str
  def extract(grid: Grid, mask: frozenset[tuple[int, int]] = frozenset()) -> list[Thing]  # ids are -1
  class Tracker:
      def reset(self) -> None
      def update(self, grid: Grid, mask: frozenset[tuple[int, int]]) -> tuple[Scene, SceneDiff | None]
          # None diff on the first frame after reset
  ```

- [ ] **Step 1: Write the failing tests**

```python
def grid_with(cells: dict[tuple[int, int], int], bg: int = 0) -> Grid:
    g = np.full((64, 64), bg, dtype=np.int8)
    for (x, y), c in cells.items():
        g[y, x] = c
    return g

def test_sig_is_position_invariant_and_color_aware():
    a = extract(grid_with({(1, 1): 3, (2, 1): 3}))[0]
    b = extract(grid_with({(10, 7): 3, (11, 7): 3}))[0]
    c = extract(grid_with({(10, 7): 4, (11, 7): 4}))[0]
    assert a.sig == b.sig and a.shape == c.shape and a.sig != c.sig
    assert a.sig.startswith("c3:")

def test_tracker_reports_move_with_stable_id():
    t = Tracker()
    s0, d0 = t.update(grid_with({(5, 5): 2, (20, 20): 7}), frozenset())
    s1, d1 = t.update(grid_with({(6, 5): 2, (20, 20): 7}), frozenset())
    assert d0 is None and d1 is not None
    mover = next(c for c in d1.changes if c.kind == "moved")
    assert (mover.dx, mover.dy) == (1, 0)
    assert s0.by_id()[mover.id].color == 2 and s1.by_id()[mover.id].x == 6

def test_tracker_recolor_and_disappear_and_appear():
    t = Tracker()
    t.update(grid_with({(5, 5): 2, (9, 9): 4}), frozenset())
    _, d = t.update(grid_with({(5, 5): 3, (30, 30): 6}), frozenset())
    kinds = sorted(c.kind for c in d.changes)
    assert kinds == ["appeared", "disappeared", "recolored"]

def test_mask_hides_cells_from_extraction():
    g = grid_with({(5, 5): 2, (40, 1): 9})
    assert {t.color for t in extract(g, frozenset({(40, 1)}))} == {2}

def test_padding_is_blocked_not_a_thing():
    g = np.full((64, 64), -1, dtype=np.int8); g[:32, :32] = 0; g[3, 3] = 5
    s, _ = Tracker().update(g, frozenset())
    assert [t.color for t in s.things] == [5] and (40, 40) in s.blocked

def test_unchanged_frame_gives_empty_diff():
    t = Tracker(); g = grid_with({(5, 5): 2})
    t.update(g, frozenset())
    assert t.update(g.copy(), frozenset())[1].empty

def test_many_small_objects_extract_and_track_fast():
    g = grid_with({(x, y): 1 + (x + y) % 3 for x in range(0, 64, 2) for y in range(0, 64, 2)})
    t = Tracker(); t.update(g, frozenset())
    start = time.perf_counter(); t.update(g.copy(), frozenset())
    assert time.perf_counter() - start < 0.2
```

- [ ] **Step 2: Run to verify they fail**

Run: `C:/arc-agent-venv/Scripts/python.exe -m pytest tests/mind/test_objects.py -v`
Expected: FAIL, `ModuleNotFoundError: arcagent.mind.objects`.

- [ ] **Step 3: Implement `arcagent/mind/objects.py`** with the matching order and `MATCH_RADIUS = 16` from "Decisions". Masked cells read as the background color before `components` runs. Ids come from a per-tracker counter starting at 1 and never reused until `reset`.

- [ ] **Step 4: Run the tests and the gates**

Run: `C:/arc-agent-venv/Scripts/python.exe -m pytest tests/mind/test_objects.py -v` then `ruff format arcagent tests && ruff check arcagent tests && pyright arcagent/mind/objects.py tests/mind/test_objects.py`
Expected: all PASS, no lint or type errors.

- [ ] **Step 5: Commit** `git add arcagent/mind/objects.py tests/mind/test_objects.py && git commit` with message "Fluid slice 1: objects, scenes, tracking and diffs" plus the trailer line.

---

### Task 2: Noise mask

**Files:**
- Create: `arcagent/mind/noise.py`
- Test: `tests/mind/test_noise.py`

**Interfaces:**
- Consumes: `Scene`, `SceneDiff`, `Change` from Task 1.
- Produces:
  ```python
  class NoiseMask:
      cells: frozenset[tuple[int, int]]
      noisy_ids: frozenset[int]
      def observe(self, before: Scene, action_id: int, diff: SceneDiff) -> list[int]  # newly flagged ids
      def reset_tracks(self) -> None   # level change: forget track history, keep cells
  ```

- [ ] **Step 1: Write the failing tests**

```python
def test_same_change_under_two_actions_is_noise():  # a bar that shrinks by one whatever is pressed
    # build a before Scene with a bar thing (id 7) and a player (id 1); diffs reshape the bar only
    m = NoiseMask()
    assert m.observe(scene, 1, reshape_diff(7, size_delta=-1)) == []
    assert m.observe(scene, 2, reshape_diff(7, size_delta=-1)) == [7]
    assert 7 in m.noisy_ids and bar_cells <= m.cells

def test_player_moving_differently_per_action_is_never_noise():
    m = NoiseMask()
    for a, (dx, dy) in [(1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0))] * 2:
        m.observe(scene, a, move_diff(1, dx, dy))
    assert 1 not in m.noisy_ids

def test_animation_every_step_across_actions_is_noise():
    m = NoiseMask()
    for a in (1, 2, 1, 2):
        m.observe(scene, a, recolor_diff(7, next_color()))
    assert 7 in m.noisy_ids

def test_change_after_only_one_action_is_never_noise():
    m = NoiseMask()
    for _ in range(6):
        m.observe(scene, 5, recolor_diff(7, 3))
    assert m.noisy_ids == frozenset()
```

Write the small `scene`, `reshape_diff`, `move_diff`, `recolor_diff` helpers in the test file over the Task 1 types. "Change descriptor" for the same-change rule is `(kind, dx, dy, color)` plus the size delta for `reshaped`.

- [ ] **Step 2: Run to verify they fail** (`pytest tests/mind/test_noise.py -v`, ModuleNotFoundError).
- [ ] **Step 3: Implement `NoiseMask`** with the rule in "Decisions" (window of 4 for the every-step rule).
- [ ] **Step 4: Run tests and gates** as in Task 1. Expected: PASS.
- [ ] **Step 5: Commit** "Fluid slice 1: noise mask over object tracks".

---

### Task 3: Fact memory with a table of contents, and learning from a transition

**Files:**
- Create: `arcagent/mind/facts.py`
- Test: `tests/mind/test_facts.py`

**Interfaces:**
- Consumes: `Scene`, `SceneDiff`, `Thing` from Task 1.
- Produces:
  ```python
  EffectKind = Literal["move", "slide", "appear", "disappear", "recolor", "reshape",
                       "toggle_group", "teleport", "none", "game_over", "level_up"]
  @dataclass(frozen=True, slots=True)
  class Effect:
      kind: EffectKind; dx: int = 0; dy: int = 0; color: int = -1; sig: str = ""
      at: tuple[int, int] = (-1, -1)
      def text(self) -> str            # "moves 1 cell left", "no change", ...
  FactKey = tuple[str, str, str]       # (action label, subject, context)
  Status = Literal["thin", "solid", "broken"]
  THIN_BELOW = 5; BROKEN_MISSES = 2
  @dataclass(slots=True)
  class Fact:
      key: FactKey; effects: tuple[Effect, ...]; evidence: list[int]
      hits: int = 0; misses: int = 0; tags: frozenset[str] = frozenset()
      @property
      def status(self) -> Status
      def line(self) -> str            # "A3 c1:ab12.. any: moves 1 cell left [solid 6/0]"
  class FactMemory:
      facts: dict[FactKey, Fact]
      toc: dict[str, set[FactKey]]     # behavior tag -> keys
      blockers: dict[str, set[int]]    # subject sig -> colors that stopped it
      def get(self, key: FactKey) -> Fact | None
      def lookup(self, label: str, subject: str, context: str) -> tuple[Fact, bool] | None
          # most specific of (label, subject, context), (label, subject, "any"),
          # then a sibling-subject fact via the toc (borrowed=True), then (label, "any", "any")
      def record(self, key: FactKey, effects: tuple[Effect, ...], move: int) -> Literal["new", "hit", "miss"]
      def by_tag(self, tag: str) -> list[Fact]
  def action_label(action: Action, before: Scene) -> str        # "A3", "A6:<clicked sig>", "A6:none"
  def learn(mem: FactMemory, before: Scene, action: Action, diff: SceneDiff,
            *, move: int, level_up: bool, game_over: bool) -> list[FactKey]  # keys touched
  ```

`learn` turns each `Change` into a per-subject observation: `moved -> move(dx, dy)`, `same -> none`, `disappeared -> disappear`, `recolored -> recolor(color)`, `reshaped -> reshape(sig)`, `appeared -> appear(sig, at)` under subject `"any"`. The clicked object's subject is `"self"`. Context rules:
- a `none` observed where `(label, sig, "any")` holds a move: record `(label, sig, "blocked") -> none` and add the colors at the would-be destination cells to `blockers[sig]`; the move fact is not charged a miss (Review Focus 5).
- a `moved` object whose new cells overlap another mover's destination: record under `"pushed"`.
- `game_over=True`: for each subject that moved, record `("any", sig, f"touch:{c}") -> game_over` for every non-background color 4-adjacent to its destination cells; nothing else is learned from that transition.
- `level_up=True`: record `("event", "level_up", "any") -> level_up`; nothing else is learned.
- A move observed with a different magnitude in the same direction, ending blocked, rewrites the fact to `slide(sign dx, sign dy)` as a hit. The same absolute destination twice with different displacements rewrites to `teleport(at)`.

Tags from effects: move -> "move", slide -> "move" and "moves until blocked", none -> "no_effect", blocked facts -> "block", pushed facts -> "push", disappear -> "collect", recolor and toggle_group -> "toggle", appear -> "spawn", game_over -> "die", level_up -> "win", reshape -> "rotate" when the cell count is unchanged.

- [ ] **Step 1: Write the failing tests**

```python
def test_new_fact_then_hits_until_solid():
    mem = FactMemory()
    key = ("A3", "c1:aaaa", "any")
    assert mem.record(key, (Effect("move", dx=-1),), 0) == "new"
    for i in range(1, 5):
        assert mem.record(key, (Effect("move", dx=-1),), i) == "hit"
    assert mem.get(key).status == "solid" and mem.get(key).evidence == [0, 1, 2, 3, 4]

def test_two_unabsorbed_misses_break_a_fact():
    mem = FactMemory(); key = ("A1", "c1:aaaa", "any")
    mem.record(key, (Effect("move", dy=-1),), 0)
    mem.record(key, (Effect("recolor", color=3),), 1)
    mem.record(key, (Effect("disappear"),), 2)
    assert mem.get(key).status == "broken"

def test_wall_makes_blocked_fact_not_a_miss():
    # before: player c1 at (5,5), wall c8 at (4,5); A3 learned as move(-1,0) earlier, now no change
    keys = learn(mem, before, Action(3), same_diff, move=7, level_up=False, game_over=False)
    assert ("A3", player.sig, "blocked") in keys
    assert mem.get(("A3", player.sig, "any")).misses == 0 and 8 in mem.blockers[player.sig]

def test_toc_finds_facts_by_behavior_and_borrows_for_siblings():
    mem.record(("A1", "c2:shape1", "any"), (Effect("move", dy=-1),), 0)
    assert [f.key for f in mem.by_tag("move")] == [("A1", "c2:shape1", "any")]
    fact, borrowed = mem.lookup("A1", "c5:shape1", "any")   # same shape, new color
    assert borrowed and fact.effects[0].kind == "move"

def test_game_over_records_touch_fact_only():
    keys = learn(mem, before, Action(4), diff_player_moved_next_to_color_9,
                 move=3, level_up=False, game_over=True)
    assert keys == [("any", player.sig, "touch:9")]

def test_click_labels_use_the_clicked_sig_and_self():
    assert action_label(Action(6, 5, 5), scene_with_thing_at_5_5) == f"A6:{thing.sig}"
    assert action_label(Action(6, 60, 60), scene_with_thing_at_5_5) == "A6:none"
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `arcagent/mind/facts.py`** with the rules above. `record` counts a miss only when the effects differ and no context rule absorbed it.
- [ ] **Step 4: Run tests and gates.** Expected: PASS.
- [ ] **Step 5: Commit** "Fluid slice 1: fact memory, table of contents, learning".

---

### Task 4: Predictor

**Files:**
- Create: `arcagent/mind/predictor.py`
- Test: `tests/mind/test_predictor.py`

**Interfaces:**
- Consumes: Task 1 types; `FactMemory`, `Effect`, `FactKey`, `action_label` from Task 3.
- Produces:
  ```python
  @dataclass(frozen=True, slots=True)
  class Prediction:
      scene: Scene | None                 # None when any subject is unknown
      effects: dict[int, tuple[Effect, ...]]   # thing id -> predicted effects
      used: tuple[FactKey, ...]           # facts consulted
      unknown: tuple[tuple[str, str], ...]     # (label, subject) with no fact
      game_over: bool
      borrowed: bool
  def predict(scene: Scene, action: Action, mem: FactMemory) -> Prediction
  def check(pred: Prediction, before: Scene, diff: SceneDiff) -> tuple[bool, list[FactKey]]
      # (hit, keys whose effect missed); a hit means every predicted effect matched its Change
  ```
- Rules: subjects are looked up with context `"pushed"` when another subject's predicted move lands on them, `"blocked"` when the move's destination hits the edge, padding or a color in `mem.blockers[sig]` (prediction becomes `none`), else `"any"`. `slide` steps one cell at a time until blocked. `toggle_group(sig)` applies its recolor to every thing with that sig. `game_over` is True when any subject's predicted destination touches a color with a `("any", sig, "touch:c")` fact. Things on noise ids are never subjects (the agent strips them before calling). Pure Python, no numpy in the hot path.

- [ ] **Step 1: Write the failing tests**

```python
def test_predicts_move_and_new_scene():
    p = predict(scene_player_at(5, 5), Action(3), mem_with(("A3", PSIG, "any"), Effect("move", dx=-1)))
    assert p.scene is not None and player_in(p.scene).x == 4 and p.unknown == ()

def test_wall_predicts_no_change():
    p = predict(scene_player_left_of_wall(), Action(4), mem_with(("A4", PSIG, "any"), Effect("move", dx=1)))
    assert player_in(p.scene).x == player_in(before).x

def test_unknown_subject_gives_no_scene():
    p = predict(scene_player_at(5, 5), Action(1), FactMemory())
    assert p.scene is None and ("A1", PSIG) in p.unknown

def test_slide_stops_at_blocker():
    # corridor: player at x=2, wall at x=9; slide right ends at x=8
    assert player_in(predict(corridor, Action(4), slide_mem).scene).x == 8

def test_predicted_touch_of_deadly_color_flags_game_over():
    assert predict(next_to_lava, Action(4), mem_move_right_and_touch_9).game_over

def test_check_hit_and_miss():
    hit, missed = check(pred_move_left, before, diff_moved_left)
    assert hit and missed == []
    hit, missed = check(pred_move_left, before, diff_recolored)
    assert not hit and missed == [("A3", PSIG, "any")]
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `arcagent/mind/predictor.py`.**
- [ ] **Step 4: Run tests and gates; add a timing check: 1000 `predict` calls on a 30-thing scene under 0.5 s.**
- [ ] **Step 5: Commit** "Fluid slice 1: predictor over facts".

---

### Task 5: Goal guesses

**Files:**
- Create: `arcagent/mind/goals.py`
- Test: `tests/mind/test_goals.py`

**Interfaces:**
- Consumes: `Scene`, `Thing`; `FactMemory`.
- Produces:
  ```python
  GoalKind = Literal["reach", "clear", "match", "count"]
  GoalStatus = Literal["open", "ruled_out", "excluded", "confirmed"]
  @dataclass(frozen=True, slots=True)
  class Goal:
      kind: GoalKind; a: str = ""; b: str = ""; color: int = -1; n: int = -1
      region_a: tuple[int, int, int, int] = (0, 0, 0, 0); region_b: tuple[int, int, int, int] = (0, 0, 0, 0)
      def id(self) -> str                # stable text id, e.g. "reach c1:ab..>c4:cd.."
      def holds(self, scene: Scene, grid: Grid | None = None) -> bool
  class GoalBook:
      status: dict[Goal, GoalStatus]
      confirmed_forms: list[Goal]        # carried across levels, first hypotheses next level
      def propose(self, scene: Scene, mem: FactMemory, *, source: str) -> list[Goal]  # newly added
      def rule_out(self, history: Sequence[Scene]) -> list[Goal]   # held on a logged frame, no level-up
      def note_reached(self, scene: Scene) -> list[Goal]            # open goals now true -> excluded
      def level_up(self, before: Scene) -> list[Goal]               # goals true in before -> confirmed
      def open(self) -> list[Goal]       # confirmed forms first, then by insertion order
  ```
- `reach(a, b)`: some thing with sig `a` has a cell inside the bbox of some thing with sig `b`, or 4-adjacent to it. `clear(color)`: no thing of that color. `count(color, n)`: exactly n things of that color. `match(region_a, region_b)`: the two grid regions are equal (needs `grid`). Heuristic proposals as in "Decisions". An excluded or ruled-out goal is never proposed again in this game (`propose` skips anything already in `status`). On `level_up`, open goals that became excluded stay excluded on the next level unless they are a confirmed form.

- [ ] **Step 1: Write the failing tests**

```python
def test_reach_proposed_for_movers_against_other_sigs():
    goals = GoalBook().propose(scene_player_and_two_targets, mem_player_moves, source="heuristic")
    assert {g.kind for g in goals} == {"reach"} and len(goals) == 2

def test_goal_true_on_logged_frame_without_level_up_is_ruled_out():
    book = GoalBook(); book.propose(...)
    out = book.rule_out([scene_player_on_target_1])
    assert [g.b for g in out] == [T1SIG] and book.status[out[0]] == "ruled_out"

def test_reached_without_level_up_is_excluded_and_never_reproposed():
    book.note_reached(scene_player_on_target_1)
    assert book.propose(scene_player_and_two_targets, mem_player_moves, source="heuristic") == []

def test_level_up_confirms_the_true_goal_and_puts_its_form_first():
    book.level_up(scene_player_on_target_2)
    assert book.confirmed_forms[0].b == T2SIG and book.open()[0] == book.confirmed_forms[0]

def test_clear_proposed_from_recolor_facts_in_click_games():
    goals = GoalBook().propose(click_scene, mem_click_recolors_4_to_9, source="heuristic")
    assert {(g.kind, g.color) for g in goals} == {("clear", 4), ("clear", 9)}
```

- [ ] **Step 2–5:** fail, implement, pass with gates, commit "Fluid slice 1: goal guesses".

---

### Task 6: Short search inside the predictions

**Files:**
- Create: `arcagent/mind/search.py`
- Test: `tests/mind/test_search.py`

**Interfaces:**
- Consumes: `predict`, `Prediction` (Task 4), `Goal` (Task 5), `Scene`.
- Produces:
  ```python
  SEARCH_TIME_S = 0.25; SEARCH_MAX_NODES = 20000
  @dataclass(frozen=True, slots=True)
  class PlanStep:
      action: Action; expect: Prediction
  @dataclass(frozen=True, slots=True)
  class SearchResult:
      goal: Goal | None; plan: tuple[PlanStep, ...]
      frontier: tuple[tuple[str, str], ...]   # unknown (label, subject) pairs met, by first meeting
      nodes: int; depth: int; stop: Literal["found", "exhausted", "time", "nodes"]
  def search(scene: Scene, actions: Callable[[Scene], list[Action]], mem: FactMemory,
             goals: Sequence[Goal], *, time_s: float = SEARCH_TIME_S,
             max_nodes: int = SEARCH_MAX_NODES, clock: Callable[[], float] = time.monotonic) -> SearchResult
  ```
- Breadth-first; the first node where any goal holds wins (fewest predicted moves, so the cheapest goal). A predicted `game_over` node is never expanded. `actions(scene)` lets clicks follow the predicted things. `goal.holds` is called with `grid=None`; `match` goals are skipped by search in slice 1.

- [ ] **Step 1: Write the failing tests**

```python
def test_finds_shortest_plan_to_reach():
    r = search(player_at(2, 2), simple_actions, mem_four_moves, [reach(PSIG, TSIG_at_5_2)])
    assert r.stop == "found" and [s.action.id for s in r.plan] == [4, 4, 4]

def test_avoids_predicted_game_over_route():
    r = search(lava_between, simple_actions, mem_four_moves_and_lava, [reach_target])
    assert r.stop == "found" and all(not s.expect.game_over for s in r.plan)

def test_unknown_actions_end_branches_and_become_frontier():
    r = search(player_at(2, 2), simple_actions, mem_only_right_known, [reach_left_target])
    assert r.plan == () and ("A3", PSIG) in r.frontier

def test_respects_time_cap_on_large_open_board():
    fake = iter(range(10_000)); clock = lambda: next(fake) * 0.01
    r = search(open_board, simple_actions, mem_four_moves, [unreachable], time_s=0.25, clock=clock)
    assert r.stop == "time"
```

- [ ] **Step 2–5:** fail, implement, pass with gates, commit "Fluid slice 1: short search over predicted scenes".

---

### Task 7: Targeted tests and wisdom vetoes

**Files:**
- Create: `arcagent/mind/tester.py`
- Test: `tests/mind/test_tester.py`

**Interfaces:**
- Consumes: Tasks 1, 3, 4, 6.
- Produces:
  ```python
  @dataclass(frozen=True, slots=True)
  class Test:
      action: Action; question: str; settles: int   # open (label, subject) pairs it answers
  def choose_test(scene: Scene, actions: list[Action], mem: FactMemory,
                  frontier: Sequence[tuple[str, str]], tried: set[tuple[tuple[str, int, int], tuple[int, int, int]]],
                  toward: Action | None) -> Test | None
  def veto(scene: Scene, action: Action, mem: FactMemory,
           tried: set[tuple[tuple[str, int, int], tuple[int, int, int]]]) -> str | None
      # reason or None: "predicted game_over", "known no effect here", "settled test"
  ```
- Ranking: the action whose prediction has the most unknown subjects first (a simple action answers every subject at once; a click answers its clicked sig), then actions touching `broken` or `thin` facts, ties broken first by `frontier` order, then by `toward` (the first step of the best partial plan), then by lowest `Action.key()`. `tried` holds `(scene.key(), action.key())`; a tried pair is never chosen again. `None` when nothing is open. `veto` uses `predict`: a predicted `game_over`; every subject predicted `none` by `solid` facts with no unknowns; or `(scene.key(), action.key())` in `tried` with no open question for that action.

- [ ] **Step 1: Write the failing tests**

```python
def test_untried_simple_action_beats_known_ones():
    t = choose_test(player_scene, acts_1_to_4, mem_knows_1_and_2, frontier=[], tried=set(), toward=None)
    assert t is not None and t.action.id == 3

def test_click_game_tests_each_untried_clicked_sig_once():
    t = choose_test(click_scene, clicks, mem_knows_click_on_sig_a, frontier=[], tried=set(), toward=None)
    assert action_label(t.action, click_scene) == f"A6:{SIG_B}"

def test_never_repeats_a_settled_test():
    tried = {(player_scene.key(), Action(3).key())}
    t = choose_test(player_scene, [Action(3)], mem_knows_1_and_2, [], tried, None)
    assert t is None

def test_nothing_open_returns_none():
    assert choose_test(player_scene, acts_1_to_4, mem_all_solid, [], set(), None) is None

def test_veto_reasons():
    assert veto(next_to_lava, Action(4), mem_lava, set()) == "predicted game_over"
    assert veto(player_left_of_wall, Action(4), mem_wall_solid, set()) == "known no effect here"
```

- [ ] **Step 2–5:** fail, implement, pass with gates, commit "Fluid slice 1: targeted tests and vetoes".

---

### Task 8: FluidAgent, config wiring and logging

**Files:**
- Create: `arcagent/mind/fluid.py`
- Modify: `arcagent/mind/config.py` (add `fluid: bool = False` to `MindConfig`, add `CONFIGS["FLUID"]`)
- Modify: `arcagent/agent/submit.py` (`MindRun.make_agent`: a `fluid` config builds `FluidAgent` on `ProgramSandbox.dead()`, no worker spawned)
- Modify: `arcagent/mind/runlog.py` docstring (new event kinds)
- Test: `tests/mind/test_fluid_agent.py`, extend `tests/test_offline_loop.py` only if it enumerates configs

**Interfaces:**
- Consumes: everything above; `Baseline` (B0) from `arcagent/mind/baseline.py`; `RunLog`; `candidate_actions` from `arcagent/mind/perception.py`.
- Produces:
  ```python
  class FluidAgent:   # implements arcagent.agent.play.Policy and the close() MindGame calls
      def __init__(self, game: str, cfg: MindConfig, *, seed: int, log: RunLog,
                   game_deadline: float, run_deadline: float,
                   clock: Callable[[], float] = time.monotonic) -> None
      def choose(self, obs: Observation) -> Action
      def close(self) -> None   # logs game_end with fact and goal counts
  CONFIGS["FLUID"] = MindConfig("FLUID", metacognition=False, inhibition=False, voi=False,
                                transfer=False, llm=False, fluid=True)
  ```
- Per `choose(obs)`, in order:
  1. `GAME_OVER` or `NOT_PLAYED`: if a previous move exists, `learn(..., game_over=True)` from it; clear the plan; return `Action(0)` (branch `reset`). The transition out of a RESET learns nothing.
  2. `obs.levels_completed` rose: `learn(..., level_up=True)`, `goals.level_up(prev_scene)`, log `level_up` (`step`, `level`) exactly as `agent.py` does, reset tracker and noise tracks, clear the plan, history and `tried`. Facts stay.
  3. Otherwise update tracker and noise (strip noisy ids from the scene before any further use), `learn` from the transition, `check` the last prediction, log the `programs` list as `[{"id": " ".join(key), "pred": text, "matched": bool, "fit": hits/(hits+misses)}]` for each used fact, `goals.note_reached`, `goals.rule_out(history)`.
  4. A miss or a fact turning `broken` clears the plan. If the plan still has steps and the last step hit: next step, branch `follow`.
  5. Else `goals.propose`, `search` over `goals.open()`; a found plan starts, branch `plan`, why `goal <id>`.
  6. Else `choose_test` with the search's frontier, branch `learn`, why the test's question.
  7. Else B0 with the vetoed keys excluded, branch `fallback`.
  8. `veto` on the chosen action; a vetoed action is logged as a `veto` event and the next option in the same order is taken (B0 last, which never vetoes). The predicted outcome of the chosen action is stored for step 3 of the next call.
  Every exception inside steps 3 to 8 is logged as `agent_error` and that move goes to B0, as in `agent.py`. Past the game or run deadline, skip search and tests and use the plan or B0.
- New events: `fact` (`step, key, status, line`) when a fact is new or changes status; `goal` (`step, id, status, source`); `test` (`step, action, question`); `noise` (`step, ids, cells`); `veto` (`step, action, reason`). `entropy` is the count of open `(label, subject)` pairs; `active` is null.

- [ ] **Step 1: Write the failing tests** with a tiny fake game (a 64x64 board, player color 1, target color 4, walls color 8, actions 1-4 move by one cell, reaching the target levels up, color 9 ends the game):

```python
def test_fluid_config_exists_and_skips_llm():
    cfg = CONFIGS["FLUID"]
    assert cfg.fluid and not cfg.llm

def test_agent_learns_moves_then_plans_to_the_target(tmp_path):
    game = FakeGame(player=(2, 2), target=(6, 2), walls=[], lava=[])
    agent = FluidAgent("fake", CONFIGS["FLUID"], seed=0, log=RunLog(tmp_path / "f.jsonl", "fake", "FLUID", 0),
                       game_deadline=1e12, run_deadline=1e12)
    actions = play(game, agent, max_steps=30)
    assert game.levels == 1 and len(actions) <= 12

def test_agent_resets_on_game_over_and_avoids_the_cause_next_time(tmp_path):
    game = FakeGame(player=(2, 2), target=(2, 8), walls=[], lava=[(3, 2)])
    ... play until two game overs or a level-up
    assert game.game_overs <= 1

def test_level_change_learns_nothing_from_the_new_screen(tmp_path):
    # after level-up the fake game moves every object; no "moved" fact may come from that transition
    ...

def test_click_only_game_tests_each_object_once(tmp_path):
    game = FakeClickGame(objects=3)   # clicking recolors the clicked object; all same color wins
    ...
    assert game.levels == 1

def test_log_scores_with_mind_metrics(tmp_path):
    ... play the fake game, then
    out = mind_metrics.compute(mind_metrics.load([tmp_path])[0], human={"fake": [10]})
    assert out["games"]["fake"]["levels_completed"] == 1
    assert out["games"]["fake"]["prediction_accuracy"] is not None
```

The fake games live in `tests/mind/fakegames.py` and return `Observation` values directly (no toolkit), stepping through `agent.choose`.

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `FluidAgent` and the wiring.**
- [ ] **Step 4: Run the full suite and gates**

Run: `C:/arc-agent-venv/Scripts/python.exe -m pytest` then `ruff format --check . && ruff check . && pyright`
Expected: every test passes (v1 tests included), no lint or type errors.

- [ ] **Step 5: Commit** "Fluid slice 1: FluidAgent, FLUID config, logging".

---

### Task 9: Score slice 1 on the 25 public games

**Files:**
- Modify: `docs/experiment-log.md` (one row under "Mind entry (stack + Genix Mind + offline LLM), from 2026-10-23")

- [ ] **Step 1: Run** `C:/arc-agent-venv/Scripts/python.exe scripts/offline_loop.py --config FLUID --game-seconds 300 --out bench/offline/fluid-s1-run1`
Expected: 25 games finish; the script prints the `mind_metrics` summary.
- [ ] **Step 2: Run it a second time** to `bench/offline/fluid-s1-run2` (paired runs, spec "Goal and measure").
- [ ] **Step 3: Compare per game** against `bench/evals/2026-10-23-full-v1/` with `scripts/mind_metrics.py`. Record levels, score estimate for both runs, and the five games with the most actions per completed level.
- [ ] **Step 4: Write the row** in `docs/experiment-log.md` with both runs, the v1 comparison, and which games stalled and why (from the `test` and `goal` events). Commit "Log fluid slice 1 on the public games".
