# Agent Memory Design (Part 1 of the refactor spec)

Status: draft for review, 2026-10-30. Nothing is built from this document yet.
Source: Kameron's "Memory" section, pasted 2026-10-30. Where the source and the existing
code disagree, the source governs. The current agent is not scoring well, so the
source is the target and the code is the starting point.

Later parts of the spec will be added as new sections. Numbers in this part refer to
its own sections only.

## 1. Requirements (from the source)

**Working memory**

- **R1 Scope.** Working memory persists through levels. It clears on a full game reset
  or on the start of the next game.
  - Hypothesis H1 (Kameron's): if working memory forces the agent to relearn each game,
    the working memory design gets redesigned. Section 6 says how to test this.
- **R2 Escalation.** A working memory item that is validated moves to long-term memory.
- **R3 Contents.**
  - Current state list: items on the grid, each mapped to its actions, events, and
    interactions with the player (a wall, an enemy, an item, a door, an exit, or
    anything like them).
  - Action list: each action mapped to the events and interactions that occur when it
    is present or called.
  - Goal list: two items in sequence, "complete the level" then "win the game".
  - Task list: a focused list of steps, ranked by sequence.
  - Level rule list: rules organized by object, event, or experience, for the current
    level. The agent identifies objects and guesses events, then pulls the relevant
    rules from the game rule list. If no rule exists during the level, the agent creates
    one.
  - Game rule list: the same organization, scoped to the current game.

**Long-term memory**

- **R4 Scope.** Long-term memory persists between games.

The source does not say what a validated item is, what long-term memory stores beyond
rules, or what gets loaded at the start of a game. Section 5 proposes answers and marks
them as decisions.

## 2. What the current code does

Code paths are the fluid agent (`arcagent/mind/fluid.py`), the path the hybrid runs
(`arcagent/agent/hybrid.py`). The older LLM and program path (`arcagent/mind/agent.py`,
`memory.py`) is not on the hybrid path.

| Requirement | Current code | Status | Gap |
|---|---|---|---|
| R1 scope: game | A new `FluidAgent` and fresh modules per game (`config.py` `build_modules`: "modules hold no cross-game state"). | Built | None for the game boundary. |
| R1 persistence: levels | `FactMemory` keeps facts across level-up and resets only positions (`fluid.py` step 2). | Built | Facts are not split into a level list and a game list. Scope is a `level` stamp on each fact. |
| R1 persistence: game over | A game-over RESET keeps facts and clears the plan (`fluid.py` step 1). | Built | Source says working memory clears on a full game reset. Whether a RESET after game over is a full reset is open (Q1). |
| R3 current state list | `objects.py` (`Thing`, `Scene`), identity across looks in `kinds.py`. | Partial | Things have no event or interaction list. Interactions live only in facts keyed by subject and label. |
| R3 action list | `FactMemory` facts keyed by subject, label, and context, with effects. | Partial | Facts are keyed by action, not organized as a per-action list. |
| R3 goal list (level, game) | `GoalBook` holds guesses (reach, clear, match, cover, sequence, match_thing). A level-up confirms a goal. | Partial | No "win the game" goal. No two-tier list. |
| R3 task list | Plan from `search.py` and `route.py`. `goals.ranked` orders goals. | Partial | The plan is a path to one goal, not a ranked task list. |
| R3 level rule creation | `FactMemory.record` creates a fact when evidence is new. | Built | The fact is not filed under a level rule list. |
| R2 escalation | None. `fluid_sleep` demotes facts a winning level relied on after misses, merges siblings, and drops facts with zero hits. | Missing | Nothing promotes a fact to a longer-lived store. |
| R4 long-term memory | None on the fluid path. `GameStore` and `WorkingMemory` (`memory.py`) belong to the LLM path. | Missing | No store survives the game. |

**Why this matters for the score.** A game starts with zero knowledge. What a click does,
what a wall blocks, and what a key opens are relearned in every game, even when the
same mechanic appeared in an earlier game. The fact keys also use the color index
(`sig` includes color). The 25 games use different color subsets (survey of first
frames, 2026-10-29), so a key built on color would not transfer across games even if
it were stored. A cross-game key has to name kind and role, not color.

## 3. Proposed structure

Three stores, with one schema for every rule.

**Rule record.** Every rule in working memory and long-term memory has the same fields.

- `id`: stable within its store.
- `kind`: `object`, `action`, `event`, or `experience`.
- `subject`: a thing kind (not a sig) for cross-game rules, or a sig within a game.
- `effect`: what happens, in the existing effect vocabulary.
- `scope`: `level`, `game`, or `long`.
- `created_level`, `created_game`: where the rule was first seen.
- `hits`, `misses`: evidence counts.
- `status`: `candidate`, `validated`, or `broken`.
- `source`: `learned` or `loaded` (from long-term memory).

**Goal record.** Scope `level` or `game`, a statement (the goal kind and its arguments),
and a status (`open`, `probing`, `reached`, `excluded`). "Win the game" is a game-scope
goal.

**Stores.**

- **Level rules** are the rules with scope `level`. `new_level` drops them, except those
  promoted to game scope.
- **Game rules** are the rules with scope `game`. They are cleared on a full game reset
  or at the start of the next game.
- **Long-term rules** are the rules with scope `long`. They are loaded at game start and
  saved at game end.

The existing `FactMemory` becomes the store for level and game rules. It already stamps
each fact with its level, and `fluid_sleep` already moves evidence between states. The
change is to add `scope` and the canonical key, not to replace the class.

## 4. Behavior

- **Creation.** A rule is created when the agent sees evidence no rule covers. It starts
  as `candidate` with scope `level`.
- **Promotion inside a game.** A candidate becomes `validated` when it reaches the solid
  threshold (`THIN_BELOW = 5` agreeing observations, the existing constant) and has no
  miss in the level where it was used. A validated rule created on one level is promoted
  to scope `game`.
- **Escalation to long-term (R2).** A game-scope rule that is validated and used in a
  winning level is written to long-term memory with scope `long`.
- **Loading.** At game start, long-term rules are loaded as `candidate` with
  `source = loaded`. They are not validated, so the first miss on a loaded rule breaks
  it. A loaded rule that the new game confirms is promoted as usual.
- **Breaking.** A rule breaks after the existing `BROKEN_MISSES` misses in a row. A
  broken long-term rule is kept with its miss count, not deleted, so the log shows it.
- **Clearing.** On a full game reset, game rules and level rules are cleared. Long-term
  rules are kept.

Every create, promote, load, break, and clear writes a log event with the rule `id`, so
an ablation can count them.

## 5. Decisions I made

These are my calls. Each one can be changed without changing the requirements in
section 1.

- **D1.** Cross-game keys use thing kind and role, not color index. Reason: color
  subsets differ per game (section 2).
- **D2.** Loaded long-term rules start as `candidate`. Reason: a rule learned in one game
  can be wrong in another, and one miss should be enough to drop it.
- **D3.** Validated means `THIN_BELOW` agreeing observations and no miss on the level
  where it was used. The source does not give a number, so I reused the existing
  constant. Reason: it keeps one definition of "solid" in the code.
- **D4.** Long-term memory is a file written at game end and read at game start. It is
  never written during scoring unless Q2 says it should be.

## 6. Test plan

**H1 (relearning cost).** For each game, record moves to the first level-up and to the
first win, from a run with an empty long-term store. Then run again with the store
filled from other games, on games the store has not seen. Compare the two.

- A drop in moves on the second run supports H1 and the long-term design.
- No drop means the relearning cost is not in the rules, and the working memory redesign
  needs a different cause.

**Level transfer (6.1, 6.3).** Moves to level n+1 against moves to level n within a game,
with and without the working memory carry.

**Held-out games.** The 25 games are all public and have been used for tuning. Q4 asks
for a split. Every memory change is judged on games not used to tune it.

**Unit tests.** Each rule transition in section 4 has a test that checks its log event.
Each store's clear rule has a test: game rules and level rules clear on game reset, long-term
rules do not.

## 7. Open questions for Kameron

- **Q1.** After a game over, does RESET restart the same level or the whole game? The
  answer decides whether working memory clears on game over, or only on a full reset as
  the source says.
- **Q2.** Should long-term memory update during a scored run, so that games in the same
  run help each other? Or should it be a file built before the run?
- **Q3.** What makes a rule validated for escalation to long-term memory: a count of
  observations, a number of levels, or a number of games?
- **Q4.** Which games are the held-out set for memory changes? All 25 games are public
  and were used for tuning.
