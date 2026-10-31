# Visual-Spatial Calculations Design (Part 4 of the refactor spec)

Status: draft for review, 2026-10-30. Nothing is built from this document yet.
Source: Kameron's "Visual-Spatial Calculations" section, pasted 2026-10-30. The source
names four tools and no requirements, so this part states what each tool would do in
our agent and what it would take to adopt it. Parts 1 to 3 define the stores, retrieval,
and metacognition that these calculations feed.

## 1. Requirements (from the source)

The source lists four tools:

- **Breadth-first search (BFS).**
- **Jump Point Search (JPS).**
- **Theta\*.**
- **Q-learning.**

It does not say which problem each tool solves for the agent. Section 3 maps each one to
a problem in the current agent. Section 5 records the calls I made.

## 2. What the current code does

The agent already runs three of the four ideas in some form, under other names. It has no
JPS, Theta*, or Q-learning.

| Tool | Where it exists | What it does | Status |
|---|---|---|---|
| BFS | `kinematics.py` `reachable` and `_walk_back`; `explore.py` `Graph.path_to_frontier`; `guided.py` path search | Shortest move count over a gait graph, or over the explorer's state graph. | Built in four places with the same loop. |
| Best-first search (A*-like) | `search.py` | Expands scenes by plan length plus weight times `Goal.distance`, using a heap. | Built. The heuristic is not checked for admissibility, so paths are not guaranteed shortest. |
| Optimal visiting order | `route.py` | Held-Karp for up to 6 targets, then nearest-neighbor with 2-opt. | Built. Not in the source list, but it is the spatial routing tool the agent uses. |
| JPS | none | | Missing. |
| Theta\* | none | | Missing. |
| Q-learning | none. I searched `arcagent` for reward, Q-table, TD, and epsilon and found nothing. | | Missing. |

Movement in the agent is not free-form. `kinematics.gaits` learns each action's effect:
a fixed step (`move`) or a slide until blocked (`slide`). Clicks are excluded from gaits.
A path is a sequence of learned actions, and only those are executable.

## 3. What each tool would do here

**BFS: keep and share.** BFS is exact for shortest move counts when every move costs one
action, which is the case here. The four copies (kinematics, explorer, guided, and the
walk back) differ only in their neighbor function. One shared function that takes a
neighbor function removes the duplication and gives one place to fix bugs.

**Best-first search: keep, and measure before changing.** `search.py` already does what
A* does, with `Goal.distance` as the heuristic. The gap is that the heuristic is not
guaranteed to be admissible, so the search can return a longer plan than the shortest one.
An admissible heuristic would fix that, but it may expand more nodes. This is a
trade-off to measure, not a bug to fix now.

**JPS: conditional, not built.** JPS speeds up uniform-cost grid search by skipping
symmetric expansions. Its pruning rules assume every move is a unit step in a fixed set of
directions. Our gaits break that assumption in two ways: a game's step size can be more
than one cell, and slides stop at obstacles. JPS applies only to a game whose learned
gaits are all unit steps. Whether any game is in that class, and whether the search time
justifies JPS, is unmeasured. Section 5 (D2) sets the gate.

**Theta\*: not adopted.** Theta* produces any-angle paths by drawing straight lines
between cells. The agent can only take the learned discrete actions, so an any-angle path
usually contains moves the game cannot make. It would fit a game with continuous movement.
I have not checked the 25 games for continuous movement. The survey from 2026-10-29
recorded action lists and colors, not movement types, so this is unverified.

**Q-learning: not in this build.** Q-learning learns the value of state-action pairs from
reward. In these games the reward is a level-up, which happens 6 to 10 times per game, and
a game over. A value table keyed on scene would be sparse. It would need the same canonical
features and store that Parts 1 and 2 define, and its cost would show up in the held-out
games. Section 5 (D4) sets the gate.

## 4. Behavior

Applies only to the BFS consolidation in this build. The other tools are gated.

- **Shared BFS.** One function takes a start, a neighbor function, a target test, and a
  visit limit, and returns the path and the depth. The limit is `KIN_LIMIT` for kinematics
  and the explorer's own bound for the state graph.
- **Equivalence.** For every caller, the shared BFS returns the same path and the same
  distances as the current copy it replaces, on the same inputs.
- **Bounds.** The shared BFS visits at most its limit of nodes and reports when it stops
  at the limit, so a caller can tell an empty result from a truncated one. Today
  `reachable` stops silently at `KIN_LIMIT` (4096 top-lefts); that silent stop is the
  behavior to change.

## 5. Decisions I made

These are my calls, changeable without changing section 1.

- **D1. BFS stays the default for distance.** Reason: exact for unit moves, and the
  simplest correct tool. Cost if wrong: none found; the gait model is already discrete.
- **D2. JPS gate.** JPS is built only if all three hold on the measured runs: (a) at
  least one game's learned gaits are all unit steps; (b) search and BFS time is a large
  share of move time in those games; (c) JPS returns the same plan length as BFS on those
  games. Reason: JPS adds code paths that only some games use. Cost if wrong: time spent
  on a tool that does not pay off.
- **D3. Theta* is out of scope.** Reason: the executable moves are discrete. Revisit only
  if a game with continuous movement is in the held-out set.
- **D4. Q-learning waits for the store.** It is built only after Part 1's store and Part
  2's retrieval exist, and only as a held-out experiment. Reason: a value table keyed on
  scene would overfit a single game, and its keys need the canonical features. Cost if
  wrong: an experiment with no result.
- **D5. Heuristic admissibility is measured, not assumed.** I will not change the
  heuristic in `search.py` until the plan-length gap is measured on real runs.

## 6. Test plan

**Shared BFS equivalence.** For each of the four current copies, a test feeds the same
inputs to the old and new function and asserts the same path and distances. Each caller
keeps its own test.

**Truncation.** A test with a limit below the reachable area checks that the result is
marked truncated and that the empty result (no path) is distinct from the truncated one.

**Search plan length.** On recorded runs, compare the plan length `search.py` returns
with the BFS shortest length over the same gait graph. The gap is the measure for D5.

**JPS gate.** Only if D2 is triggered: time BFS and JPS on the qualifying games, and check
plan lengths match.

**Profile.** Before any change, profile the move loop and report the share of time in
`kinematics`, `search`, `explore`, and `guided`. This decides whether D2 applies at all.

**Held-out games.** Same rule as Parts 1 to 3.

## 7. Open questions for Kameron

- **Q1.** Do you want an admissible heuristic in the best-first search (D5), even if it
  costs more expansions? Or is the current search, which is faster but not guaranteed
  shortest, the one to keep?
- **Q2.** Do any of the games you plan to test have continuous movement, such as a sprite
  that moves by pixels rather than cells? That decides whether Theta* stays out (D3).
- **Q3.** Is Q-learning meant as a learned policy that replaces the explorer, or as a
  value estimate that ranks frontier states? The second fits the store and retrieval
  design better, and is what D4 assumes.
- **Q4.** Should `route.py` (the visiting order over targets) be listed as a visual-spatial
  tool in the refactor? It is in the code but not in your list.
