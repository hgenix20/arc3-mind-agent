# LLM-Driven Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `DRIVE` config where the LLM writes the model of the game, the deterministic tools play moves out inside that model before any real move, and the LLM rewrites a failing planning tool, with fixes shared across the run.

**Architecture:** `DriveAgent` subclasses `FluidAgent` (`arcagent/mind/llm_drive.py`) and changes only who decides: wake rules, turn deadlines, how a reply is applied (play-out first), and when an unmodeled move needs the LLM. The LLM's model enters the existing `FactMemory` as claims plus a free-text model it keeps across turns. Patched tools live in a process-wide `Toolbox` and run in a separate tools worker process.

**Tech Stack:** Python 3.12, numpy, the existing mind modules, vLLM-served Qwen3.6-27B-FP8 through `vchat`.

**Spec:** `docs/superpowers/specs/2026-10-26-llm-driven-agent-design.md`

## Global Constraints

- Branch `llm-drive` off main; merged only after a full comparison (spec: Branch).
- No Franzen code; no game-specific code; a tool fix never hard-codes a layout, coordinates or a move list.
- Model-written code runs only in a subprocess with a deadline; on crash or timeout the original tool answers.
- Patchable tools: `search.search`, `route.route`, `goals.Goal.distance`, `planner` search. Never perception, wisdom, sandbox, budget, scheduler, submit.
- No real move the model hasn't played out, except mode 3 (an LLM experiment) and the late fallback.
- Without a chat, `DRIVE` plays exactly as `FLUID` (no-model fallback stays valid).
- ruff, pyright, pytest pass; pytest runs with `C:\arc-agent-venv\Scripts\python.exe -m pytest`.

## Review Focus

- A reply whose facts name ids not on screen or unknown effects: dropped with a reason, never raise.
- A patched tool that returns the wrong type, hangs, or crashes the worker: original answers, worker restarts.
- Two games adopting fixes to the same tool at once: versions stay ordered, no lost update.
- An LLM plan whose first step the model can't predict: only that one step goes out (an experiment), then a wake.
- No chat or turns off mid-game: the agent keeps playing on slice 1 choices.

---

### Task 1: Reply contract carries the model

**Files:** Modify `arcagent/mind/contract.py`; Test `tests/mind/test_contract.py`

**Interfaces:**
- Produces: `FactClaim(label: str, subject: str, effects: tuple[Effect, ...])`; `Reply.claims: tuple[FactClaim, ...]`, `Reply.model: str` (at most `MAX_MODEL_CHARS = 1500`), `Reply.rule: str` (code, at most `MAX_RULE_CHARS = 4000`).
- Reply JSON adds `"facts": [{"action": "A1"|"click #<id>", "object": <id>|"any", "effect": "move", "dx": 0, "dy": -1}]` with effects `move`(dx,dy), `none`, `disappear`, `recolor`(color), `game_over`, `level_up`; `"model": "<text>"`; `"rule": "<python>"`.
- A click claim's label is `action_label(Action(6,x,y), scene)`; a simple one `"A<n>"`. Subject is the thing's sig, or `"any"`.

- [ ] Tests: `test_facts_parse_into_claims` (move claim on #3 → FactClaim("A1", sig3, (Effect("move",0,-1),))), `test_bad_fact_dropped_with_reason` (unknown id, unknown effect, A0), `test_model_and_rule_clipped`.
- [ ] Implement in `parse_reply`; run tests; commit.

### Task 2: Drive prompt

**Files:** Modify `arcagent/mind/turn.py`; Test `tests/mind/test_turn.py`

**Interfaces:**
- `TurnInput` gains `model: str = ""` and `drive: bool = False`; with `drive`, `build_messages` uses `DRIVE_SYSTEM_PROMPT` and adds a `Your model so far:` block with `model`, and `MAX_PROMPT_CHARS_DRIVE = 9000`.
- `DRIVE_SYSTEM_PROMPT` asks for: goals, facts (the model of what each action does to which object), a plan of up to 8 steps with expected effects or one test, the updated `model` text, and optionally a `rule` function `rule(objects, action) -> list[dict]` over objects given as dicts `{"id","sig","color","x","y","cells"}`. It keeps "Never assume the game works like a game you know" out: the LLM's own knowledge is allowed (spec anti-goals), and asks that every plan step's expect name the object and effect.

- [ ] Tests: `test_drive_prompt_has_model_block`, `test_non_drive_prompt_unchanged`.
- [ ] Implement; run; commit.

### Task 3: The model in memory and play-out

**Files:** Create `arcagent/mind/game_model.py`; Test `tests/mind/test_game_model.py`

**Interfaces:**
- `seed_claims(mem: FactMemory, claims: Sequence[FactClaim], move: int) -> list[FactKey]`: records each claim under `(label, subject, "any")` only when that key has no fact; observation always wins. Returns the keys added.
- `PlayOut(reached: Goal | None, steps: int, unknown_at: int | None, scenes: tuple[Scene, ...])`.
- `play_out(scene: Scene, actions: Sequence[Action], mem: FactMemory, goals: Sequence[Goal], protect: frozenset[str], rule: RuleFn | None = None) -> PlayOut`: predicts step by step with `assume_still=protect`; stops at the first step with no scene (`unknown_at`) or the first scene where a goal holds (`reached`, `steps` = moves to it). `rule` (from Task 4) is applied after each fact prediction to the objects it returns.

- [ ] Tests: `test_seed_skips_observed_key`, `test_play_out_reaches_goal`, `test_play_out_stops_at_unknown`, `test_rule_moves_object`.
- [ ] Implement; run; commit.

### Task 4: Toolbox and tools worker

**Files:** Create `arcagent/mind/toolbox.py`, `arcagent/mind/_tool_worker.py`; Test `tests/mind/test_toolbox.py`

**Interfaces:**
- `TOOLS: dict[str, ToolSpec]`, `ToolSpec(name, module, func, agrees: Callable[[object, object], bool])` for `"search"`, `"route"`, `"distance"`.
- `Toolbox` (process-wide `TOOLBOX`, lock-protected): `current(name) -> ToolVersion | None` (None = original), `adopt(name, source, game, reason) -> int` (version no.), `rollback(name, reason) -> int`, `add_case(name, args, kwargs, result)` keeping the last `MAX_CASES = 8`, `cases(name)`, `history(name)`.
- `ToolWorker` (one per game): subprocess `python _tool_worker.py` with cwd on `sys.path` so it imports `arcagent`; JSON lines carrying base64 pickles. `call(name, version, source, args, kwargs, timeout_s) -> tuple[bool, object]` (ok, result or error text): the worker execs `source` into a copy of the tool's module namespace, reloads when the version changes, and calls the function. Timeout or crash: kill, restart, `(False, reason)`. `run_rule(source, objects, action, timeout_s)` for Task 3's `rule`.
- `call_tool(name, worker, *args, **kwargs)`: original in-process when no version; else the worker; on failure the original's answer.
- `SceneActions(simple: tuple[Action, ...], can_click: bool, offsets: dict[str, Cell])`, picklable and callable like `FluidAgent._actions`, so search args pickle.

- [ ] Tests: `test_patched_tool_runs_in_worker`, `test_hanging_patch_falls_back`, `test_crash_restarts_worker`, `test_adopt_and_rollback_versions`, `test_cases_capped`, `test_concurrent_adopts_ordered`.
- [ ] Implement; run; commit.

### Task 5: Tool fix turn

**Files:** Create `arcagent/mind/tool_fix.py`; Test `tests/mind/test_tool_fix.py`

**Interfaces:**
- `ToolFailure(tool: str, args: tuple, kwargs: dict, expected: str, got: str, check: Callable[[object], bool])`: `check` says whether a result solves the failing case (for search: found, plan no longer than the LLM's, and `play_out` of it reaches the goal).
- `fix_messages(failure, source) -> list[VMessage]`: tool source (`inspect.getsource`), the failing case in words, the expected and actual answers; asks for one Python function with the same name and signature, general, no coordinates.
- `try_fix(chat, failure, worker, game, seed) -> FixResult(adopted: int | None, reason: str)`: extracts the code fence, runs it on the failing case (must pass `check`) and on every stored case (must `agree` with the recorded result), within `FIX_CALL_S = 5.0` each; adopts on success.

- [ ] Tests (fake chat): `test_fix_adopted_when_it_solves_and_agrees`, `test_fix_rejected_on_regression`, `test_fix_rejected_without_code`.
- [ ] Implement; run; commit.

### Task 6: DriveAgent

**Files:** Create `arcagent/mind/llm_drive.py`; Modify `arcagent/mind/fluid.py` (search through `self._search_fn`, `_actions` via `SceneActions`, small hooks); Test `tests/mind/test_llm_drive.py`

**Interfaces / behavior:**
- Wake: no per-level or per-game turn caps and no spacing; turns allowed until `DRIVE_CUTOFF = 0.1` of the game budget is left. Wake also on an unmodeled slice 1 choice (branch `learn`, `fallback`, `forced`, `commit`, or why `explore novel` / `sequence try`), reason `no modeled move`.
- Turn deadline `DRIVE_TURN_S = 90.0`, `max_tokens` `DRIVE_MAX_TOKENS = 2048`. Timeout keeps the slice 1 choice, logged as mode `fallback`.
- Apply: seed claims (Task 3), keep `reply.model` for the next prompt, load `reply.rule`; play out the LLM plan. Reached or fully predicted: plan runs as mode `sim`. First unknown step at i: run steps up to and including i (the last is mode `experiment`), then wake. A test is mode `experiment`.
- Tool failure: when the LLM plan reaches goal g in the model, run search over [g]; no plan or a longer one is a `search` failure; the fix runs on the pool without blocking, the LLM plan is played meanwhile. A tool exception is a failure too. One fix attempt per (tool, level).
- Regression cases: when a search plan's real moves all matched their predictions, `TOOLBOX.add_case("search", ...)`.
- Rollback: a patched tool's failure that the previous version solves (run both) rolls back.
- Log: action record `mode` in `sim|experiment|fallback`; events `tool_fail`, `tool_fix` (adopted, reason), `tool_rollback`; `game_end` adds mode counts.

- [ ] Tests (fake chat, small synthetic scenes): `test_level_start_turn_before_first_move`, `test_unmodeled_choice_wakes`, `test_plan_played_out_runs_as_sim`, `test_unknown_step_is_one_experiment`, `test_search_failure_triggers_fix`, `test_no_chat_plays_like_fluid`.
- [ ] Implement; run full suite; commit.

### Task 7: Wiring and an offline smoke run

**Files:** Modify `arcagent/mind/config.py` (`CONFIGS["DRIVE"]` = `FLUID-LLM` with `drive=True`), the agent factory that builds `FluidAgent` (build `DriveAgent` when `cfg.drive`), `scripts/mind_metrics.py` (mode and tool counts); Test `tests/mind/test_config.py`, `tests/scripts/test_mind_metrics.py`.

- [ ] Tests: `test_drive_config_builds_drive_agent`, `test_metrics_count_modes`.
- [ ] Smoke: one public game offline with a scripted fake chat for 200 moves, no `agent_error`.
- [ ] Commit.

### Task 8: Measure and iterate

- [ ] Build the notebook with `--config DRIVE`, push, run the interactive commit on the 25 public games, read `metrics.json` scorecard and the per-game logs.
- [ ] Below 75: find the biggest loss in the logs (turn failures, mode mix, experiments wasted, fixes), fix on the branch, repeat.
- [ ] At 75 or above: merge to main per the repo rule, submit, log the result in `docs/experiment-log.md`.
