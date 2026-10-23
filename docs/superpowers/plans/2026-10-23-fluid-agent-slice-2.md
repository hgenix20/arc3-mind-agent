# Fluid agent, slice 2 (the executive turn) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the slice 1 agent a conscious executive: when salience wakes it, the offline LLM sees the screen as an image plus an image of what changed, with a short text of objects, facts, goal guesses and recent moves. It answers in JSON with goal guesses and either a short plan with predicted effects or one test.

**Architecture:** Four new modules under `arcagent/mind/` and changes to `fluid.py`, `config.py`, `submit.py` and the notebook builder:
- `screen.py` renders PNGs with no new dependency.
- `vchat.py` is a vision chat client with record and replay.
- `contract.py` parses and validates the reply.
- `turn.py` builds the prompt, calls the model under a deadline, and returns the parsed reply.

The agent stays fully playable without the model: a failed, late or malformed turn falls back to slice 1 behavior.

**Tech Stack:** Python 3.12, numpy, stdlib `zlib`/`struct`/`base64` for PNG, `requests` (already used by `arcagent/synth/run.py`), vLLM OpenAI-compatible API, pytest, ruff, pyright strict.

**Spec:** `docs/superpowers/specs/2026-10-23-fluid-agent-design.md` (component 6 the executive turn, component 7 salience, delivery slice 2). Slice 1 plan and its ledger rulings: `docs/superpowers/plans/2026-10-23-fluid-agent-slice-1.md`, `.superpowers/sdd/2026-10-23-fluid-agent-slice-1/progress.md`.

## Global Constraints

- Everything from slice 1's Global Constraints holds (no Franzen, no genre priors, no memorized solutions, bounded time, run log envelope, repo gates, commit trailer, no em-dashes).
- The prompt names root mechanics only (move, block, push, collect, toggle, die, win, spawn, rotate, select, no_effect). It never names or describes a known game.
- Model: Qwen3.6-27B-FP8 on vLLM 0.19 at `http://127.0.0.1:1234/v1`, served name from `scripts/kaggle_synth_notebook.py`. Thinking off (`chat_template_kwargs.enable_thinking = False`).
- Per turn: `TURN_DEADLINE_S = 30`, `TURN_MAX_TOKENS = 1024`. Per level: `MAX_TURNS_PER_LEVEL = 8`. Per game: `MAX_TURNS_PER_GAME = 40`. No turn starts past 75% of the game's budget (the same cutoff as v1's `SYNTH_CUTOFF_FRACTION = 0.25` of time left).
- A turn that fails, times out or returns invalid JSON costs nothing but the time. The move is chosen by slice 1 logic, and an `exec_failed` event is logged with the reason.
- LLM goal guesses enter the GoalBook with source `"llm"`, ahead of heuristic ones. Every rule that excludes, rules out or stalls a goal applies to them unchanged.
- LLM plan steps are executed only through the slice 1 gate: the predictor checks each step, the vetoes apply, and a step the facts can't predict counts as a test (taken, observed, learned).
- GPU: Kaggle runs only, within the 5 GPU hours a day Kameron allowed (2026-10-23). The laptop GPU is not used without a window from Kameron.

## Decisions this plan makes

- **Images:** 64x64 grid upscaled ×8 to 512x512, the standard 16-color ARC palette (hex values in Task 1), padding drawn as black. The diff image is the current frame with unchanged cells darkened to 30% and changed cells at full color with a 1-px white border per changed cell. No PIL: PNG written with `zlib` and `struct` (8-bit palette PNG).
- **Ids in the prompt:** objects are listed as `#<id> c<color> <w>x<h> at (x,y)` using the tracker ids. The model names a click as `"click #<id>"` and a simple action as `"A1"`..`"A7"`. Ids are valid for one turn only.
- **Reply contract (JSON only, one object):**
  ```json
  {"goals": [{"kind": "reach", "a": 12, "b": 7, "why": "..."},
             {"kind": "clear", "color": 9},
             {"kind": "count", "color": 8, "n": 0},
             {"kind": "match", "region_a": [x0,y0,x1,y1], "region_b": [x0,y0,x1,y1]}],
   "plan": [{"action": "A3", "expect": "player moves left"}],
   "test": {"action": "click #4", "question": "does clicking the red tile toggle it?"},
   "notes": "one line"}
  ```
  `a` and `b` are object ids, which the parser maps to sigs. Either `plan` (up to 8 steps) or `test`, never both; if both come back, the plan wins. Unknown keys are ignored.
- **Facts and code from the model:** slice 2 logs the model's proposed facts and code but doesn't store or run them. Facts are learned only from observed moves. Hints from the model and sandboxed code over objects are slice 3 or later work, so the reply contract has no `facts` or `code` keys.
- **Salience wake rules** (spec component 7), checked after every move in this order: level start; a miss that matters (slice 1's 8c rule); plan finished or blocked; a goal excluded, ruled out or stalled; a fact turning broken; no executable plan and no test. Don't wake while a plan holds.
- **Sync or async:** the turn runs on a worker pool. The game keeps acting with slice 1 logic while it's in flight and picks the reply up on a later move, as v1 does. That keeps the 15-minute scorecard idle rule safe and spends no actions waiting. A reply is applied only if it arrives on the same level it was asked on.

## Review Focus

1. Malformed or partial JSON: code fences, trailing prose, an unknown action name, an id that isn't on screen, an empty object. The parser returns what's valid and drops the rest, and never raises. Pinned in Task 3.
2. A reply that arrives after a level change or game over must be dropped and logged, never applied. Pinned in Task 5.
3. Replay offline: the same prompt replays the recorded reply, and a missing recording answers `{}`, so the run continues on slice 1 logic. Pinned in Task 2.
4. An image request against a server that rejects images (HTTP 400) logs `exec_failed` once per game and disables further turns for that game. Pinned in Task 2 and Task 5.
5. Prompt size: with 300 objects the text must stay under 6,000 characters. The object list is capped at the 40 largest-salience objects (movers first, then things changed this level, then the smallest), and the rest are summarized as counts per color. Pinned in Task 4.

---

### Task 1: Screen images

**Files:** Create `arcagent/mind/screen.py`; Test `tests/mind/test_screen.py`

**Interfaces:**
- Produces:
  ```python
  PALETTE: tuple[tuple[int, int, int], ...]  # 16 RGB triples, index = ARC color
  SCALE = 8
  def png(grid: Grid, *, scale: int = SCALE) -> bytes
  def diff_png(before: Grid, after: Grid, *, scale: int = SCALE) -> bytes
  def data_url(png_bytes: bytes) -> str     # "data:image/png;base64,..."
  ```
- PALETTE (hex): `000000 0074D9 FF4136 2ECC40 FFDC00 AAAAAA F012BE FF851B 7FDBFF 870C25 FFFFFF 3D9970 B10DC9 39CCCC 85144B F7F7F7`. Padding (-1) draws as `000000`.

- [ ] **Step 1: Write the failing tests**
```python
def test_png_signature_and_size():
    b = png(np.zeros((64, 64), dtype=np.int8))
    assert b[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", b[16:24])
    assert (w, h) == (512, 512)

def test_png_decodes_back_to_palette_indices():
    g = np.zeros((64, 64), dtype=np.int8); g[3, 5] = 9; g[0, 0] = -1
    rows = decode_palette_png(png(g, scale=1))   # helper in the test using zlib
    assert rows[3][5] == PALETTE[9] and rows[0][0] == (0, 0, 0)

def test_diff_png_darkens_unchanged_and_keeps_changed():
    a = np.zeros((64, 64), dtype=np.int8); b = a.copy(); b[10, 10] = 4
    rows = decode_rgb_png(diff_png(a, b, scale=1))
    assert rows[10][10] == PALETTE[4] and rows[0][0] == (0, 0, 0)

def test_data_url_prefix():
    assert data_url(b"x").startswith("data:image/png;base64,")
```
- [ ] **Step 2: Run, verify fail.** `C:/arc-agent-venv/Scripts/python.exe -m pytest tests/mind/test_screen.py -v` → ModuleNotFoundError.
- [ ] **Step 3: Implement.** `png` writes a palette PNG (color type 3, PLTE chunk of the 16 colors plus index 16 for padding black). `diff_png` writes an RGB PNG (color type 2). The ×8 upscale uses `np.repeat`. At scale ≥ 4, the changed-cell border in `diff_png` is drawn white.
- [ ] **Step 4: Run tests and gates** (ruff format on .py files, ruff check, pyright). PASS.
- [ ] **Step 5: Commit** "Fluid slice 2: screen images".

---

### Task 2: Vision chat with record and replay

**Files:** Create `arcagent/mind/vchat.py`; Test `tests/mind/test_vchat.py`

**Interfaces:**
- Consumes: `Completion`, `parse_completion` from `arcagent/synth/run.py`.
- Produces:
  ```python
  Part = dict[str, object]                     # {"type": "text", "text": ...} or {"type": "image_url", "image_url": {"url": ...}}
  VMessage = dict[str, object]                 # {"role": ..., "content": str | list[Part]}
  VisionChat = Callable[[list[VMessage], int], Completion]
  class ImagesRejected(RuntimeError): ...      # server answered 400 to an image request
  def http_vision_chat(endpoint: str, model: str, *, max_tokens: int, timeout_s: float) -> VisionChat
  def recording(inner: VisionChat, directory: Path) -> VisionChat
  def replaying(directory: Path) -> VisionChat  # missing recording -> Completion(text="{}")
  def scripted(replies: Sequence[str]) -> VisionChat   # offline tests: returns replies in order, then "{}"
  ```
- The request body is like `request_body` in `arcagent/synth/run.py` (temperature from there, thinking off), with `max_tokens` from the argument. The record key is SHA-1 over `json.dumps(messages, sort_keys=True) + str(seed)`, files `<sha1>.json` written atomically like `chat_store.py`. Replay looks up by key only.

- [ ] **Step 1: Write the failing tests**: `scripted` order and exhaustion; `recording` writes one file and `replaying` returns its text; a missing recording replays `"{}"`; `http_vision_chat` against a local stub server (`http.server` on a free port, in a thread) sends `content` parts with an `image_url` and parses the reply; a stub answering 400 raises `ImagesRejected`.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests and gates.**
- [ ] **Step 5: Commit** "Fluid slice 2: vision chat with record and replay".

---

### Task 3: Reply contract

**Files:** Create `arcagent/mind/contract.py`; Test `tests/mind/test_contract.py`

**Interfaces:**
- Consumes: `Goal` (goals.py), `Scene`/`Thing` (objects.py), `Action` (core/types.py), the click-cell rule from `fluid.py` (the cell of a thing nearest its centroid; move it into `objects.py` as `click_cell(thing) -> tuple[int, int]` if it isn't shared yet).
- Produces:
  ```python
  @dataclass(frozen=True, slots=True)
  class PlanItem:
      action: Action; expect: str
  @dataclass(frozen=True, slots=True)
  class Reply:
      goals: tuple[Goal, ...]; plan: tuple[PlanItem, ...]
      test: PlanItem | None; notes: str; dropped: tuple[str, ...]   # reasons for dropped parts
  def parse_reply(text: str, scene: Scene, available: Sequence[int]) -> Reply
  ```
- Rules: take the first `{...}` JSON object in the text (fences and prose around it are allowed). `"A1"`..`"A7"` map to `Action(n)` only if `n` is in `available` and isn't 0 or 6. `"click #<id>"` maps to `Action(6, *click_cell(thing))` only if 6 is in `available` and the id is on screen. Goals map object ids to sigs. A goal or step that fails validation is dropped, with a reason in `dropped`. The plan is capped at 8 steps. Never raises.

- [ ] **Step 1: Write the failing tests**: a valid reply with a reach goal and a 3-step plan; fenced JSON with prose; an unknown id dropped with a reason; an unavailable action dropped; both plan and test returns the plan only; garbage text returns an empty Reply; a 12-step plan is capped at 8; a match goal keeps its regions; clear and count map their colors.
- [ ] **Step 2–5:** fail, implement, pass with gates, commit "Fluid slice 2: reply contract".

---

### Task 4: The executive turn (prompt and call)

**Files:** Create `arcagent/mind/turn.py`; Test `tests/mind/test_turn.py`

**Interfaces:**
- Consumes: Tasks 1–3; `FactMemory.by_tag`/`facts` and `Fact.line()`; `GoalBook.status`; the agent's move-log lines.
- Produces:
  ```python
  SYSTEM_PROMPT: str
  MAX_OBJECTS = 40; MAX_PROMPT_CHARS = 6000; MOVE_LOG_LINES = 8
  @dataclass(frozen=True, slots=True)
  class TurnInput:
      grid: Grid; prev_grid: Grid | None; scene: Scene; available: tuple[int, ...]
      facts: tuple[str, ...]; goals: tuple[tuple[str, str], ...]   # (goal id, status)
      questions: tuple[str, ...]; moves: tuple[str, ...]; level: int; wake: str
  def build_messages(inp: TurnInput) -> list[VMessage]
  def take_turn(chat: VisionChat, inp: TurnInput, *, seed: int) -> Reply   # raises ImagesRejected; other errors -> empty Reply with a dropped reason
  ```
- `SYSTEM_PROMPT`, about 25 lines, says:
  - You are the planner for an agent learning an unknown game from scratch.
  - The first image is the screen. The second is what changed, with changed cells bright.
  - The object list gives exact positions.
  - Infer the goal from what you see and what changed. Propose goal guesses as predicates (reach, clear, count, match).
  - Propose either a short plan, each step with its expected effect, or one test move that answers a question.
  - Never assume the game works like a game you know. Use only these root mechanics as hypotheses: move, block, push, collect, toggle, die, win, spawn, rotate, select, no_effect.
  - Reply with JSON only, in this exact shape (the contract from "Decisions").
- User content: the text (wake reason, level, available actions, object list, facts, goals with statuses, open questions, last moves), the screen image, and the diff image when `prev_grid` exists. The object cap and summary rule follow Review Focus 5.

- [ ] **Step 1: Write the failing tests**: `build_messages` has two image parts when `prev_grid` is given and one otherwise; the text stays under `MAX_PROMPT_CHARS` with 300 objects; the system prompt contains none of a short list of known game names (use a test-local list); `take_turn` with `scripted([valid_json])` returns the parsed Reply; with `scripted(["not json"])` it returns an empty Reply.
- [ ] **Step 2–5:** fail, implement, pass with gates, commit "Fluid slice 2: executive turn".

---

### Task 5: Wire the executive into FluidAgent

**Files:** Modify `arcagent/mind/fluid.py`, `arcagent/mind/config.py` (`CONFIGS["FLUID-LLM"]`: `llm=True, fluid=True`), `arcagent/agent/submit.py` (a fluid config with `llm=True` gets a `VisionChat`: `http_vision_chat` live and wrapped in `recording`, or `replaying` with `--replay`; the turn pool is `MindRun.pool`); `arcagent/mind/runlog.py` docstring. Test `tests/mind/test_fluid_agent.py` (extend), `tests/test_agent.py` or the submit tests that enumerate configs.

**Interfaces:**
- `FluidAgent.__init__(..., chat: VisionChat | None = None, pool: Executor | None = None)`. With no chat, behavior is slice 1 exactly.
- New events: `wake` (`step, reason`), `exec_turn` (`step, call, seconds, goals, plan_len, test, dropped`), `exec_failed` (`step, call, reason`), `exec_stale` (`step, call, asked_level`).
- Applying a reply: goals go to `GoalBook` with source `"llm"` and are put first among the open ones. A plan replaces the current plan: each step's `expect` is the predictor's prediction from the then-current scene; a step with unknowns is taken as a test and the plan continues if the outcome doesn't miss on a mover. A test becomes the next move unless vetoed. Every applied move logs branch `plan` (why `llm plan`) or `learn` (why `llm test: <question>`).
- Budget rules from Global Constraints. `ImagesRejected` disables turns for the rest of the game.

- [ ] **Step 1: Write the failing tests** with the fake games from slice 1 and `scripted` chats:
  - (a) A scripted reply with a reach goal toward the target makes the agent finish the fake level in fewer actions than slice 1 on the same game.
  - (b) A reply arriving after a level-up is logged as `exec_stale` and not applied.
  - (c) `ImagesRejected` happens once and then no more turns run.
  - (d) An empty reply leaves behavior identical to slice 1 (same action sequence on the fake game with a fixed seed).
  - (e) The wake reasons log in order.
  - (f) `CONFIGS["FLUID-LLM"]` exists.
- [ ] **Step 2–5:** fail, implement, pass the full suite with gates, commit "Fluid slice 2: executive wired into the agent".

---

### Task 6: Notebook and Kaggle eval

**Files:** Modify `scripts/kaggle_mind_notebook.py` (run `--config FLUID-LLM`; the EXPLORER fallback becomes `FLUID` when the model server fails), `scripts/kaggle_synth_notebook.py` `SERVE_CELL` (add `"--limit-mm-per-prompt", '{"image": 2}'`), `tests/test_kaggle_mind_notebook.py`, `docs/experiment-log.md`.

- [ ] **Step 1:** Update the notebook builder and its tests; pass the full suite and gates; commit "Fluid slice 2: notebook runs FLUID-LLM".
- [ ] **Step 2 (controller):** Push the kernel per `docs/agent-submission.md` for an interactive commit on the 25 public games, within the 5 GPU-hour daily budget. Pull the output to `bench/evals/<date>-fluid-llm/`.
- [ ] **Step 3 (controller):** Score with `scripts/mind_metrics.py`. Compare per game with v1 and slice 1, and replay offline with `--replay` to confirm it reproduces. Log the row in `docs/experiment-log.md`. A second paired GPU run before any claim of a gain.
