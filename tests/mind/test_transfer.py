"""CarryTransfer: what it loads, when it calls a failure, and the goal-evidence hand-off."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from arcagent.core.types import Action, Grid
from arcagent.mind.config import CONFIGS
from arcagent.mind.memory import GameStore
from arcagent.mind.modules.transfer import CarryTransfer
from arcagent.mind.predictive import Belief
from arcagent.mind.sandbox import ProgramSandbox
from arcagent.mind.synthesizer import Synthesizer, SynthResult
from arcagent.mind.types import Program
from arcagent.synth.prompt import Message
from arcagent.synth.run import Completion
from tests.mind.test_mind_agent import (
    TRUE_RULES,
    Corridor,
    counting_chat,
    make_agent,
    play,
)


class LoadAll:
    """A sandbox that loads every program except those whose code is ``BAD``."""

    def __init__(self) -> None:
        self.dropped: dict[str, str] = {}
        self.loaded: list[str] = []

    def load(self, pid: str, code: str) -> str | None:
        if code == "BAD":
            return "does not load"
        self.loaded.append(pid)
        return None

    def unload(self, pid: str) -> None:
        return None

    def predict(self, pids: Sequence[str], grid: Grid, action: Action) -> dict[str, str | None]:
        return dict.fromkeys(pids)


def prog(pid: str, code: str = "ok", origin: str = "llm") -> Program:
    assert origin in ("llm", "transfer")
    return Program(pid, code, False, origin)  # pyright: ignore[reportArgumentType]  # asserted above


def test_carry_loads_procedural_and_held_with_reset_fit() -> None:
    store = GameStore()
    store.procedural["p1"] = (prog("p1"), 9, 10)
    store.held["h1"] = prog("h1")
    store.procedural["bad"] = (prog("bad", "BAD"), 5, 5)
    belief = Belief(LoadAll())
    carried = CarryTransfer().carry(store, belief)
    assert carried == ["h1", "p1"]
    for pid in carried:
        state = belief.programs[pid]
        assert (state.hits, state.trials) == (0, 0)
        assert state.program.origin == "transfer"
    assert store.procedural["p1"][0].origin == "llm"  # the store keeps its own copy
    assert "bad" not in belief.programs


def test_check_fails_only_after_check_steps_without_a_fit() -> None:
    store = GameStore()
    store.held["h1"] = prog("h1")
    belief = Belief(LoadAll())
    transfer = CarryTransfer(check_steps=5, admit_fit=0.8)
    assert transfer.check(belief, 100) is False  # nothing carried yet
    transfer.carry(store, belief)
    assert transfer.check(belief, 4) is False  # too early
    assert transfer.check(belief, 5) is True  # no fit at the deadline
    state = belief.programs["h1"]
    state.hits, state.trials = 3, 5
    assert transfer.check(belief, 5) is True  # 0.6 is below the bar
    state.hits, state.trials = 4, 5
    assert transfer.check(belief, 50) is False
    belief.remove("h1")
    assert transfer.check(belief, 50) is True  # the carried program is gone
    assert transfer.carry(GameStore(), belief) == []
    assert transfer.check(belief, 50) is False  # a new, empty carry replaces the set


class RecordingSynth(Synthesizer):
    """Delegates to the real synthesizer and records the goal evidence it was handed."""

    def __init__(self, chat: object, k: int) -> None:
        super().__init__(chat, k=k)  # pyright: ignore[reportArgumentType]  # the chat protocol
        self.evidence: list[int] = []

    def propose(  # noqa: PLR0917  # mirrors the Synthesizer.propose signature
        self,
        game: str,
        call_no: int,
        steps: Sequence[object],
        win_levels: int,
        goal_evidence: Sequence[tuple[Grid, Grid]],
        repair: object,
        sandbox: ProgramSandbox,
        seed: int,
    ) -> SynthResult:
        self.evidence.append(len(goal_evidence))
        return super().propose(
            game,
            call_no,
            steps,  # pyright: ignore[reportArgumentType]  # forwarded unchanged
            win_levels,
            goal_evidence,
            repair,  # pyright: ignore[reportArgumentType]  # forwarded unchanged
            sandbox,
            seed,
        )


def test_agent_passes_goal_evidence_to_synthesizer(tmp_path: Path) -> None:
    chat, _ = counting_chat(TRUE_RULES)

    def wrapped(messages: list[Message], seed: int) -> Completion:
        return chat(messages, seed)

    with ProgramSandbox(timeout_s=2.0) as sandbox:
        agent, _ = make_agent(tmp_path, sandbox, CONFIGS["FULL-transfer"], wrapped)
        synth = RecordingSynth(wrapped, CONFIGS["FULL-transfer"].k)
        agent._synth = synth  # pyright: ignore[reportPrivateUsage]  # swap in the recorder
        game = Corridor(win_levels=3)
        play(agent, game, 150)
        agent.close()
    assert game.levels >= 2
    assert synth.evidence[0] == 0  # no level cleared yet
    assert synth.evidence[-1] >= 1  # later calls carry the cleared level's goal pair
