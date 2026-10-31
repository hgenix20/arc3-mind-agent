"""Long-term store: rules, sequences and typical moves that outlive one game.

Spec 4.2 names a long-term store that holds rule records with scope ``long``, sequence
records and the index, lives across games and runs, and is never cleared (records are
only retired). Spec 4.12b writes to it at every level completion and 4.12c at game end;
the 4.7 horizon rule reads the typical moves per level from it. This module is that store.

Records are plain dataclasses. A rule is keyed by its canonical key, ``(action_role,
shape_class, context_tag, effect_kind)``, which never carries a colour index (spec 4.2,
Part 1 D1): two games with the same mechanic in different colours share a key, so their
evidence merges in ``LongTermStore.promote``. ``canonical_key`` builds the key from a
``facts.FactKey`` and its effects; ``statement_for`` words the same key as one sentence.

Contract. The file is one JSON object ``{"version", "rules", "sequences", "completions",
"net_state"}``; tuples become lists on save and tuples again on load. ``load`` treats a
missing file as an empty store and moves a file it cannot read to ``<name>.bad``, then
starts empty (``load_error`` says why). ``save`` writes a temp file beside the target and
``os.replace``s it, so a reader never sees a half-written store. Every public method holds
``lock`` (a re-entrant lock), and a caller holds it across ``load``, changes and ``save``
to make the three one step. The public dicts are the live content; writing to them
directly bypasses the lock and the bounds.

Bounds: ``MAX_RULES`` (the lowest probability is dropped when full), ``MAX_SEQUENCES``
(fewest successes dropped), ``MAX_COMPLETIONS_PER_LEVEL`` (oldest dropped) and
``MAX_GAMES_SEEN`` per rule. Statements are cut at ``STATEMENT_MAX_CHARS``.

Failure modes: ``ValueError`` for negative counts or malformed moves; ``OSError`` from the
file system is raised as is (a missing file is the one case handled). Nothing here logs;
the agent that calls the store logs ``load_error`` and what it promoted.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast, get_args

from arcagent.core.types import COORD_ACTION_ID
from arcagent.mind.facts import (
    BAND_CERTAIN,
    BAND_LOW,
    PRIOR_A,
    PRIOR_B,
    PROMOTE_HITS,
    SPAWN,
    Effect,
    FactKey,
    State,
)
from arcagent.mind.kinds import base_sig

RuleKind = Literal["object", "action", "event", "experience"]
RuleState = State  # hypothesis, fact or retired: the same vocabulary as a Fact (spec 4.2)
Scope = Literal["working", "long"]
Source = Literal["observed", "loaded", "label"]
CanonicalKey = tuple[str, str, str, str]  # (action_role, shape_class, context_tag, effect_kind)

STORE_VERSION = 1
MAX_RULES = 5000  # rules kept; the lowest probability goes when a new key arrives
MAX_SEQUENCES = 2000  # sequence records kept; the fewest successes goes
MAX_COMPLETIONS_PER_LEVEL = 50  # move counts kept per game and level; the oldest goes
MAX_GAMES_SEEN = 500  # game ids kept per rule; the oldest goes
STATEMENT_MAX_CHARS = 120
CLICK_ROLE = "click"  # the action role of every click label, whatever was clicked
NO_HARM = "none"
_CLICK_LABEL = f"A{COORD_ACTION_ID}"
_COLOUR_PREFIX = re.compile(r"^c\d+:")  # the colour half of a sig, "c<colour>:"
_COLOUR_CONTEXT = re.compile(r"^(touch|near):-?\d+$")  # contexts that name a colour
_FIXED_SUBJECTS = frozenset({"self", "any", SPAWN})
_RULE_KINDS: tuple[str, ...] = get_args(RuleKind)
_RULE_STATES: tuple[str, ...] = get_args(RuleState)
_SCOPES: tuple[str, ...] = get_args(Scope)
_SOURCES: tuple[str, ...] = get_args(Source)


@dataclass(slots=True)
class RuleRecord:
    """One regularity, in the shape every store shares (spec 4.2 rule record).

    ``canonical_key`` is what merges records across games; ``id`` is stable for the rule
    (``rule_id`` derives one from the key). ``statement`` is the rule in plain words for
    the briefing; ``effect`` the effect in the words of ``effect_text``. ``probability``
    and ``state`` are derived from ``confirmations`` and ``contradictions`` by
    ``rule_probability`` and ``rule_state``; ``promote`` recomputes both, so a stored
    record never carries a caller's guess. ``games_seen`` always holds ``created_game``.
    ``harm`` is ``"none"`` or the harm kind the adverse book names (spec 4.10). ``source``
    says where the rule came from: observed in play, loaded from this store, or an
    agent-made label (spec 4.11a).
    """

    id: str
    kind: RuleKind
    statement: str
    canonical_key: CanonicalKey
    effect: str
    probability: float
    confirmations: int
    contradictions: int
    state: RuleState
    scope: Scope
    games_seen: list[str]
    created_game: str
    created_level: int
    last_revised: float
    harm: str
    source: Source


@dataclass(slots=True)
class SequenceRecord:
    """A move sequence that served a goal (spec 4.2 sequence record).

    ``moves`` are ``[action id, x, y]`` triples (``-1`` coordinates for a non-click).
    ``runs`` counts the times it was executed and ``successes`` the times it reached its
    goal; ``add_sequence`` sums both when the same ``id`` comes again. ``deterministic``
    marks a named skill's sequence (spec 4.9).
    """

    id: str
    goal_id: str
    moves: list[list[int]]
    game: str
    level: int
    runs: int
    successes: int
    deterministic: bool


# --- probability and state ------------------------------------------------------------


def rule_probability(confirmations: int, contradictions: int) -> float:
    """Beta mean ``(confirmations + 1) / (confirmations + contradictions + 2)`` to four
    decimals (spec 4.4, flat prior); ``ValueError`` on a negative count."""
    if confirmations < 0 or contradictions < 0:
        raise ValueError(f"negative count: {confirmations}/{contradictions}")
    total = confirmations + contradictions + PRIOR_A + PRIOR_B
    return round((confirmations + PRIOR_A) / total, 4)


def rule_state(probability: float, confirmations: int, contradictions: int) -> RuleState:
    """``retired`` below ``BAND_LOW``; ``fact`` at or above ``BAND_CERTAIN`` with at least
    ``PROMOTE_HITS`` confirmations and no contradiction; else ``hypothesis`` (spec 4.2
    status rules, 4.4b). The store has no count since the last refinement, so "none since"
    reads as a lifetime zero here.
    """
    if probability < BAND_LOW:
        return "retired"
    if probability >= BAND_CERTAIN and confirmations >= PROMOTE_HITS and contradictions == 0:
        return "fact"
    return "hypothesis"


# --- canonical key --------------------------------------------------------------------


def role_of(label: str) -> str:
    """The action role of a fact label: every ``A6`` label (``A6``, ``A6:none``,
    ``A6:<sig>@<ring>``) is ``click``; ``A3``, ``event`` and ``any`` stay as they are."""
    if label == _CLICK_LABEL or label.startswith(_CLICK_LABEL + ":"):
        return CLICK_ROLE
    return label


def shape_of(subject: str) -> str:
    """The shape class of a fact subject: ``c3:<shape>`` and ``kc3:<shape>`` give
    ``<shape>``; ``self``, ``any`` and ``spawn`` stay as they are, as does any subject
    with no colour prefix (``level``, ``level_up``)."""
    if subject in _FIXED_SUBJECTS:
        return subject
    return _COLOUR_PREFIX.sub("", base_sig(subject), count=1)


def tag_of(context: str) -> str:
    """The context tag of a fact context: ``touch:<c>`` and ``near:<c>`` become ``touch``
    and ``near``; a ``c<colour>:`` prefix is dropped; other contexts (``any``, ``blocked``,
    ``selected:1,0``) stay as they are, since their numbers are offsets, not colours."""
    match = _COLOUR_CONTEXT.match(context)
    if match:
        return match.group(1)
    return _COLOUR_PREFIX.sub("", context, count=1)


def canonical_key(fact_key: FactKey, effects: Sequence[Effect]) -> CanonicalKey:
    """``(action_role, shape_class, context_tag, effect_kind)`` for a fact key and its
    effects; ``effect_kind`` is the first effect's kind, ``none`` with no effects. Never
    contains a colour index (spec 4.2, D1)."""
    label, subject, context = fact_key
    kind = effects[0].kind if effects else "none"
    return (role_of(label), shape_of(subject), tag_of(context), kind)


def rule_id(key: CanonicalKey) -> str:
    """A stable id for a canonical key: ``r`` plus twelve hex digits of its SHA-1."""
    digest = hashlib.sha1("\x1f".join(key).encode("utf-8"), usedforsecurity=False)
    return "r" + digest.hexdigest()[:12]


# --- words ------------------------------------------------------------------------------

_ROLE_WORDS: dict[str, str] = {CLICK_ROLE: "clicking", "any": "any action", "event": "an event"}
_CONTEXT_WORDS: dict[str, str] = {
    "any": "",
    "blocked": " when blocked",
    "pushed": " when pushed",
    "active": " while active",
    "touch": " while touching something",
    "near": " while near something",
    "selected": " while selected",
    "sel": " while something is selected",
    "between": " while between the selection and the click",
}


def _object_words(shape: str) -> str:
    if shape == "self":
        return "it"
    if shape == "any":
        return "any thing"
    if shape == SPAWN:
        return "a new thing"
    if shape in ("level", "level_up"):
        return "the level"
    return f"a {shape} thing"


def _context_words(tag: str) -> str:
    head = tag.partition(":")[0]
    words = _CONTEXT_WORDS.get(head)
    return f" in context {tag}" if words is None else words


def _effect_words(effect: Effect, obj: str) -> str:
    sig_shape = shape_of(effect.sig) if effect.sig else ""
    match effect.kind:
        case "move":
            out = f"moves {obj} by ({effect.dx}, {effect.dy})"
        case "slide":
            out = f"slides {obj} by ({effect.dx}, {effect.dy}) until blocked"
        case "teleport":
            out = f"teleports {obj}"
        case "appear":
            out = f"makes a {sig_shape} thing appear" if sig_shape else "makes a new thing appear"
        case "disappear":
            out = f"makes {obj} disappear"
        case "recolor":
            out = f"recolours {obj}"
        case "reshape":
            out = f"reshapes {obj}" + (f" to {sig_shape}" if sig_shape else "")
        case "toggle_group":
            out = f"recolours every {sig_shape} thing" if sig_shape else "recolours a group"
        case "none":
            out = f"does nothing to {obj}"
        case "game_over":
            out = "ends the game" if obj == "it" else f"ends the game for {obj}"
        case "level_up":
            out = "completes the level"
        case "reset_level":
            out = "puts the level back as it started"
    return out


def statement_for(fact_key: FactKey, effects: Sequence[Effect]) -> str:
    """One plain sentence for the canonical key of ``fact_key`` and its first effect, for
    example ``A3 moves a <shape> thing by (1, 0)`` or ``clicking a thing recolours it``.
    No colour index; at most ``STATEMENT_MAX_CHARS`` characters."""
    role, shape, tag, _ = canonical_key(fact_key, effects)
    actor = _ROLE_WORDS.get(role, role)
    obj = _object_words(shape)
    effect = _effect_words(effects[0], obj) if effects else f"has no recorded effect on {obj}"
    core = f"a thing {effect}" if shape == "self" else effect
    text = f"{actor} {core}{_context_words(tag)}"
    if len(text) <= STATEMENT_MAX_CHARS:
        return text
    return text[: STATEMENT_MAX_CHARS - 3] + "..."


def _effect_word(effect: Effect) -> str:
    match effect.kind:
        case "move" | "slide":
            out = f"{effect.kind} ({effect.dx}, {effect.dy})"
        case "appear" | "reshape" | "toggle_group":
            out = f"{effect.kind} {shape_of(effect.sig)}" if effect.sig else effect.kind
        case _:
            out = effect.kind
    return out


def effect_text(effects: Sequence[Effect]) -> str:
    """Colour-free words for ``RuleRecord.effect``: ``move (1, 0)``, ``appear <shape>``,
    ``recolor``, joined by ``; ``; ``none`` with no effects."""
    if not effects:
        return "none"
    return "; ".join(_effect_word(e) for e in effects)


# --- merging ----------------------------------------------------------------------------


def _union(*groups: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for group in groups:
        for game in group:
            seen.setdefault(game, None)
    games = list(seen)
    return games[-MAX_GAMES_SEEN:] if len(games) > MAX_GAMES_SEEN else games


def _derived(confirmations: int, contradictions: int) -> tuple[float, RuleState]:
    p = rule_probability(confirmations, contradictions)
    return p, rule_state(p, confirmations, contradictions)


def _check_key(key: CanonicalKey) -> None:
    """``ValueError`` when a key part still names a colour (spec 4.2, D1)."""
    for part in key:
        if _COLOUR_PREFIX.match(part) or _COLOUR_CONTEXT.match(part):
            raise ValueError(f"canonical key {key} carries a colour index in {part!r}")


def _fresh(record: RuleRecord, now: float) -> RuleRecord:
    """A new store entry from ``record``: scope long, derived fields recomputed."""
    p, state = _derived(record.confirmations, record.contradictions)
    return replace(
        record,
        probability=p,
        state=state,
        scope="long",
        games_seen=_union([record.created_game], record.games_seen),
        last_revised=now,
    )


def _merge(current: RuleRecord, incoming: RuleRecord, now: float) -> RuleRecord:
    """``current`` with ``incoming``'s evidence added; identity fields stay ``current``'s."""
    confirmations = current.confirmations + incoming.confirmations
    contradictions = current.contradictions + incoming.contradictions
    p, state = _derived(confirmations, contradictions)
    harm = current.harm if current.harm != NO_HARM else incoming.harm
    return replace(
        current,
        confirmations=confirmations,
        contradictions=contradictions,
        probability=p,
        state=state,
        scope="long",
        games_seen=_union(current.games_seen, incoming.games_seen, [incoming.created_game]),
        last_revised=now,
        harm=harm,
    )


def _rule_rank(record: RuleRecord) -> tuple[float, int, str]:
    """Lowest first: the record to drop when the store is full."""
    return (record.probability, record.confirmations, record.id)


def _sequence_rank(record: SequenceRecord) -> tuple[int, int, str]:
    return (record.successes, record.runs, record.id)


def _check_sequence(record: SequenceRecord) -> None:
    if record.runs < 0 or record.successes < 0 or record.successes > record.runs:
        raise ValueError(f"successes {record.successes} must be within 0..runs {record.runs}")
    if any(len(move) != 3 for move in record.moves):
        raise ValueError("moves must be [id, x, y] triples")


# --- the store ------------------------------------------------------------------------


class LongTermStore:
    """The long-term store of spec 4.2, one JSON file (module docstring for the contract).

    ``rules`` by canonical key, ``sequences`` by id, ``completions`` as game to level to
    the moves of each completion, and ``net_state`` kept opaque for the inhibition
    network (spec 4.10b). ``clock`` stamps ``last_revised`` and is ``time.time`` unless a
    test injects one.
    """

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self.lock = threading.RLock()
        self.rules: dict[CanonicalKey, RuleRecord] = {}
        self.sequences: dict[str, SequenceRecord] = {}
        self.completions: dict[str, dict[int, list[int]]] = {}
        self.net_state: dict[str, object] | None = None
        self.load_error: str | None = None
        self._clock = clock

    # -- file --

    def load(self) -> None:
        """Replace the content with the file's. A missing file gives an empty store. A
        file that is not a store this module wrote is moved to ``<name>.bad`` and the
        store starts empty, with the reason in ``load_error``."""
        with self.lock:
            self.load_error = None
            try:
                text = self.path.read_text(encoding="utf-8")
            except FileNotFoundError:
                self._reset()
                return
            try:
                rules, sequences, completions, net_state = _decode(text)
            except ValueError as exc:
                os.replace(self.path, self.path.with_name(self.path.name + ".bad"))
                self.load_error = f"{type(exc).__name__}: {exc}"[:300]
                self._reset()
                return
            self.rules = rules
            self.sequences = sequences
            self.completions = completions
            self.net_state = net_state
            self._bound_rules()
            self._bound_sequences()

    def save(self) -> None:
        """Write the store to ``path`` through a temp file in the same directory and
        ``os.replace``, creating parent directories; the temp file is removed on failure."""
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(self._payload())
            fd, tmp = tempfile.mkstemp(
                dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(text)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, self.path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise

    def _reset(self) -> None:
        self.rules = {}
        self.sequences = {}
        self.completions = {}
        self.net_state = None

    def _payload(self) -> dict[str, object]:
        return {
            "version": STORE_VERSION,
            "rules": [asdict(r) for r in sorted(self.rules.values(), key=lambda r: r.id)],
            "sequences": [asdict(s) for s in sorted(self.sequences.values(), key=lambda s: s.id)],
            "completions": {
                game: {str(level): list(moves) for level, moves in levels.items()}
                for game, levels in self.completions.items()
            },
            "net_state": self.net_state,
        }

    # -- rules --

    def promote(self, record: RuleRecord) -> RuleRecord:
        """Insert ``record`` with scope long, or merge it into the record with the same
        canonical key: counts summed, ``games_seen`` united, probability and state
        recomputed, ``last_revised`` set to now. Counts add up, so pass the evidence
        gathered since the last promote of the same rule, not the lifetime counts again.
        Returns the stored record; when the store was full and this record had the lowest
        probability, it is the one dropped and is returned unstored. ``ValueError`` when
        the key still carries a colour index."""
        key = record.canonical_key
        _check_key(key)
        now = self._clock()
        with self.lock:
            current = self.rules.get(key)
            stored = _fresh(record, now) if current is None else _merge(current, record, now)
            self.rules[key] = stored
            self._bound_rules()
            return stored

    def rules_for(
        self,
        *,
        action_role: str | None = None,
        shape_class: str | None = None,
        context_tag: str | None = None,
        effect_kind: str | None = None,
        limit: int = 20,
    ) -> list[RuleRecord]:
        """Rules whose key matches every given field (``None`` matches anything), by
        probability descending then id, at most ``limit``."""
        wanted = (action_role, shape_class, context_tag, effect_kind)
        with self.lock:
            found = [
                r
                for key, r in self.rules.items()
                if all(w is None or w == k for w, k in zip(wanted, key, strict=True))
            ]
        found.sort(key=lambda r: (-r.probability, r.id))
        return found[: max(0, limit)]

    def lessons(self, limit: int = 12) -> list[str]:
        """One line per rule, ``"<statement> (p=0.8571, 3 games)"`` (``1 game`` for one),
        by probability then games seen (both descending) then id, at most ``limit``."""
        with self.lock:
            ranked = sorted(
                self.rules.values(), key=lambda r: (-r.probability, -len(r.games_seen), r.id)
            )
        out: list[str] = []
        for r in ranked[: max(0, limit)]:
            n = len(r.games_seen)
            out.append(f"{r.statement} (p={r.probability:.4f}, {n} game{'' if n == 1 else 's'})")
        return out

    def _bound_rules(self) -> None:
        excess = len(self.rules) - MAX_RULES
        if excess <= 0:
            return
        if excess == 1:
            victim = min(self.rules.values(), key=_rule_rank)
            del self.rules[victim.canonical_key]
            return
        keep = sorted(self.rules.values(), key=_rule_rank, reverse=True)[:MAX_RULES]
        self.rules = {r.canonical_key: r for r in keep}

    # -- completions (the horizon rule, spec 4.7) --

    def record_completion(self, game: str, level: int, moves: int) -> None:
        """Note that ``game``'s level ``level`` was completed in ``moves`` moves; the last
        ``MAX_COMPLETIONS_PER_LEVEL`` are kept. ``ValueError`` on a negative value."""
        if level < 0:
            raise ValueError(f"level must be 0 or more, got {level}")
        if moves < 0:
            raise ValueError(f"moves must be 0 or more, got {moves}")
        with self.lock:
            runs = self.completions.setdefault(game, {}).setdefault(level, [])
            runs.append(moves)
            if len(runs) > MAX_COMPLETIONS_PER_LEVEL:
                del runs[: len(runs) - MAX_COMPLETIONS_PER_LEVEL]

    def typical_moves(self, game: str, level: int) -> float | None:
        """Mean moves over ``game``'s completions of ``level``; ``None`` with none. The
        horizon rule (spec 4.7) rounds this up; the caller does the rounding."""
        with self.lock:
            runs = self.completions.get(game, {}).get(level, [])
            return sum(runs) / len(runs) if runs else None

    def typical_for_level(self, level: int) -> float | None:
        """Mean moves over every game's completions of level index ``level``, pooled;
        ``None`` with none."""
        with self.lock:
            runs = [m for levels in self.completions.values() for m in levels.get(level, [])]
        return sum(runs) / len(runs) if runs else None

    # -- sequences --

    def add_sequence(self, record: SequenceRecord) -> None:
        """Insert ``record``, or add its ``runs`` and ``successes`` to the stored record
        with the same id (whose moves and goal stay). ``ValueError`` when successes exceed
        runs, a count is negative, or a move is not an ``[id, x, y]`` triple."""
        _check_sequence(record)
        with self.lock:
            current = self.sequences.get(record.id)
            if current is None:
                stored = replace(record, moves=[list(move) for move in record.moves])
            else:
                stored = replace(
                    current,
                    runs=current.runs + record.runs,
                    successes=current.successes + record.successes,
                )
            self.sequences[record.id] = stored
            self._bound_sequences()

    def sequences_for(self, goal_id: str, limit: int = 5) -> list[SequenceRecord]:
        """Sequences that served ``goal_id``, by successes descending then id, at most
        ``limit``."""
        with self.lock:
            found = [s for s in self.sequences.values() if s.goal_id == goal_id]
        found.sort(key=lambda s: (-s.successes, s.id))
        return found[: max(0, limit)]

    def _bound_sequences(self) -> None:
        if len(self.sequences) <= MAX_SEQUENCES:
            return
        keep = sorted(self.sequences.values(), key=_sequence_rank, reverse=True)[:MAX_SEQUENCES]
        self.sequences = {s.id: s for s in keep}


# --- decoding --------------------------------------------------------------------------


class _CorruptError(ValueError):
    """The file is not a store this module wrote."""


def _as_dict(value: object, what: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _CorruptError(f"{what}: expected an object")
    out: dict[str, object] = {}
    for key, item in cast(dict[object, object], value).items():
        if not isinstance(key, str):
            raise _CorruptError(f"{what}: key {key!r} is not a string")
        out[key] = item
    return out


def _as_list(value: object, what: str) -> list[object]:
    if not isinstance(value, list):
        raise _CorruptError(f"{what}: expected a list")
    return list(cast(list[object], value))


def _field(obj: dict[str, object], key: str, what: str) -> object:
    if key not in obj:
        raise _CorruptError(f"{what}: missing {key}")
    return obj[key]


def _str(obj: dict[str, object], key: str, what: str) -> str:
    value = _field(obj, key, what)
    if not isinstance(value, str):
        raise _CorruptError(f"{what}: {key} is not a string")
    return value


def _int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _CorruptError(f"{what}: not an integer")
    return value


def _int_field(obj: dict[str, object], key: str, what: str) -> int:
    return _int(_field(obj, key, what), f"{what}: {key}")


def _float_field(obj: dict[str, object], key: str, what: str) -> float:
    value = _field(obj, key, what)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _CorruptError(f"{what}: {key} is not a number")
    return float(value)


def _bool_field(obj: dict[str, object], key: str, what: str) -> bool:
    value = _field(obj, key, what)
    if not isinstance(value, bool):
        raise _CorruptError(f"{what}: {key} is not a boolean")
    return value


def _choice(obj: dict[str, object], key: str, what: str, options: tuple[str, ...]) -> str:
    value = _str(obj, key, what)
    if value not in options:
        raise _CorruptError(f"{what}: {key} {value!r} is not one of {options}")
    return value


def _str_list(value: object, what: str) -> list[str]:
    out: list[str] = []
    for item in _as_list(value, what):
        if not isinstance(item, str):
            raise _CorruptError(f"{what}: item is not a string")
        out.append(item)
    return out


def _key_from(value: object, what: str) -> CanonicalKey:
    parts = _str_list(value, what)
    if len(parts) != 4:
        raise _CorruptError(f"{what}: canonical key needs 4 parts, got {len(parts)}")
    return (parts[0], parts[1], parts[2], parts[3])


def _moves_from(value: object, what: str) -> list[list[int]]:
    out: list[list[int]] = []
    for move in _as_list(value, what):
        parts = [_int(p, what) for p in _as_list(move, what)]
        if len(parts) != 3:
            raise _CorruptError(f"{what}: a move needs 3 parts, got {len(parts)}")
        out.append(parts)
    return out


def _rule_from(obj: dict[str, object]) -> RuleRecord:
    what = "rule"
    return RuleRecord(
        id=_str(obj, "id", what),
        kind=cast(RuleKind, _choice(obj, "kind", what, _RULE_KINDS)),
        statement=_str(obj, "statement", what),
        canonical_key=_key_from(_field(obj, "canonical_key", what), what),
        effect=_str(obj, "effect", what),
        probability=_float_field(obj, "probability", what),
        confirmations=_int_field(obj, "confirmations", what),
        contradictions=_int_field(obj, "contradictions", what),
        state=cast(RuleState, _choice(obj, "state", what, _RULE_STATES)),
        scope=cast(Scope, _choice(obj, "scope", what, _SCOPES)),
        games_seen=_str_list(_field(obj, "games_seen", what), what),
        created_game=_str(obj, "created_game", what),
        created_level=_int_field(obj, "created_level", what),
        last_revised=_float_field(obj, "last_revised", what),
        harm=_str(obj, "harm", what),
        source=cast(Source, _choice(obj, "source", what, _SOURCES)),
    )


def _sequence_from(obj: dict[str, object]) -> SequenceRecord:
    what = "sequence"
    record = SequenceRecord(
        id=_str(obj, "id", what),
        goal_id=_str(obj, "goal_id", what),
        moves=_moves_from(_field(obj, "moves", what), what),
        game=_str(obj, "game", what),
        level=_int_field(obj, "level", what),
        runs=_int_field(obj, "runs", what),
        successes=_int_field(obj, "successes", what),
        deterministic=_bool_field(obj, "deterministic", what),
    )
    try:
        _check_sequence(record)
    except ValueError as exc:
        raise _CorruptError(f"{what}: {exc}") from exc
    return record


def _completions_from(value: object) -> dict[str, dict[int, list[int]]]:
    out: dict[str, dict[int, list[int]]] = {}
    for game, levels in _as_dict(value, "completions").items():
        per_level: dict[int, list[int]] = {}
        for level_text, runs in _as_dict(levels, f"completions of {game}").items():
            if not level_text.isdigit():
                raise _CorruptError(f"completions of {game}: level {level_text!r} is not a number")
            moves = [_int(m, "completion moves") for m in _as_list(runs, "completion moves")]
            if any(m < 0 for m in moves):
                raise _CorruptError(f"completions of {game}: negative moves")
            per_level[int(level_text)] = moves[-MAX_COMPLETIONS_PER_LEVEL:]
        out[game] = per_level
    return out


def _decode(
    text: str,
) -> tuple[
    dict[CanonicalKey, RuleRecord],
    dict[str, SequenceRecord],
    dict[str, dict[int, list[int]]],
    dict[str, object] | None,
]:
    """The four parts of a store file; ``ValueError`` (``_CorruptError`` or a JSON error) when
    the text is not one."""
    top = _as_dict(json.loads(text), "store")
    version = _int_field(top, "version", "store")
    if version != STORE_VERSION:
        raise _CorruptError(f"store: version {version} is not {STORE_VERSION}")
    rules: dict[CanonicalKey, RuleRecord] = {}
    for item in _as_list(_field(top, "rules", "store"), "rules"):
        record = _rule_from(_as_dict(item, "rule"))
        if record.canonical_key in rules:
            raise _CorruptError(f"rules: duplicate key {record.canonical_key}")
        rules[record.canonical_key] = record
    sequences: dict[str, SequenceRecord] = {}
    for item in _as_list(_field(top, "sequences", "store"), "sequences"):
        record = _sequence_from(_as_dict(item, "sequence"))
        if record.id in sequences:
            raise _CorruptError(f"sequences: duplicate id {record.id}")
        sequences[record.id] = record
    completions = _completions_from(_field(top, "completions", "store"))
    net_raw = top.get("net_state")
    net_state = None if net_raw is None else _as_dict(net_raw, "net_state")
    return rules, sequences, completions, net_state
