"""CLI for the rule-synthesis experiment.

    python -m arcagent.synth.run --env-dir C:\\arc-study\\data\\environment_files \\
        --games ls20 ft09 vc33 --steps 120 --rounds 3 \\
        --endpoint http://127.0.0.1:8001/v1 --model <name> \\
        --max-transitions 40 --out bench/synth/results.json

Per game: collect a trace, compute the identity baseline, then up to
``--rounds`` rounds of synthesize (round 1) or repair (later rounds), each
verified out of process. Repair stops early once held-out exact rate reaches
``REPAIR_TARGET``. Outputs live next to ``--out``: ``traces/``, ``code/``,
``prompts/``. The JSON schema is documented in ``docs/synth-experiment.md``.

``--dry-run`` writes the prompts and calls nothing. ``--fake`` replaces the
model with the identity candidate so the whole pipeline runs offline.

Failure handling: an HTTP or decoding error ends that game's rounds and is
recorded in the round's ``error`` field; a reply without a code fence is
scored as a candidate that fails to load and also ends that game's rounds
(``no_code`` is true), since a repair turn has no code to repair. Other games
still run.

Requests disable the model's thinking channel (``chat_template_kwargs``), and
every round records ``finish_reason``, ``reasoning_chars`` and ``reply_chars``
and saves the raw reply under ``replies/``: the v4 Kaggle run returned 75
replies that all hit ``max_tokens`` with empty content, and nothing in the
output could show why.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from arcagent.synth.prompt import Message, build_messages, extract_code
from arcagent.synth.trace import Trace, collect_trace
from arcagent.synth.verify import CELLS, IDENTITY_CODE, identity_exact_rate, verify

REPAIR_TARGET = 0.95
TEMPERATURE = 0.2
# 16000 leaves room for a long answer even if a template ignores ENABLE_THINKING;
# the v4 run spent all of a 6000 budget thinking and returned no content at all.
MAX_TOKENS = 16000
ENABLE_THINKING = False
# A reply without a code fence is re-sampled once with the next seed: in Kaggle run v7 one
# repair reply cycled between two formulas for 16000 tokens and never reached the code.
NO_CODE_RETRIES = 1
HTTP_TIMEOUT_S = 900.0
DEFAULT_ENV_DIR = r"C:\arc-study\data\environment_files"
NO_CODE = "# the model reply contained no code fence\n"


@dataclass(frozen=True, slots=True)
class Completion:
    """One model reply with its accounting.

    ``finish_reason`` is the server's value ("stop", "length", ...) or "" when
    absent; ``reasoning_chars`` is the length of ``reasoning_content`` when the
    server splits thinking out of the reply, else 0.
    """

    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    finish_reason: str = ""
    reasoning_chars: int = 0


Chat = Callable[[list[Message], int], Completion]
"""``chat(messages, seed)``: the seed makes one sample reproducible on a server that
honors it (vLLM does); identical prompts gave different replies across Kaggle runs v6/v7."""


def request_seed(trace_seed: int, round_no: int, attempt: int) -> int:
    """Distinct per (trace, round, attempt) and stable across runs."""
    return trace_seed * 1000 + round_no * 10 + attempt


def request_body(model: str, messages: list[Message], seed: int) -> dict[str, Any]:
    """The JSON body of one chat request; thinking is switched off at the template."""
    return {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
    }


def parse_completion(data: dict[str, Any], seconds: float) -> Completion:
    """Read one ``/chat/completions`` response document; raises KeyError/IndexError/TypeError
    on a document without ``choices[0].message``."""
    usage: dict[str, Any] = data.get("usage") or {}
    choice: dict[str, Any] = data["choices"][0]
    message: dict[str, Any] = choice["message"]
    return Completion(
        text=str(message.get("content") or ""),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        seconds=seconds,
        finish_reason=str(choice.get("finish_reason") or ""),
        reasoning_chars=len(str(message.get("reasoning_content") or "")),
    )


def http_chat(endpoint: str, model: str) -> Chat:
    """A ``Chat`` over an OpenAI-compatible ``/chat/completions`` endpoint."""
    url = endpoint.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("SYNTH_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    def call(messages: list[Message], seed: int) -> Completion:
        t0 = time.perf_counter()
        resp = requests.post(
            url, json=request_body(model, messages, seed), headers=headers, timeout=HTTP_TIMEOUT_S
        )
        secs = time.perf_counter() - t0
        resp.raise_for_status()
        return parse_completion(resp.json(), secs)

    return call


def fake_chat(_messages: list[Message], _seed: int) -> Completion:
    """Offline stand-in that always answers with the identity candidate."""
    return Completion(f"```python\n{IDENTITY_CODE}```", 0, 0, 0.0, "stop", 0)


def same_program(a: str, b: str) -> bool:
    """True when two modules have the same AST, so comment or formatting edits count as no
    change. Falls back to exact text equality when either fails to parse."""
    try:
        return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
    except SyntaxError:
        return a == b


def _write_prompt(path: Path, messages: list[Message]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(messages, indent=1), encoding="utf-8")
    text = "\n\n".join(f"=== {m['role']} ===\n{m['content']}" for m in messages)
    path.with_suffix(".txt").write_text(text, encoding="utf-8")


def trace_record(trace: Trace) -> dict[str, Any]:
    """The ``trace`` and ``baseline`` fields shared by every game record."""
    return {
        "trace": {
            "steps": len(trace.transitions),
            "train": len(trace.train),
            "heldout": len(trace.heldout),
            "seed": trace.seed,
        },
        "baseline": {
            "identity_exact_rate_train": identity_exact_rate(trace.train),
            "identity_exact_rate_heldout": identity_exact_rate(trace.heldout),
        },
    }


def _ask(
    chat: Chat, messages: list[Message], trace: Trace, round_no: int, root: Path
) -> tuple[Completion, str | None, int]:
    """Call the model for one round, re-sampling up to ``NO_CODE_RETRIES`` times with the
    next seed when the reply has no code fence. Returns the last completion with its
    accounting summed over attempts, the extracted code (None if every attempt lacked a
    fence) and the number of attempts. Every reply is saved: the kept one as
    ``replies/<game>-r<round>.txt``, earlier ones with a ``-retry<n>`` suffix."""
    replies = root / "replies"
    replies.mkdir(parents=True, exist_ok=True)
    prompt_tokens = completion_tokens = 0
    seconds = 0.0
    attempt = 0
    while True:
        comp = chat(messages, request_seed(trace.seed, round_no, attempt))
        prompt_tokens += comp.prompt_tokens
        completion_tokens += comp.completion_tokens
        seconds += comp.seconds
        extracted = extract_code(comp.text)
        if extracted is not None or attempt == NO_CODE_RETRIES:
            break
        (replies / f"{trace.game}-r{round_no}-retry{attempt}.txt").write_text(
            comp.text, encoding="utf-8"
        )
        attempt += 1
    (replies / f"{trace.game}-r{round_no}.txt").write_text(comp.text, encoding="utf-8")
    total = Completion(
        comp.text,
        prompt_tokens,
        completion_tokens,
        seconds,
        comp.finish_reason,
        comp.reasoning_chars,
    )
    return total, extracted, attempt + 1


def run_game(
    trace: Trace,
    chat: Chat,
    rounds: int,
    max_transitions: int,
    root: Path,
) -> dict[str, Any]:
    """Synthesize/verify/repair loop for one game; returns its JSON record."""
    out = trace_record(trace)
    out["rounds"] = []
    code: str | None = None
    mismatches = None
    exact_train: tuple[int, int] | None = None
    for r in range(1, rounds + 1):
        messages = build_messages(
            trace, max_transitions, prior_code=code, mismatches=mismatches, exact_train=exact_train
        )
        _write_prompt(root / "prompts" / f"{trace.game}-r{r}.json", messages)
        rec: dict[str, Any] = {"round": r, "error": None}
        try:
            comp, extracted, attempts = _ask(chat, messages, trace, r, root)
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"[:500]
            out["rounds"].append(rec)
            break
        unchanged = code is not None and extracted is not None and same_program(code, extracted)
        code = extracted or NO_CODE
        code_path = root / "code" / f"{trace.game}-r{r}.py"
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(code, encoding="utf-8")
        res = verify(code, trace)
        mismatches = res.mismatches
        train_results = [x for x in res.results if x.split == "train"]
        exact_train = (sum(x.exact for x in train_results), len(train_results))
        n_train_cells = len(train_results) * CELLS
        rec.update(
            exact_rate_train=res.exact_rate_train,
            exact_rate_heldout=res.exact_rate_heldout,
            cell_accuracy=res.cell_accuracy,
            cell_accuracy_train=(
                1.0 - sum(x.cells_wrong for x in train_results) / n_train_cells
                if n_train_cells
                else 0.0
            ),
            crashed=res.crashed,
            timed_out=res.timed_out,
            load_error=res.load_error,
            no_code=extracted is None,
            unchanged=unchanged,
            attempts=attempts,
            prompt_tokens=comp.prompt_tokens,
            completion_tokens=comp.completion_tokens,
            finish_reason=comp.finish_reason,
            reasoning_chars=comp.reasoning_chars,
            reply_chars=len(comp.text),
            seconds=round(comp.seconds, 3),
            prompt_chars=sum(len(m["content"]) for m in messages),
            code_path=code_path.as_posix(),
        )
        out["rounds"].append(rec)
        # A repair that re-emits the same program would get the same feedback and, at this
        # temperature, the same reply again: the v6 Kaggle run spent round 3 on that.
        if extracted is None or unchanged or res.exact_rate_heldout >= REPAIR_TARGET:
            break
    out["final"] = select_final(out["rounds"])
    return out


def select_final(rounds: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The candidate an agent would keep: the round that fits the observations it has seen
    best, by training exact rate then training cell accuracy, earliest round on a tie.
    Selection never reads the held-out split, so held-out scores stay a fair report of
    generalization. Rounds without code are skipped (a no-code round scores 0 as a load
    failure, and a repair can make a candidate worse: vc33 fell 0.53 -> 0.39 in Kaggle run v8).
    """
    with_code = [x for x in rounds if "exact_rate_heldout" in x and not x["no_code"]]
    if not with_code:
        return None
    return max(
        with_code, key=lambda x: (x["exact_rate_train"], x["cell_accuracy_train"], -x["round"])
    )


def dry_run_game(trace: Trace, max_transitions: int, root: Path) -> dict[str, Any]:
    """Write the round-1 prompt and a sample repair prompt (built from the identity
    candidate's mismatches) without calling any model."""
    first = build_messages(trace, max_transitions)
    _write_prompt(root / "prompts" / f"{trace.game}-r1.json", first)
    res = verify(IDENTITY_CODE, trace)
    repair = build_messages(trace, max_transitions, IDENTITY_CODE, res.mismatches)
    _write_prompt(root / "prompts" / f"{trace.game}-repair-sample.json", repair)
    return {
        "prompt_chars_r1": sum(len(m["content"]) for m in first),
        "prompt_chars_repair_sample": sum(len(m["content"]) for m in repair),
    }


def format_table(results: dict[str, Any]) -> str:
    """Compact fixed-width summary, one row per game and round."""
    head = (
        f"{'game':<6}{'rnd':>4}{'train':>7}{'held':>7}{'cells':>8}{'crash':>6}"
        f"{'p_tok':>8}{'c_tok':>7}{'sec':>7}{'finish':>8}{'reason':>8}{'code':>6}  identity_held"
    )
    lines = [head, "-" * len(head)]
    for game, g in results["games"].items():
        ident = g["baseline"]["identity_exact_rate_heldout"]
        for r in g.get("rounds", []):
            if r.get("error"):
                lines.append(f"{game:<6}{r['round']:>4}  error: {r['error'][:60]}")
                continue
            lines.append(
                f"{game:<6}{r['round']:>4}{r['exact_rate_train']:>7.2f}"
                f"{r['exact_rate_heldout']:>7.2f}{r['cell_accuracy']:>8.4f}{r['crashed']:>6}"
                f"{r['prompt_tokens']:>8}{r['completion_tokens']:>7}{r['seconds']:>7.1f}"
                f"{r.get('finish_reason', '')[:7]:>8}{r.get('reasoning_chars', 0):>8}"
                f"{'none' if r.get('no_code') else 'same' if r.get('unchanged') else 'yes':>6}"
                f"  {ident:.2f}"
            )
        if "dry_run" in g:
            d = g["dry_run"]
            lines.append(
                f"{game:<6} dry-run prompt chars r1={d['prompt_chars_r1']} "
                f"repair={d['prompt_chars_repair_sample']} identity_held={ident:.2f}"
            )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m arcagent.synth.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--env-dir", default=os.environ.get("ARC_ENV_DIR", DEFAULT_ENV_DIR))
    p.add_argument("--games", nargs="+", default=["ls20", "ft09", "vc33"])
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--endpoint", default="http://127.0.0.1:8001/v1")
    p.add_argument("--model", default="")
    p.add_argument("--max-transitions", type=int, default=40)
    p.add_argument("--out", default="bench/synth/results.json")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="write prompts, call nothing")
    mode.add_argument("--fake", action="store_true", help="identity candidate, no model")
    args = p.parse_args(argv)
    if args.rounds < 1 or args.steps < 4 or args.max_transitions < 1:
        p.error("need --rounds >= 1, --steps >= 4, --max-transitions >= 1")
    if not (args.dry_run or args.fake or args.model):
        p.error("--model is required unless --dry-run or --fake")
    return args


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Run the experiment; returns (and writes) the results document."""
    args = parse_args(argv)
    out_path = Path(args.out)
    root = out_path.parent
    chat: Chat = fake_chat if args.fake else http_chat(args.endpoint, args.model)
    results: dict[str, Any] = {
        "config": {
            "games": args.games,
            "steps": args.steps,
            "seed": args.seed,
            "rounds": args.rounds,
            "max_transitions": args.max_transitions,
            "endpoint": None if (args.fake or args.dry_run) else args.endpoint,
            "model": "fake-identity" if args.fake else args.model,
            "dry_run": args.dry_run,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "enable_thinking": ENABLE_THINKING,
            "repair_target": REPAIR_TARGET,
        },
        "games": {},
    }
    for game in args.games:
        trace = collect_trace(args.env_dir, game, args.steps, args.seed)
        trace_path = trace.save(root / "traces")
        if args.dry_run:
            g = trace_record(trace)
            g["dry_run"] = dry_run_game(trace, args.max_transitions, root)
        else:
            g = run_game(trace, chat, args.rounds, args.max_transitions, root)
        g["trace"]["path"] = trace_path.as_posix()
        results["games"][game] = g
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(format_table(results))
    print(f"wrote {out_path.as_posix()}")
    return results


if __name__ == "__main__":
    main()
