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
scored as a candidate that fails to load. Other games still run.
"""

from __future__ import annotations

import argparse
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
from arcagent.synth.verify import IDENTITY_CODE, identity_exact_rate, verify

REPAIR_TARGET = 0.95
TEMPERATURE = 0.2
MAX_TOKENS = 6000
HTTP_TIMEOUT_S = 600.0
DEFAULT_ENV_DIR = r"C:\arc-study\data\environment_files"
NO_CODE = "# the model reply contained no code fence\n"


@dataclass(frozen=True, slots=True)
class Completion:
    """One model reply with its accounting."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


Chat = Callable[[list[Message]], Completion]


def http_chat(endpoint: str, model: str) -> Chat:
    """A ``Chat`` over an OpenAI-compatible ``/chat/completions`` endpoint."""
    url = endpoint.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("SYNTH_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    def call(messages: list[Message]) -> Completion:
        body = {
            "model": model,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }
        t0 = time.perf_counter()
        resp = requests.post(url, json=body, headers=headers, timeout=HTTP_TIMEOUT_S)
        secs = time.perf_counter() - t0
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        usage: dict[str, Any] = data.get("usage") or {}
        text = str(data["choices"][0]["message"].get("content") or "")
        return Completion(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            seconds=secs,
        )

    return call


def fake_chat(_messages: list[Message]) -> Completion:
    """Offline stand-in that always answers with the identity candidate."""
    return Completion(f"```python\n{IDENTITY_CODE}```", 0, 0, 0.0)


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
    for r in range(1, rounds + 1):
        messages = build_messages(trace, max_transitions, prior_code=code, mismatches=mismatches)
        _write_prompt(root / "prompts" / f"{trace.game}-r{r}.json", messages)
        rec: dict[str, Any] = {"round": r, "error": None}
        try:
            comp = chat(messages)
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"[:500]
            out["rounds"].append(rec)
            break
        code = extract_code(comp.text) or NO_CODE
        code_path = root / "code" / f"{trace.game}-r{r}.py"
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(code, encoding="utf-8")
        res = verify(code, trace)
        mismatches = res.mismatches
        rec.update(
            exact_rate_train=res.exact_rate_train,
            exact_rate_heldout=res.exact_rate_heldout,
            cell_accuracy=res.cell_accuracy,
            crashed=res.crashed,
            timed_out=res.timed_out,
            load_error=res.load_error,
            prompt_tokens=comp.prompt_tokens,
            completion_tokens=comp.completion_tokens,
            seconds=round(comp.seconds, 3),
            prompt_chars=sum(len(m["content"]) for m in messages),
            code_path=code_path.as_posix(),
        )
        out["rounds"].append(rec)
        if res.exact_rate_heldout >= REPAIR_TARGET:
            break
    scored = [x for x in out["rounds"] if "exact_rate_heldout" in x]
    out["final"] = scored[-1] if scored else None
    return out


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
        f"{'p_tok':>8}{'c_tok':>7}{'sec':>7}  identity_held"
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
