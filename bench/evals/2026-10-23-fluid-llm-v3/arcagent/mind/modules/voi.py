"""Value of information: act where the live rule programs disagree most.

HCQM 5.1/8.4. Salience calls this when no plan exists and at least two
programs are live, so the next action separates hypotheses instead of
repeating one they all agree on.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from arcagent.core.types import Action, Observation
from arcagent.mind.predictive import Belief
from arcagent.mind.sandbox import ProgramSandbox
from arcagent.mind.types import ActionKey


class DisagreementVoi:
    """Scores each candidate by the fit-weighted entropy of the programs' predictions.

    Contract of ``pick``:
    - Fewer than two live programs: ``(None, {})``. A lone program is handled by
      the executive's plan-to-novel branch.
    - Only the first ``max_candidates`` candidates are scored, in the order given,
      so one ``predict_many`` request stays bounded.
    - A program's weight is its fit. A program with no trials weighs the mean fit
      of the programs that have trials (1.0 if none do), so a fresh program still
      counts. If that leaves every weight at 0 the weights are made equal, since
      zero total weight would switch VOI off exactly when nothing fits.
    - Predicted frame hashes form the buckets; a None prediction (crash, timeout,
      junk) is one extra bucket. Entropy is natural log.
    - The best score wins and ties go to the earlier candidate. If the best score
      is 0 the pick is None so the executive falls through to B0. Scores are
      returned for every scored candidate either way, for the run log.
    - ``visited`` is part of the Voi protocol and is not used: disagreement is the
      whole criterion.

    Failure mode: if the sandbox raises, the exception propagates; the agent
    already falls back to B0 on mind-side errors.
    """

    def __init__(self, max_candidates: int = 48) -> None:
        self.max_candidates = max_candidates

    def pick(
        self,
        belief: Belief,
        sandbox: ProgramSandbox,
        obs: Observation,
        candidates: Sequence[Action],
        visited: frozenset[str],
    ) -> tuple[Action | None, dict[ActionKey, float]]:
        if len(belief.programs) < 2:
            return None, {}
        acts = list(candidates[: self.max_candidates])
        if not acts:
            return None, {}
        weights = self._weights(belief)
        pids = sorted(weights)
        preds = sandbox.predict_many(pids, obs.grid, acts)

        scores: dict[ActionKey, float] = {}
        best: Action | None = None
        best_score = 0.0
        for a in acts:
            key = a.key()
            buckets: defaultdict[str | None, float] = defaultdict(float)
            for pid in pids:
                buckets[preds.get((pid, key))] += weights[pid]
            score = _entropy(list(buckets.values()))
            scores[key] = score
            if score > best_score:
                best, best_score = a, score
        return best, scores

    @staticmethod
    def _weights(belief: Belief) -> dict[str, float]:
        with_trials = [s.fit for s in belief.programs.values() if s.trials]
        fresh = sum(with_trials) / len(with_trials) if with_trials else 1.0
        w = {pid: (s.fit if s.trials else fresh) for pid, s in belief.programs.items()}
        if sum(w.values()) <= 0.0:
            return dict.fromkeys(w, 1.0)
        return w


def _entropy(masses: list[float]) -> float:
    total = sum(masses)
    return -sum((m / total) * math.log(m / total) for m in masses if m > 0.0)
