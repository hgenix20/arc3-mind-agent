"""Confidence-based metacognition: drop programs that stopped earning their place."""

from __future__ import annotations

from arcagent.mind.predictive import Belief


class ConfidenceMetacognition:
    """Reports programs to drop; the agent removes and logs them.

    ``misses_in_row`` applies at any trial count. ``low_fit`` applies only once
    ``trials >= min_trials`` so a young program is not judged on one miss. When
    both apply the reason is ``misses_in_row``. Output is sorted by pid.
    """

    def __init__(
        self, min_trials: int = 3, drop_fit: float = 0.6, max_misses_in_row: int = 3
    ) -> None:
        self.min_trials = min_trials
        self.drop_fit = drop_fit
        self.max_misses_in_row = max_misses_in_row

    def review(self, belief: Belief, step_index: int) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for pid in sorted(belief.programs):
            st = belief.programs[pid]
            if st.misses_in_row >= self.max_misses_in_row:
                out.append((pid, "misses_in_row"))
            elif st.trials >= self.min_trials and st.fit < self.drop_fit:
                out.append((pid, "low_fit"))
        return out
