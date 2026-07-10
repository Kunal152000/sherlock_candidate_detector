"""Scoring engine: fuse weak signals into a candidate verdict.

This is where the many independent signals (``signals.py`` + the LLM signal in
``llm.py``) are combined into a single, explainable answer. Per the plan the
engine is a *pure function of accumulated state* -- it reads the current
:class:`~app.models.Meeting`, recomputes every signal, and writes back the
latest scoring output. It is invoked by the event buffer after each flush
(``buffer.recompute = engine.recompute``).

Three concerns, one pass (plan phases 7-9):

* **Score** -- a weighted sum of each signal's per-participant opinion. Weights
  live in ``config.py`` when that phase lands; until then we use the built-in
  :data:`DEFAULT_WEIGHTS`.
* **Confidence** -- scores are turned into a probability-like share across
  participants with a softmax. That yields a value in ``[0, 1]`` that naturally
  *drops when two people score similarly*. When the top two are within a small
  margin we additionally raise an :attr:`~app.models.Meeting.ambiguous` flag so
  callers can surface graceful uncertainty rather than a false-confident pick.
* **Explanation** -- the reasons are essentially free: every signal already
  returns a human-readable justification, so we just collect the
  positively-contributing ones for each participant (leader included).

Design notes:

* Signals are run *outside* the store lock (the LLM signal may block on the
  network); only the fast write-back of results is guarded, so API readers never
  observe a half-updated verdict and never wait on a network call.
* A single broken signal must not sink the verdict -- each is wrapped in
  ``try/except`` and simply omitted on failure.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from . import signals as signals_module
from .llm import llm_signal
from .meeting_state import store
from .models import Meeting, Participant, SignalContribution

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tunables (overridable via app.config.settings once that phase exists)
# --------------------------------------------------------------------------- #

# Per-signal weights. Identity + self-introduction + the holistic LLM read are
# the strongest; camera/join are gentle tie-breakers. Signals not listed here
# fall back to ``_DEFAULT_WEIGHT`` so new signals contribute without edits.
DEFAULT_WEIGHTS: dict[str, float] = {
    "name_signal": 1.5,
    "intro_signal": 1.5,
    "conversation_signal": 1.2,
    "speaker_signal": 1.0,
    "camera_signal": 0.5,
    "join_signal": 0.5,
    "llm_signal": 2.0,
}
_DEFAULT_WEIGHT = 1.0

# Softmax sharpness: higher = more decisive confidence, lower = more cautious.
DEFAULT_SOFTMAX_BETA = 1.0

# If the leader's confidence exceeds the runner-up's by less than this, the
# verdict is flagged ambiguous (top-2 too close to call).
DEFAULT_AMBIGUITY_MARGIN = 0.15

# How many reason strings to keep per participant (strongest first).
_MAX_REASONS = 4


def _tunable(attr: str, default: float) -> float:
    """Read a float tunable from ``app.config.settings`` if present, else default."""
    try:  # config.py is a later phase; work standalone until it exists.
        from .config import settings  # type: ignore

        val = getattr(settings, attr, None)
        if val is not None:
            return float(val)
    except Exception:
        pass
    return default


def _weights() -> dict[str, float]:
    """Effective signal weights: defaults merged with any config override."""
    weights = dict(DEFAULT_WEIGHTS)
    try:  # optional config override: settings.signal_weights = {name: weight}
        from .config import settings  # type: ignore

        override = getattr(settings, "signal_weights", None)
        if isinstance(override, dict):
            for name, value in override.items():
                try:
                    weights[str(name)] = float(value)
                except (TypeError, ValueError):
                    continue
    except Exception:
        pass
    return weights


# --------------------------------------------------------------------------- #
# Signal execution
# --------------------------------------------------------------------------- #


def _run_signals(meeting: Meeting) -> dict[str, dict[str, tuple[float, str]]]:
    """Run every signal (pure registry + LLM) grouped by participant.

    Returns ``{participant_id: {signal_name: (score, reason)}}``. Each signal is
    isolated: a failure is logged and skipped so the rest still produce a
    verdict. The LLM signal (``llm.py``) is appended to the pure registry here
    because it does network I/O with a deterministic fallback.
    """
    all_signals = list(signals_module.SIGNALS) + [llm_signal]

    per_participant: dict[str, dict[str, tuple[float, str]]] = {
        pid: {} for pid in meeting.participants
    }
    for signal in all_signals:
        try:
            outcome = signal(meeting)
        except Exception:
            logger.exception("Signal %s failed; skipping", getattr(signal, "__name__", signal))
            continue
        for pid, contribution in outcome.items():
            if pid in per_participant:  # ignore ids no longer in the meeting
                per_participant[pid][signal.__name__] = contribution

    return per_participant


# --------------------------------------------------------------------------- #
# Phase 7: weighted score
# --------------------------------------------------------------------------- #


def _weighted_scores(
    per_participant: dict[str, dict[str, tuple[float, str]]],
    weights: dict[str, float],
) -> tuple[dict[str, float], dict[str, list[SignalContribution]]]:
    """Combine each participant's signal opinions into one weighted score.

    Returns ``(scores, contributions)`` where ``contributions`` preserves every
    signal's raw score + reason for explainability and the ranked API view.
    """
    scores: dict[str, float] = {}
    contributions: dict[str, list[SignalContribution]] = {}

    for pid, signal_map in per_participant.items():
        total = 0.0
        contribs: list[SignalContribution] = []
        for signal_name, (raw, reason) in signal_map.items():
            total += weights.get(signal_name, _DEFAULT_WEIGHT) * raw
            contribs.append(
                SignalContribution(signal=signal_name, score=raw, reason=reason)
            )
        scores[pid] = total
        contributions[pid] = contribs

    return scores, contributions


# --------------------------------------------------------------------------- #
# Phase 8: normalized confidence + ambiguity
# --------------------------------------------------------------------------- #


def _softmax(scores: dict[str, float], beta: float) -> dict[str, float]:
    """Normalise scores into a probability-like share summing to 1.

    Shifting by the max keeps ``exp`` numerically stable and, because the values
    are negative or zero, safely bounded. With no participants (or a degenerate
    partition) we return a uniform distribution.
    """
    if not scores:
        return {}
    top = max(scores.values())
    exps = {pid: math.exp(beta * (s - top)) for pid, s in scores.items()}
    z = sum(exps.values())
    if z <= 0:
        uniform = 1.0 / len(scores)
        return {pid: uniform for pid in scores}
    return {pid: e / z for pid, e in exps.items()}


# --------------------------------------------------------------------------- #
# Phase 9: reason aggregation
# --------------------------------------------------------------------------- #


def _positive_reasons(
    contribs: list[SignalContribution],
    weights: dict[str, float],
    limit: int = _MAX_REASONS,
) -> list[str]:
    """Collect the reasons of positively-contributing signals, strongest first.

    A signal contributes positively when ``weight * raw_score > 0``. We rank by
    that weighted magnitude so the most decisive evidence leads the explanation.
    """
    weighted: list[tuple[float, str]] = []
    for c in contribs:
        value = weights.get(c.signal, _DEFAULT_WEIGHT) * c.score
        if value > 0:
            weighted.append((value, c.reason))
    weighted.sort(key=lambda item: item[0], reverse=True)
    return [reason for _, reason in weighted[:limit]]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def recompute(meeting: Meeting) -> Meeting:
    """Recompute the full verdict for ``meeting`` and write it back into state.

    Runs all signals, applies weights, normalises to confidence, flags
    ambiguity, and aggregates reasons -- then stores the results on each
    :class:`~app.models.Participant` and the meeting's verdict fields. Returns
    the same ``meeting`` for convenience.

    Safe to call repeatedly: it is a pure function of accumulated state, so
    re-running never double-counts.
    """
    # Run signals outside the lock: the LLM signal may block on the network and
    # we must not hold up API readers (or deadlock via the re-entrant lock).
    per_participant = _run_signals(meeting)
    weights = _weights()
    scores, contributions = _weighted_scores(per_participant, weights)
    confidences = _softmax(scores, _tunable("softmax_beta", DEFAULT_SOFTMAX_BETA))
    margin = _tunable("ambiguity_margin", DEFAULT_AMBIGUITY_MARGIN)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    # Guard only the fast write-back so readers see a consistent snapshot.
    with store.lock:
        for pid, participant in meeting.participants.items():
            participant.score = round(scores.get(pid, 0.0), 4)
            participant.confidence = round(confidences.get(pid, 0.0), 4)
            participant.contributions = contributions.get(pid, [])
            participant.reasons = _positive_reasons(
                contributions.get(pid, []), weights
            )
            participant.is_candidate = False

        if ranked:
            winner_id, _ = ranked[0]
            winner_conf = confidences.get(winner_id, 0.0)

            ambiguous = False
            if len(ranked) >= 2:
                runner_up_conf = confidences.get(ranked[1][0], 0.0)
                ambiguous = (winner_conf - runner_up_conf) < margin

            meeting.candidate_id = winner_id
            meeting.confidence = round(winner_conf, 4)
            meeting.ambiguous = ambiguous
            meeting.participants[winner_id].is_candidate = True
        else:
            meeting.candidate_id = None
            meeting.confidence = 0.0
            meeting.ambiguous = False

    return meeting


def rank_participants(meeting: Meeting) -> list[Participant]:
    """Return participants ordered by score (highest first) for the API view.

    Reads the scores written by the most recent :func:`recompute`; does not
    itself recompute. Ties break by presence then speaking time so the ranking
    is stable and sensible even before the first flush.
    """
    with store.lock:
        return sorted(
            meeting.participants.values(),
            key=lambda p: (p.score, p.is_present, p.speaking_seconds),
            reverse=True,
        )
