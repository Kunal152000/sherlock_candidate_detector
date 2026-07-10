"""Weak-signal detectors for candidate identification.

Each *signal* is an independent **pure function of meeting state**:

    signal(meeting: Meeting) -> dict[participant_id, (score, reason)]

* ``score`` is a raw, roughly ``[-1.0, +1.0]`` opinion where **positive means
  "more likely the candidate"** and negative means "less likely" (e.g. looks
  like an interviewer/observer). Magnitudes are only loosely comparable across
  signals -- the :mod:`app.engine` applies per-signal *weights* and normalises.
* ``reason`` is a short human-readable justification used for explainability.

Signals never mutate state and never depend on each other, so they can be
added, removed, reordered, unit-tested, or run in any order. They are collected
in :data:`SIGNALS`; the engine iterates that registry. The LLM signal lives in
``llm.py`` (network I/O + fallback) to keep this module pure and offline-testable.

Design rationale (see plan): fuse *many* weak signals rather than trust one
rule, so the system degrades gracefully under device-style names ("MacBook
Pro"), nicknames, wrong names typed by the interviewer, silent observers, and
renames.
"""

from __future__ import annotations

import re
from typing import Callable

from .models import Meeting

# A signal maps participant_id -> (raw_score, human_reason).
SignalResult = dict[str, tuple[float, str]]
Signal = Callable[[Meeting], SignalResult]


# --------------------------------------------------------------------------- #
# Fuzzy-matching helper (rapidfuzz if present, difflib fallback otherwise)
# --------------------------------------------------------------------------- #

try:  # rapidfuzz is in requirements, but stay importable without it.
    from rapidfuzz import fuzz as _fuzz

    def _ratio(a: str, b: str) -> float:
        """Token-set similarity in ``[0, 1]`` (order/duplication insensitive)."""
        if not a or not b:
            return 0.0
        return _fuzz.token_set_ratio(a, b) / 100.0

except Exception:  # pragma: no cover - exercised only when rapidfuzz missing
    from difflib import SequenceMatcher

    def _ratio(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _best_ratio(name: str, candidates: list[str]) -> float:
    """Highest similarity of ``name`` against any string in ``candidates``."""
    return max((_ratio(name, c) for c in candidates if c), default=0.0)


# --------------------------------------------------------------------------- #
# Text / identity helpers
# --------------------------------------------------------------------------- #

# Substrings that mark a display name as a device/auto-generated label rather
# than a human name (candidates frequently join as their device).
_DEVICE_MARKERS = (
    "macbook", "imac", "iphone", "ipad", "ipod", "android", "samsung", "galaxy",
    "pixel", "oneplus", "redmi", "xiaomi", "huawei", "oppo", "vivo", "nokia",
    "windows", "surface", "laptop", "desktop", "notebook", " pc", "tablet",
    "phone", "device", "user", "guest", "participant", "anonymous", "unknown",
    "caller", "'s ", "s iphone", "s macbook", "s galaxy", "sm-", "moto ",
)

_WORD_RE = re.compile(r"[a-zA-Z']+")

# Phrases people use to introduce themselves.
_INTRO_RE = re.compile(
    r"\b("
    r"my name is|i am|i'm|this is|myself|i go by|you can call me|"
    r"name's|i will be|here is"
    r")\b",
    re.IGNORECASE,
)

# Interviewer-ish phrasing (asking, directing the interview).
_INTERVIEWER_CUE_RE = re.compile(
    r"\b("
    r"tell me about|can you|could you|walk me through|why don'?t you|"
    r"let'?s start|shall we|do you have any questions|we'll be|"
    r"take your time|let'?s begin|introduce yourself"
    r")\b",
    re.IGNORECASE,
)


def _looks_like_device(name: str) -> bool:
    """True if the display name reads like a device/placeholder, not a person."""
    n = name.strip().lower()
    if not n:
        return True
    if any(marker in n for marker in _DEVICE_MARKERS):
        return True
    # Names that are mostly non-alphabetic (e.g. "SM-G991B", "+1 555...").
    letters = sum(c.isalpha() for c in n)
    return letters < max(2, len(n) // 2)


def _email_local(email: str | None) -> str:
    """Local part of an email as a space-separated, name-ish string."""
    if not email or "@" not in email:
        return ""
    local = email.split("@", 1)[0]
    return re.sub(r"[._\-+]+", " ", local).strip()


def _name_tokens(text: str) -> set[str]:
    """Lower-cased alphabetic word tokens of length >= 2."""
    return {w.lower() for w in _WORD_RE.findall(text) if len(w) >= 2}


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #


def name_signal(meeting: Meeting) -> SignalResult:
    """Fuzzy-match display name / email against candidate vs interviewer identity.

    * Strong positive when name or email matches the known candidate.
    * Negative when it matches an interviewer.
    * Neutral (~0) for device-like/blank names -- explicitly *uninformative*
      rather than misleading, so other signals decide.
    """
    results: SignalResult = {}

    cand_name = (meeting.candidate_name or "").strip()
    cand_email = (meeting.candidate_email or "").strip().lower()
    cand_local = _email_local(cand_email)
    interviewer_names = [n.strip() for n in meeting.interviewer_names if n.strip()]
    interviewer_emails = [e.strip().lower() for e in meeting.interviewer_emails if e.strip()]

    for pid, p in meeting.participants.items():
        name = (p.display_name or "").strip()
        email = (p.email or "").strip().lower()

        # Exact email matches are the most reliable identity signal.
        if email and cand_email and email == cand_email:
            results[pid] = (1.0, f"Email matches the candidate ({email})")
            continue
        if email and email in interviewer_emails:
            results[pid] = (-1.0, "Email matches an interviewer")
            continue

        if _looks_like_device(name):
            results[pid] = (0.0, "Display name is device-like/blank (uninformative)")
            continue

        cand_score = _ratio(name, cand_name)
        if cand_local:
            cand_score = max(cand_score, _ratio(name, cand_local))
        interviewer_score = _best_ratio(name, interviewer_names)

        if interviewer_score >= 0.85 and interviewer_score > cand_score:
            results[pid] = (
                -0.8 * interviewer_score,
                f"Name '{name}' matches an interviewer",
            )
        elif cand_score >= 0.6:
            results[pid] = (
                cand_score,
                f"Name '{name}' matches the candidate name",
            )
        else:
            results[pid] = (0.0, f"Name '{name}' has no clear identity match")

    return results


def intro_signal(meeting: Meeting) -> SignalResult:
    """Detect self-introductions in the transcript.

    Candidates almost always introduce themselves ("my name is ...", "I'm ...").
    This is robust to a wrong/device display name. If the introduced text also
    contains the known candidate name, the score is boosted.
    """
    results: SignalResult = {}
    cand_tokens = _name_tokens(meeting.candidate_name or "")

    for pid, p in meeting.participants.items():
        text = p.full_transcript
        if not text:
            results[pid] = (0.0, "No transcript yet")
            continue

        intros = len(_INTRO_RE.findall(text))
        if not intros:
            results[pid] = (0.0, "No self-introduction detected")
            continue

        score = min(0.7, 0.5 + 0.1 * (intros - 1))
        reason = "Self-introduced in the meeting"

        # Boost if what they said includes the expected candidate name.
        if cand_tokens and cand_tokens & _name_tokens(text):
            score = min(1.0, score + 0.3)
            reason = "Self-introduced using the candidate's name"

        results[pid] = (score, reason)

    return results


def conversation_signal(meeting: Meeting) -> SignalResult:
    """Interviewers ask, candidates answer.

    Uses the share of a participant's utterances that are questions plus explicit
    interviewer phrasing ("tell me about...", "can you..."). A high question /
    prompting ratio reads as an interviewer (negative); mostly-answering with
    real content reads as a candidate (positive).
    """
    results: SignalResult = {}

    for pid, p in meeting.participants.items():
        chunks = [c for c in p.transcript if c.strip()]
        if not chunks:
            results[pid] = (0.0, "No transcript yet")
            continue

        questions = sum(1 for c in chunks if "?" in c)
        cues = sum(1 for c in chunks if _INTERVIEWER_CUE_RE.search(c))
        n = len(chunks)
        prompt_ratio = (questions + cues) / (2 * n)  # 0 = pure answerer, ~1 = pure asker

        if prompt_ratio >= 0.5:
            results[pid] = (
                -min(1.0, prompt_ratio),
                "Mostly asks questions / directs the interview (interviewer-like)",
            )
        elif n >= 2:
            # Substantial speech that is mostly answers -> candidate-like.
            score = min(1.0, (1.0 - prompt_ratio) * min(1.0, n / 4.0))
            results[pid] = (score, "Mostly answering questions (candidate-like)")
        else:
            results[pid] = (0.0, "Too little conversation to judge")

    return results


def speaker_signal(meeting: Meeting) -> SignalResult:
    """Reward substantial (but not monopolising) speaking time.

    Candidates typically speak a lot (answering) but not the entire meeting.
    Near-zero speech reads as a silent observer (negative); a lone monologue is
    slightly discounted (could be a presenter/interviewer).
    """
    results: SignalResult = {}

    present = [p for p in meeting.participants.values() if p.is_present]
    pool = present or list(meeting.participants.values())
    total = sum(p.speaking_seconds for p in pool)

    if total <= 0:
        return {pid: (0.0, "No speech recorded yet") for pid in meeting.participants}

    multi = len(pool) > 1
    for pid, p in meeting.participants.items():
        share = p.speaking_seconds / total if total else 0.0

        if share < 0.05:
            results[pid] = (-0.6, "Speaks very little (likely an observer)")
        elif multi and share > 0.85:
            results[pid] = (0.3, "Dominates the conversation (possible monologue)")
        else:
            results[pid] = (
                min(1.0, 0.3 + share),
                f"Substantial speaking time ({share:.0%} of talk)",
            )

    return results


def camera_signal(meeting: Meeting) -> SignalResult:
    """Small positive for camera-on (candidates are usually asked to enable it)."""
    results: SignalResult = {}
    for pid, p in meeting.participants.items():
        if p.camera_on:
            results[pid] = (0.4, "Camera is on")
        else:
            results[pid] = (0.0, "Camera is off")
    return results


def join_signal(meeting: Meeting) -> SignalResult:
    """Mild timing heuristic: the earliest joiner is usually the host/interviewer.

    Kept intentionally weak -- it only nudges ties. Interviewers/hosts tend to
    open the room first; the candidate usually joins around the scheduled start.
    """
    results: SignalResult = {}

    ordered = [p for p in meeting.participants.values() if p.join_order is not None]
    if not ordered:
        return {pid: (0.0, "Join order unknown") for pid in meeting.participants}

    first_order = min(p.join_order for p in ordered)  # type: ignore[type-var]

    for pid, p in meeting.participants.items():
        if p.join_order is None:
            results[pid] = (0.0, "Join order unknown")
        elif p.join_order == first_order:
            results[pid] = (-0.3, "Joined first (likely the host/interviewer)")
        else:
            results[pid] = (0.2, "Joined after the host")

    return results


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

# Ordered registry the engine iterates. Add/remove signals here; each is
# independent so order does not affect correctness. The LLM signal (llm.py) is
# registered separately by the engine because it does network I/O with fallback.
SIGNALS: list[Signal] = [
    name_signal,
    intro_signal,
    conversation_signal,
    speaker_signal,
    camera_signal,
    join_signal,
]


def evaluate(meeting: Meeting) -> dict[str, dict[str, tuple[float, str]]]:
    """Run every registered signal and group results by participant.

    Returns ``{participant_id: {signal_name: (score, reason)}}``. This is a
    convenience for the engine (which then applies weights + normalisation); the
    individual signal functions remain the pure, independently-testable units.
    """
    per_participant: dict[str, dict[str, tuple[float, str]]] = {
        pid: {} for pid in meeting.participants
    }
    for signal in SIGNALS:
        try:
            outcome = signal(meeting)
        except Exception:  # a broken signal must not sink the whole verdict
            continue
        for pid, contribution in outcome.items():
            per_participant.setdefault(pid, {})[signal.__name__] = contribution
    return per_participant
