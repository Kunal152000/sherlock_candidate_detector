"""LLM signal: OpenRouter (OpenAI-compatible SDK) with a heuristic fallback.

This is the one signal that does network I/O, so it lives here rather than in
``signals.py`` (which stays pure and offline-testable). It exposes the exact
same contract as every other signal::

    llm_signal(meeting: Meeting) -> dict[participant_id, (score, reason)]

Flow:

* If an OpenRouter key is configured *and* the ``openai`` SDK imports, we ask a
  chat model which participant is the interview candidate and why, then hand the
  chosen participant a positive score carrying the model's own explanation.
* Otherwise -- no key, SDK missing, network error, timeout, or an unparseable
  reply -- we fall back to :func:`_heuristic`, a *deterministic* pure-Python
  guess. That guarantees the signal always returns something sensible and keeps
  the whole system working (and tests passing) fully offline.

Config is read from :mod:`app.config` if that phase exists yet, otherwise from
environment variables, so this module is import-safe on its own.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from .models import Meeting
from .signals import (
    SignalResult,
    _INTRO_RE,
    _email_local,
    _looks_like_device,
    _ratio,
)

logger = logging.getLogger(__name__)

# Stay importable even if the SDK isn't installed; we simply fall back.
try:  # pragma: no cover - trivial import guard
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Configuration (prefer app.config.settings, fall back to env vars)
# --------------------------------------------------------------------------- #

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
_DEFAULT_TIMEOUT = 8.0

# Keep prompts small: cap how much transcript per participant we send.
_MAX_TRANSCRIPT_CHARS = 800


def _cfg(attr: str, env: str, default: Optional[str] = None) -> Optional[str]:
    """Read a setting from ``app.config.settings`` if present, else the env.

    ``config.py`` is a later phase; until it lands we transparently use the
    environment (matching ``.env.example``). Empty strings are treated as unset.
    """
    try:  # config phase may not exist yet
        from .config import settings  # type: ignore

        val = getattr(settings, attr, None)
        if val not in (None, ""):
            return str(val)
    except Exception:
        pass
    val = os.getenv(env)
    return val if val not in (None, "") else default


def _api_key() -> Optional[str]:
    return _cfg("openrouter_api_key", "OPENROUTER_API_KEY")


def _base_url() -> str:
    return _cfg("openrouter_base_url", "OPENROUTER_BASE_URL", _DEFAULT_BASE_URL) or _DEFAULT_BASE_URL


def _model() -> str:
    return _cfg("llm_model", "LLM_MODEL", _DEFAULT_MODEL) or _DEFAULT_MODEL


def _timeout() -> float:
    raw = _cfg("llm_timeout_seconds", "LLM_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT))
    try:
        val = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT
    return val if val > 0 else _DEFAULT_TIMEOUT


# --------------------------------------------------------------------------- #
# Public signal
# --------------------------------------------------------------------------- #


def llm_signal(meeting: Meeting) -> SignalResult:
    """Ask the LLM (or the heuristic fallback) which participant is the candidate.

    Returns the standard ``{participant_id: (score, reason)}`` mapping. The
    LLM's pick gets a strong positive score with the model's explanation; the
    rest get a neutral ``0.0`` so this signal only ever *promotes* one person and
    never fights the other signals over the losers.
    """
    if not meeting.participants:
        return {}

    verdict = _query_llm(meeting)
    if verdict is None:
        return _heuristic(meeting)

    candidate_id, reason = verdict
    reason = reason.strip() or "Selected by the LLM as the interview candidate"
    results: SignalResult = {}
    for pid in meeting.participants:
        if pid == candidate_id:
            results[pid] = (0.9, f"LLM: {reason}")
        else:
            results[pid] = (0.0, "Not selected by the LLM")
    return results


# --------------------------------------------------------------------------- #
# OpenRouter call
# --------------------------------------------------------------------------- #


def _query_llm(meeting: Meeting) -> Optional[tuple[str, str]]:
    """Call OpenRouter and return ``(candidate_id, reason)``, or None to fall back.

    Any problem -- missing SDK/key, network error, timeout, malformed reply, or
    an id the model invented -- returns None so the caller uses the heuristic.
    """
    if OpenAI is None:
        return None
    key = _api_key()
    if not key:
        return None

    try:
        client = OpenAI(api_key=key, base_url=_base_url())
        response = client.chat.completions.create(
            model=_model(),
            timeout=_timeout(),
            temperature=0,  # deterministic-ish; this is a classification task
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(meeting)},
            ],
            # OpenRouter forwards this for OpenAI-compatible providers; the free
            # models mostly honour it, and we parse defensively regardless.
            extra_headers={
                "HTTP-Referer": "https://github.com/sherlock/candidate-detector",
                "X-Title": "Sherlock Candidate Detector",
            },
        )
        content = response.choices[0].message.content if response.choices else None
    except Exception as exc:  # network / auth / timeout / provider error
        logger.warning("LLM signal call failed (%s); using heuristic fallback", exc)
        return None

    return _parse_reply(content, set(meeting.participants))


def _parse_reply(content: Optional[str], valid_ids: set[str]) -> Optional[tuple[str, str]]:
    """Extract ``(candidate_id, reason)`` from a model reply, tolerantly.

    Accepts a bare JSON object or JSON embedded in prose. Requires the returned
    id to be a real participant; otherwise returns None to trigger the fallback.
    """
    if not content:
        return None

    obj = None
    try:
        obj = json.loads(content)
    except Exception:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return None

    candidate_id = str(obj.get("candidate_id") or obj.get("id") or "").strip()
    reason = str(obj.get("reason") or "").strip()
    if candidate_id in valid_ids:
        return candidate_id, reason
    return None


_SYSTEM_PROMPT = (
    "You are an assistant that identifies which meeting participant is the "
    "interview CANDIDATE (the person being interviewed), as opposed to "
    "interviewers, hosts, or silent observers. Display names may be misleading "
    "(device names like 'MacBook Pro', nicknames, or a name the interviewer "
    "typed incorrectly), so weigh behaviour -- self-introductions, answering "
    "(not asking) questions, and substantial speaking time -- alongside the "
    "provided candidate metadata. Reply with ONLY a JSON object of the form "
    '{"candidate_id": "<id>", "reason": "<short justification>"} and nothing '
    "else. The candidate_id MUST be one of the provided participant ids."
)


def _build_user_prompt(meeting: Meeting) -> str:
    """Serialise interview metadata + participant snapshots into a JSON prompt."""
    participants = []
    for pid, p in meeting.participants.items():
        transcript = p.full_transcript
        if len(transcript) > _MAX_TRANSCRIPT_CHARS:
            transcript = transcript[-_MAX_TRANSCRIPT_CHARS:]
        participants.append(
            {
                "id": pid,
                "display_name": p.display_name,
                "email": p.email,
                "is_present": p.is_present,
                "camera_on": p.camera_on,
                "speaking_seconds": round(p.speaking_seconds, 1),
                "join_order": p.join_order,
                "transcript": transcript,
            }
        )

    payload = {
        "interview": {
            "candidate_name": meeting.candidate_name,
            "candidate_email": meeting.candidate_email,
            "interviewer_names": meeting.interviewer_names,
            "interviewer_emails": meeting.interviewer_emails,
        },
        "participants": participants,
    }
    return (
        "Identify the interview candidate from the meeting below.\n\n"
        + json.dumps(payload, ensure_ascii=False, default=str)
    )


# --------------------------------------------------------------------------- #
# Deterministic heuristic fallback (pure, offline)
# --------------------------------------------------------------------------- #


def _heuristic(meeting: Meeting) -> SignalResult:
    """A dependency-free stand-in for the LLM when it's unavailable.

    Combines a few strong cues (candidate-name match, self-introduction,
    speaking share, camera) into a per-participant opinion. Fully deterministic,
    so tests and the offline demo behave identically every run.
    """
    results: SignalResult = {}

    present = [p for p in meeting.participants.values() if p.is_present]
    pool = present or list(meeting.participants.values())
    total_speaking = sum(p.speaking_seconds for p in pool)
    multi = len(pool) > 1

    cand_name = (meeting.candidate_name or "").strip()
    cand_local = _email_local(meeting.candidate_email)

    for pid, p in meeting.participants.items():
        score = 0.0
        cues: list[str] = []

        name = (p.display_name or "").strip()
        if name and not _looks_like_device(name):
            name_match = _ratio(name, cand_name)
            if cand_local:
                name_match = max(name_match, _ratio(name, cand_local))
            if name_match >= 0.6:
                score += 0.4
                cues.append("name matches the candidate")

        if _INTRO_RE.search(p.full_transcript):
            score += 0.3
            cues.append("self-introduced")

        share = p.speaking_seconds / total_speaking if total_speaking else 0.0
        if share >= 0.15:
            score += min(0.3, share)
            cues.append(f"{share:.0%} of speaking time")
        elif multi and 0 < share < 0.05:
            score -= 0.3
            cues.append("barely speaks (observer-like)")

        if p.camera_on:
            score += 0.1
            cues.append("camera on")

        score = max(-1.0, min(1.0, score))
        detail = ", ".join(cues) if cues else "no strong cues"
        results[pid] = (score, f"Heuristic (LLM offline): {detail}")

    return results
